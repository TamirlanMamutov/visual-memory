# TempRet and EAGLE re-implementations

Neither paper releases a usable training recipe in full, so both pipelines are **re-implementations from the paper text**. Every place where the paper is silent is listed under *Deviations / assumptions*; treat results as "our TempRet" / "our EAGLE", not as the authors' numbers.

## Environment

GPU training uses a separate environment outside Google Drive (Python 3.11 with CUDA PyTorch; Python 3.15 has no CUDA wheels):

```powershell
& 'C:\Users\<you>\AppData\Local\Programs\Python\Python311\python.exe' -m venv E:\cs230-venv\env
E:\cs230-venv\env\Scripts\python.exe -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128
E:\cs230-venv\env\Scripts\python.exe -m pip install -r requirements-gpu.txt
```

The launcher uses `<drive root>\cs230-venv\env` automatically (override with `CS230_PYTHON`). Decoded frames, feature caches and checkpoints go to `<drive root>\cs230-cache` (override with `CS230_CACHE`) so tens of GB are not synced to Drive. Metrics and logs go to `visual-memory/output/<run>/`. The first run downloads CLIP / DINOv2 / SAM weights from their official hubs.

## TempRet: EK-100 Multi-Instance Retrieval (arXiv 2605.24470)

```powershell
# decode 16 frames/clip from the raw MP4s + cache CLIP features (resumable)
E:\cs230-venv\env\Scripts\python.exe run_tempret.py --extract-only
# zero-shot CLIP baseline (mean pooling)
E:\cs230-venv\env\Scripts\python.exe run_tempret.py --epochs 0
# train + evaluate (dual encoder each epoch, two-stage reranking at the end)
E:\cs230-venv\env\Scripts\python.exe run_tempret.py --epochs 10
# paper ablations
E:\cs230-venv\env\Scripts\python.exe run_tempret.py --epochs 10 --no-temporal
E:\cs230-venv\env\Scripts\python.exe run_tempret.py --epochs 10 --no-rerank
# (re)train only the reranker on top of a trained first stage (~3 min)
E:\cs230-venv\env\Scripts\python.exe run_tempret.py --epochs 0 --rerank-only --checkpoint E:\cs230-cache	empret\checkpoints\<run>est.pt
```

Reranker training. Training the cross-encoder jointly inside the 10 dual-encoder epochs (`--reranker-epochs 0`, the first version) left it near chance (ITM loss 0.50 vs 0.69 for guessing). With the paper's alpha = 0.002 its scores then override the first stage, and reranking dropped ViT-L/14 from 15.6 to 5.1 mAP (old metric; nDCG 33.3 to 23.0) on the local subset. The reranker is therefore trained in its own stage after the first stage:

- the first stage is frozen;
- it trains for 40 epochs, each anchor clip getting its positive, 3 hard negatives per direction drawn from the top 32 first-stage hits over the whole training set, and 1 random negative;
- the target is the soft EK-100 relevance (BCE; `--itm-target binary` restores the paper's binary ITM label);
- alpha is chosen from {0.002 ... 10} on held-out training videos (15 %) that the reranker never sees.

Caveat: the first stage did train on those held-out clips, so its held-out scores are optimistic, and alpha is biased towards the first stage. The test-set alpha sweep is logged as a diagnostic only.

Results on the P01 + P02 subset (1,256 test clips x 1,351 sentences; ViT-L/14, frozen image tower; best epoch picked on this same subset):

| Setting | train clips | mAP | nDCG |
|---|---|---|---|
| zero-shot CLIP, mean pooling | 0 | 22.51 | 28.67 |
| dual encoder | 1,220 | 28.07 | 32.63 |
| dual encoder | 2,440 | 29.20 | 33.09 |
| dual encoder, no temporal transformer | 4,881 | 27.01 | 32.42 |
| dual encoder | 4,881 | 30.09 | 33.28 |
| + joint ITM reranker, alpha 0.002 (first version) | 4,881 | not re-scored (5.10 with the old metric) | 22.97 |
| **+ separate reranker stage, alpha 10** | 4,881 | **30.99** | **33.89** |

Metric note: runs before 2026-11-09 13:00 computed mAP with binary precision@k, which under-reports mAP compared with the official code (the official `calculate_mAP` accumulates the soft relevance when computing precision). `tempret/metrics.py` now matches the official JPoSE / C5 functions exactly on random test matrices, and the launcher marks older runs as "pre-fix metric". nDCG was always identical to the official code. The relevance matrix is computed from the class annotations with the official formula. It has not been checked against the official `caption_relevancy_EPIC_100_retrieval_test.pkl`, which is not downloaded.

Implemented as described: CLIP ViT-L/14 at 224, per-frame CLS embedding after the visual projection; a 4-layer, 8-head pre-LN temporal transformer with learnable positional encodings, 4x GELU MLP, residual `X~ = T(X) + X` and masked mean pooling; EOS text embedding; Symmetric Multi-Similarity loss (m = 0.2, tau = 0.1) on the soft relevance; AdamW, lr 1.8e-5, batch 64, cosine schedule, 2x lr for temporal layers; Top-K = 1000 reranking with a cross-encoder + ITM head, row-minimum miss penalty, `S = norm(ITM) + 0.002 * norm(S_init)`. Relevance is the official `0.5 * verb IoU + 0.5 * noun-set IoU`; metrics follow the official mAP / nDCG code.

Deviations / assumptions:

| Item | Paper | Here |
|---|---|---|
| Embedding size | "D = 512" with ViT-L/14 | CLIP's own width (768 for ViT-L/14); 512 would break the pretrained alignment |
| Visual tower | fine-tuned | default `--mode frozen` (cached features; 12 GB GPU). `--mode full` fine-tunes both towers end to end at the paper's batch 64 via gradient caching (`--chunk-size 8`, exact full-batch SMS gradient, verified against an unchunked backward) and 8-bit AdamW (`--optimizer adamw8bit`): ViT-L/14, 8 frames, peak 6.4 GB, about 13.5 s/step on an RTX 4070 (about 4 h/epoch on all 67k clips) |
| Frames per clip | not stated | 8 (from 16 stored segment centres; random within segment during training) |
| SMS neutral threshold eps | not stated | 0.1 |
| Cross-encoder | not specified | 4-layer, 512-wide transformer over [CLS; temporal frame tokens; CLIP text tokens], trained jointly on detached features with in-batch hard negatives |
| Score normalisation | not specified | per-query min-max |
| Reranker training | jointly with the dual encoder, binary ITM | separate stage on the frozen first stage, soft relevance targets, global hard negatives (see above) |
| Fusion weight alpha | 0.002 | chosen on held-out training videos (`--alpha auto`); with only ~3.7k training clips the reranker is much weaker than the paper's, so the chosen alpha is 10 |
| Checkpoint selection | not stated | best epoch by dual-encoder test score (subset); **this uses the test set**, so treat the epoch choice as optimistic |
| Epochs, warm-up, weight decay | not stated | 10, 5 %, 0.01 |

Data: the evaluation uses only clips whose video is present. With P01 + P02 only, that is **1,334 of 9,668** test clips (and the sentences relevant to them), and 12,173 of 67,217 training clips. Results are labelled *subset* and are not comparable to the leaderboard. Full comparability needs the 138 validation videos (and ideally all 495 training videos).

## EAGLE: Ego4D VQ2D, 2D branch (arXiv 2511.08007)

```powershell
# SAM pseudo-masks for every local train query (resumable; rerun as clips download)
E:\cs230-venv\env\Scripts\python.exe run_eagle.py --stage prepare
# meta-train AMM + GLM + decoder (25k iterations)
E:\cs230-venv\env\Scripts\python.exe run_eagle.py --stage train
# online inference on local val clips, official VQ2D metrics
E:\cs230-venv\env\Scripts\python.exe run_eagle.py --stage evaluate --checkpoint E:\cs230-cache\eagle\checkpoints\<run>\last.pt
# ablations
E:\cs230-venv\env\Scripts\python.exe run_eagle.py --stage all --no-amm      # or --no-glm
# resume an interrupted run (any stage); its own settings are reused
E:\cs230-venv\env\Scripts\python.exe run_eagle.py --stage all --resume output\eagle-<run>
```

Interruptions: every stage resumes with `--resume <run folder>`.
- **Prepare** writes one file per query and skips finished ones.
- **Training** saves a resumable `last.pt` every 500 iterations (`--save-every`). It holds the weights, optimiser, learning-rate schedule, RNG states and the exact list of training files, which is also in `train_files.json`. A permanent `iter_N.pt` is kept every 2,500 iterations (`--keep-every`).
- **Evaluation** saves each clip's predictions to `eval_parts/` as it finishes, and skips those clips on restart.

All writes go through a temporary file and a rename, so an interruption never leaves a half-written file under its final name.

Implemented as described: frozen DINOv2 ViT-B/14 at 448 (long side + zero padding); SAM mask of the visual crop (box + centre-point prompt) as M0; AMM with pseudo-label modulator P, target re-weighting W, a K x K x C x D query-segment kernel optimised by steepest descent with learnable delta (Eq. 1-2; 10 initial / 3 update iterations, differentiable during training); GLM DCF with the mixed least-squares / hinge residual (Eq. 4) and Gauss-Newton step length; `F_N = F_A + kappa(H_J)` and a mask decoder; Lovasz loss + rho x DCF hinge loss; AdamW, peak lr 2.5e-3, wd 0.05, 25k iterations with 2.5k warm-up and linear decay. Inference: augmented query (vertical flip, translation, blur), memories of up to 50 samples, entry threshold s_conf >= 0.6, updates in the first 100 frames then every 25, static-snapshot fallback when the filter fails on >60 % of frames, reset after confidence stays < 0.4, 5-frame median filter, 0.8 x max peak, last interval above threshold, boxes from the mask bounding rectangle. Metrics come from the official VQ2D evaluation code in `../VQLoC`.

Deviations / assumptions:

| Item | Paper | Here |
|---|---|---|
| Backbone | DINOv2 fine-tuned on EgoTracks, then frozen | pretrained DINOv2, frozen (no EgoTracks fine-tuning yet) |
| Decoder, P/W nets, kernel sizes, D | not specified | LWL-style choices: 256-ch reduced features, D = 16, 3 x 3 sigma, 5 x 5 DCF, light RGB stem for boundaries |
| Tracking during training | frozen pretrained tracker | no pretrained tracker exists here; its feature reducer is trained with L_tck |
| rho, batch size, sequence length N | not stated | 1.0, 4 sequences, 4 frames |
| Training negatives | not mentioned | one object-free frame per VQ2D sequence, so confidence is low when the object is absent |
| Memory crops | crop-and-resize before insertion | full frames, which keeps query and search frames at the same scale |
| SAM model | not stated | `facebook/sam-vit-base` (change with `--sam facebook/sam-vit-huge`) |
| 3D (VGGT) branch | yes | not implemented |

Data: VQ2D clips go in `../Ego4D/v2/clips/` (5,823 clips, 346 GB for all splits; see `../Ego4D/download-plans`). VISOR (28.4 GiB, sparse annotations + RGB frames) goes in `../EPIC-KITCHENS-VISOR/` so that `GroundTruth-SparseAnnotations/annotations/train/*.json` and `.../rgb_frames/train/Pxx/*.zip` exist. It is used automatically when present. EgoTracks is only needed for the backbone fine-tuning step, which is not implemented.
