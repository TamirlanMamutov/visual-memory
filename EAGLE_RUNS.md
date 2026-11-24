# EAGLE VQ2D runs: commands and commits

The commands to reproduce each EAGLE run on Ego4D VQ2D val, with the git commit each one used.

Run everything from the `visual-memory` folder with the project's Python environment active (see REPRODUCTION.md for setup and data). Our runs used `..\cs230-venv\env\Scripts\python.exe`; below it is written as `python`.

Each run writes to `output\eagle-<start time>\`, which holds `config.json`, `log.txt`, `history.jsonl`, `metrics.json` and `predictions.json`. Any interrupted run continues exactly with `--resume`:

```bash
python run_eagle.py --stage all --resume output\eagle-<start time>
```

Resume runs whatever code is checked out, so keep the same commit while a run is in progress.

All runs use the defaults: 25,000 iterations, batch 4, sequence length 4, LR 2.5e-3, seed 230, DINOv2-base and SAM ViT-B. Times are Pacific.

## All runs at a glance

One row per run or paper result across the three tasks: Ego4D VQ2D, the multimodal localization prototype and EPIC-KITCHENS-100 retrieval. Each metric has its own column, and a dash means the metric doesn't apply or wasn't measured. Evaluation sets differ, so compare rows only when Task and Evaluated on match. The sections further down give the commands and details.

- **Code:** the commit the run used. When a run started before its code was committed, the commit that later recorded that code is in brackets.
- **Init:** where the weights came from.
- **Change:** what differs from the run it was built on.
- **Metrics:** VQ2D metrics use the official Ego4D setting, which scores every query (`vc_rt_pairings = {"all": ("all", "all")}`, the default in `EGO4D/episodic-memory`'s `evaluate_vq.py`). Before Nov 24 our numbers came from VQLoC's copy of the evaluator, whose default `("all", "large")` silently keeps only the 2,336 of 4,504 val queries whose ground-truth box reaches 192² px. Those large-only numbers were higher (run 6: stAP25 0.531 vs 0.423) and are no longer used here; `vq_metrics(..., pairing=VQLOC_PAIRING)` reproduces them. Paper rows are as each paper reports them: VQLoC's code uses the large-only filter, PRVQL's and REN's score all queries, and EAGLE and RELOCATE released no evaluation code, so their setting is unknown. Frame mIoU is the mean box IoU on frames that contain the object, and presence F1 scores whether each frame is predicted to contain it. Retrieval mAP and nDCG are in percent; V→T is video→text, T→V is text→video. "Dual" columns score the text–video encoder alone, and "Rerank" columns add TempRet's reranking stage on top of it. The TempRet paper reports a single final score, listed under Dual.
- **Multimodal localization prototype:** from the teammate's branch `tamirlan/multimodal-localization-model`. It is scored on 27 VQ2D val queries from 8 videos, with sparsely sampled frames, so its frame mIoU and F1 compare only roughly with ours on 4,501 queries.
- **Old mAP:** these rows were scored before the official EK-100 mAP fix (`52f857e`), so their mAP is understated; nDCG is unaffected.

| Task | Run | Code | Change | Init | Evaluated on | tAP25 | stAP25 | tAP | stAP | Succ | Rec | Frame mIoU | Presence F1 | Dual mAP V→T | Dual mAP T→V | Dual mAP avg | Dual nDCG V→T | Dual nDCG T→V | Dual nDCG avg | Rerank mAP V→T | Rerank mAP T→V | Rerank mAP avg | Rerank nDCG V→T | Rerank nDCG T→V | Rerank nDCG avg |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| VQ2D | EAGLE paper | — | — | DINOv2 fine-tuned on EgoTracks, then frozen; segmentation trained on Ego4D, EgoTracks and VISOR | val | 0.47 | 0.42 | — | — | 61.29 | 52.09 | — | — | — | — | — | — | — | — | — | — | — | — | — | — |
| VQ2D | REN paper | — | — | frozen DINOv2 ViT-L/14 with REN region tokens and SAM 2; no VQ2D training | test server | 0.52 | 0.40 | — | — | 61.2 | 49.3 | — | — | — | — | — | — | — | — | — | — | — | — | — | — |
| VQ2D | RELOCATE (from EAGLE's Table 1) | — | — | SAM-based region tokens; no VQ2D training | val | 0.41 | 0.33 | — | — | 58.03 | 50.50 | — | — | — | — | — | — | — | — | — | — | — | — | — | — |
| VQ2D | 1. Baseline `eagle-20261110-084401` | train `2125dd1`, eval `1f0a52c` | first full run: VQ2D only, after the mask-decoder collapse fix; per-frame BatchNorm at evaluation | DINOv2-base pretrained and frozen; EAGLE memory and decoder random; SAM ViT-B only makes training pseudo-masks | val, 4,501 queries | 0.318 | 0.229 | 0.179 | 0.097 | 52.32 | 36.72 | 0.312 | 0.452 | — | — | — | — | — | — | — | — | — | — | — | — |
| VQ2D | 2. EgoTracks `eagle-20261111-221053` | `7d14dbd` | 1 + EgoTracks long-term tracks in training | as 1 | val, 4,501 queries | 0.380 | 0.281 | 0.223 | 0.124 | 56.10 | 41.18 | 0.351 | 0.490 | — | — | — | — | — | — | — | — | — | — | — | — |
| VQ2D | 3. Fine-tuned 4 blocks `eagle-20261114-002928` | `120cd72` | 2 + training the last 4 DINOv2 blocks (LR 2.5e-5) | as 1, with the last 4 blocks trainable | val, 4,501 queries | 0.374 | 0.301 | 0.220 | 0.138 | 56.81 | 40.57 | 0.345 | 0.511 | — | — | — | — | — | — | — | — | — | — | — | — |
| VQ2D | 4. Run 2 + SAM 2 | `8ed086d` | 2 + SAM 2 `extend` box refinement, margin 10; no training | run 2's weights + SAM 2.1 Hiera Small | val clips 0–350, 1,348 queries | 0.437 | 0.353 | 0.308 | 0.190 | 58.68 | 52.41 | 0.465 | 0.529 | — | — | — | — | — | — | — | — | — | — | — | — |
| VQ2D | 5. VISOR `eagle-20261114-204835` | `aee6859` | 2 + VISOR (mix: VQ2D 50%, EgoTracks 25%, VISOR 25%) | as 1 | val, 4,501 queries | 0.354 | 0.258 | 0.204 | 0.113 | 54.12 | 40.03 | 0.343 | 0.476 | — | — | — | — | — | — | — | — | — | — | — | — |
| VQ2D | 6. Ensemble `ens3-mask0` + SAM 2 | `422f0c0` | ensemble of 2, 3 and 5, visual-crop constraint, SAM 2 `extend` with margin 30; no training | runs 2, 3 and 5 + SAM 2.1 Hiera Small | val, 4,501 queries | **0.503** | **0.423** | **0.371** | **0.229** | **64.12** | **58.07** | **0.518** | **0.586** | — | — | — | — | — | — | — | — | — | — | — | — |
| Multimodal localization | Temporal prototype | `9c47669` | first real Ego4D run: 117 training queries, 10 epochs, checkpoint chosen by val loss | frozen DINOv2 ViT-S/14 at 224 px (official weights); heads random | VQ2D val, 27 queries | — | — | — | — | — | — | 0.006 | 0.043 | — | — | — | — | — | — | — | — | — | — | — | — |
| Multimodal localization | Spatial matcher, Ego4D only | `dfce151` | per-frame spatial matcher; checkpoint chosen by IoU | as above | VQ2D val, 27 queries | — | — | — | — | — | — | 0.089 | 0.059 | — | — | — | — | — | — | — | — | — | — | — | — |
| Multimodal localization | Spatial matcher + EPIC (joint) | `dfce151` | + EPIC noun classification as an auxiliary task (31 segments, weight 0.1) | as above | VQ2D val, 27 queries | — | — | — | — | — | — | 0.089 | 0.059 | — | — | — | — | — | — | — | — | — | — | — | — |
| Multimodal localization | Assigned boxes, 20 epochs | `7f0eb48` | longer training with assigned box targets; 96/21 fitting/development split | as above | VQ2D val, 27 queries | — | — | — | — | — | — | 0.081 | 0.066 | — | — | — | — | — | — | — | — | — | — | — | — |
| Multimodal localization | Box prior, 43 epochs | `7f0eb48` | + a box-size prior from the fitting set | as above | VQ2D val, 27 queries | — | — | — | — | — | — | 0.073 | 0.082 | — | — | — | — | — | — | — | — | — | — | — | — |
| Retrieval | TempRet paper | — | — | — | EK-100 test server | — | — | — | — | — | — | — | — | — | — | 67.97 | — | — | 82.92 | — | — | — | — | — | — |
| Retrieval | JPoSE, pretrained (`epic-baseline-*`, 6 identical runs) | `1d0fd38` (first two runs before it) | evaluation only | downloaded JPoSE checkpoint | EPIC-100 full val, JPoSE's own scorer | — | — | — | — | — | — | — | — | 75.7 | 71.2 | 73.4 | 70.7 | 67.4 | 69.0 | — | — | — | — | — | — |
| Retrieval | JPoSE, 1 epoch `epic-baseline-20261107-000802` | before `1d0fd38` | trained for 1 epoch | random | EPIC-100 full val, JPoSE's own scorer | — | — | — | — | — | — | — | — | 52.4 | 43.3 | 47.8 | 55.5 | 51.7 | 53.6 | — | — | — | — | — | — |
| Retrieval | TempRet `005058` | before `6a873d0` | first run: ViT-B/16 frozen, 2 epochs, 320 training clips | OpenAI CLIP ViT-B/16, frozen; temporal module and reranker random | local subset, 160 test clips; old mAP | — | — | — | — | — | — | — | — | 12.3 | 14.5 | 13.4 | 32.7 | 30.9 | 31.8 | 6.2 | 5.6 | 5.9 | 26.7 | 20.7 | 23.7 |
| Retrieval | TempRet `020438` | before `6a873d0` | ViT-L/14 frozen, 10 epochs, 4,881 training clips | OpenAI CLIP ViT-L/14, frozen; temporal module and reranker random | local subset, 1,256 test clips; old mAP | — | — | — | — | — | — | — | — | 15.5 | 15.6 | 15.6 | 33.3 | 33.2 | 33.3 | 5.2 | 5.0 | 5.1 | 22.8 | 23.2 | 23.0 |
| Retrieval | TempRet `100605` | before `6a873d0` | reranker retrained only | `020438` checkpoint | local subset, 1,256 test clips; old mAP | — | — | — | — | — | — | — | — | 15.5 | 15.6 | 15.6 | 33.3 | 33.2 | 33.3 | 16.1 | 16.6 | 16.3 | 33.9 | 33.9 | 33.9 |
| Retrieval | TempRet `130804` | `6a873d0` (+ `52f857e`) | official EK-100 mAP; no encoder training | OpenAI CLIP ViT-L/14 with an untrained temporal module | local subset, 1,256 test clips | — | — | — | — | — | — | — | — | 22.9 | 22.2 | 22.5 | 28.8 | 28.5 | 28.7 | — | — | — | — | — | — |
| Retrieval | **TempRet `130858`** | `6a873d0` (+ `52f857e`) | `100605` re-scored with the official mAP | `100605` checkpoint | local subset, 1,256 test clips | — | — | — | — | — | — | — | — | 30.4 | 29.8 | 30.1 | 33.3 | 33.2 | 33.3 | **31.1** | **30.9** | **31.0** | **33.9** | **33.9** | **33.9** |
| Retrieval | TempRet `131038` | `6a873d0` (+ `52f857e`) | 10 epochs on 1,220 training clips; no reranker | OpenAI CLIP ViT-L/14, frozen | local subset, 1,256 test clips | — | — | — | — | — | — | — | — | 29.4 | 26.7 | 28.1 | 33.1 | 32.2 | 32.6 | — | — | — | — | — | — |
| Retrieval | TempRet `131140` | `6a873d0` (+ `52f857e`) | 10 epochs on 2,440 training clips; no reranker | as above | local subset, 1,256 test clips | — | — | — | — | — | — | — | — | 30.2 | 28.2 | 29.2 | 33.3 | 32.9 | 33.1 | — | — | — | — | — | — |
| Retrieval | TempRet `131307` | `6a873d0` (+ `52f857e`) | 10 epochs on 4,881 clips without the temporal module; no reranker | as above | local subset, 1,256 test clips | — | — | — | — | — | — | — | — | 27.7 | 26.3 | 27.0 | 32.9 | 31.9 | 32.4 | — | — | — | — | — | — |
| Retrieval | TempRet `162636` (smoke test) | `2d5ddc3` (+ `11744e1`) | full fine-tuning of CLIP, 1 epoch, 8-bit AdamW, 640 training clips | OpenAI CLIP ViT-L/14, all layers trained | local subset, 128 test clips | — | — | — | — | — | — | — | — | 34.4 | 38.9 | 36.6 | 40.4 | 37.9 | 39.2 | 33.9 | 38.9 | 36.4 | 39.1 | 37.5 | 38.3 |

Not in the table, to keep one row per run: EAGLE on the test server scores tAP25 0.46, stAP25 0.40, Succ 62.70 and Rec 53.51, the fair comparison for REN. Run 2 scored on clips 0–350 only, the subset run 4 used, gives tAP25 0.403, stAP25 0.308, tAP 0.243, stAP 0.138, Succ 57.64 and Rec 42.77. `100605` reused the `020438` encoder unchanged, so its Dual columns repeat `020438`'s.

## 1. Baseline (VQ2D only): `eagle-20261110-084401`

Trained Nov 10 08:44–13:30; evaluated until Nov 11 02:13.

- **Training: commit `2125dd1`.** The run started 4 minutes before this commit, from the same code.
  ```bash
  git checkout 2125dd1
  python run_eagle.py --stage all
  ```
- **Evaluation: commit `1f0a52c`,** which uses per-frame BatchNorm statistics. The first evaluation, done with `2125dd1`, was discarded. The finished checkpoint was then evaluated again:
  ```bash
  git checkout 1f0a52c
  python run_eagle.py --stage evaluate --resume output\eagle-20261110-084401
  ```

VISOR wasn't on disk yet, so this run trained on VQ2D only.

## 2. VQ2D + EgoTracks: `eagle-20261111-221053`

Started Nov 11 22:10 and finished Nov 13 10:13, at commit `7d14dbd`.

```bash
git checkout 7d14dbd
python run_eagle.py --stage all
```

It was resumed three times with `--resume output\eagle-20261111-221053`. VISOR wasn't on disk yet.

## 3. Fine-tuning the last 4 DINOv2 blocks: `eagle-20261114-002928`

Started Nov 14 00:29 and finished Nov 14 19:56, at commit `120cd72`. Trained on VQ2D + EgoTracks.

```bash
git checkout 120cd72
python run_eagle.py --stage all --finetune-blocks 4 --no-visor
```

It was stopped once and resumed with `--resume output\eagle-20261114-002928`. The backbone LR was the default 2.5e-5.

## 4. SAM 2 refinement of run 2 (no retraining): `eagle-20261111-221053\sam2\small-m10`

Run on Nov 14 01:07–04:23, over the first 351 val clips. The code is now in commit `8ed086d`.

```bash
git checkout 8ed086d
python eagle_sam2.py output\eagle-20261111-221053 --clips 0:351 --model facebook/sam2.1-hiera-small --margin 10 --tag small-m10
```

We passed a local download of `facebook/sam2.1-hiera-small` as `--model`; the Hugging Face ID shown above gives the same weights. The script reports three variants: `replace`, `trim` and `extend`. The `extend` variant gave the best results.

## 5. VQ2D + EgoTracks + VISOR: `eagle-20261114-204835`

Started Nov 14 20:48 at commit `aee6859`. Training finished Nov 15 01:49 and evaluation Nov 15 14:27, with no resumes. The training mix is VQ2D 50%, EgoTracks 25%, VISOR 25% (`--mix 2,1,1`, the default).

```bash
git checkout aee6859
python run_eagle.py --stage all
```

VISOR must be unzipped at `..\EPIC-KITCHENS-VISOR\GroundTruth-SparseAnnotations`.

## 6. Ensemble of runs 2, 3, 5 + visual-crop constraint + SAM 2: `ens3-mask0`

Run on Nov 19 with no retraining, from the per-frame traces that runs 2, 3 and 5 saved during evaluation.

```bash
python eagle_ensemble.py output\eagle-20261111-221053 output\eagle-20261114-002928 output\eagle-20261114-204835 --out output\ens3-mask0
python eagle_sam2.py output\ens3-mask0 --clips 0: --model facebook/sam2.1-hiera-small --margin 30 --tag small-m30
```

- **Trace ensembling:** the three runs' per-frame confidences are averaged with equal weights, and each frame keeps the box of the most confident run. The paper's default post-processing rule then builds the track. Averaging removes many wrong time windows: Success, which ignores track scores, rises from 56.10 for run 2 to 62.14.
- **Visual-crop constraint:** the crop is a sighting of the object, and the response track is the *last* occurrence before the query. When the crop's frame comes before the query frame, frames up to the crop frame get confidence 0. This uses only the query input. It applies to 42% of val queries.
- **SAM 2 `extend`:** the same refinement as run 4, with a margin of 30 frames instead of 10.

No setting was tuned on val. Equal weights, crop margin 0, the paper's post-processing and SAM 2 margin 30 were all fixed in advance. A sweep on clips 351+ found no setting that clearly beat them on clips 0–350; that sweep was scored with VQLoC's large-only filter and has not been rerun with the official setting.

Full val ablation, official setting (all 4,501 queries):

| Step | tAP25 | stAP25 | tAP | stAP | Succ | Rec |
|---|---|---|---|---|---|---|
| Run 2 alone | 0.380 | 0.281 | 0.223 | 0.124 | 56.10 | 41.18 |
| Run 2 + crop constraint | 0.392 | 0.290 | 0.234 | 0.128 | 56.34 | 41.33 |
| Ensemble of 2 + 3 + 5 | 0.446 | 0.355 | 0.270 | 0.163 | 62.14 | 45.37 |
| Ensemble + crop constraint | 0.469 | 0.370 | 0.287 | 0.171 | 62.74 | 45.59 |
| + SAM 2 `extend`, margin 30 | 0.503 | 0.423 | 0.371 | 0.229 | 64.12 | 58.07 |

Caveats: the ensemble runs three EAGLE models plus SAM 2, about 3× the inference cost of one EAGLE model. These are val numbers, and the test set has not been scored.

## Results on Ego4D VQ2D val

Our rows use the official Ego4D setting (every query). Paper rows are as reported; which setting EAGLE used is unknown (see the Metrics note at the top).

| Run | tAP25 | stAP25 | tAP | stAP | Succ | Rec |
|---|---|---|---|---|---|---|
| EAGLE paper (val) | 0.47 | 0.42 | — | — | 61.29 | 52.09 |
| REN paper (test) | 0.52 | 0.40 | — | — | 61.2 | 49.3 |
| 1. Baseline | 0.318 | 0.229 | 0.179 | 0.097 | 52.32 | 36.72 |
| 2. EgoTracks | 0.380 | 0.281 | 0.223 | 0.124 | 56.10 | 41.18 |
| 3. Fine-tuned 4 blocks | 0.374 | 0.301 | 0.220 | 0.138 | 56.81 | 40.57 |
| 2 on clips 0–350 only | 0.403 | 0.308 | 0.243 | 0.138 | 57.64 | 42.77 |
| 4. 2 + SAM 2 `extend`, clips 0–350 | 0.437 | 0.353 | 0.308 | 0.190 | 58.68 | 52.41 |
| 5. VISOR | 0.354 | 0.258 | 0.204 | 0.113 | 54.12 | 40.03 |
| 5 on clips 0–350 only | 0.370 | 0.278 | 0.215 | 0.126 | 55.79 | 41.39 |
| 6. Ensemble 2+3+5 + crop constraint | 0.469 | 0.370 | 0.287 | 0.171 | 62.74 | 45.59 |
| **6. + SAM 2 `extend`, margin 30** | **0.503** | **0.423** | **0.371** | **0.229** | **64.12** | **58.07** |

All full-val rows score 4,501 queries; the clips 0–350 rows score 1,348.

With the official setting, run 3 has a higher stAP25 than run 2 (0.301 vs 0.281); under the large-only filter it was the other way round. Run 2 still has the higher tAP25.

Adding VISOR (run 5) lowered every metric compared with run 2, on both the full val set and clips 0–350. A likely cause is that VISOR's 25% share cut EgoTracks from about 50% of the training mix to 25%. Training metrics didn't predict this: run 5 was ahead of run 2 throughout training.

## Text–video retrieval: EPIC-KITCHENS-100 Multi-Instance Retrieval

This section covers a different task, with different metrics: mAP and nDCG, in each direction (video→text, text→video) and averaged. Higher is better. The rows below are on three different test sets, so only rows in the same group compare directly.

### Papers, on the EK-100 test server

| Method | mAP avg | nDCG avg |
|---|---|---|
| TempRet paper (arXiv 2605.24470) | 67.97 | 82.92 |

### JPoSE baseline, on the full EPIC-100 validation set
Scored by JPoSE's own evaluation code, which may define mAP differently from the official EK-100 code.

| Run | Started | mAP V→T / T→V / avg | nDCG V→T / T→V / avg |
|---|---|---|---|
| Downloaded pretrained JPoSE (`epic-baseline-*`, 6 identical runs) | Nov 7–8 | 75.7 / 71.2 / **73.4** | 70.7 / 67.4 / **69.0** |
| JPoSE trained from scratch, 1 epoch (`epic-baseline-20261107-000802`) | Nov 7 00:08 | 52.4 / 43.3 / 47.8 | 55.5 / 51.7 / 53.6 |

The pretrained result matches the JPoSE README. The one-epoch model started from random weights, so it is only a sanity check.

### TempRet re-implementation, on a local subset
Unless the Setup column says otherwise, these runs use 4,881 training clips, and 1,256 test clips with 1,351 sentences; the full benchmark has 67,217 training and 9,668 test clips. "Dual" is the text–video encoder alone; "+ rerank" adds TempRet's reranking stage. All runs use 8 frames per clip and the OpenAI CLIP weights.

The Code column gives the latest commit before the run started. Several runs used changes that were committed only later; those commits are in brackets.

| Run | Started (Nov 9) | Code | Setup | Dual mAP V→T / T→V / avg | Dual nDCG avg | + rerank mAP avg | + rerank nDCG avg |
|---|---|---|---|---|---|---|---|
| `tempret-20261109-130858` | 13:08 | 6a873d0 (+ mAP fix, `52f857e`) | ViT-L/14 frozen, 10 epochs; the `100605` checkpoint re-scored | 30.4 / 29.8 / **30.1** | 33.3 | **31.0** | **33.9** |
| `tempret-20261109-131140` | 13:11 | 6a873d0 (+ `52f857e`) | ViT-L/14 frozen, 10 epochs, 2,440 training clips | 30.2 / 28.2 / 29.2 | 33.1 | — | — |
| `tempret-20261109-131038` | 13:10 | 6a873d0 (+ `52f857e`) | ViT-L/14 frozen, 10 epochs, 1,220 training clips | 29.4 / 26.7 / 28.1 | 32.6 | — | — |
| `tempret-20261109-131307` | 13:13 | 6a873d0 (+ `52f857e`) | ViT-L/14 frozen, 10 epochs, no temporal module | 27.7 / 26.3 / 27.0 | 32.4 | — | — |
| `tempret-20261109-130804` | 13:08 | 6a873d0 (+ `52f857e`) | ViT-L/14 frozen, no training | 22.9 / 22.2 / 22.5 | 28.7 | — | — |
| `tempret-20261109-162636` (smoke test) | 16:26 | 2d5ddc3 (+ `11744e1`) | full fine-tuning, 1 epoch, only 640 training and 128 test clips | 34.4 / 38.9 / 36.6 | 39.2 | 36.4 | 38.3 |

Runs from before the mAP fix: their mAP is understated, but nDCG is unaffected.

| Run | Started (Nov 9) | Code | Setup | Dual mAP / nDCG avg | + rerank mAP / nDCG avg |
|---|---|---|---|---|---|
| `tempret-20261109-005058` | 00:50 | before 6a873d0, uncommitted | ViT-B/16 frozen, 2 epochs, 320 training and 160 test clips | 13.4 / 31.8 | 5.9 / 23.7 |
| `tempret-20261109-020438` | 02:04 | before 6a873d0, uncommitted | ViT-L/14 frozen, 10 epochs | 15.6 / 33.3 | 5.1 / 23.0 |
| `tempret-20261109-100605` | 10:06 | before 6a873d0, uncommitted | the `020438` checkpoint, with the reranker retrained | 15.6 / 33.3 | 16.3 / 33.9 |

The best fair TempRet result is `130858`: 31.0 mAP and 33.9 nDCG, on the local subset. That is far below the paper's 67.97 and 82.92, which were scored on the test server, so the two aren't directly comparable. The `162636` smoke test is too small to mean anything. There are no full-benchmark TempRet runs yet.
