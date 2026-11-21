# EAGLE VQ2D runs: commands and commits

The commands to reproduce each EAGLE run on Ego4D VQ2D val, with the git commit each one used.

Run everything from the `visual-memory` folder with the project's Python environment active (see REPRODUCTION.md for setup and data). Our runs used `..\cs230-venv\env\Scripts\python.exe`; below it is written as `python`.

Each run writes to `output\eagle-<start time>\`, which holds `config.json`, `log.txt`, `history.jsonl`, `metrics.json` and `predictions.json`. Any interrupted run continues exactly with `--resume`:

```bash
python run_eagle.py --stage all --resume output\eagle-<start time>
```

Resume runs whatever code is checked out, so keep the same commit while a run is in progress.

All runs use the defaults: 25,000 iterations, batch 4, sequence length 4, LR 2.5e-3, seed 230, DINOv2-base and SAM ViT-B. Times are Pacific.

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

- **Trace ensembling:** the three runs' per-frame confidences are averaged with equal weights, and each frame keeps the box of the most confident run. The paper's default post-processing rule then builds the track. Averaging removes many wrong time windows: with every track given the same score, the ensemble still beats run 2 by 0.077 stAP25.
- **Visual-crop constraint:** the crop is a sighting of the object, and the response track is the *last* occurrence before the query. When the crop's frame comes before the query frame, frames up to the crop frame get confidence 0. This uses only the query input. It applies to 42% of val queries.
- **SAM 2 `extend`:** the same refinement as run 4, with a margin of 30 frames instead of 10.

No setting was tuned on val. Equal weights, crop margin 0, the paper's post-processing and SAM 2 margin 30 were all fixed in advance. A sweep on clips 351+ found no setting that clearly beat them on clips 0–350.

Full val ablation:

| Step | tAP25 | stAP25 | tAP | stAP | Succ | Rec |
|---|---|---|---|---|---|---|
| Run 2 alone | 0.472 | 0.386 | 0.283 | 0.179 | 64.51 | 46.35 |
| Run 2 + crop constraint | 0.477 | 0.390 | 0.291 | 0.181 | 64.51 | 46.52 |
| Ensemble of 2 + 3 + 5 | 0.540 | 0.464 | 0.332 | 0.222 | 70.46 | 49.21 |
| Ensemble + crop constraint | 0.553 | 0.473 | 0.344 | 0.229 | 70.85 | 49.39 |
| + SAM 2 `extend`, margin 30 | 0.598 | 0.531 | 0.448 | 0.301 | 71.70 | 62.07 |

Caveats: the ensemble runs three EAGLE models plus SAM 2, about 3× the inference cost of one EAGLE model. These are val numbers, and the test set has not been scored.

## Results on Ego4D VQ2D val

| Run | tAP25 | stAP25 | tAP | stAP | Succ | Rec |
|---|---|---|---|---|---|---|
| EAGLE paper (val) | 0.47 | 0.42 | — | — | 61.29 | 52.09 |
| REN paper (test) | 0.52 | 0.40 | — | — | 61.2 | 49.3 |
| 1. Baseline | 0.406 | 0.319 | 0.232 | 0.142 | 60.74 | 40.35 |
| 2. EgoTracks | 0.472 | 0.386 | 0.283 | 0.179 | 64.51 | 46.35 |
| 3. Fine-tuned 4 blocks | 0.445 | 0.378 | 0.265 | 0.180 | 62.89 | 43.58 |
| 2 on clips 0–350 only | 0.486 | 0.400 | 0.300 | 0.189 | 64.63 | 47.40 |
| 4. 2 + SAM 2 `extend`, clips 0–350 | 0.533 | 0.448 | 0.377 | 0.250 | 65.46 | 55.52 |
| 5. VISOR | 0.450 | 0.351 | 0.260 | 0.161 | 63.18 | 44.14 |
| 5 on clips 0–350 only | 0.456 | 0.364 | 0.266 | 0.173 | 62.69 | 44.69 |
| 6. Ensemble 2+3+5 + crop constraint | 0.553 | 0.473 | 0.344 | 0.229 | 70.85 | 49.39 |
| **6. + SAM 2 `extend`, margin 30** | **0.598** | **0.531** | **0.448** | **0.301** | **71.70** | **62.07** |

All full-val rows score 4,501 queries; the clips 0–350 rows score 1,348.

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
These runs use 4,881 training clips, and 1,256 test clips with 1,351 sentences; the full benchmark has 67,217 training and 9,668 test clips. "Dual" is the text–video encoder alone; "+ rerank" adds TempRet's reranking stage. All runs use 8 frames per clip and the OpenAI CLIP weights.

The Code column gives the latest commit before the run started. Several runs used changes that were committed only later; those commits are in brackets.

| Run | Started (Nov 9) | Code | Setup | Dual mAP V→T / T→V / avg | Dual nDCG avg | + rerank mAP avg | + rerank nDCG avg |
|---|---|---|---|---|---|---|---|
| `tempret-20261109-130858` | 13:08 | 6a873d0 (+ mAP fix, `52f857e`) | ViT-L/14 frozen, 10 epochs; the `100605` checkpoint re-scored | 30.4 / 29.8 / **30.1** | 33.3 | **31.0** | **33.9** |
| `tempret-20261109-131140` | 13:11 | 6a873d0 (+ `52f857e`) | ViT-L/14 frozen, 10 epochs, 2,440 training clips | 30.2 / 28.2 / 29.2 | 33.1 | — | — |
| `tempret-20261109-131038` | 13:10 | 6a873d0 (+ `52f857e`) | ViT-L/14 frozen, 10 epochs | 29.4 / 26.7 / 28.1 | 32.6 | — | — |
| `tempret-20261109-131307` | 13:13 | 6a873d0 (+ `52f857e`) | ViT-L/14 frozen, 10 epochs | 27.7 / 26.3 / 27.0 | 32.4 | — | — |
| `tempret-20261109-130804` | 13:08 | 6a873d0 (+ `52f857e`) | ViT-L/14 frozen, no training | 22.9 / 22.2 / 22.5 | 28.7 | — | — |
| `tempret-20261109-162636` (smoke test) | 16:26 | 2d5ddc3 (+ `11744e1`) | full fine-tuning, 1 epoch, only 640 training and 128 test clips | 34.4 / 38.9 / 36.6 | 39.2 | 36.4 | 38.3 |

Runs from before the mAP fix: their mAP is understated, but nDCG is unaffected.

| Run | Started (Nov 9) | Code | Setup | Dual mAP / nDCG avg | + rerank mAP / nDCG avg |
|---|---|---|---|---|---|
| `tempret-20261109-005058` | 00:50 | before 6a873d0, uncommitted | ViT-B/16 frozen, 2 epochs | 13.4 / 31.8 | 5.9 / 23.7 |
| `tempret-20261109-020438` | 02:04 | before 6a873d0, uncommitted | ViT-L/14 frozen, 10 epochs | 15.6 / 33.3 | 5.1 / 23.0 |
| `tempret-20261109-100605` | 10:06 | before 6a873d0, uncommitted | the `020438` checkpoint, with the reranker retrained | 15.6 / 33.3 | 16.3 / 33.9 |

The best fair TempRet result is `130858`: 31.0 mAP and 33.9 nDCG, on the local subset. That is far below the paper's 67.97 and 82.92, which were scored on the test server, so the two aren't directly comparable. The `162636` smoke test is too small to mean anything. There are no full-benchmark TempRet runs yet.
