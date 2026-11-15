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

## 5. VQ2D + EgoTracks + VISOR: `eagle-20261114-204835` (in progress)

Started Nov 14 20:48 at commit `aee6859`. The training mix is VQ2D 50%, EgoTracks 25%, VISOR 25% (`--mix 2,1,1`, the default).

```bash
git checkout aee6859
python run_eagle.py --stage all
```

VISOR must be unzipped at `..\EPIC-KITCHENS-VISOR\GroundTruth-SparseAnnotations`.

## Results on Ego4D VQ2D val

| Run | tAP25 | stAP25 | tAP | stAP | Succ | Rec |
|---|---|---|---|---|---|---|
| EAGLE paper (val) | 0.47 | 0.42 | — | — | 61.29 | 52.09 |
| REN paper (likely test) | 0.52 | 0.40 | — | — | 61.2 | 49.3 |
| 1. Baseline | 0.406 | 0.319 | 0.232 | 0.142 | 60.74 | 40.35 |
| 2. EgoTracks | 0.472 | 0.386 | 0.283 | 0.179 | 64.51 | 46.35 |
| 3. Fine-tuned 4 blocks | 0.445 | 0.378 | 0.265 | 0.180 | 62.89 | 43.58 |
| 2 on clips 0–350 only | 0.486 | 0.400 | 0.300 | 0.189 | 64.63 | 47.40 |
| 4. 2 + SAM 2 `extend`, clips 0–350 | 0.533 | 0.448 | 0.377 | 0.250 | 65.46 | 55.52 |
| 5. VISOR | pending | | | | | |
