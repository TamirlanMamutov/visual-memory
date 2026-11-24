# VQ2D results summary: ensemble + SAM 2

Nov 24, 2026. Shared copy: [VQ2D results summary on claude.ai](https://claude.ai/artifact/GgqbDhcwDEAtMJyBoPbbK8). Per-run commands and commits are in [EAGLE_RUNS.md](EAGLE_RUNS.md).

## TL;DR

Averaging three of our trained EAGLE models, adding a visual-crop constraint and refining tracks with SAM 2 lifts Ego4D VQ2D val stAP25 from 0.281 to 0.423, with no retraining. Scored on all queries with the official Ego4D setting, that ties EAGLE on stAP25 (+0.003) and beats it on tAP25, Success and Recovery.

**Correction:** the 0.531 stAP25 we circulated earlier was scored on only the 2,336 large-object queries, not all 4,501. It is not a valid comparison with the papers, and the claim that we beat EAGLE by about 0.11 stAP25 is withdrawn.

| Ego4D VQ2D val, official evaluator (4,501 queries) | tAP25 | stAP25 | Succ (%) | Rec (%) |
|---|---|---|---|---|
| EAGLE paper (AAAI'26), best published | 0.47 | 0.42 | 61.29 | 52.09 |
| Our run 2, best single model | 0.380 | 0.281 | 56.10 | 41.18 |
| **Our run 6: ensemble + crop constraint + SAM 2** | **0.503** | **0.423** | **64.12** | **58.07** |

EAGLE's numbers are as reported in its paper. EAGLE released no evaluation code, so we can't tell whether it scored all queries or only large ones (next section).

## The evaluation gap: large-only vs all queries

Until Nov 24 we scored with VQLoC's copy of the evaluator, which silently drops every query whose object stays small: it keeps only the 2,336 of 4,504 val queries (52%) whose ground-truth box reaches 192×192 px in some frame. Its default is `vc_rt_pairings = {"all": ("all", "large")}`. The official Ego4D evaluator (`EGO4D/episodic-memory`, `evaluate_vq.py`) defaults to `("all", "all")` and scores every query. We didn't choose the filter; we used VQLoC's default without checking it against the official code, and we wrongly described the numbers as official. Small objects are harder, so dropping them inflated every score.

Our code now passes the official setting explicitly. All numbers below and in `EAGLE_RUNS.md` are on all 4,501 scored queries.

| Our run (full val) | stAP25, all queries (official) | stAP25, large only (old) | Gap |
|---|---|---|---|
| 1. Baseline, VQ2D only | 0.229 | 0.320 | −0.091 |
| 2. + EgoTracks | 0.281 | 0.386 | −0.105 |
| 3. + fine-tuned last 4 DINOv2 blocks | 0.301 | 0.378 | −0.077 |
| 5. + VISOR | 0.258 | 0.351 | −0.093 |
| 6a. Ensemble 2+3+5 + crop constraint | 0.370 | 0.473 | −0.103 |
| **6. 6a + SAM 2 `extend`** | **0.423** | 0.531 | −0.108 |

### Which setting did the papers use?

Only two papers' settings are known from their code: PRVQL scores all queries and VQLoC large objects only. For EAGLE, RELOCATE and HERO-VQL it can't be determined, so we show our runs under both settings; the right comparison lies between our two rows. Paper numbers are as reported, val unless marked test.

| Method | Evaluator setting | Evidence | tAP25 | stAP25 | Succ (%) | Rec (%) |
|---|---|---|---|---|---|---|
| [EAGLE](https://arxiv.org/abs/2511.08007) (AAAI'26) | **Unknown** | No public code; the repo linked in the paper returns 404 | 0.47 | 0.42 | 61.29 | 52.09 |
| [RELOCATE](https://arxiv.org/abs/2412.01826) (CVPR'25) | **Unknown** | No code released; paper says "official metrics", ~4,500 val queries | 0.41 | 0.33 | 58.03 | 50.50 |
| [HERO-VQL](https://github.com/KHU-VLL/HERO-VQL) (BMVC'25) | **Ambiguous** | `ltvu/metrics.py` computes and prints both an all-queries and a large-only score; paper doesn't say which it reports | 0.38 | 0.28 | 61.10 | 44.90 |
| [PRVQL](https://github.com/fb-reps/PRVQL) (ICCV'25) | **All queries** | `metrics/metrics.py` default `("all", "all")` | 0.35 | 0.27 | 57.93 | 47.87 |
| [VQLoC](https://github.com/hwjiang1510/VQLoC) (NeurIPS'23) | **Large only** | `metrics/metrics.py` default `("all", "large")` | 0.31 | 0.22 | 55.89 | 47.05 |
| [REN](https://github.com/savya08/REN) (NeurIPS'25), test server | Test server's own | Its own code scores all queries, but these numbers come from the hidden test server | 0.52 | 0.40 | 61.2 | 49.3 |
| EAGLE, test server | Test server's own | Hidden test server | 0.46 | 0.40 | 62.70 | 53.51 |
| Our run 2 (single model) | All queries | `eagle/infer.py`, `OFFICIAL_PAIRING` | 0.380 | 0.281 | 56.10 | 41.18 |
| Our run 2 (single model) | Large only | `VQLOC_PAIRING` | 0.472 | 0.386 | 64.51 | 46.35 |
| **Our run 6** | **All queries** | `OFFICIAL_PAIRING` | **0.503** | **0.423** | **64.12** | **58.07** |
| **Our run 6** | **Large only** | `VQLOC_PAIRING` | **0.598** | **0.531** | **71.70** | **62.07** |

How to read it: against PRVQL, compare our all-queries rows; against VQLoC, our large-only rows. Against EAGLE, run 6 is at least a tie on stAP25 (0.423 all-queries vs 0.42) and ahead on the other three metrics under either setting. The paper rows come from EAGLE's Table 1, except REN's. Our test-server numbers don't exist: the VQ2D challenge has closed.

**What this means for the comparison:**

- **The method works either way.** Run 6 beats our single model run 2 by +0.142 stAP25 on all queries (0.281 → 0.423).
- **The fair comparison is all queries against EAGLE's 0.42,** because the official setting is the benchmark's definition and two of the papers whose code we could read score all queries. On that basis, run 6 ties EAGLE on stAP25 (0.423 vs 0.42) and leads on tAP25 (+0.03), Success (+2.8) and Recovery (+6.0).
- **If EAGLE did use VQLoC's filter,** its 0.42 would be large-only, and our large-only 0.531 would be the like-for-like number. We can't confirm that, so we don't claim it.
- **VQLoC's own reported numbers are probably large-only,** so rows copied from VQLoC (as EAGLE's Table 1 does) may not be on the same basis as the others.

## What we changed

Three inference-time steps on top of models we had already trained; nothing is retrained. Error analysis motivated them: in 36% of queries, run 2 picked a time window that misses the object entirely, while in 90% its per-frame score was high inside the correct window.

1. **Trace ensembling.** Average the per-frame confidence traces of runs 2, 3 and 5, which were trained differently (frozen vs fine-tuned DINOv2, with and without VISOR). Each frame keeps the box of the most confident run, then EAGLE's own post-processing builds the track. Different models fire on different false peaks, so averaging cancels many of them.
2. **Visual-crop constraint.** The visual crop is itself a sighting of the object, and the answer is its *last* sighting before the query. When the crop's frame comes before the query frame (42% of val queries), every frame up to the crop frame gets confidence 0. This uses only the crop's frame number, which is part of the query input.
3. **SAM 2 `extend`, margin 30.** From the track's most confident frame and box, SAM 2.1 tracks the object forward and backward. Its boxes replace EAGLE's, and the track grows at both ends while SAM 2 still sees the object, up to 30 frames. This idea comes from REN and RELOCATE.

| Step (full val, official evaluator) | stAP25 | tAP25 | Succ (%) | Rec (%) |
|---|---|---|---|---|
| Run 2 alone | 0.281 | 0.380 | 56.10 | 41.18 |
| Run 2 + crop constraint | 0.290 | 0.392 | 56.34 | 41.33 |
| Ensemble of runs 2, 3, 5 | 0.355 | 0.446 | 62.14 | 45.37 |
| Ensemble + crop constraint (run 6a) | 0.370 | 0.469 | 62.74 | 45.59 |
| + SAM 2 `extend` (run 6) | **0.423** | **0.503** | **64.12** | **58.07** |

Ensembling gives the largest gain (+0.074 stAP25), SAM 2 the next (+0.053, mostly Recovery), and the crop constraint a small but consistent one (+0.009 to +0.015).

## Fairness and caveats

Nothing was trained on val, but val was also our development set, so the result is a val number for a method designed while looking at val.

- **No val leakage in training.** Runs 2, 3 and 5 train on VQ2D `train` and EgoTracks `train` (0 videos and 0 clips shared with the 422 val videos) and on EPIC-KITCHENS VISOR `train`. Run 6 trains nothing.
- **Settings fixed in advance.** Equal ensemble weights, crop margin 0, EAGLE's default post-processing and SAM 2 margin 30 were chosen before scoring. A sweep tuned on clips 351+ and reported on clips 0–350 found nothing that clearly beat them.
- **Same split as the papers.** All rows use the official VQ2D val split: 422 videos, 1,157 clips. We score 4,501 of 4,504 queries; 3 queries from one clip are skipped because their crop frame can't be decoded.
- **EAGLE's evaluator is unknown.** We use EAGLE's reported numbers; we did not re-run its code. Whether it scored all queries or only large ones decides whether our stAP25 lead is +0.003 or +0.11.
- **More compute.** Run 6 runs three EAGLE models plus SAM 2, about 3× EAGLE's inference cost. A single-model version (run 2 + crop constraint + SAM 2) hasn't been measured.
- **No test-set score.** The VQ2D test labels are hidden, the EvalAI challenge closed on Dec 31, 2025, and VQ2D is not among the 2026 Ego4D challenges on CodaBench.
- **Pretrained models.** DINOv2-base, SAM ViT-B and SAM 2.1-hiera-small are Meta's public weights; none was fine-tuned except run 3's last 4 DINOv2 blocks.

## How to reproduce

Options A and B below were tested on Nov 24 on a slice of the data and reproduced our saved results exactly: all 4,501 ensemble tracks, 1,657 of 1,657 SAM 2 frame boxes on 5 clips, and 9 of 9 tracks for each of the three checkpoints on 2 clips. Option C is the same commands the original runs used, but it takes days and was not re-run.

### Setup

1. Clone `git@github.com:TamirlanMamutov/visual-memory.git` and check out branch `zr` at commit `07d4365` or later. `main` is older and lacks the scoring fix (`ae35b68`).
2. Create the Python environment (transformers 5.22 or later, CUDA PyTorch) as in `REPRODUCTION.md`, section "Environment". Run every command from the `visual-memory` folder with `..\cs230-venv\env\Scripts\python.exe`, written `python` below.
3. Put these folders next to `visual-memory`, inside one project folder. The scripts find them by these relative paths:

| Folder | Needed for | What it is |
|---|---|---|
| `VQLoC` | A, B, C | A clone of `hwjiang1510/VQLoC`; our scoring imports its metric code |
| `Ego4D\v2\annotations`, `Ego4D\v2\clips` | A, B, C | VQ2D annotations and video clips, under the Ego4D license |
| `visual-memory\output\eagle-20261111-221053`, `eagle-20261114-002928`, `eagle-20261114-204835` | A | Saved per-frame evaluations of runs 2, 3 and 5 (shared Google Drive) |
| `cs230-cache\eagle\checkpoints` | B | Trained weights of every run (shared Google Drive) |
| `Ego4D\v2\egotracks` | C | EgoTracks annotations (`egotracks_train.json`) |
| `EPIC-KITCHENS-VISOR` | C, run 5 only | VISOR sparse annotations and frames |

All times are measured on one RTX 4070 with 12 GB of memory. SAM 2.1, DINOv2 and SAM download from Hugging Face on first use.

### A. From our saved evaluations (about 10 hours)

Write to a **new** folder. Reusing `output\ens3-mask0` would not reproduce anything: `eagle_sam2.py` skips clips already refined there and would only re-score our saved boxes.

```powershell
python eagle_ensemble.py output\eagle-20261111-221053 output\eagle-20261114-002928 output\eagle-20261114-204835 --out output\ens3-mask0-repro
python eagle_sam2.py output\ens3-mask0-repro --clips 0: --model facebook/sam2.1-hiera-small --margin 30 --tag small-m30
```

- **The first command** takes a few minutes on the CPU and should print exactly `stAP25 0.3695 tAP25 0.4686 stAP 0.1714 tAP 0.2869 Succ 62.7416 Rec 45.5860` (run 6a).
- **The second** takes about 30 seconds per clip, roughly 10 hours for 1,156 clips. It resumes if rerun with the same command. Its final table's `extend` row is run 6: stAP25 0.4228, tAP25 0.5033, stAP 0.2286, tAP 0.3712, Succ 64.12, Rec 58.07.

### B. From our checkpoints (about 14 hours per run, then A)

Each command evaluates one checkpoint into a new `output\eagle-<start time>` folder:

```powershell
python run_eagle.py --stage evaluate --checkpoint ..\cs230-cache\eagle\checkpoints\eagle-20261111-221053\last.pt
python run_eagle.py --stage evaluate --checkpoint ..\cs230-cache\eagle\checkpoints\eagle-20261114-002928\last.pt
python run_eagle.py --stage evaluate --checkpoint ..\cs230-cache\eagle\checkpoints\eagle-20261114-204835\last.pt
```

- Run 3's fine-tuned DINOv2 blocks load from its checkpoint automatically; its log says `loaded 74 fine-tuned DINOv2 tensors`.
- Each run's `metrics.json` should match: stAP25 0.281 for run 2, 0.301 for run 3 and 0.258 for run 5.
- If one is interrupted, continue it with `python run_eagle.py --stage evaluate --resume output\eagle-<that new folder>`. Don't use `--resume` on our original folders: it reuses their saved predictions instead of recomputing them.
- Then run option A with the three new folders in place of ours.

### C. Train from scratch (about 1 day per run, then A)

Run `python run_eagle.py --stage prepare` once first; it writes SAM pseudo-masks for every training query and resumes if interrupted. Then train and evaluate each run at the commit it used:

| Run | Commit | Command | Note |
|---|---|---|---|
| 2 | `7d14dbd` | `python run_eagle.py --stage all --no-visor` | `--no-visor` is required: this commit uses VISOR automatically when it is on disk, and run 2 was trained without it |
| 3 | `120cd72` | `python run_eagle.py --stage all --finetune-blocks 4 --no-visor` | |
| 5 | `aee6859` | `python run_eagle.py --stage all` | VISOR must be at `..\EPIC-KITCHENS-VISOR\GroundTruth-SparseAnnotations` |

Those old commits print large-only metrics at the end. Then check out `07d4365` or later and run option A on the three new folders, which scores all queries. Fresh trainings on a GPU aren't bit-identical, so expect numbers close to ours, not exact.

### Comparing other predictions

`python vq2d_compare.py --pred ours=run6 --pred theirs=<predictions.json>` scores any prediction file against run 6 on exactly the same queries, with a paired bootstrap interval.

## Box refinement from Qiaomiao's branch

The code checks out on a second machine; reproducing the 51-query result now needs only the exported input files. Checked at commit `a35f617` of `codex/ren-hints-box-refinement-20261120`; the branch has since moved to `2be55a2`, which we haven't looked at yet.

- **Same code: passed.** All 14 unit tests pass, and the toy example behaves as the README says: `toy_query_b` switches to `better_box` with frames and score unchanged, and `toy_query_a` stays put.
- **Scorer: matches the official one.** On run 6, clips 0–350, its stAP25 (0.4348) equals our official-setting rescore exactly. That check is how the large-only filter came to light.
- **Same results: waiting on files.** The committed results are aggregates only, with no query keys. To rerun the pilot we need the `bundle.json` and `ground_truth.json` that `export_local.py` wrote for the 51 queries, plus the commit they were made at.
- **Their numbers can't be compared with ours yet.** The pilot uses 51 queries from the VQ2D training split; our run 6 only has val predictions.
- **One fix before running it on our data.** `adapt_eagle.py` takes the first run's saved tracks as the baseline, and `output\ens3-mask0\eval_parts` holds run 6a's tracks, before SAM 2. Refinement keeps the baseline's time window, so run 6's SAM 2 extension would be lost. The baseline should be run 6's final tracks.

Next step, if the pilot reproduces: run image-mode and name-mode refinement on run 6 over clips 0–350 (about 1–1.5 hours), then full val (3–5 hours) if it helps.
