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

Check out branch `zr` (commit `ae35b68`) of `TamirlanMamutov/visual-memory`. `main` is still at `422f0c0` and lacks `vq2d_compare.py`. Read three files in this order:

| File | What it covers |
|---|---|
| `README.md` | Shared folder layout and how to launch |
| `REPRODUCTION.md` | Python environment setup and the EAGLE pipeline: pseudo-masks, training, evaluation, resume |
| `EAGLE_RUNS.md` | The command and git commit for every run, 1 to 6 |

Run from `visual-memory` with `..\cs230-venv\env\Scripts\python.exe` (written `python` below). Pick the option that matches how much you want to redo:

1. **From saved evaluations, about 5 hours.** Rebuild the ensemble, then run SAM 2:

   ```bash
   python eagle_ensemble.py output\eagle-20261111-221053 output\eagle-20261114-002928 output\eagle-20261114-204835 --out output\ens3-mask0
   python eagle_sam2.py output\ens3-mask0 --clips 0: --model facebook/sam2.1-hiera-small --margin 30 --tag small-m30
   ```
2. **From checkpoints, about 12 hours per run.** Re-evaluate runs 2, 3 and 5, then option 1:

   ```bash
   python run_eagle.py --stage evaluate --resume output\eagle-20261111-221053
   ```
3. **From scratch, 2 to 3 days of GPU.** Check out each run's commit and use its command in `EAGLE_RUNS.md` sections 2, 3 and 5, then option 1.

To score any prediction files against ours on exactly the same queries: `python vq2d_compare.py --pred ours=run6 --pred theirs=<predictions.json>`. Use commit `ae35b68` or later: from that commit every script scores all queries with the official setting. Older commits print the large-only numbers.

Options 1 and 2 need the shared Google Drive project folder, which isn't in git:

- `visual-memory\output\`: saved evaluations (`eval_parts`) for runs 2, 3, 5 and `ens3-mask0`
- `cs230-cache\eagle\checkpoints\`: trained weights for every run
- `Ego4D\v2\`: VQ2D annotations and clips, under the Ego4D license
- `hf-models\`: a local copy of SAM 2.1-hiera-small

## Box refinement from Qiaomiao's branch

The code checks out on a second machine; reproducing the 51-query result now needs only the exported input files. Checked at commit `a35f617` of `codex/ren-hints-box-refinement-20261120`; the branch has since moved to `2be55a2`, which we haven't looked at yet.

- **Same code: passed.** All 14 unit tests pass, and the toy example behaves as the README says: `toy_query_b` switches to `better_box` with frames and score unchanged, and `toy_query_a` stays put.
- **Scorer: matches the official one.** On run 6, clips 0–350, its stAP25 (0.4348) equals our official-setting rescore exactly. That check is how the large-only filter came to light.
- **Same results: waiting on files.** The committed results are aggregates only, with no query keys. To rerun the pilot we need the `bundle.json` and `ground_truth.json` that `export_local.py` wrote for the 51 queries, plus the commit they were made at.
- **Their numbers can't be compared with ours yet.** The pilot uses 51 queries from the VQ2D training split; our run 6 only has val predictions.
- **One fix before running it on our data.** `adapt_eagle.py` takes the first run's saved tracks as the baseline, and `output\ens3-mask0\eval_parts` holds run 6a's tracks, before SAM 2. Refinement keeps the baseline's time window, so run 6's SAM 2 extension would be lost. The baseline should be run 6's final tracks.

Next step, if the pilot reproduces: run image-mode and name-mode refinement on run 6 over clips 0–350 (about 1–1.5 hours), then full val (3–5 hours) if it helps.
