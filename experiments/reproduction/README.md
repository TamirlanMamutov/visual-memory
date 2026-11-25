# Qiaomiao's REN extensions: reproducible handoff

This branch adds experiments under `experiments/` only. Teammate EAGLE training,
evaluation and report files remain unchanged. Start here; the root
[VQ2D_SUMMARY.md](../../VQ2D_SUMMARY.md) describes the teammate's EAGLE work.

## Results at a glance — 2026-11-24

**The stats-only quality verifier is our strongest current validation lead by
point estimate.** It remains a predeclared secondary analysis, and the AP gain
is uncertain. All methods below used the same **50 reserved validation queries
from 25 sources**, with models and thresholds frozen before evaluation. These
are official **all/all** metrics on a subset, not full-benchmark results.

| Method | Analysis role | stAP25 | tAP25 | Recovery | Success |
|---|---|---:|---:|---:|---:|
| Native released REN pipeline | Baseline | 0.3917 | 0.4622 | 43.40% | 54.00% |
| Expanded candidates + stats/image/name verifier | Primary | 0.4038 | 0.4761 | 48.43% | 56.00% |
| Expanded candidates + stats-only verifier | Secondary | **0.4447** | **0.5216** | **52.99%** | **60.00%** |
| Expanded candidates + stats/image verifier | Secondary | 0.4038 | 0.4761 | 48.43% | 56.00% |

Stats-only improves stAP25 by **0.0530** and Recovery by **9.59 percentage points**.
Its paired source-bootstrap 95% AP-difference interval is **[-0.0027, +0.1314]**;
the primary arm's AP difference is +0.0121, interval **[-0.0391, +0.0678]**.
Both AP intervals include zero. Secondary comparisons are not multiplicity-adjusted.
Keep the primary/secondary labels and use fresh sources to confirm any method
selected from these results. Image/name and image-only made identical final
choices, so this does not demonstrate useful text fusion.

### What looks worth trying next

| Priority | Experiment | Evidence so far / next action |
|---|---|---|
| High | Stats-only verifier | Strongest reserved-validation point gain. Audit changed cases, retain native/latest-center controls, then confirm on a fresh cohort. EAGLE transfer requires its own training-source features and retraining. |
| High | Window-preserving image box refinement | On a separate 51-query/36-source development cohort, stAP25 **0.4712 → 0.4857**, Recovery **56.55% → 63.94%**. Test on final EAGLE+SAM tracks while keeping their time window and AP confidence fixed; transfer gain is still unknown. |
| Exploratory | Timing hints and multimodal ranking | Precise GT-derived timing simulations can help, but wrong/earlier clues can hurt. The first real-clue pilot lowered AP; fixed image/text fusion showed no consistent benefit. Improve candidate coverage and test independent, fallible human inputs. |
| Lower | Hand-interaction proxies | Mixed development results: all-hand features reduced all-60-query stAP25 **0.5096 → 0.4509** versus the matched stats control. No hand variant was used in reserved validation. |

The teammate's EAGLE checkpoints, saved evaluations and final SAM outputs have
now been located in the shared Drive. They have **not yet been downloaded or
rescored here**. Our subset scores cannot establish a head-to-head win over
their full 4,501-query result. See [RESULTS.md](RESULTS.md) for complete tables,
uncertainty and cohort definitions, and [EAGLE_TRANSFER.md](EAGLE_TRANSFER.md)
for the proposed matched add-on test.

## Work ownership and status

| Work | Owner / model | Status and scope |
|---|---|---|
| EAGLE training, EgoTracks/VISOR ablations, ensemble and SAM2 extension | Teammate, `zr` | Full 4,501-query validation reported; see their summary |
| Candidate quality verifier | Qiaomiao, released REN candidate tracks + small trained heads | Training-source CV complete; separate frozen 50-query validation complete |
| Window-preserving box refinement | Qiaomiao, cached CLIP crop similarities | 51-query development replay verified; EAGLE transfer not tested yet |
| Timing hints / selective clarification | Qiaomiao, fixed candidate ranking policies | Simulations complete; limited legacy human-clue pilot; new scoped/confidence study pending |
| Image, text, combined and transcript query ranking | Qiaomiao, CLIP over fixed image-generated candidates | Cached 20-query pilot; no consistent fusion gain; real recorded speech not evaluated |
| Hand proximity / co-motion proxies | Qiaomiao, MediaPipe + verifier ablations | 60-query development complete; mixed results, not deployed into reserved validation |

See [results and priorities](RESULTS.md), [artifact checklist](ARTIFACTS.md),
[EAGLE transfer](EAGLE_TRANSFER.md), [candidate schema](SCHEMA.md), and the
timestamped [run status](RUN_STATUS.json). GitHub status is a snapshot, not a live
monitor. Repeated development results are not independent validation results.

## 1. Get an isolated checkout

Use a fresh folder so an existing teammate checkout stays untouched:

```text
git clone --branch codex/ren-reproducible-experiments-20261124 https://github.com/TamirlanMamutov/visual-memory.git visual-memory-ren-experiments
cd visual-memory-ren-experiments
python -m venv .venv-ren-experiments
```

Activate with `source .venv-ren-experiments/bin/activate` on macOS/Linux, or
`.venv-ren-experiments\Scripts\Activate.ps1` in PowerShell. Use Python 3.12.

```text
python -m pip install -r experiments/reproduction/requirements-cached.txt
git clone https://github.com/hwjiang1510/VQLoC.git ../VQLoC-official
git -C ../VQLoC-official checkout 59460cd5ef605066fdd31a55afa6577d23ab0828
```

Pass the **actual evaluator folder** to `--evaluator-root` below, e.g.
`../VQLoC-official`. No EAGLE, REN, CLIP or SAM model is loaded for scoring cached
predictions. The wrapper explicitly uses `vc_rt_pairings={'all': ('all','all')}`;
some upstream defaults silently restrict response tracks to large objects.

## 2. Obtain and verify the private handoff

Get `ren_handoff_v1.zip` from Qiaomiao through authorized private team storage.
Each recipient needs their own Ego4D access. Extract it to
`../ren_handoff_v1`, **outside this Git checkout**. The ZIP contains our small
verifier checkpoints in portable JSON, cached candidate features, saved
evaluations, separate targets and geometry-reference crops. It contains no raw
video, credentials or large pretrained backbone/SAM/CLIP weights.

```text
python -m experiments.reproduction.verify_handoff --root ../ren_handoff_v1
python -m experiments.reproduction.check_replay --handoff ../ren_handoff_v1 --evaluator-root ../VQLoC-official --out output/ren-handoff/replay
```

Expected: `REPLAY_CHECK_PASS 16 comparisons`. This independently rescores the
51-query box study, checks three frozen verifier heads against saved validation
tracks **exactly**, and checks seven cached multimodal rankings plus the real-clue pilot and all-declined interface check. Inspect
`output/ren-handoff/replay/REPLAY_CHECK.json`. The validation folder name identifies
the closed scored batch included in this archive; it is not the full benchmark.
This is replay of saved inputs, not a fresh end-to-end REN inference claim.

To independently reproduce the source-paired confidence intervals:

```text
python -m experiments.reproduction.bootstrap --bundle ../ren_handoff_v1/validation50/bundle.json --truth ../ren_handoff_v1/validation50/ground_truth.json --baseline ../ren_handoff_v1/validation50/REN_baseline.json --predictions ../ren_handoff_v1/validation50/expanded_image_name.json ../ren_handoff_v1/validation50/expanded_stats.json ../ren_handoff_v1/validation50/expanded_stats_image.json --evaluator-root ../VQLoC-official --out output/ren-handoff/source-bootstrap.json
```

Defaults:2,000 source draws, seed2026112101. Compare with the private
`validation50/expected_analysis.json`; interpret the intervals as in
[RESULTS.md](RESULTS.md).

## 3. Retrain the small heads yourself

This needs only cached training candidates/targets and CPU Torch. It does **not**
train REN, DINOv2, CLIP, EAGLE or SAM. These are our added quality classifiers.

```text
python -m experiments.track_verifier.run_cv --data ../ren_handoff_v1/development60/dataset.json --recipes ../ren_handoff_v1/development60/hand_recipes.json --names stats hand_all hand_zero_control --evaluator-root ../VQLoC-official --out output/ren-handoff/hand-cv
python -m experiments.reproduction.check_cv --expected ../ren_handoff_v1/development60/expected.json --group hand --results output/ren-handoff/hand-cv --out output/ren-handoff/hand-cv-check.json
```

Omit `--names` for all nine hand/control recipes. For the earlier 20- and 14-recipe
hyperparameter sweeps, replace `hand_recipes.json` with `v10_recipes.json` or
`v11_recipes.json`, use fresh output folders, and set `check_cv --group v10` or
`v11`. Each recipe uses five source outer folds, three inner threshold folds,
and seeds 0/1/2. `COMPLETE.json`, fold audits, candidate choices and metrics are
written per recipe. `progress.json` is updated on fold/recipe completion.

The original, smaller study that reached stAP25 **0.6233** can be replayed with
`development20/dataset.json` and `development20/recipes.json`; compare its result
to `development20/expected_metrics.json`. It used 20 development queries, not
the full benchmark. Do not present it as higher performance than EAGLE's full-val
score. `old`/`new` labels in the 60-query package define reporting subsets, not
unseen data relative to the released REN model's own training.

## 4. Run individual methods and extend to new data

| Task | Instructions |
|---|---|
| Apply our frozen verifier without training or targets | [track_verifier](../track_verifier/README.md) |
| Refine boxes while keeping time window and AP confidence | [box_refinement](../box_refinement/README.md) |
| Timing clues, fallible memories, simulated hints | [additional_hints](../additional_hints/README.md) |
| Image/text/fused/transcript candidate rankings | [multimodal_queries](../multimodal_queries/README.md) |
| Extract hand proxies from selected licensed videos | [hand_interaction](../hand_interaction/README.md) |
| Use final teammate EAGLE+SAM predictions as candidates | [EAGLE transfer](EAGLE_TRANSFER.md) |
| Export additional REN tracks or another model's tracks | [schema and boundaries](SCHEMA.md) |

Training recipes may be changed on the previous 60 training queries/40 sources.
The 50 reserved queries/25 other sources remain evaluation only. Do not tune on
their outcomes, replace frozen heads, or pick the best validation arm after
looking at scores. New development-selected policies require another prospective
cohort. Retain native REN and simple latest-center/ranking controls.

## 5. Checks and interpretation

```text
python -m unittest experiments.candidate_tools.test_core experiments.track_verifier.test_contracts experiments.multimodal_queries.test_ranking -v
```

With optional hand dependencies installed, also run
`python -m unittest experiments.hand_interaction.test_proxies -v`.
See [tested evidence](TESTED.md). Exact numerical parity is checked on the
recorded Mac CPU environment; Windows/CUDA training equivalence is untested.
On another platform compare selected candidates and official metrics, and report
any differences rather than silently loosening tolerances.

Always report cohort, source count, selection procedure, model/checkpoint,
input modality, evaluator pairing, and whether method development used that
cohort. Inspect successes **and failures** manually. Candidate oracle/Top-3
coverage is not automatic Top-1 stAP25 or official Success. Our subset scores and
the teammate's 4,501-query scores are not a head-to-head comparison.
