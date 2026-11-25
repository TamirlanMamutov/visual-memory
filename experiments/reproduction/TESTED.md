# Packaging checks actually completed

Test date: 2026-11-24; Mac Apple M4 / 16 GB, Python 3.12, CPU Torch 2.5.1,
NumPy 2.5.4, pandas 3.1.1. No claim that a fresh Windows/CUDA environment was tested.
The official VQLoC evaluator was freshly cloned at
`59460cd5ef605066fdd31a55afa6577d23ab0828` and used for the final replay.
See machine-readable [REPRODUCTION_CHECKS.json](REPRODUCTION_CHECKS.json).

| Check | Observed outcome |
|---|---|
| Private archive integrity |361 files verified against SHA256 manifest |
| Cached replay |16 comparisons pass: box51all/new metrics, all three frozen heads on validation50, seven modality rankings, first human pilot, synthetic all-declined fallback |
| Frozen JSON checkpoint export |All saved validation50 tracks and AP confidences reproduced exactly; official metrics match within1e-10 |
| Hand/control training |All 9 recipes reproduce original candidate choices exactly and all/new metrics within1e-10 |
| v10 hyperparameter training |All 20 recipes reproduce original candidate choices exactly and metrics within1e-10 |
| v11 corrected training |All 14 recipes reproduce original candidate choices exactly and metrics within1e-10 |
| Original20-query head |stAP25 .6233333333 and other reported metrics reproduced within1e-10 |
| Source bootstrap |All CIs reproduced exactly using the portable command,2,000 draws/25 sources |
| Timing/clarification simulation |396 recorded timing metric means match within1e-10; clarification summaries, source intervals and hint costs match exactly |
| Fresh hand extraction |2 queries / 3 candidates from raw licensed videos; reference crops pixel-exact and all 12 features identical to original cached vectors |
| Behavior contracts |32 tests pass: window/confidence preservation, hint fallibility, source exclusion, future-frame rejection, baseline fallback, matched initialization, ranking and hand missingness |
| Teammate root files |Unmodified; branch changes are restricted to our experiment namespaces |

The completed hand study originally checked geometry on all 60 queries. The new
portable raw-video extractor was freshly tested on **two**, not all 60; the full
cached60-query training replay is independently verified. New EAGLE decoding,
fresh CLIP text export and EAGLE-trained proxy heads have not been run here because
the teammate's saved evaluations/checkpoints are absent. Synthetic adapter tests
check that using `extend` retains the final SAM time window.

The setup requirements record installed versions; a second fresh virtualenv
installation and GPU speed/memory benchmark were not performed during packaging.
The shared archive enables CPU numerical replay immediately; fresh upstream REN
candidate generation remains a separate, substantially heavier step.
