# Shared candidate export and evaluation

This folder contains common data conversion, EAGLE adaptation, input validation,
unchanged-baseline export and official scoring. Experiment-specific algorithms live
in [additional_hints](../additional_hints/README.md) and
[box_refinement](../box_refinement/README.md).

For the complete private-artifact handoff, training replay, final50-query results
and priorities, start with [reproduction/README.md](../reproduction/README.md).
For the teammate's **final run6 with SAM extension**, use
[EAGLE_TRANSFER.md](../reproduction/EAGLE_TRANSFER.md): specify
`--sam2-baseline-variant extend` as well as `--sam2-dir`. The commands below
otherwise use the first run's pre-SAM `eval_parts` as baseline.

## Quick start for the teammate's EAGLE runs

Run these commands from the repository root in the **existing project Python
environment** described in [REPRODUCTION.md](../../REPRODUCTION.md). Use a small
20-query export first. Commands are single lines and work in PowerShell or a shell.

### 1. Check available saved runs without loading models

The first run supplies the baseline. Other runs supply alternative tracks.

```text
python experiments/candidate_tools/adapt_eagle.py --runs output/ens3-mask0 output/eagle-20261111-221053 output/eagle-20261114-002928 output/eagle-20261114-204835 --annotations ../Ego4D/v2/annotations/vq_val.json --clips ../Ego4D/v2/clips --max-queries 20 --out output/eagle-context-pilot --inventory-only
```

Selection is deterministic, spreading source videos before extra queries; it does
not use scores or outcomes. Missing run files reduce the eligible pool and are
reported. A selected query missing inside a file raises an error instead of being
silently dropped.

### 2. Export candidate boxes and CLIP scores once

Use the same command without `--inventory-only`, adding an explicit CLIP model:

```text
python experiments/candidate_tools/adapt_eagle.py --runs output/ens3-mask0 output/eagle-20261111-221053 output/eagle-20261114-002928 output/eagle-20261114-204835 --annotations ../Ego4D/v2/annotations/vq_val.json --clips ../Ego4D/v2/clips --clip-model openai/clip-vit-base-patch32 --max-queries 20 --out output/eagle-context-pilot
```

An HF model ID may download CLIP weights on first use. To avoid downloads, pass an
existing **local model directory** and `--local-files-only`. CPU is the default;
`--device cuda` uses the existing GPU environment. No EAGLE/SAM checkpoint is loaded.

If already computed, append
`--sam2-dir output/ens3-mask0/sam2/small-m30` to include saved SAM 2 replace/trim/extend
tracks. This reconstructs the saved `eagle_sam2.variants` boxes without propagating
SAM again. Add `--sam2-baseline-variant extend` to make the final extended track
the baseline; otherwise the original pre-SAM track remains the baseline.

The exporter uses the repository's `FullResReader`, annotation frame numbers at
5 FPS, and original annotation pixel coordinates. It encodes the reference crop
and up to three distinct sampled frames per candidate. Empty predictions remain
in the cohort. Identical tracks are deduplicated.

Outputs:

- `bundle.json`: query inputs, candidates, scores, and sample scores; **no GT**.
- `ground_truth.json`: separate evaluation targets. Never pass this to real-hint
  inference or box refinement.

Only object-name text scores are exported by the EAGLE adapter. Detailed
description scores require independently prepared text/encoding; they are not
invented. Without description scores, the simulation's description arm duplicates
image scores and is explicitly flagged as such.

### 3. Export the unchanged baseline; choose an experiment

```text
python experiments/candidate_tools/baseline.py --bundle output/eagle-context-pilot/bundle.json --out output/eagle-context-pilot/baseline.json
```

Then follow either the [box-refinement guide](../box_refinement/README.md) or the
[additional-hint guide](../additional_hints/README.md). The experiment outputs
use the same prediction format and official evaluator.

### 4. Re-score with the official evaluator yourself

```text
python experiments/candidate_tools/evaluate.py --bundle output/eagle-context-pilot/bundle.json --predictions output/eagle-context-pilot/baseline.json --truth output/eagle-context-pilot/ground_truth.json --evaluator-root ../VQLoC --out output/eagle-context-pilot/baseline_metrics.json
python experiments/candidate_tools/evaluate.py --bundle output/eagle-context-pilot/bundle.json --predictions output/eagle-context-pilot/refined.json --truth output/eagle-context-pilot/ground_truth.json --evaluator-root ../VQLoC --out output/eagle-context-pilot/refined_metrics.json
```

The evaluator root must contain `metrics/metrics.py` (VQLoC) or
`vq2d/metrics/metrics.py` (Ego4D layout). It needs the existing NumPy/pandas evaluator
dependencies and reports pooled `all` metrics only, not object-size breakdowns.
Query keys must match exactly; missing predictions raise an error. Inspect actual
successes and failures yourself in addition to re-scoring the saved exports.

## Reproduce the local REN pilot

On the machine with the existing REN artifacts:

```text
python experiments/candidate_tools/export_local.py --runs-root PATH_TO_EXPERIMENTS --out output/ren-context-bundle
```

`PATH_TO_EXPERIMENTS` contains `ren_multimodal_cached_v3/` and
`ren_multimodal_descriptions_v4/`. Then substitute `output/ren-context-bundle`
for the EAGLE bundle directory in the commands above. Add `--subset new` to
`evaluate.py` to score the 31 queries tagged new; omit it for all 51.

Export files are intentionally **not committed**. Share licensed artifacts only
through your team's authorized data storage. They contain derived scores, boxes,
and separate labels, not images/videos/checkpoints. Keep all model outputs under
`output/`, which the main repository already ignores.

## Validation and limitations

```text
python -m unittest discover -s experiments/candidate_tools -p test_core.py -v
```

Behavior tests cover seconds/frame units, future-frame rejection, immutable
baseline windows/confidence, empty predictions, candidate deduplication and source
sampling. On the original REN artifacts, this portable export reproduced **all
51 refined tracks exactly** and all simulation choices/metric means from the
local robustness study. See [box results](../box_refinement/RESULTS.md) and [hint results](../additional_hints/RESULTS.md).

The EAGLE pool/selection adapter has synthetic behavior tests. Actual EAGLE video
decoding/CLIP export has not been run on this machine; begin with 20 queries and
inspect decoded query crops and box alignment. The adapter uses model variants
as candidates, so this is a new EAGLE ablation, not a guaranteed transfer of REN
performance. Cross-model confidence/similarity calibration may differ.

No existing training entry points or teammate files are edited. No weights,
credentials, reference crops, videos, or query-level licensed labels are in this
folder's committed files.

## Updating from the combined folder

The old `experiments/ren_context_refinement/` command paths remain compatibility
wrappers. Update the new experiment/shared folders together. Keep code revisions
with frozen result files: after upgrading code, use a fresh simulation output
folder instead of overwriting a run whose input/code hashes are frozen.
