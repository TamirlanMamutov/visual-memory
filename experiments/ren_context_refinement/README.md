# Timing hints and window-preserving box refinement

Additive experiment utilities for saved REN **or EAGLE** candidates. Existing
training scripts, checkpoints, runs, and teammates' files are unchanged.

Two separate experiments are included:

1. **Timing hints:** rank existing candidate tracks with image similarity plus
   an approximate sighting time or an early/middle/late clue.
2. **Box refinement:** compare alternative boxes **inside the original predicted
   time window**; keep every baseline frame ID and AP confidence unchanged.

Neither operation runs REN, EAGLE, or SAM again. Building a candidate bundle may
need one-time CLIP crop encoding. Hints cannot recover a target missing from the
candidate bank. Appearance-description and speech inputs are not automatically
provided by this code.

## No-data smoke test

These examples are invented geometry, with no dataset files or model downloads.
The core needs only NumPy.

```text
python experiments/ren_context_refinement/run.py hint --bundle experiments/ren_context_refinement/examples/toy_bundle.json --hints experiments/ren_context_refinement/examples/toy_hints.json --out output/context-demo/hinted.json
python experiments/ren_context_refinement/run.py refine --bundle experiments/ren_context_refinement/examples/toy_bundle.json --out output/context-demo/refined.json
```

The first hint selects the `later` candidate in `toy_query_a`; refinement selects
`better_box` for `toy_query_b` while retaining its original frame IDs/confidence.
These are format/behavior checks, not measured retrieval results.

## Quick start for the teammate's EAGLE runs

Run these commands from the repository root in the **existing project Python
environment** described in [REPRODUCTION.md](../../REPRODUCTION.md). Use a small
20-query export first. Commands are single lines and work in PowerShell or a shell.

### 1. Check available saved runs without loading models

The first run supplies the baseline. Other runs supply alternative tracks.

```text
python experiments/ren_context_refinement/adapt_eagle.py --runs output/ens3-mask0 output/eagle-20261111-221053 output/eagle-20261114-002928 output/eagle-20261114-204835 --annotations ../Ego4D/v2/annotations/vq_val.json --clips ../Ego4D/v2/clips --max-queries 20 --out output/eagle-context-pilot --inventory-only
```

Selection is deterministic, spreading source videos before extra queries; it does
not use scores or outcomes. Missing run files reduce the eligible pool and are
reported. A selected query missing inside a file raises an error instead of being
silently dropped.

### 2. Export candidate boxes and CLIP scores once

Use the same command without `--inventory-only`, adding an explicit CLIP model:

```text
python experiments/ren_context_refinement/adapt_eagle.py --runs output/ens3-mask0 output/eagle-20261111-221053 output/eagle-20261114-002928 output/eagle-20261114-204835 --annotations ../Ego4D/v2/annotations/vq_val.json --clips ../Ego4D/v2/clips --clip-model openai/clip-vit-base-patch32 --max-queries 20 --out output/eagle-context-pilot
```

An HF model ID may download CLIP weights on first use. To avoid downloads, pass an
existing **local model directory** and `--local-files-only`. CPU is the default;
`--device cuda` uses the existing GPU environment. No EAGLE/SAM checkpoint is loaded.

If already computed, append
`--sam2-dir output/ens3-mask0/sam2/small-m30` to include saved SAM 2 replace/trim/extend
tracks. This reuses `eagle_sam2.variants`; it does not propagate SAM again.

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

### 3. Run baseline and box refinement

```text
python experiments/ren_context_refinement/run.py baseline --bundle output/eagle-context-pilot/bundle.json --out output/eagle-context-pilot/baseline.json
python experiments/ren_context_refinement/run.py refine --bundle output/eagle-context-pilot/bundle.json --mode image --out output/eagle-context-pilot/refined.json
```

Refinement requires an alternative to overlap at least half the baseline frames,
with at least two distinct crop samples inside that window (or one for a one-frame
track). Its mean image similarity must exceed the baseline by **0.025**. Only
shared-frame boxes are copied. The time window, frame IDs and AP confidence stay
fixed. `--mode name` or `--mode description` additionally requires compatible image
evidence and the corresponding sample text scores. These are fixed ablations,
not claims of a demonstrated text gain.

### 4. Re-score with the official evaluator yourself

```text
python experiments/ren_context_refinement/evaluate.py --bundle output/eagle-context-pilot/bundle.json --predictions output/eagle-context-pilot/baseline.json --truth output/eagle-context-pilot/ground_truth.json --evaluator-root ../VQLoC --out output/eagle-context-pilot/baseline_metrics.json
python experiments/ren_context_refinement/evaluate.py --bundle output/eagle-context-pilot/bundle.json --predictions output/eagle-context-pilot/refined.json --truth output/eagle-context-pilot/ground_truth.json --evaluator-root ../VQLoC --out output/eagle-context-pilot/refined_metrics.json
```

The evaluator root must contain `metrics/metrics.py` (VQLoC) or
`vq2d/metrics/metrics.py` (Ego4D layout). It needs the existing NumPy/pandas evaluator
dependencies and reports pooled `all` metrics only, not object-size breakdowns.
Query keys must match exactly; missing predictions raise an error. Inspect actual
successes and failures yourself in addition to re-scoring the saved exports.

## Real timing hints

Create a template without reading ground truth:

```text
python experiments/ren_context_refinement/run.py hint-template --bundle output/eagle-context-pilot/bundle.json --out output/eagle-context-pilot/hints.json
```

Fill every query with either:

```json
{"key":"actual_query_key","kind":"time","seconds":20.0,"sigma_seconds":5.0}
```

or:

```json
{"key":"actual_query_key","kind":"third","value":"late"}
```

Times are **seconds into the benchmark clip**, not a camera clock or frame number.
The 5-second value is a Gaussian standard deviation, not a guaranteed +/-5-second
bound. For `third`, use `early`, `middle`, or `late` relative to searchable history.
Record top-level `provenance` accurately. Real clues should be written independently
of target boxes/timestamps. These are sighting-time clues, not action recognition.

```text
python experiments/ren_context_refinement/run.py hint --bundle output/eagle-context-pilot/bundle.json --hints output/eagle-context-pilot/hints.json --out output/eagle-context-pilot/hinted.json
```

Hint ranking uses image cosine + `0.05 * temporal_log_density`; AP confidence stays
unchanged. It returns the winning track and a ranked candidate-ID list. A robust
mixture is optional through `--prior-json`, which must contain externally fitted
training-only candidate log densities; never use evaluation GT to construct a
real-user prior. No action parser or speech recognizer is included. A speech
transcript of the same words is an interface test, not extra semantic evidence.

## Oracle-derived timing simulation — keep separate from real-user results

```text
python experiments/ren_context_refinement/simulate.py --bundle output/eagle-context-pilot/bundle.json --truth output/eagle-context-pilot/ground_truth.json --evaluator-root ../VQLoC --out output/eagle-context-pilot/simulation
```

This **deliberately derives hints from GT** to test the potential value of timing
information. It is not a real-user evaluation or a benchmark improvement. It tests:

- 2-, 5- and 10-second Gaussian noise;
- 20% wrong hints;
- hints biased 5 seconds earlier;
- synthetic early/middle/late categories;
- 200 independently shuffled source-disjoint controls, random controls, latest-center
  and last-third baselines, and an out-of-fold timing prior.

Five seeds x 40 hint draws = 200 simulations, **not 200 independent videos**.
The marginal timing prior uses only other source folds, but these are still
development analyses. On a validation bundle, fitting a prior on other validation
sources does not make the result an untouched validation score. Use separate
training sources for any deployable prior. `old` tags merely define a reporting
subset; they do not establish independence from pretrained model training.

The robust mixture is fixed at 80% hint density + 20% prior density. No parameter
sweep is performed. Full official metric calls check the cached-statistic fast
path on 12 fixed scenario/cohort combinations. To re-score frozen simulation
rankings without regenerating hints, repeat the command with `--evaluate-only`.

## Reproduce the local REN pilot

On the machine with the existing REN artifacts:

```text
python experiments/ren_context_refinement/export_local.py --runs-root PATH_TO_EXPERIMENTS --out output/ren-context-bundle
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
python -m unittest discover -s experiments/ren_context_refinement -p test_core.py -v
```

Behavior tests cover seconds/frame units, future-frame rejection, immutable
baseline windows/confidence, empty predictions, candidate deduplication and source
sampling. On the original REN artifacts, this portable export reproduced **all
51 refined tracks exactly** and all simulation choices/metric means from the
local robustness study. See [RESULTS.md](RESULTS.md).

The EAGLE pool/selection adapter has synthetic behavior tests. Actual EAGLE video
decoding/CLIP export has not been run on this machine; begin with 20 queries and
inspect decoded query crops and box alignment. The adapter uses model variants
as candidates, so this is a new EAGLE ablation, not a guaranteed transfer of REN
performance. Cross-model confidence/similarity calibration may differ.

No existing training entry points or teammate files are edited. No weights,
credentials, reference crops, videos, or query-level licensed labels are in this
folder's committed files.
