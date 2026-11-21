# Window-preserving box refinement

This experiment changes **spatial boxes inside the baseline time window**. It uses
no additional timing hint. Every baseline frame ID and AP confidence stays fixed.
The observed REN gain is image-based; it is not evidence of useful text fusion.

- [Recorded refinement results](RESULTS.md)
- [Aggregate numbers](results/ren_box_refinement.json)
- [Shared export and evaluation tools](../candidate_tools/README.md)

## No-data format check

```text
python experiments/box_refinement/run.py --bundle experiments/candidate_tools/examples/toy_bundle.json --out output/refinement-demo/predictions.json
```

The invented `toy_query_b` selects `better_box` while preserving its frames and
confidence. The wrong-time alternative for `toy_query_a` cannot change its window.
These are behavior checks, not measured retrieval results.

## Run on an exported REN or EAGLE candidate bundle

Prepare `bundle.json` with the [shared instructions](../candidate_tools/README.md),
then run:

```text
python experiments/candidate_tools/baseline.py --bundle output/eagle-context-pilot/bundle.json --out output/eagle-context-pilot/baseline.json
python experiments/box_refinement/run.py --bundle output/eagle-context-pilot/bundle.json --mode image --out output/eagle-context-pilot/refined.json
```

An alternative must overlap at least half the baseline frames and have at least
`min(2, baseline length)` distinct cached crop frames inside the baseline window.
Its mean image similarity must exceed baseline by the fixed margin **0.025**.
Only boxes on shared frames are copied; baseline boxes fill other frames.

Optional `--mode name` or `--mode description` requires corresponding sample text
scores and compatible image evidence. These are separate negative/control
ablations; description scoring is not available automatically. Do not describe the
image-based result as an added-hint or language gain.

## Re-score the paired predictions

```text
python experiments/candidate_tools/evaluate.py --bundle output/eagle-context-pilot/bundle.json --predictions output/eagle-context-pilot/baseline.json --truth output/eagle-context-pilot/ground_truth.json --evaluator-root ../VQLoC --out output/eagle-context-pilot/baseline_metrics.json
python experiments/candidate_tools/evaluate.py --bundle output/eagle-context-pilot/bundle.json --predictions output/eagle-context-pilot/refined.json --truth output/eagle-context-pilot/ground_truth.json --evaluator-root ../VQLoC --out output/eagle-context-pilot/refined_metrics.json
```

`tAP25` should be unchanged because frame IDs and AP confidence are preserved.
Inspect successful and worsened examples yourself. REN's small development result
is not a guarantee of EAGLE transfer. No model is trained or decoded by this script.
