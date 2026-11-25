# Candidate quality verifier

This is our added small classifier, not REN representation training. It chooses
among fixed native/expanded candidate tracks using 12 quality statistics, optionally
plus CLIP image/name scores or hand proxies. It cannot recover an object outside
the candidate pool. Selection can change the time interval; AP confidence stays
at the native prediction's value.

## Apply the frozen validation head

Follow [setup](../reproduction/README.md). Example for the archive's completed50-query batch:

```text
python -m experiments.track_verifier.apply --bundle ../ren_handoff_v1/validation50/bundle.json --model ../ren_handoff_v1/models/expanded_stats.json --out output/verifier/stats.json
python experiments/candidate_tools/evaluate.py --bundle ../ren_handoff_v1/validation50/bundle.json --predictions output/verifier/stats.json --truth ../ren_handoff_v1/validation50/ground_truth.json --evaluator-root ../VQLoC-official --out output/verifier/stats_metrics.json
```

Use the validation folder actually listed in the ZIP. Three small JSON model
files are included: `expanded_stats`, `expanded_stats_image`, and
`expanded_image_name`. These are the pre-reserved-validation models, lr 0.001,
threshold 0.65, mean-track-IoU inner selection, three seeds. The later lr 0.003,
source-balanced, AP-selected development recipe is a different experiment and
was never substituted into this validation run.

`apply` reads no targets. It requires `feature_family=ren_quality_v1`; it rejects
applying REN weights to different EAGLE/CLIP proxy evidence. Checkpoints are JSON
tensors, so replay does not unpickle arbitrary Torch objects.

## Train or tune on new training-source candidates

Prepare a label-free candidate bundle and separate target file using the
[schema](../reproduction/SCHEMA.md). For exact REN features, export its original
12-dimensional evidence. For EAGLE use the separate proxy family and **retrain**:

```text
python -m experiments.track_verifier.build_dataset --bundle output/eagle-training/bundle.json --truth output/eagle-training/ground_truth.json --feature-family clip_quality_proxy_v1 --confirm-training-split --evaluator-root ../VQLoC-official --out output/eagle-training/dataset.json
python -m experiments.track_verifier.run_cv --data output/eagle-training/dataset.json --recipes experiments/track_verifier/recipes.json --evaluator-root ../VQLoC-official --fit-full-model --out output/eagle-training/quality-head
```

`--excluded-sources FILE.json` can supply the reserved source-ID list. Training
rejects known `val`/`test` bundles; the confirmation flag still requires the
operator to verify provenance. Need at least six training sources and both track
classes. Labels are stIoU≥0.25 positive, stIoU<0.05 negative; ambiguous tracks are
excluded from BCE. No evaluation targets enter inference.

The default recipe is the later development control: 12→8→1 GELU head, AdamW
lr 0.003, weight decay 0.01, 200 epochs, source/query weighting, seeds0/1/2.
Source-group outer CV assesses the added head; three inner folds select the
override threshold. Added inputs use matched base initialization and zero initial
added columns. `--fit-full-model` subsequently fits all training sources and
writes `model.json` for a **new prospective evaluation**.

Outputs: per-recipe `prediction_freeze.json`, fold audit, `choices.npz`,
`COMPLETE.json` metrics/seed variation and optional `model.json`. Repeat a command
to reuse completed recipes with identical hashes; use a fresh folder after code,
input or recipe changes. Repeated recipe selection still carries development
overfitting risk. The simple latest-center control is competitive and belongs in
comparisons. See [results](../reproduction/RESULTS.md).
