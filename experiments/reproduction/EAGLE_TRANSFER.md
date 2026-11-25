# Reproduce teammate EAGLE; then test our add-ons

The [teammate summary](../../VQ2D_SUMMARY.md) reports run6 **0.4228 stAP25** on
4,501 validation queries using the corrected all/all evaluator. The earlier0.531
was large-object-only. Run2 alone is0.281 all/all. These are different cohorts
from our REN pilots. Exact published-paper comparability still depends on the
paper's evaluator filter/protocol; do not claim a confirmed paper tie/beat.

## Replay existing teammate artifacts

Get the [artifact checklist](ARTIFACTS.md) first. Follow their unchanged
[REPRODUCTION.md](../../REPRODUCTION.md) / [EAGLE_RUNS.md](../../EAGLE_RUNS.md)
for EAGLE dependencies. On their machine, rebuild into a **new** output folder:

```text
python eagle_ensemble.py output/eagle-20261111-221053 output/eagle-20261114-002928 output/eagle-20261114-204835 --out output/ens3-mask0-replay
python eagle_sam2.py output/ens3-mask0-replay --clips 0: --model facebook/sam2.1-hiera-small --margin 30 --tag small-m30
```

The second command is fresh SAM propagation, requiring clips/model. With saved
SAM JSON already available, use the adapter below to reconstruct `extend` tracks
and rescore without model inference. Match all 4,501 query IDs and investigate
the three documented unavailable crop frames; do not silently discard queries.

## First useful transfer test:20 source-spread queries

`ens3-mask0/eval_parts` stores **pre-SAM** tracks. Always specify the final SAM
variant when assessing our box refiner on the best final run6 baseline:

```text
python experiments/candidate_tools/adapt_eagle.py --runs output/ens3-mask0 output/eagle-20261111-221053 output/eagle-20261114-002928 output/eagle-20261114-204835 --sam2-dir output/ens3-mask0/sam2/small-m30 --sam2-baseline-variant extend --annotations ../Ego4D/v2/annotations/vq_val.json --clips ../Ego4D/v2/clips --max-queries 20 --out output/eagle-ren-transfer20 --inventory-only
```

For no-model/no-video scoring replay, replace `--inventory-only` with
`--baseline-only`. The zero appearance scores are placeholders: this export is
**not** suitable for ranking/refinement/training.

For the actual box-refinement experiment, replace `--inventory-only` with
`--clip-model openai/clip-vit-base-patch32`; optionally add `--device cuda` or a
local model directory and `--local-files-only`. It decodes only selected queries
and samples, reuses saved EAGLE/SAM tracks, and never retrains EAGLE/SAM.

```text
python experiments/candidate_tools/baseline.py --bundle output/eagle-ren-transfer20/bundle.json --out output/eagle-ren-transfer20/native.json
python experiments/box_refinement/run.py --bundle output/eagle-ren-transfer20/bundle.json --mode image --out output/eagle-ren-transfer20/refined.json
python experiments/candidate_tools/evaluate.py --bundle output/eagle-ren-transfer20/bundle.json --predictions output/eagle-ren-transfer20/native.json --truth output/eagle-ren-transfer20/ground_truth.json --evaluator-root ../VQLoC-official --out output/eagle-ren-transfer20/native_metrics.json
python experiments/candidate_tools/evaluate.py --bundle output/eagle-ren-transfer20/bundle.json --predictions output/eagle-ren-transfer20/refined.json --truth output/eagle-ren-transfer20/ground_truth.json --evaluator-root ../VQLoC-official --out output/eagle-ren-transfer20/refined_metrics.json
```

Baseline window/confidence is the final SAM-extended window/confidence; our
refiner keeps these fixed. `tAP25` must stay identical. Inspect crops/box
coordinates and changed predictions yourself before scaling up. Exact EAGLE
decoding/CLIP transfer has **not** been run on our laptop; saved artifacts are
missing. Synthetic adapter tests verify variant construction and window retention.

For quality-head transfer, first export **training-split** EAGLE predictions and
train the separate `clip_quality_proxy_v1` head as in
[track_verifier](../track_verifier/README.md). Do not blindly apply our REN-quality
weights. Timing/modality methods can reuse the same label-free bundle; compare
each added input or postprocessor independently before combining them.
