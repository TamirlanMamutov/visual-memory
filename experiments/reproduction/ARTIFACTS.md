# Artifact exchange checklist

## Our private package

`ren_handoff_v1.zip` is prepared locally for authorized private sharing, not
uploaded by this branch. `MANIFEST.json` records every included file's SHA256.
The ZIP checksum/size and closed cohort are recorded in `HANDOFF_ARCHIVE.json`.
The public Git branch contains code, instructions and aggregate results only.

| Folder | Contents | Needed for |
|---|---|---|
| `models` | Three frozen three-seed verifier heads, JSON weights/normalizers/thresholds | Replay our reserved-validation inference |
| `validationNN` | Closed scored batch: label-free candidates, separate targets, saved predictions and expected metrics | Independently score our current held-out readout |
| `development60` | 60 queries/40 sources, 160 candidates; labels, features, folds, recipes, expected OOF selections; separate input bundle | Retrain our heads and reproduce controls |
| `development60/hand_input_bundle.json` | Video-relative paths, source/clip offsets and pixel coordinate transforms | Fresh hand extraction |
| `development60/videos_needed.json` | Selected source/clip download identifiers and sizes | Download only necessary licensed media |
| `development60/geometry_refs` | Private small crops + hashes, no full frames/videos | Verify pixel geometry before hand features |
| `development20` | Original20 quality-head dataset/folds/expected metrics | Reproduce historical0.6233 development score |
| `box_hints51` |51-query cached candidate bundle, sample scores, separate targets | Box and simulated timing experiments |
| `multimodal20` |20-query cached modality scores/rankings and targets | Exact original query-modality pilot replay |
| `human_pilot7` |First seven requested human-clue records (five clues/two declines),31-query bundle/targets and frozen predictions | Replay actual limited human pilot |
| `human_clues8` |Eight all-declined synthetic interface records | Interface/fallback check only; no human-study evidence |

Do not put these derived licensed targets/crops or the ZIP in public Git. No raw
video or large pretrained weights are included. Inference reads input bundles
only; targets are separate for scoring. Training datasets deliberately contain
training targets. The owner-only `export_legacy.py` assembles this archive from
our existing study folders; teammates do not need the owner's folder layout.

## What to request from the teammate for EAGLE

Either saved evaluation traces or trained weights suffice for different levels
of replay; obtain the run configuration/commit as well:

1. **Fastest exact scoring/ensemble replay:** each run's
   `output/<run>/eval_parts/*.json`, plus the same `vq_val.json` annotations.
2. **Final run6 replay without propagating SAM again:** also
   `output/ens3-mask0/sam2/small-m30/*.json`, and any saved final prediction file.
   `output/ens3-mask0/eval_parts` by itself is the pre-SAM6a prediction!
3. **Fresh model inference:** `last.pt` for all three run IDs below, with the
   matching config/commit/dependency versions and licensed validation clips.
   This costs more than rescoring saved traces.

Run IDs: `eagle-20261111-221053`, `eagle-20261114-002928`,
`eagle-20261114-204835`. In the teammate layout checkpoints live at
`../cs230-cache/eagle/checkpoints/<run>/last.pt`.

```text
python -m experiments.reproduction.inspect_eagle --project-root .. --repo-root . --out output/eagle-artifact-inventory.json
```

These artifacts were not found on our laptop at packaging time. Public code
alone does not reproduce their exact trained model. Retraining from scratch is
a new reproduction with seed/configuration differences, not saved-run replay.
EgoTracks and VISOR are needed only if retraining the corresponding teammate
EAGLE variants; our cached replay and postprocessing do not require them.

## Model provenance and reproduction boundary

Our REN candidates were generated using the released REN checkpoint, upstream
`savya08/REN` commit `91876b67ce98e8a7ac2a2176cf6c90be6945105e`,
DINOv2 ViT-L/14 and SAM2.1-hiera-large. This was inference with released weights,
not retraining REN's region encoder. The local runtime used float32 Apple MPS;
author CUDA/BF16 numerical equivalence has not been established.

| Upstream asset | SHA256 on our machine |
|---|---|
| REN `checkpoint.pth` |`17e14eac4052ec5d10f2dc36f1b0368b9edc1a3f33bc199ed2fbb32ca01e1ce5`|
| `dinov2_vitl14_pretrain.pth` |`d5383ea8f4877b2472eb973e0fd72d557c7da5d3611bd527ceeb1d7162cbf428`|
| `sam2.1_hiera_large.pt` |`2647878d5dfa5098f2f8649825738a9345572bae2d4350a2468587ece47dd318`|

This branch starts from exported candidate tracks. It does not package the
entire owner-specific REN preprocessing/scan scheduler or claim a fresh full
REN benchmark reproduction on another machine. For new REN inference, use the
[upstream implementation](https://github.com/savya08/REN), record config/weights/
commit, then export its candidates using [our schema](SCHEMA.md). For adding our
methods to the teammate's already-computed EAGLE candidates, no REN setup is needed.
