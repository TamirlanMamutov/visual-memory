# Hand-interaction proxy ablation

An optional development experiment adds hand presence, object/hand proximity,
camera-corrected co-motion and conservative release proxies to the quality head.
It is **not** a contact classifier. No hand/contact labels or action recognition
are used. Completed60-query results were mixed; see
[the aggregate table](../reproduction/RESULTS.md). The reserved validation used
no hand features.

## Replay training without video extraction

Use `development60/dataset.json` and `hand_recipes.json` in the private handoff;
follow [the main guide](../reproduction/README.md). This needs only CPU Torch and
cached numbers, not MediaPipe, FFmpeg, videos or hand-model weights.

## Extract fresh proxies from selected videos

Use a separate Python 3.12 environment to avoid changing an existing EAGLE stack:

```text
python -m pip install -r experiments/hand_interaction/requirements.txt
python -m experiments.hand_interaction.download_model --out ../hand-assets/hand_landmarker.task
python -m experiments.hand_interaction.extract --bundle ../ren_handoff_v1/development60/hand_input_bundle.json --video-root ../Ego4D/v2 --model ../hand-assets/hand_landmarker.task --ffmpeg PATH_TO_INSTALLED_FFMPEG --out output/hands-first2 --max-queries 2
```

The downloader checks the recorded 7.8 MB model SHA256. FFmpeg must already be
installed; no executable is downloaded. For the full60-query replay, remove
`--max-queries`, use a fresh output folder, and obtain only the files listed in
`development60/videos_needed.json`. Paths are relative to `--video-root`:
`full_scale/<source_uid>.mp4` or `clips/<clip_uid>.mp4`. No full dataset is needed.
Each recipient needs licensed Ego4D access; this script does not bypass it.

Coordinates must match the existing candidate tracks, **not just the raw
video resolution**. The handoff includes saved crop bounds, processed dimensions,
full-source offsets and small reference crops. The extractor checks those crops
pixel-for-pixel before computing features; a mismatch stops instead of silently
using misaligned boxes. For new bundles specify the transform/dimensions in
[the schema](../reproduction/SCHEMA.md) and check alignment yourself.

Sampling: six uniform anchors plus adjacent candidate frames, at most18 samples
per candidate; all strictly before `query_frame`,5FPS in our exports. CPU
MediaPipe uses two hands and 0.3detection/presence thresholds. On macOS its image
preprocessing may initialize OpenGL even with the CPU delegate. Use a normal
terminal session if a restricted environment cannot initialize that service.

`features/<key>.json` caches per-query vectors, missingness counts and hashes.
`bundle_with_hands.json` is a label-free bundle with12 hand features per candidate.
Existing geometry/weights/video pixels may cause platform differences in fresh
extraction; cached CV replay is the exact numerical reproduction route.

For a new training study, pass that bundle and separate training targets to
`track_verifier.build_dataset`, then train hand variants. Do not use validation
targets for head training. Zero-added-feature, presence-only and within-query
shuffled controls are important: detector coverage alone can correlate with
visibility. Camera-only movement is not object/hand co-motion, and disappearance
of a detected hand is missing evidence, **not a release event**.

```text
python -m unittest experiments.hand_interaction.test_proxies -v
```
