# Candidate bundle contract and adding data

The shared methods accept exported tracks, not raw videos alone. The generator
may be REN or the teammate's EAGLE. Its original prediction is always an explicit
baseline; each additional candidate is a contiguous, pre-query track.

```json
{
  "version":1,
  "annotation_split":"train",
  "feature_family":"ren_quality_v1",
  "queries":[{
    "key":"unique_clip_annotation_query",
    "source":"original_source_video_uid",
    "query_frame":100,
    "fps":5,
    "score":0.7,
    "baseline_id":"native",
    "candidates":[{
      "id":"native",
      "bboxes":[[10,20,30,80,100],[11,21,30,81,100]],
      "image_score":0.4,
      "name_score":0.3,
      "samples":[{"frame":10,"image_score":0.4,"name_score":0.3}],
      "quality_features":[0,0,0,0,0,0,0,0,0,0,0,0]
    }]
  }]
}
```

The zeros above illustrate dimension only; they are **not valid substitutes for
measured REN evidence**. Boxes are `[frame,x1,y1,x2,y2]`, not xywh. Frame numbers
are integer indices on the declared5FPS timeline; all predicted frames satisfy
`0<=frame<query_frame`. All coordinates in a bundle/targets share one declared
pixel space. The reference crop can occur later; it is a query input, not a
permitted frame in the returned response track. `source` groups clips from the
same original video and must not be replaced by query ID.

Separate scoring file:

```json
{"evaluation_only":true,"queries":[
  {"key":"unique_clip_annotation_query","bboxes":[[10,20,30,80,100]]}
]}
```

Exactly one target and one prediction per selected query are required. Never
silently remove empty predictions. `old:true` is a historical reporting tag only.
Inference commands do not read targets; `build_dataset` deliberately reads **train**
targets to fit our added heads.

## New REN candidates

Generate native and alternative tracks with your licensed subset and the released
REN pipeline. Save original confidence, pixel/timeline geometry, candidate crop
similarities and original 12 quality features. For legacy artifacts in our existing
study layout, `candidate_tools/export_local.py --runs-root ... --out ...` exports
the51-query box/hint study. It is **not** a generic raw-video REN runner.

For a new REN generator, write a small converter into the schema above. Candidate
generation and scanning must match the intended baseline; cache and identify
their configuration/weights. Hand-constructing a better candidate from a target
box is not a deployable preprocessing method. Candidate quality/oracle coverage
may be measured after prediction freezing for error diagnosis only.

Our12base features are named in `track_verifier/core.py::FEATURE_NAMES`:
REN score, query-similarity mean/p10/std, valid sample fraction, log-area std,
relative center shift, neighbor-box IoU, duration/end/start fractions and frame
density. Reimplementing a similarly named statistic differently changes the
feature family; do not use a frozen head without a numerical parity check.

## EAGLE candidates

Use [the adapter](EAGLE_TRANSFER.md), which decodes reference/candidate crops at
annotation coordinates and5FPS. It supports saved SAM variants, three sampled
crop frames per candidate, image/name scores and optional mean CLIP vectors.
`clip_quality_proxy_v1` statistics are explicitly a new feature family; fit a
new head on training sources. Full EAGLE inference/checkpoints are not bundled.

## New hand inputs

Additionally provide `video_path` relative to `--video-root`, `frame_width`,
`frame_height`, and `geometry.bounds=[left,top,right,bottom]` if raw borders must
be cropped before resize. Set `full_source_video=true` with `video_start_sec`
when reading a full source rather than its benchmark clip. Clip files use
`full_source_video=false`. Verify decoded crop alignment before extracting.
Never mix full-source seconds, clip seconds, source FPS and5FPS frame IDs.

## Select more queries without introducing leakage

Choose them before examining outcomes, spread source videos first, record split
and query IDs, keep all selected cases including failures, and reserve new
sources for prospective evaluation. More queries from one video are correlated:
report both query and original-source counts, and bootstrap by source.
Additional train data/variants are optional; none require downloading all Ego4D.
