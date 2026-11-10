# Dataset analysis and shared model-input contract

This note describes the project workspace as inspected on 2026-11-09. Dataset
counts are a local snapshot, not official dataset totals.

## November 10 update: Ego4D subset added

The earlier missing-media statements below describe the November 9 snapshot.
There are now 39 non-empty MP4 clips (2,131,649,333 bytes) under the sibling
`Ego4D/v2/clips` directory. Clip IDs match **117 training queries in 31 clips**
and **27 validation queries in 8 clips** in the existing processed VQ2D
annotations. This is not the complete dataset.

`prepare_ego4d_localization.py` now reads the matching clips, reconstructs the
canonical query crops using visual-crop annotations, maps annotation frame
numbers to the media FPS, and extracts frozen DINOv2 patch features. Separate
query-crop images and full-scale source videos are not required for this new
clip-based path. The original upstream VQLoC loader is unchanged.

The new localization trainer uses a task-specific feature bundle:
float32 video patch features `[N,T,P,C]`, visual query patch features `[N,Q,C]`,
normalized padded-image XYXY boxes `[N,T,4]`, and boolean presence/validity
masks. This spatial feature bundle is distinct from JPoSE's pooled retrieval
features and is documented in the README. It is not a conversion of all
three datasets to interchangeable supervision.

Training frame selection balances annotated response and non-response
frames. Validation uniformly samples only the history before the query time,
without consulting response boxes; short response tracks may be missed.
These are sparse frame-level prototype diagnostics, not the official
exhaustive VQ2D benchmark. Only the image-query path is prepared; text and
combined query training still need encoders and verified query/target labels.
Ego-Exo4D and EPIC localization-label limitations below remain unchanged.

## Project shape

The workspace combines several related but distinct projects:

- The CS230 proposal targets episodic object retrieval/localization with text,
  image, or combined queries. It names EPIC-KITCHENS-100 and Ego4D VQ2D and
  proposes 5 FPS clips, 448x448 frames/query crops, pretrained visual features,
  and response boxes over time.
- `visual-memory` is a local launcher, not a unified training repository. Its
  EPIC path evaluates/trains the JPoSE retrieval baseline from precomputed
  feature files. Its Ego-Exo4D path checks files and prepares an unlabelled
  manual review page; it does not run localization inference.
- `Joint-Part-of-Speech-Embeddings` and `C5-Multi-Instance-Retrieval` provide
  retrieval code, relations, relevance matrices, and pretrained checkpoints.
  JPoSE converts numeric batches to float32 tensors. Its video/text embeddings
  are not interchangeable with decoded RGB frames.
- `VQLoC` is a separate Ego4D VQ2D localization implementation. Its dataset
  item contains clips `[T,3,H,W]`, query images `[3,H,W]`, per-frame boxes,
  and presence/query-time masks. Processed train/validation annotations are
  local; the required clips, query crops, and full-scale videos are not.
- `Ego-Exo4D` is not the same dataset as Ego4D VQ2D. Pose annotations are not
  VQ2D response-track labels or EPIC object boxes.

## Current local data coverage

- EPIC-KITCHENS has nine local MP4 files; one is zero-byte (`P01_02.MP4`). It also has
  the EPIC-KITCHENS-100 annotations and the JPoSE feature/data bundle.
- `JPoSE_data/data` contains train/validation/test dataframes, text and video
  feature pickles, relational dictionaries, a validation relevance matrix, and
  pretrained JPoSE/MMEN checkpoints. The feature data totals about 2 GB in this
  workspace.
- Ego-Exo4D contains 183 downscaled MP4 streams (about 1.98 GB), but the local
  folder has no `takes.json` or annotation JSON. Consequently, the README's
  coverage check and query-review generation cannot run against this local
  copy; the video-only samples are not supervised training examples.
- VQLoC source code and processed Ego4D VQ2D annotation JSON are present:
  `train_anno_new.json` has 13,607 records and `val_anno_new.json` has 4,504.
  The local VQLoC folder has no query-crop images, clip videos, or full-scale
  videos, so its dataset loader cannot produce training examples. Do not treat
  the unrelated Ego-Exo4D MP4s as substitutes.

Raw video files and upstream data are left untouched. The adapters operate on
decoded/model-ready values and do not download data, deserialize pickle files,
or rewrite source annotations.

## Why one model or one tensor dtype is not the goal

Retrieval and localization have different target semantics and model heads.
JPoSE consumes feature vectors and relevance/relational supervision; VQLoC
consumes visual-query/video correspondence and per-frame boxes/presence. A
single `torch.float32` conversion cannot make those tasks equivalent. Instead,
`data_contract.py` defines one sample/batch container with explicit task tags,
source IDs, named modalities, targets, masks, and metadata. Batches must contain
one task and one set of modalities at a time.

The shared convention is:

| Field | Type and shape | Convention |
|---|---|---|
| `video_frames` | float32 `[T,3,H,W]` | RGB, values in `[0,1]` |
| `video_features` | float32 `[D]` or `[N,D]` | precomputed model features |
| `query_image` | float32 `[3,H,W]` | RGB, values in `[0,1]` |
| `query_features` | float32 `[D]` or `[N,D]` | text/query embedding(s) |
| `query_token_ids` | int64 `[L]` | token IDs, padded by the batch collator |
| `boxes_xyxy` | float32 `[T,4]` | normalized `[x1,y1,x2,y2]` |
| `relevance` | float32 | retrieval supervision |
| temporal/query masks | bool | validity, object presence, or before-query state |

`adapt_vqloc_sample` explicitly reorders VQLoC's `[y1,x1,y2,x2]` boxes into the
shared XYXY convention. `make_retrieval_sample` adapts JPoSE/MMEN video/query
features. `make_video_only_sample` represents decoded unlabelled footage for
review or inference only; it does not fabricate targets. `collate_model_samples`
pads variable-length video/box sequences and emits validity masks, while
rejecting mixed-task batches and incompatible image/feature shapes.

## Setup and validation status

The `visual-memory/README.md` setup was applied in an ignored `.venv` using
Python 3.11.9 and `requirements.txt`; dependency consistency, the launcher UI
smoke test, and the shared-contract tests passed. The initial pandas wheel had
a native-extension import failure; replacing it with pandas 2.2.3 resolved the
issue, and the EPIC pretrained evaluation and one-epoch train/validation run
completed. The documented Ego-Exo4D checker still cannot run because the local
copy lacks `takes.json`. No machine policy was bypassed, and upstream assets
were not altered.

## Remaining work to train across sources

1. Make the authorized Ego4D VQ2D query crops and clip/full-scale videos
   available; the processed train/validation annotation JSON is already local.
2. Obtain the matching Ego-Exo4D take manifest and annotations if Ego-Exo4D is
   intended for supervised training; decide which labels map to the task.
3. Decide which EPIC annotations provide object-query/response-box supervision.
   The existing JPoSE retrieval labels alone do not provide VQ2D boxes.
4. Run each source adapter with real samples, then train/evaluate task-specific
   heads and metrics. Keep retrieval and localization batches separate even
   though they share the sample container.
