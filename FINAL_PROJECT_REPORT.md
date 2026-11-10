# Visual Memory Project: Dataset Readiness and Training Report

**Initial report date:** 2026-11-09; **latest update:** 2026-11-10.
**Initial repository snapshot:** `1d0fd3804ab98ac0f1404b3fcc6258fac31cae88`.
**Current implementation branch:** `tamirlan/multimodal-localization-model`.
Changes are committed and pushed on that branch. Existing local changes in
`GUI-README.md` and `experiment_gui.py` remain untouched and uncommitted.

## Latest status: real Ego4D subset prepared and trained

The user added `Ego4D/v2/clips` on November 10. There are **39 non-empty MP4
clips**, totaling 2,131,649,333 bytes. Their clip IDs match **117 training
queries across 31 source videos** and **27 validation queries across 8 source
videos** in the existing processed VQ2D annotations. There is no source-video
overlap between these splits. This is a small local subset of 13,607 training
and 4,504 validation annotation records, not the complete dataset.

### What now works end to end

1. `prepare_ego4d_localization.py` decodes the clips, crops canonical visual
   queries from the annotated query-image frames, and maps the 5-FPS annotation
   indices to the actual media FPS (the inspected clips are 30 FPS).
2. RGB frames and query crops are centered in square padding, bicubically
   resized and ImageNet-normalized. Target boxes are clipped to image bounds
   and transformed into normalized XYXY coordinates on that same padded canvas.
3. A frozen, pretrained DINOv2 ViT-S/14 at 224 pixels produces patch features.
   This is an explicitly smaller CPU baseline than the proposed ViT-B/14 at
   448 pixels; the preparer supports the latter as its default configuration.
4. Float32 features/boxes and boolean masks are persisted as trainer-compatible
   `.npy` bundles, with per-query frame IDs, FPS, clip IDs, and encoder provenance.
5. The localization model trains on real image-query examples and validates
   on the separate local validation subset. Checkpoints and metrics are saved.
6. `inspect_localization.py` reloads the chosen checkpoint and exports all
   sampled-frame predictions, clip timestamps, original/padded coordinates,
   query-crop thumbnails, and predicted/ground-truth box overlays.

Training selects up to 16 balanced response/non-response frames per query.
Validation uniformly selects 64 frames from the pre-query history without
using response boxes. The bundles contain **1,870 valid training frames**,
including 787 response frames, and **1,728 validation frames**, including
68 response frames. Response frames were sampled for 26 of 27 validation
queries; the remaining short response was missed by sparse sampling.
Labels outside the annotated response track mean "not part of this response,"
not independently verified physical absence of the object.

### Measured ten-epoch run

Configuration: CPU, four threads, seed 42, AdamW, learning rate 0.0003,
weight decay 0.0001, batch size 2, hidden dimension 64, four attention heads,
one temporal layer. The visual backbone remains frozen. Model selection uses
lowest validation loss; this selected **epoch 2**, not epoch 10.

| Model / checkpoint | Validation loss | Presence F1 | Mean box IoU on positive frames |
|---|---:|---:|---:|
| Randomly initialized localization head, same seed/configuration | 2.0454 | 0.0000 | 0.0041 |
| Selected checkpoint, epoch 2 | **1.8862** | 0.0426 | 0.0061 |
| Final checkpoint, epoch 10 | 2.2780 | 0.1095 | 0.0058 |
| Always predict presence | Not applicable | 0.0757 | Not applicable |
| Constant mean training-target box | Not applicable | Not applicable | 0.0000 |

These are **poor localization results**: 0.0061 IoU is about 0.61% overlap,
not 61%. The selected model's presence F1 is below the always-present baseline.
Training loss fell from 2.0290 to 1.3479, while validation loss worsened after
the early minimum. This is consistent with overfitting/distribution mismatch;
the run does not demonstrate useful localization or justify more epochs alone.
Presence and spatial predictions need further architectural/loss investigation.

All metrics are sparse sampled-frame diagnostics. Mean positive IoU evaluates
boxes on labeled positive frames regardless of the model's presence decision.
These results are **not stAP25**, not official VQ2D scores, and not held-out
test-set results. The validation subset was used for checkpoint selection.
The model does not yet produce official scored response-track proposals.

### Validation and artifacts

- All **19 unit tests passed**, covering annotation geometry, actual video
  decoding/FPS conversion, query modes, masks, gradients, visual review export,
  normalized output bounds, and batch-size-invariant validation metrics.
- `pip check` passed. Added dependencies are declared in
  `requirements-localization.txt`; pretrained encoder code/weights are cached
  locally outside the repository.
- A regression test exposed float32 weighted-average coordinates exceeding
  1 by approximately 1.2e-7; final box output is now clamped to its contract.
- Validation loss now aggregates presence by valid-frame count and box losses
  by positive-frame count, so changing evaluation batch size does not change
  the reported objective. The recorded ten-epoch run was repeated after this
  fix; exploratory earlier run folders are not the authoritative results.

Authoritative local artifacts:

- Features and preprocessing provenance: `output/ego4d-dinov2-small-224/`
- Recorded run: `output/ego4d-localizer-validated-run/` (`best.pt`, `last.pt`,
  `metrics.jsonl`, `run.json`, and `COMPLETE`)
- Visual review: `output/ego4d-localizer-review-validated/review.html`
- Prediction export: `output/ego4d-localizer-review-validated/predictions.json`
- Small committed aggregate summary: `results/ego4d-localization.json`

Raw data, generated features, weights and review images stay local and ignored
by Git. The new review is separate from the existing experiment GUI.

### What is still missing

The **downloaded Ego4D subset is now ready for image-query feature training**;
the statement that all localization media is missing no longer applies.
However, the model is not yet a useful finished system. Next steps are to
improve spatial/query matching and temporal presence prediction, check the
effect of balanced training versus sparse-history validation, run exhaustive
history/official track evaluation, and scale up the training data.
Text/combined-query feature extraction, verified text-query targets,
interaction cues and integration of appropriately labeled EPIC/Ego-Exo4D
data remain unimplemented. This run does **not** train one model on all three
datasets.

The remaining sections below preserve the November 9 retrieval experiments
and earlier development stages; their missing-Ego4D-media statements are
historical and superseded by this latest update.

## Executive summary

The datasets are **not all ready in one common training format**. The project
now has a validated, shared PyTorch sample/batch contract, but adapters do not
replace missing source media/annotations or turn separate tasks into one
model. The existing EPIC-KITCHENS JPoSE **retrieval** path is runnable; its
pretrained checkpoint reproduced the expected validation scores. Ego4D VQ2D
annotation JSON is present, but its query images and videos are absent locally.
The local Ego-Exo4D videos lack the take manifest and annotations. Therefore
neither localization dataset is ready for supervised training/evaluation here.

A one-epoch scratch JPoSE run and two longer three-epoch learning-rate trials
were used to compare retrieval hyperparameters. The longer trials use the
same seed, batch size, margin, triplet count, and validation split, changing
only the learning rate. In this small comparison, 0.01 outperformed 0.003,
but neither scratch model beat the pretrained checkpoint. Three epochs are a
limited screening run, not a complete optimization or convergence study.

## What “one format” means here

`data_contract.py` standardizes **in-memory samples**, not files on disk. It
defines task-tagged inputs, targets, masks, IDs, and metadata. The conventions
are:

| Field | Type/shape | Convention |
|---|---|---|
| Video frames | float32 `[T,3,H,W]` | RGB, values in `[0,1]` |
| Video features | float32 `[D]` or `[N,D]` | Precomputed feature representation |
| Query image | float32 `[3,H,W]` | RGB, values in `[0,1]` |
| Query features | float32 `[D]` or `[N,D]` | Feature representation |
| Query token IDs | int64 `[L]` | Token IDs; padded with a boolean attention mask |
| Localization boxes | float32 `[T,4]` | Normalized `[x1,y1,x2,y2]` |
| Relevance targets | float32 | Retrieval supervision |
| Validity/presence masks | bool | Frame, query-time, token, or object validity |

The VQLoC adapter reorders the source box convention `[y1,x1,y2,x2]` into
`[x1,y1,x2,y2]`. The collator pads temporal sequences and prevents mixing
retrieval, localization, and unlabelled video-only examples in the same batch.
Retrieval and localization still require task-specific models/heads and
supervision; a shared tensor type does not make their learning objectives
interchangeable.

## Dataset readiness

Counts below describe this local workspace, not official dataset totals.

| Source | Available locally | Ready for current supervised training? | Blockers/notes |
|---|---|---|---|
| EPIC-KITCHENS JPoSE retrieval data | Required train/validation dataframes, pooled video features, text features, relational dictionaries, validation relevance matrix, and pretrained checkpoints are present under `JPoSE_data/data`. | **Yes, for the existing retrieval baseline.** It ran successfully. | This is feature-based retrieval, not object-box localization. |
| Raw EPIC-KITCHENS videos | 9 MP4 files; `P01_02.MP4` is zero bytes. | Not required for JPoSE feature retrieval; **not sufficient for localization training** on its own. | Existing retrieval labels/features do not provide the proposal's query/response box tracks. |
| Ego4D VQ2D / VQLoC | `train_anno_new.json` has 13,607 records and `val_anno_new.json` has 4,504 records under `VQLoC/data`. | **No, not end-to-end.** Annotation records exist, but the loader cannot form samples from the current assets. | No local query-crop images, clip videos, or full-scale videos were found under `VQLoC`. |
| Ego-Exo4D | 183 downscaled MP4 streams (about 1.98 GB). | **No, not for supervised localization.** | No local `takes.json` or pose annotation tree. Pose data would not by itself provide VQ2D response tracks. |
| EPIC annotation and retrieval code repositories | Source and annotation files are present. | Useful supporting code/data, not an additional unified model-ready dataset. | Upstream repositories remain separate. |

The current data layout therefore supports the EPIC/JPoSE retrieval experiment
and sample-contract tests. It does **not** make every dataset ready for the
proposal's text/image localization experiments or for training one combined
network.

## Training and evaluation

### Environment

- Python 3.11.9 in the repository's ignored `.venv`
- PyTorch 2.15.0 CPU build; CUDA unavailable
- NumPy 2.4.6 and pandas 2.2.3
- Four CPU threads used by the experiment runner
- `pip check`, launcher UI smoke test, Python syntax compilation, and all 7
  shared-contract unit tests passed

An initial pandas wheel failed to import its native extension. Installing
pandas 2.2.3 in the local virtual environment fixed that issue; no dependency
manifest was changed.

### Measured results

| Run | Initialization | Epochs | Learning rate | Batch size | Avg. nDCG | Avg. mAP | Status |
|---|---|---:|---:|---:|---:|---:|---|
| Official pretrained checkpoint | Pretrained JPoSE | — | — | — | **0.690** | **0.734** | Rerun passed; matches upstream README |
| Scratch reference | Random initialization | 1 | 0.01 | 64 | 0.537 | 0.479 | Completed |
| Scratch trial A | Random initialization | 3 | 0.01 | 64 | 0.595 | 0.570 | Completed; better than the one-epoch scratch run |
| Scratch trial B | Random initialization | 3 | 0.003 | 64 | 0.534 | 0.474 | Completed |

The 0.690 nDCG / 0.734 mAP result is a good reproduction of the published
JPoSE **retrieval validation baseline**. The scratch runs start from random
initialization, not from the pretrained checkpoint. The default 0.01 trial
improved on the one-epoch scratch result; the 0.003 trial did not. The 0.01 run
is the best setting measured among these two three-epoch trials, but both
remain below the pretrained model. This small validation-only comparison is
not a broad search, a convergence study, or a test-set result.

No localization metric (stAP25, tAP25, recovery, or success) has been measured.

### Run artifacts

Prior successful runs:

- Fresh pretrained evaluation: `output/epic-baseline-20261109-203431/`
- One-epoch scratch training and validation:
  `output/epic-baseline-20261109-203713/`

Three-epoch sweep:

- Trial A: `output/epic-baseline-20261109-211424/`
- Trial B: `output/epic-baseline-20261109-213757/`

Trial A trained for 1,399 seconds and trial B for 1,336 seconds. Their
respective summed triplet losses by epoch were `456.04, 300.71, 260.45` and
`608.45, 411.09, 356.41` (sums, not mean losses). Both decreased during
training. Trial A improved over the one-epoch scratch result but remained
below the pretrained model. Trial B also remained below both trial A and the
pretrained model.

Each completed run stores `run.json`, stage logs, metric JSON, the trained
checkpoint, and a `COMPLETE` marker.

## Code and documentation changes

- `data_contract.py`: shared typed sample/batch records, validation, JPoSE
  feature adapter, VQLoC adapter, unlabelled-video adapter, and padded
  same-task collation.
- `tests/test_data_contract.py`: seven unit tests for dtypes, box-coordinate
  conversion, padding, unlabelled samples, and task separation.
- `run_epic_baseline.py`: command-line flags for epochs, batch size, learning
  rate, margin, and number of triplets; run metadata records the settings.
- `README.md`: documented hyperparameter command and shared contract.
- `DATASET_ANALYSIS.md`: initial inventory and task/data limitations.

The runner's hyperparameter flags let each run be repeated, for example:

```powershell
python .\run_epic_baseline.py --epochs 3 --learning-rate 0.003 --batch-size 64
```

## What should happen next

1. Treat learning rate 0.01 as the better of these two tested settings; keep
   the pretrained checkpoint as the strongest measured model.
2. For a meaningful from-scratch comparison, continue training the 0.01
   candidate for substantially more epochs, track validation scores by epoch,
   and use early stopping. The upstream default is 100 epochs; this short
   screening run does not justify claiming convergence.
3. If the goal is to improve the pretrained retrieval result, add a separate,
   controlled fine-tuning path; the current runner intentionally trains fresh
   models and does not fine-tune the pretrained checkpoint.
4. To train localization, make the authorized Ego4D VQ2D query crops and
   clip/full-scale videos available and connect the processed annotations to
   the VQLoC loader.
5. To use Ego-Exo4D for supervised localization, obtain the matching take
   manifest and appropriate object/query/response labels. Its MP4s alone are
   not sufficient.
6. Confirm the EPIC zero-byte video is an incomplete download before relying on
   it for any raw-video processing.

## Viewing changes and results

Open the nested `visual-memory` folder in VS Code. Source Control will show the
new files (`data_contract.py`, `DATASET_ANALYSIS.md`, this report, and `tests`)
as untracked, and the modified files as changes. New files may not have green
line-diff gutters until opened or staged; **do not stage/commit unless desired**.

The existing GUI is launched with `Start Experiments.cmd`; its results table
loads completed EPIC run metrics. Individual JSON files under each run folder
are the authoritative metrics. Generated outputs and the `.venv` are ignored
by Git.

## Multimodal localization implementation update

The separate branch `tamirlan/multimodal-localization-model` adds an initial
trainable localization prototype without combining EPIC retrieval labels with
object-box labels:

- `multimodal_localizer.py` cross-attends video patch features to image and/or
  text query features, models temporal context, and predicts per-frame
  presence plus normalized `[x1,y1,x2,y2]` boxes.
- `localization_loss` combines presence BCE, positive-frame box L1, and GIoU
  losses, with padding and absent-object frames masked appropriately.
- `train_multimodal_localizer.py` loads memory-mapped feature bundles, trains
  and validates the model, and saves `best.pt`, `last.pt`, and JSONL metrics.
- `tests/test_multimodal_localizer.py` covers each query mode, batch masks,
  gradient flow, empty-object frames, and a synthetic train/validation pass.

This is a **feature-level model prototype**, not an end-to-end video/text
system. It expects precomputed features from fixed image/video and text
encoders; it does not yet provide those encoders, generate features from raw
media, or implement the official VQ2D/stAP25 benchmark. The existing VQ2D
annotation files remain locally present, but the query crops and paired clips
are missing. Therefore no real localization training or quality claim is
reported. The synthetic training pass verifies that the implemented data
contract and optimizer path execute, not that the model localizes real objects.

To progress to a measured model, first make the licensed VQ2D query crops and
paired clips available, define and verify train/validation feature extraction
for all enabled modalities, generate the documented `.npy` bundles, and add
official localization evaluation. Until then, the strong EPIC/JPoSE numbers
above remain retrieval results only.
