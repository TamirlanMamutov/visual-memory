# Visual Memory Project: Dataset Readiness and Training Report

**Report date:** 2026-11-09
**Repository:** `visual-memory` at `1d0fd3804ab98ac0f1404b3fcc6258fac31cae88`
**Commit status:** No commits made. Existing local changes in `GUI-README.md` and
`experiment_gui.py` were preserved.

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
