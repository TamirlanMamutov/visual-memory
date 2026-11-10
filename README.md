# Visual memory experiments

Local launcher and runners for the EPIC-KITCHENS retrieval baseline and Ego-Exo4D data review. The upstream code repositories and datasets are kept outside this repository.

## Shared folder layout

Use the same folder names on each teammate's machine; the parent can be anywhere:

```text
CS230 Project/
  visual-memory/                    # This repository
    Start Experiments.cmd
    experiment_gui.py
    results/                        # Small committed reference measurements
    output/                         # Generated runs; ignored by Git
  Joint-Part-of-Speech-Embeddings/
    src/
  JPoSE_data/
    data/
      models/JPoSE_BEST/
      video_features/
      text_features/
      dataframes/
      relational/
      relevancy/
  Ego-Exo4D/
    takes.json
    takes/
    annotations/ego_pose/
  EPIC-KITCHENS/                     # Optional raw videos for retrieval error review
    P01/videos/
    P02/videos/
```

`experiment_paths.py` finds data relative to the repository's parent. No personal Drive path is required. Existing `JPoSE_data/data` contains the prepared features and pretrained checkpoint; do not download a duplicate. Raw EPIC videos are not needed for feature-based evaluation. Make shared Drive data locally available before running.

If repositories and data have a different parent, set `CS230_PROJECT_ROOT` to that parent. Otherwise no path settings are needed.

## Setup and launch

Use Python 3.10 or newer with Tkinter and open a terminal in `visual-memory`:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
python .\experiment_gui.py
```

On macOS/Linux activate with `source .venv/bin/activate` and launch with `python experiment_gui.py`. Tkinter may require a separate system package on Linux. The already-tested environment on the original Windows PC can be used directly. `requirements.txt` lists dependencies; it is not an exact environment lock. The recorded original torch/NumPy versions are in `results/baseline.json`.

On Windows, **Start Experiments.cmd** uses the local `.venv` if present, otherwise Python on PATH.

## Commands

```powershell
# Evaluate the downloaded pretrained retrieval checkpoint
python .\run_epic_baseline.py --epochs 0

# Train a fresh model for one epoch and validate it
python .\run_epic_baseline.py --epochs 1

# Train a fresh model for 3 epochs with a candidate learning rate
python .\run_epic_baseline.py --epochs 3 --learning-rate 0.003

# Export the pretrained model's validation retrieval errors
python .\inspect_epic_retrieval.py

# Check Ego-Exo4D videos and pose annotation coverage
python .\check_egoexo_dataset.py --sample-json 10

# Prepare the local 10-video query-labeling page
python .\prepare_egoexo_review.py
```

Open `output/egoexo-query-review/review.html` after preparation. HTML video links are relative to that page; JSON/CSV data paths are relative to `CS230 Project`. Export annotations before closing or reloading the page. Review generation refuses to overwrite existing labels.

The GUI displays commands, live logs, saved metrics and committed reference measurements. See [GUI-README.md](GUI-README.md) for controls and [results/README.md](results/README.md) for measured results.

## Evaluation scope

EPIC evaluation uses Joint-Part-of-Speech-Embeddings with the supplied pooled features. Expected README averages are nDCG 0.690 and mAP 0.734. One epoch checks training and validation; it does not reproduce every EPIC paper experiment or the paper's Table 11. The runner currently uses four CPU threads, even with CUDA available.

The Ego-Exo4D command checks local availability, nonzero video sizes and sampled annotation JSON syntax; it does not run an object-localization model or verify full video integrity. Current pose annotations cannot substitute for object query boxes and response tracks. Localization scoring still requires verified ground truth and model integration.

The EPIC runner accepts `--batch-size`, `--learning-rate`, `--margin`, and
`--num-triplets` to compare fresh JPoSE runs. Runs always initialize a new
model; they do not resume or fine-tune the pretrained checkpoint. Use the same
validation set and seed when comparing settings, and do not treat a short run
as a converged model.

Datasets, pretrained checkpoints, trained weights, generated logs and review pages stay in shared Drive and are ignored by Git. Each teammate needs their own authorized access to the data. Historical logs/configuration snapshots in `output/` retain original absolute paths for provenance; executable scripts and new review data use the portable layout.

## Shared sample contract

`data_contract.py` defines the common PyTorch sample/batch boundary for retrieval
features, VQLoC localization samples, and decoded video without labels. It
standardizes tensor dtypes, image ranges, and box coordinates without converting
the source files or pretending the tasks have the same labels. Retrieval and
localization must still use separate model heads and homogeneous batches.
See [DATASET_ANALYSIS.md](DATASET_ANALYSIS.md) for dataset scope, local coverage,
and setup limitations.
The current results and next steps are summarized in
[FINAL_PROJECT_REPORT.md](FINAL_PROJECT_REPORT.md).

## Multimodal temporal localization prototype

`multimodal_localizer.py` adds a trainable feature-level model that predicts an
object box and presence score for each video frame. It accepts image-query
features, text-query features, or both. The model requires precomputed features;
it does not yet include or download the image/video backbone or text encoder.
The same frozen feature encoders and preprocessing must be used for train and
validation data.

The trainer reads each split from a directory of `.npy` arrays:

| File | Required shape | Meaning |
|---|---|---|
| `video_features.npy` | `[N,T,P,Cv]` | Per-frame visual patch features |
| `boxes_xyxy.npy` | `[N,T,4]` | Normalized target boxes `[x1,y1,x2,y2]`; use zero boxes when absent |
| `object_present.npy` | `[N,T]` | Boolean/0-1 object-presence labels |
| `image_query_features.npy` | `[N,Q,Ci]` | Optional per-query visual features |
| `image_query_mask.npy` | `[N,Q]` | Optional boolean mask for image-query tokens |
| `text_query_features.npy` | `[N,L,Ct]` | Optional per-query text-encoder features |
| `text_query_mask.npy` | `[N,L]` | Optional boolean mask for text-query tokens |
| `frame_mask.npy` | `[N,T]` | Optional boolean mask for padded video frames |

At least one query-feature array is required. Every sample needs one valid query
token and at least one valid frame. Array dtypes may be standard numeric types;
the loader converts features and boxes to float32 and masks to bool. The loader
memory-maps arrays and copies samples into batches.

```powershell
python .\train_multimodal_localizer.py `
  --train-features .\data\localization\train `
  --val-features .\data\localization\val `
  --output-dir .\output\localizer `
  --epochs 20 --batch-size 8 --learning-rate 0.0003
```

The run writes `last.pt`, the lowest-validation-loss `best.pt`, per-epoch
`metrics.jsonl`, the configuration in `run.json`, and a `COMPLETE` marker.
It refuses to overwrite an existing output directory or use the same bundle
for both training and validation. It reports presence F1 and mean IoU on
positive frames; these are prototype diagnostics, **not** official VQ2D/stAP25
evaluation.

### Preparing downloaded Ego4D clips

The November 10 local download contains 39 non-empty clips under
`Ego4D/v2/clips`, matching 117 training queries (31 clips) and 27 validation
queries (8 clips) in the sibling `VQLoC/data/*_anno_new.json` files.
This is a small subset, not the full Ego4D dataset.

```powershell
python -m pip install -r .\requirements-localization.txt

# CPU-sized initial experiment: frozen DINOv2 ViT-S/14 at 224 pixels.
python .\prepare_ego4d_localization.py `
  --output-dir .\output\ego4d-dinov2-small-224 `
  --backbone dinov2_vits14 --image-size 224

python .\train_multimodal_localizer.py `
  --train-features .\output\ego4d-dinov2-small-224\train `
  --val-features .\output\ego4d-dinov2-small-224\val `
  --output-dir .\output\ego4d-localizer-validated-run `
  --epochs 10 --batch-size 2 --hidden-dim 64 --num-heads 4 --temporal-layers 1

python .\inspect_localization.py `
  --features .\output\ego4d-dinov2-small-224\val `
  --checkpoint .\output\ego4d-localizer-validated-run\best.pt `
  --output-dir .\output\ego4d-localizer-review-validated
```

Open `output/ego4d-localizer-review-validated/review.html` to view query crops,
predicted boxes in red, and ground-truth boxes in green. The review shows
the highest-presence sampled frame and a separately labeled ground-truth
positive diagnostic frame when one was sampled. It also exports all sampled
frame predictions to `predictions.json`, with clip timestamps and boxes in
both padded-image and original-image coordinates.

The preparer:

- Selects only annotations whose clip files exist, reporting matched and
  total counts; rejects overlapping source videos across train/validation.
- Converts the annotations' 5-FPS frame IDs to each video's actual FPS
  rather than using annotation indices directly on 30-FPS media.
- Extracts the canonical visual query crop directly from its annotated
  clip frame. Separate query-image downloads are therefore unnecessary.
- Applies centered square padding, bicubic resize, and ImageNet normalization;
  adjusts response boxes for the identical padding geometry.
- Freezes a pretrained DINOv2 encoder downloaded from Meta's public weights,
  using pinned upstream code. First use needs internet access and disk space.
- Uses 16 balanced positive/negative training frames per query, and 64
  uniformly sampled pre-query validation frames. **Validation frame selection
  never uses response boxes.** Sparse sampling can miss short response tracks;
  `preparation.json` reports positive-frame/query coverage. No temporal
  interpolation or padding frames are scored as labeled positives.
- Saves float32 feature/box arrays, boolean masks, per-sample provenance in
  `samples.json`, and preprocessing details in `preparation.json`. Only use
  bundles after their parent directory contains `COMPLETE`.

Defaults are ViT-B/14 at 448 pixels, matching the proposed backbone/size;
the explicit ViT-S/14/224 command above is a cheaper CPU baseline, not the
full proposed configuration. Both runs here are image-query only: text
encoding, verified text-query labels, interaction cues, exhaustive temporal
search and official benchmark evaluation remain unimplemented. EPIC retrieval
and Ego-Exo4D videos are **not** silently mixed into this localization run.
Generated features, checkpoints, review images, and licensed media stay local,
outside Git. Use a fresh output directory for each command rerun.

### Improved spatial matching and joint EPIC/Ego4D training

`spatial_localizer.py` is the current image-query experiment. It replaces
unanchored global box regression with normalized query/patch matching and
box offsets anchored to the patch grid. A shared residual feature projection
also feeds an auxiliary EPIC noun classifier. The old temporal prototype
remains available for reproducibility; the new spatial head is framewise,
not a completed temporal tracking or text-query system.

EPIC supplies **noun labels, not boxes**. The extractor reads four frames per
action segment from canonical non-empty local training videos, using the
same frozen ViT-S/14/224 preprocessing as the Ego4D bundle. It excludes
zero-byte files and suffixed video exports whose annotation alignment has
not been established. The verified subset has 31 segments from `P01_10`
and `P02_05`, covering 12 noun classes. No EPIC validation/test segment is
used by the default command.

```powershell
python .\prepare_epic_localization.py --output-dir .\output\epic-dinov2-nouns

# Matched Ego4D-only ablation
python .\train_spatial_localizer.py `
  --train-features .\output\ego4d-dinov2-small-224\train `
  --val-features .\output\ego4d-dinov2-small-224\val `
  --output-dir .\output\spatial-ego-only --epochs 10

# One model: Ego4D localization + 0.1 * EPIC noun-classification loss
python .\train_spatial_localizer.py `
  --train-features .\output\ego4d-dinov2-small-224\train `
  --val-features .\output\ego4d-dinov2-small-224\val `
  --epic-features .\output\epic-dinov2-nouns `
  --output-dir .\output\spatial-epic-ego-joint --epochs 10

python .\inspect_localization.py `
  --features .\output\ego4d-dinov2-small-224\val `
  --checkpoint .\output\spatial-epic-ego-joint\best.pt `
  --output-dir .\output\spatial-joint-review
```

Outputs already exist for the recorded experiments; choose fresh directories
when rerunning. The joint trainer selects `best.pt` by maximum validation
positive-frame mean IoU (unlike the older trainer's minimum loss). It combines
one Ego4D batch of four queries and one EPIC batch of up to 16 segments per
step, cycling the small EPIC set. AdamW uses learning rate 0.0001 and seed 42;
both ablations preserve the same Ego4D order and initialization. Epoch logs
record EPIC loss and update counts to make joint learning auditable.

The joint checkpoint reaches **0.08854 mean positive-frame IoU**, compared
with 0.00607 for the previously selected temporal prototype. The Ego4D-only
spatial ablation also reaches 0.08854: **EPIC learning is wired and measured,
but extra localization benefit has not been demonstrated**. Joint response
F1 is only 0.05896, below the always-positive 0.07572 baseline. These remain
poor, exploratory sparse-validation results, not official VQ2D scores.
See [results/spatial-joint-localization.json](results/spatial-joint-localization.json)
and [FINAL_PROJECT_REPORT.md](FINAL_PROJECT_REPORT.md) for precise comparisons.
Open `output/spatial-joint-review/review.html` to inspect improved predictions.

### Longer-training diagnosis and safer model selection

The later experiments did **not** establish an overall improvement over the
recorded 0.08854-IoU joint checkpoint. Two joint-training candidates stopped
after 20 and 43 epochs (an 80-epoch cap with patience 15). False positives
decreased, but box IoU decreased too. A subsequent selection audit includes
epoch zero; none of its trained epochs beats the starting matcher on
development IoU, so the initialized candidate is retained in that audit.
Do not substitute these experimental weights as an improved production model.
See [results/longer-training.json](results/longer-training.json) for all outcomes.

The changes address specific training/evaluation weaknesses:

- `spatial_loss(..., assigned_boxes=True)` trains box refinement at the patch
  closest to the target center, rather than at an incorrect predicted peak.
- `--positive-presence-weight 0.1` downweights positive presence examples in
  the response-balanced fitting sample. This is a tested heuristic, not a
  guarantee of calibrated probabilities or a proof that label shift is solved.
- Seven training-source videos supply 21 natural-history development queries.
  Their videos are excluded from fitting, leaving 96 queries in 24 videos.
  Neither the checkpoint nor the presence threshold is chosen on the 27-query
  official-validation subset during these runs. That validation subset has
  already been used in prior research iterations; it is not a pristine test set.
- Early stopping monitors development IoU. The current trainer includes the
  initialized model as a selection candidate so worse trained weights are not
  automatically promoted. Presence thresholds maximize development F1 only.
- An optional `--training-box-prior` initializes sizes from fitting-only
  positive boxes. It did not improve development results enough to recommend.
- `evaluate_spatial_localizer.py` adds response-frame AP, precision/recall,
  false-positive counts, positive-frame IoU25 fraction, joint frame-level
  precision/recall, and highest-score-frame query counts. None are official
  VQ2D track metrics or stAP25.

```powershell
python .\prepare_ego4d_localization.py `
  --output-dir .\output\ego4d-development-history `
  --backbone dinov2_vits14 --image-size 224 `
  --development-videos 7 --seed 42 --val-frames 64

python .\train_spatial_localizer.py `
  --train-features .\output\ego4d-dinov2-small-224\train `
  --val-features .\output\ego4d-dinov2-small-224\val `
  --development-features .\output\ego4d-development-history\development `
  --epic-features .\output\epic-dinov2-nouns `
  --output-dir .\output\spatial-development-audit `
  --epochs 80 --patience 15 --assigned-boxes --positive-presence-weight 0.1
```

Recorded output directories already exist; choose fresh names for reruns.
`best.pt` stores the development-selected raw model, `calibrated.pt` also
stores its development-derived presence offset, and `final-validation.json`
contains the final checkpoint's evaluation. Epoch log field `validation`
means the **development** split when `--development-features` is provided;
it does not contain per-epoch official-validation results.

### Published weights and larger GPU experiments

Frozen DINOv2 weights are already in use. They are pretrained visual features,
not a paper-trained VQ2D localization checkpoint. The official
[VQLoC README](https://github.com/hwjiang1510/VQLoC) advertises a checkpoint,
but its Box URL returned HTTP 404 to both HEAD and GET during this investigation
on November 10. A search of the public upstream issues and Hugging Face model
list did not yield a verified replacement. The partner's exact model/config
is still unknown. No official localization weights were loaded or reproduced.
VQLoC uses a different architecture, so its checkpoint cannot simply be loaded
into `SpatialLocalizer`; reproducing it requires that architecture, preprocessing,
and evaluator. The existing pretrained JPoSE scores are retrieval-only.

An AWS GPU VM is reasonable for a larger authorized-data experiment, not a
prerequisite for cloning/running this repository. Start with a budgeted
single-GPU pilot for feature extraction/inference and measure GPU memory needs
before sizing training. Upstream VQLoC lists a much larger default training
configuration (roughly 200 GB aggregate GPU memory); do not assume one GPU
reproduces it unchanged.

`train_spatial_localizer.py --device auto` or `--device cuda` supports placing
the shared-head training on a GPU; CUDA requests fail explicitly if unavailable.
The Ego4D feature extractor already selects CUDA when available. The EPIC
extractor currently remains CPU-based. GPU execution is **not validated on
this CPU-only laptop**. Install an appropriate CUDA-enabled PyTorch build
on the GPU machine using PyTorch's official installation instructions.

Clone the implementation branch, provide licensed media and annotations
separately (they are not in Git), and regenerate or copy the compatible
feature bundles. Use private storage, persistent checkpoints, budget alerts,
and stop the instance when not needed; stopped-instance storage can still
cost money. Start with more representative labeled VQ2D training clips,
not all of Ego4D's unrelated tasks or all three datasets at once. More data
may help, but poor matching, response labels, temporal modeling, and evaluation
must also be addressed. No cloud resources have been provisioned.

Run all tests with:

```powershell
python -m unittest discover -s .\tests
```
