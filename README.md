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

The run writes `last.pt`, the lowest-validation-loss `best.pt`, and per-epoch
`metrics.jsonl`. It reports presence F1 and mean IoU on positive frames; these
are prototype diagnostics, **not** the official VQ2D/stAP25 evaluation. The
feature-bundle contract and a synthetic train/validation pass are unit tested.
No real localization training was run: the local VQ2D annotations are not
paired with the required query crops and videos, and this worktree has no
localization feature bundles. A complete raw-media feature extractor and
official benchmark adapter remain future work.

Run the contract tests with:

```powershell
python -m unittest discover -s .\tests
```
