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

# Export the pretrained model's validation retrieval errors
python .\inspect_epic_retrieval.py

# Check Ego-Exo4D videos and pose annotation coverage
python .\check_egoexo_dataset.py --sample-json 10

# Prepare the local 10-video query-labeling page
python .\prepare_egoexo_review.py
```

Open `output/egoexo-query-review/review.html` after preparation. HTML video links are relative to that page; JSON/CSV data paths are relative to `CS230 Project`. Export annotations before closing or reloading the page. Review generation refuses to overwrite existing labels.

The GUI displays commands, live logs, saved metrics and committed reference measurements. See [GUI-README.md](GUI-README.md) for controls and [results/README.md](results/README.md) for measured results.

## TempRet and EAGLE

`run_tempret.py` (EK-100 Multi-Instance Retrieval, arXiv 2605.24470) and `run_eagle.py` (Ego4D VQ2D, arXiv 2511.08007) are re-implementations trained from the launcher's first two tabs. They need the GPU environment described in [REPRODUCTION.md](REPRODUCTION.md), which also lists every assumption made where the papers are silent.

## Evaluation scope

EPIC evaluation uses Joint-Part-of-Speech-Embeddings with the supplied pooled features. Expected README averages are nDCG 0.690 and mAP 0.734. One epoch checks training and validation; it does not reproduce every EPIC paper experiment or the paper's Table 11. The runner currently uses four CPU threads, even with CUDA available.

The Ego-Exo4D command checks local availability, nonzero video sizes and sampled annotation JSON syntax; it does not run an object-localization model or verify full video integrity. Current pose annotations cannot substitute for object query boxes and response tracks. Localization scoring still requires verified ground truth and model integration.

Datasets, pretrained checkpoints, trained weights, generated logs and review pages stay in shared Drive and are ignored by Git. Each teammate needs their own authorized access to the data. Historical logs/configuration snapshots in `output/` retain original absolute paths for provenance; executable scripts and new review data use the portable layout.
