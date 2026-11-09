# Desktop experiment launcher

Double-click **Start Experiments.cmd** in this folder, or open PowerShell here and run:

```powershell
python .\experiment_gui.py
```

The launcher uses Python's built-in Tkinter interface. EPIC commands additionally require `torch`, `numpy`, `pandas`, `tqdm` and `tensorboard`. Launch from the Python environment that has these dependencies. On Linux, Tkinter may need a separate system package. Project paths are inferred from the parent of `visual-memory`; `CS230_PROJECT_ROOT` overrides the dataset/repository location.

## Training tabs (TempRet and EAGLE)

The **TempRet · EK-100 retrieval** and **EAGLE · Ego4D VQ2D** tabs run `run_tempret.py` and `run_eagle.py` with the GPU environment at `<drive root>\cs230-venv\env` (or `CS230_PYTHON`). Each field maps to a command-line flag; the full command appears in the command box before the run starts.

- **TempRet:** *Decode frames + cache features* (resumable), *Train + evaluate*, *Zero-shot CLIP eval*, *Evaluate checkpoint…*, *Continue from checkpoint…* and *Train reranker on checkpoint…* (retrains only the cross-encoder, about 3 minutes, and picks the fusion weight on held-out training videos). The ablation boxes reproduce the paper's "w/o temporal" and "w/o reranking" rows. Results are labelled *subset* unless all 9,668 test clips are local.
- **EAGLE:** *Prepare SAM pseudo-masks* (resumable, so rerun it as clips download), *Train*, *Prepare + train + evaluate*, *Evaluate checkpoint…* and *Continue from checkpoint…*, plus no-AMM / no-GLM ablations.
- A progress bar follows the current step or iteration. The results table shows finished metrics, the latest epoch or iteration of running or stopped runs, and the papers' reference numbers. Checkpoints are written to `<drive root>\cs230-cache\<method>\checkpoints`, which **Open checkpoints folder** opens.
- The header shows how much data is local (EPIC videos, Ego4D VQ2D clips and downloads in progress, VISOR).

See [REPRODUCTION.md](REPRODUCTION.md) for the implementation, its deviations from the papers, and the data each method needs.

## EPIC-KITCHENS (JPoSE baseline)

- **Evaluate pretrained model** runs `python run_epic_baseline.py --epochs 0` and displays the saved validation nDCG and mAP. Expected README averages: 0.690 and 0.734.
- **Train from scratch + validate** runs the same script with your chosen epoch count. The default is one full epoch. The script evaluates the downloaded checkpoint first, trains a fresh model, then evaluates the new checkpoint. It is not fine-tuning the downloaded model.
- **Export retrieval errors** runs `python inspect_epic_retrieval.py`, saving the pretrained model's top-five predictions for every validation text query.

This is the Joint-Part-of-Speech-Embeddings retrieval baseline using the already downloaded features. It does not reproduce all EPIC paper tasks or establish the paper's Table 11 numbers. Training currently uses four CPU threads. Historical run metrics load immediately, and new rows appear when a command finishes. Double-click a result row to open its folder.

## Ego-Exo4D

- **Check videos & pose annotations** runs `python check_egoexo_dataset.py --sample-json 10`. It counts local MP4 streams and takes, checks nonzero video sizes, matches pose annotations to takes, and parses ten JSON files. This checks availability and sampled annotation syntax; it does not decode or checksum all videos.
- **Prepare 10-video query review** runs `python prepare_egoexo_review.py`. It references existing RGB files and hand annotations without copying videos. It refuses to overwrite query records containing labels.
- **Open query review / label objects** opens the local HTML video review. Export labels before closing the page. These custom queries are still unlabelled until manually annotated.

There is currently no Ego-Exo4D object-localization inference or stAP score in this launcher. The downloaded pose annotations cannot serve as object query/response tracks. Such evaluation needs the model integration and verified object ground truth.

## Running commands

The command appears above the results table and can be copied. The code stores relative script/data locations, while copied commands expand the interpreter and script paths to absolute paths on the current machine. This lets you paste a generated command from any working folder. Absolute paths are computed at runtime, never hardcoded for a particular teammate. Windows commands use PowerShell-safe quoting and the `&` call operator; macOS/Linux use shell quoting. Restart the launcher after updating its code. Logs stream into the lower panel and are saved under `output/launcher-logs`. EPIC stage logs and metrics stay in their timestamped run folders. **Stop current run** terminates only the launched Python command and retains partial outputs. It does not mark a partial run complete. Opening the launcher starts no training and downloads nothing.

Each teammate runs their own local copy against locally available shared data. This interface does not change Drive sharing permissions or edit upstream Git repositories.

## Verification on this PC

The interface was constructed successfully with Tk 9.0. The integrated launcher test ran pretrained EPIC evaluation and the Ego-Exo4D dataset check through the actual GUI callbacks; both completed successfully and refreshed the displayed results. The data check found 805 takes, 3,847 MP4 streams, 50.00 GB of video, and no takes missing pose annotations. These checks do not establish video decode integrity or object-localization accuracy. Visual inspection through the computer-use helper was unavailable because its runtime failed to initialize.
