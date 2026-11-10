# Visual Memory Project: Dataset Readiness and Training Report

**Initial report date:** 2026-11-09; **latest update:** 2026-11-10.
**Initial repository snapshot:** `1d0fd3804ab98ac0f1404b3fcc6258fac31cae88`.
**Current implementation branch:** `tamirlan/multimodal-localization-model`.
Changes are committed and pushed on that branch. Existing local changes in
`GUI-README.md` and `experiment_gui.py` remain untouched and uncommitted.

## Latest investigation: published weights, longer training, and cloud readiness

The user requested pretrained-paper settings and more training. The exact
partner checkpoint/configuration is unknown. We investigated public sources,
fixed concrete supervision/selection weaknesses, and ran longer experiments.
**No overall localization improvement is demonstrated by these additional
runs; the earlier joint checkpoint remains the reference.**

### What pretrained weights do and do not mean

The project already uses official pretrained DINOv2 feature weights, and the
older EPIC/JPoSE retrieval checkpoint reproduced its expected retrieval scores.
Neither constitutes loading a trained Ego4D localization model into our head.
Dataset papers do not define a single interchangeable set of model weights.

The current upstream [VQLoC README](https://github.com/hwjiang1510/VQLoC)
advertises a trained checkpoint at its Box URL. On November 10, both HEAD
and a range GET returned **HTTP 404**. We checked the public README, relevant
upstream issues, and a Hugging Face model search without finding a verified
replacement. [Upstream issue 11](https://github.com/hwjiang1510/VQLoC/issues/11)
also asks about reproducing reported results; it is not evidence that our
partner used the same setup. No official localization checkpoint was loaded.
The advertised checkpoint belongs to a different architecture, preprocessing,
and evaluator and cannot be safely loaded into our custom spatial head.

### Diagnosed weaknesses and implemented changes

- The response-balanced fitting data contains about 42.1% positive frames,
  while the sampled validation history contains about 3.94%. This is a
  measurable distribution mismatch, not proof it is the only cause of errors.
- Box regression previously trained the patch selected by the model even
  when it was spatially wrong. Optional assigned supervision now trains the
  patch nearest the annotated center instead. A regression test verifies the
  target patch gets box gradients rather than the incorrect winning patch.
- Presence BCE can downweight positive examples (0.1 in the experiments),
  and a score threshold is selected on development F1 only. These are tested
  mitigations, not a claim that detection or probability calibration is solved.
- Seven videos from the official training subset are reserved for a
  history-sampled development bundle: 21 queries, 1,344 frames, 49 response
  frames, and response coverage for 19 queries. Those source videos are
  excluded from fitting, leaving 96 queries in 24 source videos.
- Checkpoint selection and early stopping use this development bundle.
  The original 27-query validation subset is evaluated after selection, not
  at every training epoch. It has already informed earlier research choices,
  so it is still not an untouched test set.
- The final selection implementation includes epoch zero. A trained model
  that performs worse than the starting matcher is no longer automatically
  promoted merely because it is the best trained epoch.
- Additional diagnostics report response-frame AP, precision/recall, false
  positives, IoU25 fraction, joint frame-level box/presence metrics, and
  highest-score-frame success counts. They are not official track metrics.
  Presence labels mean membership in the annotated response track, not
  exhaustive independent confirmation that the object is absent elsewhere.
- The shared-head trainer now accepts CPU/CUDA/automatic device placement.
  CPU behavior and unavailable-CUDA errors are the locally verifiable scope;
  GPU execution is not exercised on this CPU-only laptop.

### Actual longer-training results

All new trained runs keep the shared EPIC noun task (31 segments), frozen
DINOv2 features, seed 42, AdamW learning rate 0.0001, and an 80-epoch cap.
Patience is 15 epochs without improved development mean IoU.

| Model / experiment | Epochs executed | Selected epoch | Validation mean positive IoU | Presence F1 | False positives |
|---|---:|---:|---:|---:|---:|
| Previous joint reference | 10 | 3 | 0.08854 | 0.05896 | 360 |
| Assigned-box / reweighted candidate | 20 | 5 | 0.08101 | 0.06584 | 167 |
| Same, with fitting-only box-size initialization | 43 | 28 | 0.07295 | 0.08163 | 120 |
| Selection audit including initialized model | 15 | **0** | 0.05361 | 0.00000 | 21 |

These are **separate runs**, not a single combined 78-epoch training run.
The previous model used all 117 training queries and official-validation
selection, whereas the new protocol reserves development videos. Comparisons
therefore include both model and training-protocol changes, not a matched
causal ablation.

The assigned-box candidate executed 480 EPIC auxiliary updates and its fitting
loss fell from 6.3852 to 4.5094. The size-prior candidate executed 1,032 EPIC
updates and fitting loss fell from 6.4033 to 4.0751. Its initial width/height
came from fitting-only median boxes, approximately 0.0825 and 0.0861,
excluding development and validation labels.

Fewer false positives do not mean solved detection: the two candidates find
only 8 of 68 response frames, down from 13 for the previous model. Their
response-frame AP values are 0.04786 and 0.05016 versus 0.04660 previously.
These scores remain low. The first candidate has just one of 27 queries
whose highest-scoring sampled frame both belongs to the response and reaches
IoU 0.25; the prior reference and size-prior candidate have zero. The first
candidate can be inspected at `output/spatial-assigned-review/review.html`.

The first attempt originally selected among trained epochs only. Its initial
development IoU was 0.19681, slightly above the selected trained epoch's
0.19454. After adding initialization to selection and rerunning, the audit
correctly selects epoch zero and stops after 15 epochs. That checkpoint has
**no learned EPIC/Ego4D head updates**; the 360 EPIC updates occurred in
rejected trained candidates. It is not promoted over our earlier joint model.
The size-prior run improves over its own lower initial development IoU,
but remains below the standard initialized development matcher.

This is evidence that more epochs alone are not delivering useful
generalization. It does not prove a specific amount of data will fix the
model, or that all architectures have converged.

### Should we move to AWS? Is insufficient data the cause?

An AWS GPU VM is reasonable for a larger experiment, but it is not required
just to clone the code, and cloud hardware does not repair supervision,
matching, or evaluation. We have only 96 fitting image queries under the
new split and 31 EPIC segments; this limited diversity is a plausible
bottleneck. We also have measured architecture and distribution weaknesses,
no temporal track model, and no paper-trained localization checkpoint.
It would be inaccurate to attribute all poor results to dataset size.

Recommended order:

1. Obtain a verified trained localization checkpoint and its exact model,
   configuration, preprocessing, split, and metric. Reproduce its inference
   on our labeled clips before claiming parity with a paper or a partner.
2. Budget a single-GPU feature-extraction/inference pilot on Linux, using
   compatible CUDA-enabled PyTorch. The current head can request CUDA; the
   Ego4D extractor already auto-selects it, while EPIC extraction remains CPU.
3. Clone the implementation branch and provide licensed annotations/media
   through private storage. Git does not contain datasets, feature bundles,
   or trained weights. Begin with a larger representative labeled VQ2D
   training subset, not all of Ego4D's unrelated tasks or all datasets at once.
4. Measure RAM/VRAM, disk, throughput, and cost before sizing longer training.
   The [VQLoC authors](https://github.com/hwjiang1510/VQLoC) describe roughly
   200 GB aggregate GPU memory for their default training configuration;
   one commodity GPU is not a promise of full-setting reproduction.
5. Keep budget alerts, persistent checkpoints, private dataset access, and
   explicit shutdown procedures. EC2 compute, storage and transfer may incur
   charges, and persistent storage can cost money after an instance is stopped.
   Consult [AWS instance specifications](https://docs.aws.amazon.com/ec2/latest/instancetypes/ac.html)
   and current region-specific pricing before provisioning.
6. Add denser search, temporal tracking and official VQ2D evaluation; use a
   fresh held-out evaluation split before judging final quality.

No AWS resources were created, no credentials were requested, and no spending
was authorized. A GPU pilot and additional licensed data would remove
practical constraints, not guarantee good results.

### Persistent artifacts

- Development features: `output/ego4d-development-history/development/`
- Twenty-epoch candidate: `output/spatial-assigned-development/`
- Forty-three-epoch candidate: `output/spatial-assigned-box-prior/`
- Fifteen-epoch baseline-selection audit: `output/spatial-development-audit/`
- Previous-model extended diagnostics: `output/spatial-joint-extended-metrics.json`
- Aggregate provenance and results: `results/longer-training.json`

The milestone's earlier experiment numbers remain valid historical results;
they were not replaced with the worse follow-up runs.

Validation after the final code changes: all **28 unit tests pass**, Pylance
finds no syntax errors in the five changed/new Python files, `pip check`
reports no broken requirements, and `git diff --check` passes. The aggregate
results were compared against saved run metrics and checkpoints. Tests cover
assigned-patch regression, tied-score thresholds, development-video exclusion
from fitting/box priors, checkpoint reloading, and rejection of worse trained
weights in favor of initialization. An explicit CUDA request on this laptop
raises the intended unavailable-device error without creating a run folder.

## Earlier experiment: spatial matching with shared EPIC/Ego4D training

The subsequent November 10 experiment improves spatial overlap and makes one
model learn from **both** sources through shared trainable parameters. This
section supersedes the earlier single-source results below.

### Changes and supervision

The original temporal prototype regressed global boxes without explicit patch
coordinates. The new `spatial_localizer.py` instead preserves a 16-by-16 spatial
grid, scores video patches against a centrally pooled visual query using
normalized DINOv2 features, and refines a box anchored to the highest-scoring
patch. A zero-initialized residual linear projection preserves pretrained
similarities initially and is learned by both datasets. A spatial cross-entropy
loss with soft Gaussian center targets supplies matching supervision even when
initial box predictions select an incorrect patch. The head is framewise:
temporal track reasoning is future work, not an implemented improvement.

EPIC supplies **segment noun labels only**, not fabricated object boxes.
`prepare_epic_localization.py` decodes four frames per training action segment
and pools frozen ViT-S/14 features using the same 224-pixel preprocessing as
Ego4D. Only two canonical, non-empty local videos match trusted filenames and
timestamps: `P01_10` and `P02_05`, yielding 31 segments and 12 distinct noun
classes. Suffixed video exports were not used because their time alignment
was not established; the zero-byte video was excluded.

The shared projection feeds both query/patch matching and a 300-way EPIC noun
classifier. Every joint optimizer step combines Ego4D presence/L1/GIoU/spatial
losses with **0.1 times EPIC noun cross-entropy**. The small EPIC set is cycled,
with a batch of up to 16 segments paired with each Ego4D batch of four queries.
This is genuine multi-task shared-parameter training, not separate JPoSE
retrieval training and not a claim that the datasets have interchangeable labels.

### Controlled validation comparison

Both spatial-model runs use the same frozen features, 117 training queries,
27 validation queries, sampled frames, seed 42, parameter initialization, and
Ego4D batch order. AdamW uses learning rate 0.0001 and weight decay 0.0001
for ten epochs. The comparison changes only whether EPIC auxiliary batches
contribute to the objective. Their common selection rule maximizes validation
positive-frame mean IoU and selects epoch 3 in both cases.

| Model | Positive-frame mean IoU | Presence F1 |
|---|---:|---:|
| Previous temporal checkpoint (selected by validation loss) | 0.00607 | 0.04255 |
| Initial spatial matcher, before training | 0.05361 | 0.04412 |
| Spatial matcher trained on Ego4D only, epoch 3 | 0.08854 | 0.05882 |
| Spatial matcher trained jointly on Ego4D + EPIC, epoch 3 | 0.08854 | 0.05896 |

Spatial overlap improves from approximately **0.61% to 8.85%** against the
previously selected checkpoint. Selection criteria changed, so a less favorable
comparison for the new model is also recorded: the old temporal run's maximum
IoU at any epoch was 0.00754, still below the new result. These are exploratory
comparisons on a reused validation subset, not an unbiased test-set estimate.
An initial matching probe compared full-query, center-query, and patchwise
matching on this same validation subset; architecture choices therefore also
used validation feedback. No confidence interval or multi-seed claim is made.

**The extra localization benefit of EPIC is not demonstrated.** The two
spatial models differ by less than 0.000001 mean IoU, marginally favoring
Ego4D-only. Joint EPIC training accuracy rises from 0 to 38.7% and full-EPIC
training loss falls from 5.7184 to 4.8274. These are training-set diagnostics,
not EPIC held-out scores. There are 300 logged EPIC auxiliary updates over
ten epochs; tests verify that both losses send nonzero gradients into the
same projection. The selected projection weights differ between ablations,
confirming EPIC is not an unused side head.

### Limitations and next steps

The joint model still performs poorly: only 11 of the 68 annotated positive
validation frames reach IoU at least 0.25, and presence F1 remains below the
always-present baseline (0.07572). At the fixed 0.5 presence threshold it
produces 13 true positives, 360 false positives, and 55 false negatives.
Positive-frame IoU ignores these presence decisions, so it is not overall
retrieval success, temporal AP, or stAP25. Sparse pre-query sampling still
misses one of the 27 annotated response tracks.

Priorities are better response-frame discrimination, denser temporal search
and official track evaluation, larger/diverse EPIC training data, and
multi-seed transfer ablations. The improved model does not yet support raw-text
or combined-query inference, interaction cues, or official VQ2D track output.
Ego-Exo4D remains excluded for lack of matching localization supervision.

### Artifacts and milestone

- Joint model: `output/spatial-epic-ego-joint/`
- Matched Ego4D-only ablation: `output/spatial-ego-only/`
- EPIC feature/label provenance: `output/epic-dinov2-nouns/`
- New query/box review: `output/spatial-joint-review/review.html`
- Committed measured summary: `results/spatial-joint-localization.json`
- Twenty-five unit tests pass, including spatial anchoring, losses from
  both datasets reaching the shared projection, and a reloadable joint run.
- The sibling milestone document's Dataset, Approach, Remaining Work, and
  Partial Results sections now describe this measured local implementation,
  rather than template scores or unverified full-dataset/CLIP claims. Its
  bibliography was updated for the GIoU loss and duplicate citation keys removed.
- The milestone compiles locally to `Project Milestone Document/main.pdf`:
  three pages total, comprising two pages of content and one of references.
  BibTeX resolves the citations and no overfull boxes were reported.

The milestone is outside this Git repository and is edited in place; repository
commits contain the implementation, aggregate results, and documentation,
not a duplicate copy of the milestone.

## Earlier experiment: real Ego4D subset prepared and trained

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
