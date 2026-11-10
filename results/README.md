# Reference results

| Model | Average nDCG | Average mAP |
|---|---:|---:|
| Downloaded pretrained checkpoint | 0.690 | 0.734 |
| From scratch after one epoch | 0.536 | 0.478 |

Measured on November 7, 2026 with the upstream validation evaluator. The pretrained scores match the Joint-Part-of-Speech-Embeddings README. A one-epoch model is not converged and these results do not reproduce Table 11 of the EPIC paper. Training starts from random initialization rather than fine-tuning the supplied checkpoint.

`baseline.json` contains the portable reference configuration and scores. The GUI labels these rows as reference measurements. New local results are written under `../output/`, which is excluded from Git. Existing shared Drive checkpoints and detailed logs remain there after the move.

The full epoch completed before an upstream diagnostic export error. The experiment runner restored missing validation IDs, successfully evaluated the saved checkpoint and completed the export without repeating training. The upstream source repository was not modified.

## Ego4D localization subset: November 10

[ego4d-localization.json](ego4d-localization.json) records the first real
image-query localization run: 117 training queries and 27 validation queries,
frozen DINOv2 ViT-S/14 features at 224 pixels, and ten training epochs.
The checkpoint selected by validation loss is epoch 2, with presence F1
0.0426 and mean positive-frame box IoU 0.0061. These are **poor prototype
results**, not official VQ2D/stAP25 scores or the EPIC retrieval metrics above.
The existing GUI does not display this localization run.

See [the latest project report](../FINAL_PROJECT_REPORT.md) for limitations,
baseline comparisons and generated artifact paths. Open the local
`../output/ego4d-localizer-review-validated/review.html` for the query/box review.
