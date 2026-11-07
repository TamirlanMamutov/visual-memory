# Reference results

| Model | Average nDCG | Average mAP |
|---|---:|---:|
| Downloaded pretrained checkpoint | 0.690 | 0.734 |
| From scratch after one epoch | 0.536 | 0.478 |

Measured on November 7, 2026 with the upstream validation evaluator. The pretrained scores match the Joint-Part-of-Speech-Embeddings README. A one-epoch model is not converged and these results do not reproduce Table 11 of the EPIC paper. Training starts from random initialization rather than fine-tuning the supplied checkpoint.

`baseline.json` contains the portable reference configuration and scores. The GUI labels these rows as reference measurements. New local results are written under `../output/`, which is excluded from Git. Existing shared Drive checkpoints and detailed logs remain there after the move.

The full epoch completed before an upstream diagnostic export error. The experiment runner restored missing validation IDs, successfully evaluated the saved checkpoint and completed the export without repeating training. The upstream source repository was not modified.
