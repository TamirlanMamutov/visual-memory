# Candidate-based visual memory experiments

| Folder | Experiment | What changes? |
|---|---|---|
| [additional_hints](additional_hints/README.md) | Extra timing/context input and oracle-derived simulations | Candidate choice, potentially the time interval |
| [box_refinement](box_refinement/README.md) | Image-based localization refinement | Boxes only; baseline times/confidence stay fixed |
| [candidate_tools](candidate_tools/README.md) | Shared REN/EAGLE export and official evaluation | Common data format and scoring |
| [ren_context_refinement](ren_context_refinement/README.md) | Legacy compatibility paths | Old commands still work |

Keep additional-hint and image-only refinement results in separate tables.
Oracle-derived timing inputs are not real user-memory or ordinary benchmark input.
Licensed data, weights, crops and query-level ground truth are not committed.

To retrieve only these experiment files without switching your working branch:

```text
git fetch origin
git restore --source=origin/codex/ren-hints-box-refinement-20261120 --worktree -- experiments/__init__.py experiments/README.md experiments/additional_hints experiments/box_refinement experiments/candidate_tools experiments/ren_context_refinement
```

The original model-training entry points and teammate branches are not modified.
