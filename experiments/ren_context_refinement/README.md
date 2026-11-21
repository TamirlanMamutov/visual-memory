# Compatibility entry points

The two experiments now have separate folders and results:

- [Additional hints](../additional_hints/README.md): candidate selection using
  independently provided context, or a clearly labeled oracle-derived simulation.
- [Box refinement](../box_refinement/README.md): spatial changes inside the original
  predicted time window, without an additional timing hint.
- [Shared candidate tools](../candidate_tools/README.md): REN/EAGLE export,
  unchanged baseline, validation, tests and official evaluation.

Existing `run.py`, `simulate.py`, `evaluate.py`, export and import paths remain thin
compatibility wrappers. Old example JSON files and the historical aggregate result
snapshot remain available. New work should use the separated entry points.

Update the new folders together if you previously copied only this legacy folder.
Use a fresh output directory after changing code revisions; frozen run hashes are
not bypassed or overwritten.
