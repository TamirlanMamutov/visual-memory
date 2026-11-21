# Recorded REN pilot results

These are **development experiments**, not full-benchmark or real-user results.
There are 51 official-training queries from 36 source videos: the original 20
queries and 31 newly completed queries. The new subset has 16 sources. The released
REN checkpoint may already have seen training examples. All comparisons below use
the same cached candidate pool (135 tracks), and preserve native REN AP confidence.

## Window-preserving image box refinement

| Cohort | Method | stAP25 | tAP25 | Recovery | Success |
|---|---|---:|---:|---:|---:|
| All 51 | REN baseline | 0.4712 | 0.6121 | 56.55% | 58.82% |
| All 51 | Window image refinement | 0.4857 | 0.6121 | 63.94% | 62.75% |
| New 31 | REN baseline | 0.4644 | 0.6102 | 51.90% | 58.06% |
| New 31 | Window image refinement | 0.4925 | 0.6102 | 65.53% | 64.52% |

This gain is **image-only**, not evidence of useful text fusion. Six queries' boxes
changed overall. Serviettes and pliers explain much of the new-subset gain; tin
worsened. A source-bootstrap interval for the new-subset mean track-IoU difference
was [-0.0016, +0.0529], so this is not confirmed generalization. Time windows and
confidence scores are unchanged. Text/name refinements did not improve new-subset
stAP25; description-based refinement recovered only one additional frame overall.

## Oracle-derived simulated timing hints

**GT target midpoints generate these hints. They are privileged synthetic inputs,
not real remembered interactions, not a user study, and not a benchmark gain.**

Expanded study: five simulation seed blocks x 40 draws = 200 draws. It retains
51 queries, rather than treating those draws as independent data. Source-disjoint
permutations vary per draw, and the timing-density prior uses other source folds.

New 31, mean stAP25:

| Condition | Native REN | Latest-center rule | Image + hint | Robust mixture | Image + shuffled hint |
|---|---:|---:|---:|---:|---:|
| 2-second noise | 0.4644 | 0.5394 | 0.6300 | 0.6244 | 0.4703 |
| **5-second noise (primary)** | 0.4644 | 0.5394 | 0.5761 | 0.5827 | 0.5099 |
| 10-second noise | 0.4644 | 0.5394 | 0.4653 | 0.4674 | 0.4475 |
| 20% wrong hints, 5-second noise | 0.4644 | 0.5394 | 0.5629 | 0.5701 | 0.5099 |
| 5-second earlier bias, 2-second noise | 0.4644 | 0.5394 | 0.4516 | 0.4234 | 0.4307 |
| Synthetic early/middle/late | 0.4644 | 0.5394 | 0.5946 | 0.5946 | 0.3923 |

At the primary condition, correct versus shuffled hints increased average Top1 hit
rate by 0.055, with a source-bootstrap 95% interval [0.004, 0.106]. This is an
interval for **Top1 hit-rate difference**, not stAP. There are 16 new source videos,
multiple exploratory conditions, and no independent user evaluation.

The first, smaller simulation used one fixed shuffled assignment and 30 noise
draws; its 5-second shuffled result (0.604) exceeded the correct-hint result
(0.571). That finding motivated this expanded control study. Both studies remain
part of the experimental record. More shuffled assignments reduce Monte Carlo
sensitivity; they do not replace independent video sources.

Accurate time information can help in this simulation, but a generic recency rule
is already strong and earlier biased clues can hurt. Natural-language action
understanding and recorded speech have not been demonstrated. The source-grouped
prior and synthetic hints must not be represented as evaluation-time information
available from ordinary image queries.

## Reproducibility checks

- Portable export: all 51 window-refined tracks match the original experiment
  exactly, including frame IDs, boxes and AP confidence.
- Portable simulation: all Top1/Top2 selections and all reported metric means match
  the original expanded study (metric tolerance 1e-12).
- Official evaluator: candidate track IoUs checked individually; full metric calls
  verify 12 scenario/cohort combinations in each simulation run.
- Seven behavioral tests cover time units, frame cutoffs, window/confidence
  invariants, empty predictions and EAGLE pool/source selection.
- The EAGLE adapter has not been used for an actual EAGLE inference/export on this
  machine. Its video/CLIP path needs a small paired check on the teammate's machine.

Aggregate numeric results are in [results/ren_pilot_summary.json](results/ren_pilot_summary.json).
Raw licensed annotations, query IDs, crops, videos, weights and credentials are
not committed.
