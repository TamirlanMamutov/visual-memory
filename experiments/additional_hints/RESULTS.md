# Additional timing-hint simulation results

51official-training development queries/36sources; new subset31queries/16sources; pretrained REN may have seen these examples. No comparison with teammate full-val EAGLE scores.

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


These results must remain separate from ordinary image-query box refinement.
