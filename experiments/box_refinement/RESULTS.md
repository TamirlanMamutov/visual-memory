# Box-refinement development results

51official-training development queries/36sources; new subset31queries/16sources; pretrained REN may have seen these examples. No comparison with teammate full-val EAGLE scores.

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


This is the box-refinement experiment, not evidence of useful additional timing/text input.
