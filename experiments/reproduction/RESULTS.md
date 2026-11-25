# Results, limitations and experiment priorities

Snapshot: 2026-11-24. All our stAP25/tAP25 figures below use the official evaluator
with **all/all** pairing. Recovery/Success are percentages. Cohorts differ between
tables: do not rank methods across them as a matched benchmark comparison.

## Reserved validation: complete 50 queries /25 disjoint sources

Models/thresholds were frozen before evaluation, trained only on the previous60
training queries/40 sources. Every planned query completed. No final-set tuning
or outcome-based stopping. This is a fixed subset of official validation, **not
the full Ego4D benchmark**.

| Method | Role | stAP25 | tAP25 | Recovery | Success | Spatial mIoU | Mean track IoU |
|---|---|---:|---:|---:|---:|---:|---:|
| Native released REN pipeline | Baseline |0.3917|0.4622|43.40|54.00|0.3870|0.3292|
| Expanded candidates + stats/image/name verifier | Predeclared primary |0.4038|0.4761|48.43|56.00|0.4254|0.3508|
| Expanded candidates + stats-only verifier | Predeclared secondary |0.4447|0.5216|52.99|60.00|0.4717|0.3919|
| Expanded candidates + stats/image verifier | Predeclared secondary |0.4038|0.4761|48.43|56.00|0.4254|0.3508|

Primary stAP25 difference **+0.0121**, paired source-bootstrap95% CI
**[-0.0391,+0.0678]**. Stats-only secondary difference **+0.0530**, CI
**[-0.0027,+0.1314]**. Both AP intervals include zero. Stats-only Recovery
difference **+9.59 percentage points**, CI **[+0.96,+21.45]**; secondary analyses
are not adjusted for multiple comparisons.2,000 source draws retain both queries
and recompute pooled AP. Intervals condition on fixed models, excluding
training/method-selection variability. Encouraging point gains do not establish
a full-benchmark or statistically secure AP improvement.

Image/name and image-only arms made identical final choices. Annotation object
names are supplementary text, not independently collected natural queries.
There is no demonstrated semantic-fusion benefit. Stats-only wins on this subset
but must remain labeled secondary; do not retrospectively rename it primary.

## Repeated training-source development

60 queries/40 sources,160 fixed candidates. Five source outer folds, three inner
threshold folds and three seeds. Hyperparameter selection used these data
repeatedly; they are development results. A released REN backbone may have seen
these official-training examples. Source separation applies to our added head.

| Method | All60 stAP25 | New40 reporting subset stAP25 | New40 Recovery |
|---|---:|---:|---:|
| Native REN |0.4463|0.4325|52.34|
| Stats:lr 0.003, source-balanced, AP threshold selection |0.5096|0.4803|64.56|
| Simple latest-center candidate control |0.5152|0.5147|—|
| Stats + all hand proxies |0.4509|0.5195|58.52|
| Stats + zero-added-feature control |0.5096|0.4803|64.56|
| Stats + within-query shuffled hand control |0.4508|0.4587|64.15|

The simple timing control is competitive; do not attribute all gains to learned
quality. Actual names did not outperform the corresponding image-only head in
the corrected v11 study. Nine hand/control recipes were negative/mixed overall.
Hands detected48/60; usable camera-compensated motion31/60. These are proximity/
motion/release **proxies**, not labeled contact events. No hand variant was
promoted into the reserved evaluation.

Completed sweeps: v10 tested20 recipes (lr, decay, epochs, width, dropout,
ranking loss, seed disagreement, overlap gate, robust normalization, clipping,
image/name/shuffled/relative features), plus12 postprocessing configurations.
v11 tested14 matched-initialization/AP/source-balancing recipes. The portable
training package reproduces the head recipes; the12 historical postprocessing
configurations are documented analyses, not a new generic sweep command here.
The original20-query quality-head study reached0.6233 stAP25, a small repeatedly
developed cohort; larger source-diverse development reduced it. It is not a
full-benchmark result or evidence of beating published REN.

## Separate spatial and extra-input pilots

| Experiment | Cohort | Observation | Interpretation |
|---|---|---|---|
| Image box refinement, same time/confidence |51 development queries/36 sources |stAP25 .4712→.4857; Recovery56.55→63.94 |Small spatial gain; source-IoU interval includes zero |
| Same refiner, new31 reporting subset |31 queries/16 sources |stAP25 .4644→.4925; Recovery51.90→65.53 |Reporting subset, not new untouched validation |
| Cached multimodal ranking |20 development queries |Image Top1 7/20; name8/20; nameTop3 14/20; candidate oracle15/20 |No consistent fixed-fusion gain; candidate ceiling limits retrieval |
| Typed description / TTS transcript |Same20 |Assistant-written text; synthetic speech transcription had1word error/222 |Interface feasibility, not natural user-query or acoustic-event evaluation |
| Noisy/incorrect/earlier timing hints |51 development queries |Precise simulated time can help; wrong/biased clues hurt |Explicitly GT-derived simulation, not benchmark improvement |
| First real coarse-hint pilot |7 review requests,5 clues / 2 declines; scored31development queries |stAP25 .4644→.4416; Recovery51.90→54.91 |Three changes:1 helped / 2 harmed; truthful earlier memories can conflict with last appearance |
| New fallible-clue policy |8 new review sources selected |All-declined synthetic interface check only |Scoped/confidence-annotated human study remains pending |

Keep automatic Top1, shortlist coverage, oracle ceiling, real clues and simulated
clues separate. The [hint results](../additional_hints/RESULTS.md) retain noisy-
hint and request-budget controls. The [box results](../box_refinement/RESULTS.md)
give exact spatial-study numbers. Detailed aggregate JSON accompanies this file.

## Priority for the next experiments

1. **Transfer image box refinement to final EAGLE+SAM run6 candidates.** First
  20 source-spread queries; fixed final SAM window/confidence; manually check crops
  and all changed cases. Promising compatible add-on, transfer gain still unknown.
2. **Audit the completed50 and preserve its frozen comparison.** Share evaluations,
  weights, source-bootstrap intervals and failure cases. Further model selection
  needs a fresh prospective cohort. No full-benchmark expansion queued automatically.
3. **Quality-head EAGLE transfer:** derive training-source proxy features, retrain
  separately, include native/latest-center controls. REN weights cannot be reused
  blindly across feature families.
4. **Fallible human hints / multimodal:** collect independent scope/confidence;
  report paired harm/fallback and matched-size shortlist controls. Improve candidate
  coverage before claiming text/audio fusion. Real recorded speech remains pending.
5. **Hand cues:** lower priority because coverage and overall development gains are
  weak. Do not spend full-validation compute on the current hand head.

Teammate's corrected full 4,501-query run6 reports0.4228stAP25. Our50-query
stats-only0.4447 is **not a head-to-head win**. A useful merge is a common report
with separate cohort/model tables and a later matched EAGLE add-on ablation,
not pooling predictions or combining unlike scores.
