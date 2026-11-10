# Visual episodic memory experiment summary

November 9, 2026. The milestone records the team's JPoSE, EPIC-KITCHENS, and VISOR pilots. Current localization experiments use Ego4D VQ2D and a frozen DINOv2 patch matcher. The VQLoC checkpoint link was unavailable when attempted. Published methods provide literature references.

| Experiment | Result |
| --- | --- |
| Crop reranking and distributed proposals | Better Success on the initial 33 queries. Distributed proposals underperformed the patch baseline on the later 316-query comparison. |
| Fixed trim and expansion | On 132 queries, stAP25 rose from 0.0073 to 0.0193. The bootstrap interval included zero and Recovery fell. |
| Trim length and box scale sweep | On 316 tuning queries, 20 frames and 15% expansion scored 0.035 stAP25 versus 0.027 raw. On 42 different queries, 15% and 25% tied at 0.0402 versus 0.0357 raw. |
| Adaptive trimming | 0.047 versus 0.035 on tuning and 0.0433 versus 0.0402 on a previously examined 42-query check. Recovery declined slightly on the check. |
| High-resolution crops, track linking, and SAM | Some spatial diagnostics improved. Linking fell short of the existing trim-and-expand baseline, and SAM added no qualifying tracks in seven diagnostic cases. |
| Confidence rescoring | The development gain did not transfer to 59 fresh queries. |
| Adaptive box sizing | Success rose from 11/82 to 13/82. stAP25 stayed at 0.0275. |
| Candidate threshold sweep | The selected threshold did not improve temporal shortlist recall on 42 fresh queries. |
| Recency reselection | Success rose from 10.8% to 16.2%, while stAP25 fell from 0.040 to 0.030 on 74 queries. |
| Learned window gate | On 150 fresh queries, stAP25 rose from 0.0531 to 0.0548. The bootstrap interval included zero. |
| Hand interaction | Appearance plus recency, generic hand presence, and fingertip proximity each found 4/30 qualifying tracks. Dense review found no verified release. |
| Image, text, and synthetic speech | On the shared image-derived shortlist, each mode and the fusion rules found 3/30 at rank one and 4/30 in the top three. An annotation-derived time hint raised rank-one hits to 4/30. A manually written interaction hint did not correct the bottle example. |

The frozen validation queue had completed 23 batches and 866 queries at this snapshot. Batch 023 had started. Fresh confirmation of adaptive trimming and the standalone ranked text evaluation remain pending. The latter generates candidates directly from text. The validation queue uses frozen inference. The learned rerankers were fitted separately on small development sets.

See [the milestone draft](main.tex) and the accompanying diagnostics for measurements and scripts. Some scripts require the project's local cached predictions and clips.

## Follow-up on Recovery and standalone text

- On 316 queries, the raw matcher omitted 2,237 of 4,868 annotated frames and supplied inaccurate boxes for 2,307. Fixed trim and expansion increased omitted frames to 3,893.
- On 162 sampled frames that the original tracks covered, 27 top-ranked boxes met box IoU 0.50. The original top 20 contained an accurate box on 42 frames. Adding smaller proposals raised coverage to 67, and an aspect-ratio prior selected an accurate box on 34. On a 15-query engineering subset selected for short predicted tracks, that variant reduced Recovery from 30.22% to 19.63% and stAP25 from 0.207 to 0.173. Applying it only to small reference crops left both metrics unchanged. CLIP fusion improved the sampled-frame count from 27 to 30. SAM mask refinement tied the box baseline at two hits across 12 frames.
- The frozen adaptive trim improved stAP25 from 0.083 to 0.102 and Recovery from 17.27% to 17.79% on 36 eligible queries from one new source video in batch 023. The earlier 80-query check favored the raw baseline. The frozen raw matcher remains the default.
- Standalone CLIP text search returned zero qualifying tracks in the top three on 30 queries from five videos. Five queries had a qualifying temporal interval among the three results. A separate prompt-ensemble run also had zero spatio-temporal hits. On that run, smoothing reduced temporal top-three hits from two to zero. Restricting text or speech to breaking close image-score ties left the shared shortlist result at 3/30 at rank one and 4/30 in the top three.

## Full-track CLIP reranking and text detector

- On 15 queries from five videos selected for short predicted tracks, DINOv2 plus CLIP reranking held stAP25 at 0.207 and raised Recovery from 30.22% to 33.02% while keeping time windows and track confidence fixed. The nine additional recovered frames came from a net gain across two videos (+10 and −1).
- OWLv2 box refinement at the standalone CLIP text shortlist's timestamps yielded zero qualifying tracks in the top three, tying the original grid boxes on the 22 uniquely matched queries. Eight queries with duplicate video and object-name keys were excluded. The separate standalone OWLv2 search found 1/30 qualifying tracks at rank one, with 3/30 temporal hits in the top three and stAP25 of 0.0021.

- The 18-query CLIP confirmation raised Recovery from 3.32% to 4.80% on six unseen source videos with stAP25 tied at 0.028. The paired Recovery interval was 0.00 to 4.65 percentage points. Detection linking and denser rescanning have no scored result yet.
