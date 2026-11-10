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
