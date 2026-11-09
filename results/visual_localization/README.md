# Visual localization pilot

These experiments use frozen pretrained models. They evaluate image-crop object localization separately from the team's JPoSE video–text retrieval experiments.

## EPIC-KITCHENS VISOR

On 20 single-frame queries from P02_09, Top-1 hits at bounding-box IoU ≥ 0.5 were 7/20 for DINOv2 patch pooling, 9/20 for 160-candidate crop rescoring, and 10/20 for a single-frame REN + SAM 2.1 adaptation. Patch pooling and 160-candidate crop rescoring each reached 14/20 Recall@40. Crop rescoring took a median 0.638 s per frame, versus 5.151 s for rescoring 1,333 crops on the same CPU. Timing excludes decoding and shared query encoding.

## Ego4D VQ2D

The 51-query development pilot and 33-query holdout used distinct source videos. The holdout used one clip per source video, a 100 MB clip cap, upright query crops, and 5 FPS. All rows use the official evaluator. The distributed variant proposes one short interval per temporal bin, uses a 40-box shortlist balanced by window shape, and ranks intervals by mean crop similarity. Recency chooses the latest interval within 0.03 of the best mean crop score.

The holdout showed a small stAP gain for distributed proposals with mean crop scoring, with lower tAP. Fixed-interval crop rescoring improved Success on both subsets. The subsets are preliminary; the full-validation comparison is pending. The published VQLoC and RELOCATE scores are full-split literature references.

Only aggregate results are included here.
