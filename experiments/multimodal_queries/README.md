# Image/text/transcript ranking over a fixed candidate pool

These experiments rank existing object tracks using CLIP similarities. The
candidate pool was generated with an **image query**. Text-only ranking here is
therefore conditional on that pool; it is not a complete text-only localization
system. Extra inputs can be ambiguous: the command returns ranked candidate IDs
as well as a single automatic prediction.

## Reproduce the cached20-query pilot

Follow [setup and private handoff](../reproduction/README.md), then:

```text
python -m experiments.multimodal_queries.run --bundle ../ren_handoff_v1/multimodal20/bundle.json --mode name --out output/modalities/name.json
python -m experiments.multimodal_queries.evaluate --bundle ../ren_handoff_v1/multimodal20/bundle.json --truth ../ren_handoff_v1/multimodal20/ground_truth.json --predictions output/modalities/name.json --evaluator-root ../VQLoC-official --out output/modalities/name_metrics.json
```

Modes: `baseline`, `image`, `name`, `description`, `image_name`,
`image_description`, `speech_transcript`, `image_speech`. Run each separately.
The cached pilot retains its original fused scores for exact replay. On a fresh
bundle, fusion is the fixed0.5sum of within-query standardized image/text scores.
Native confidence is preserved; ranking scores are not calibrated probabilities.

Expected image Top1:7/20; name Top1:8/20; name Top3:14/20; candidate oracle:15/20.
These count candidate tracks with stIoU≥0.25; they are **not official Success**.
Full official Top1 metrics are also emitted. The oracle uses targets only to
diagnose the candidate ceiling and is not a selectable inference method.
There is no consistent fusion improvement. Descriptions were assistant-authored;
speech was20 TTS clips transcribed with Whisper tiny.en,1 word error / 222 words.
This is not a recorded-user speech or acoustic-event study.

## New independently written descriptions or transcripts

The EAGLE adapter can export `appearance_embedding` per candidate with the
explicit CLIP model/revision. Install transformers/Torch using the existing
model environment; see the teammate root reproduction instructions. This optional
fresh encoding path is not required for cached replay and has not yet been
validated on actual EAGLE data on our laptop.

Prepare a text file with exact query coverage:

```json
{"provenance":"independent_human_description_before_model_outputs","queries":[
  {"key":"actual_query_key","text":"the pink paper bag with dark handles"}
]}
```

```text
python -m experiments.multimodal_queries.encode_text --bundle output/eagle-pilot/bundle.json --texts descriptions.json --field description_score --model openai/clip-vit-base-patch32 --out output/eagle-pilot/with-descriptions.json
python -m experiments.multimodal_queries.run --bundle output/eagle-pilot/with-descriptions.json --mode image_description --out output/eagle-pilot/image-description.json
```

Use `--field speech_score` for externally obtained ASR transcripts, then rank
`speech_transcript`/`image_speech`. This tool does not perform ASR itself. Record
whether speech is natural, read, synthetic, corrected, and whether the user saw
model predictions or targets. Text and candidate embeddings must use the same
model identifier/revision. Encoding new text invalidates old cached fused scores.

Timing memories are a **separate input** with fallible scope/confidence; follow
[additional_hints](../additional_hints/README.md). Do not infer last-sighting times
from GT for a real-user test. For multiple matching objects, compare automatic
Top1 and a fixed-size ranked shortlist separately, retaining a same-size native+
image-alternative control. A correct alternative does not mean the user selected it.
