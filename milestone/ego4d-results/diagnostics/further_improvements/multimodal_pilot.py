"""Compare CLIP query modalities on the same cached VQ2D tracks."""
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
from PIL import Image
import torch
from transformers import CLIPModel, CLIPProcessor, pipeline
from vq2d.structures import BBox, ResponseTrack
from vq2d.metrics.utils import interpolated_prec_rec, spatio_temporal_iou_response_track

root = Path(sys.argv[1])
out = Path("/tmp/vq-multimodal")
frames = Path("/tmp/vq-interaction/images")
ffmpeg = root.parent / "dinov2_assets/ffmpeg"
torch.set_num_threads(4)
name = "openai/clip-vit-base-patch32"
model = CLIPModel.from_pretrained(name).eval()
processor = CLIPProcessor.from_pretrained(name)
asr = pipeline("automatic-speech-recognition", model="openai/whisper-tiny.en", device=-1)


def read(path):
    return json.loads(path.read_text())


def embed(inputs, encoder):
    with torch.inference_mode():
        result = encoder(**inputs)
        if not isinstance(result, torch.Tensor):
            result = result.pooler_output
        return torch.nn.functional.normalize(result, dim=-1)


def images(items):
    return embed(processor(images=items, return_tensors="pt"), model.get_image_features)


def crop_image(image, box, width, height):
    x1, y1, x2, y2 = box
    return image.crop((x1 * image.width / width, y1 * image.height / height,
                       x2 * image.width / width, y2 * image.height / height))


batches = root / "full_validation/batches"
cache = {}
results = []
for index in range(10, 14):
    batch = batches / f"{index:03}"
    gt = read(batch / "annotations.json")
    predictions = read(batch / "evidence/predictions.json")
    for video, pred in zip(gt["videos"], predictions["results"]["videos"], strict=True):
        assert video["video_uid"] == pred["video_uid"]
        for clip, output in zip(video["clips"], pred["clips"], strict=True):
            uid = clip["clip_uid"]
            assert uid == output["clip_uid"]
            path = Path(f"/tmp/vq-gate/v2/clips/{uid}.mp4")
            if not path.is_file():
                continue
            for annotation, response in zip(clip["annotations"], output["predictions"], strict=True):
                for key, track in response["query_sets"].items():
                    q = annotation["query_sets"][key]
                    title = q.get("object_title", "").strip()
                    candidates = track["candidates"]
                    if not q["is_valid"] or not title or not candidates:
                        continue
                    crop = q["visual_crop"]
                    w, h = crop["original_width"], crop["original_height"]
                    png = subprocess.check_output([str(ffmpeg), "-v", "error", "-ss",
                        str(crop["frame_number"] / clip["clip_fps"]), "-i", str(path),
                        "-frames:v", "1", "-f", "image2pipe", "-c:v", "png", "-threads", "1", "-"])
                    from io import BytesIO
                    image = Image.open(BytesIO(png)).convert("RGB")
                    query_image = crop_image(image, (crop["x"], crop["y"], crop["x"] + crop["width"],
                                             crop["y"] + crop["height"]), w, h)
                    if title not in cache:
                        audio = out / "query.aiff"
                        subprocess.run(["/usr/bin/say", "-o", str(audio), "--", title], check=True)
                        raw = subprocess.check_output([str(ffmpeg), "-v", "error", "-i", str(audio),
                                                       "-ar", "16000", "-ac", "1", "-f", "f32le", "-"])
                        transcript = asr(np.frombuffer(raw, dtype=np.float32).copy())["text"].strip()
                        cache[title] = (transcript, embed(processor(text=[title, transcript or "unknown"],
                                                                   padding=True, return_tensors="pt"),
                                                           model.get_text_features))
                    transcript, texts = cache[title]
                    query_vectors = torch.cat((images([query_image]), texts))
                    candidate_scores = []
                    outcomes = []
                    target = ResponseTrack([BBox(b["frame_number"], b["x"], b["y"], b["x"] + b["width"],
                                                 b["y"] + b["height"]) for b in q["response_track"]])
                    for candidate in candidates:
                        boxes = candidate["bboxes"][::3]
                        crops = [crop_image(Image.open(frames / uid / f"{box['fno']}.png").convert("RGB"),
                                             [box[k] for k in ("x1", "y1", "x2", "y2")], w, h) for box in boxes]
                        candidate_scores.append((images(crops) @ query_vectors.T).mean(0).tolist())
                        outcomes.append(spatio_temporal_iou_response_track(ResponseTrack.from_json(candidate), target))
                    scores = np.asarray(candidate_scores)
                    selected = scores.argmax(0)
                    results.append(dict(video=video["video_uid"], title=title, transcript=transcript,
                                        iou=[outcomes[i] for i in selected], scores=scores.max(0).tolist(),
                                        candidate_scores=scores.tolist(), outcomes=outcomes, fps=clip["clip_fps"],
                                        intervals=[[c["bboxes"][0]["fno"], c["bboxes"][-1]["fno"]]
                                                   for c in candidates],
                                        target_start=q["response_track"][0]["frame_number"]))
            print(uid, len(results), flush=True)

summary = {}
for i, modality in enumerate(("image", "text", "spoken")):
    order = sorted(results, key=lambda row: row["scores"][i], reverse=True)
    hits = np.cumsum([row["iou"][i] >= .25 for row in order])
    ap = interpolated_prec_rec(hits / np.arange(1, len(hits) + 1), hits / len(hits))
    summary[modality] = dict(hits=int(hits[-1]), shortlist_stAP25=float(ap))
print(json.dumps(dict(queries=len(results), videos=len({r["video"] for r in results}),
                      metrics=summary, transcripts={k: v[0] for k, v in cache.items()}), indent=2))
(out / "per_query.json").write_text(json.dumps(results, indent=2))
