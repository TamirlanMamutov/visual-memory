"""Generate text-only candidates on a uniform grid over pre-query video."""
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
from PIL import Image
import torch
from transformers import CLIPModel, CLIPProcessor
from vq2d.structures import BBox, ResponseTrack
from vq2d.metrics.utils import interpolated_prec_rec, spatio_temporal_iou_response_track

root = Path(sys.argv[1])
ffmpeg = root.parent / "dinov2_assets/ffmpeg"
torch.set_num_threads(4)
model_id = "openai/clip-vit-base-patch32"
model = CLIPModel.from_pretrained(model_id, local_files_only=True).eval()
processor = CLIPProcessor.from_pretrained(model_id, local_files_only=True)


def normalized(result):
    return torch.nn.functional.normalize(result, dim=-1)


rows = []
for batch in range(10, 14):
    path = root / f"full_validation/batches/{batch:03}/annotations.json"
    for video in json.loads(path.read_text())["videos"]:
        for clip in video["clips"]:
            uid = clip["clip_uid"]
            movie = Path(f"/tmp/vq-gate/v2/clips/{uid}.mp4")
            if not movie.is_file():
                continue
            queries = [q for a in clip["annotations"] for q in a["query_sets"].values()
                       if q["is_valid"] and q["query_frame"] > 0 and q.get("object_title", "").strip()]
            if not queries:
                continue
            fps = clip["clip_fps"]
            stride = max(1, round(fps))
            stop = max(q["query_frame"] for q in queries)
            raw = subprocess.check_output([ffmpeg, "-v", "error", "-i", movie, "-an", "-vf",
                f"fps={fps}:start_time=0:round=up,select=not(mod(n\\,{stride})),scale=320:180",
                "-fps_mode", "passthrough", "-frames:v", str((stop + stride - 1) // stride),
                "-pix_fmt", "rgb24", "-f", "rawvideo", "-threads", "1", "-"])
            frames = np.frombuffer(raw, dtype=np.uint8).reshape(-1, 180, 320, 3)
            # Overlapping coarse and fine tiles are independent of query identity.
            tiles = [(x, y, x+w, y+h) for w, h in [(160, 90), (96, 60)]
                     for y in sorted({0, (180-h)//2, 180-h})
                     for x in sorted({0, (320-w)//2, 320-w})]
            scores = []
            with torch.inference_mode():
                texts = normalized(model.get_text_features(**processor(
                    text=[f"a photo of a {q['object_title']}" for q in queries],
                    padding=True, return_tensors="pt")))
                for frame in frames:
                    image = Image.fromarray(frame)
                    features = normalized(model.get_image_features(**processor(
                        images=[image.crop(tile) for tile in tiles], return_tensors="pt")))
                    scores.append((features @ texts.T).numpy())
            scores = np.asarray(scores)
            for j, query in enumerate(queries):
                valid = scores[:(query["query_frame"] + stride - 1) // stride, :, j]
                peak = valid.max()
                eligible = np.flatnonzero(valid.max(axis=1) >= peak - .025)
                selected = int(eligible[-1])
                crop = query["visual_crop"]
                width, height = crop["original_width"], crop["original_height"]
                target = ResponseTrack([BBox(b["frame_number"], b["x"], b["y"], b["x"] + b["width"],
                                            b["y"] + b["height"]) for b in query["response_track"]])
                truth = {b["frame_number"] for b in query["response_track"]}
                ranked = []
                order = [selected, *np.argsort(valid.max(axis=1))[::-1]]
                for index in order:
                    index = int(index)
                    if any(abs(index - match["sample"]) <= 1 for match in ranked):
                        continue
                    x1, y1, x2, y2 = tiles[valid[index].argmax()]
                    start = max(0, index * stride - stride // 2)
                    end = min(query["query_frame"], start + stride)
                    box = [int(x1*width/320), int(y1*height/180),
                           int(x2*width/320), int(y2*height/180)]
                    prediction = ResponseTrack([BBox(f, *box) for f in range(start, end)])
                    predicted = set(range(start, end))
                    ranked.append(dict(sample=index, seconds=index * stride / fps, box=box,
                        score=float(valid[index].max()),
                        iou=spatio_temporal_iou_response_track(prediction, target),
                        temporal_iou=len(truth & predicted)/len(truth | predicted)))
                    if len(ranked) == 3:
                        break
                rows.append(dict(video=video["video_uid"], title=query["object_title"],
                    **{k: ranked[0][k] for k in ("score", "iou", "temporal_iou")}, ranked=ranked))
            print(uid, len(rows), flush=True)
rows.sort(key=lambda r: r["score"], reverse=True)
hits = np.cumsum([r["iou"] >= .25 for r in rows])
print(json.dumps(dict(queries=len(rows), videos=len({r["video"] for r in rows}),
                      temporal_hits=sum(r["temporal_iou"] >= .25 for r in rows),
                      spatial_temporal_hits=int(hits[-1]),
                      recall_at_3=sum(any(m["iou"] >= .25 for m in r["ranked"]) for r in rows) / len(rows),
                      stAP25=float(interpolated_prec_rec(hits / np.arange(1, len(rows)+1), hits / len(rows))),
                      per_query=rows), indent=2))
