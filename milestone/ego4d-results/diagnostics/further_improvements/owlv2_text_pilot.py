"""Generate text-conditioned boxes on the standalone Ego4D text pilot clips."""
import json
from pathlib import Path
import subprocess
import sys

import numpy as np
from PIL import Image
import torch
from transformers import Owlv2ForObjectDetection, Owlv2Processor
from vq2d.metrics.utils import interpolated_prec_rec, spatio_temporal_iou_response_track
from vq2d.structures import BBox, ResponseTrack
from owl_track_linking import link

root = Path(sys.argv[1])
ffmpeg = root.parent / "dinov2_assets/ffmpeg"
model_id = "google/owlv2-base-patch16-ensemble"
torch.set_num_threads(3)
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
print(f"Detector device: {device}", flush=True)
model = Owlv2ForObjectDetection.from_pretrained(model_id, local_files_only=True).eval().to(device)
processor = Owlv2Processor.from_pretrained(model_id, local_files_only=True)
rows = []
linked_rows = []
dense_rows = []
shortlist = {}
if len(sys.argv) > 2:
    log = Path(sys.argv[2]).read_text()
    entries = json.loads(log[log.index("\n{")+1:])["per_query"]
    for entry in entries:
        key = entry["video"], entry["title"]
        shortlist.setdefault(key, []).append([m["sample"] for m in entry["ranked"]])
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
            count = (stop + stride - 1) // stride
            ids = list(range(count))
            if shortlist:
                query_ids = [shortlist.get((video["video_uid"], q["object_title"]), []) for q in queries]
                selections = [matches[0] if len(matches) == 1 else [] for matches in query_ids]
                ids = sorted({i for matches in selections for i in matches})
                if not ids:
                    continue
            selection = '+'.join(f'eq(n\\,{i*stride})' for i in ids) if shortlist else f'not(mod(n\\,{stride}))'
            prompts = [f"a photo of a {q['object_title']}" for q in queries]

            def scan(selection, count):
                raw = subprocess.check_output([ffmpeg, "-v", "error", "-i", movie, "-an", "-vf",
                    f"fps={fps}:start_time=0:round=up,select={selection},scale=448:448",
                    "-fps_mode", "passthrough", "-frames:v", str(count), "-pix_fmt", "rgb24",
                    "-f", "rawvideo", "-threads", "1", "-"])
                frames = np.frombuffer(raw, dtype=np.uint8).reshape(-1, 448, 448, 3)
                assert len(frames) == count
                proposals = []
                for offset in range(0, len(frames), 4):
                    images = [Image.fromarray(frame) for frame in frames[offset:offset+4]]
                    inputs = processor(text=[prompts]*len(images), images=images,
                                       do_resize=False, return_tensors="pt")
                    with torch.inference_mode():
                        result = model(**inputs.to(device), interpolate_pos_encoding=True)
                    scores, indices = result.logits.sigmoid().topk(5, dim=1)
                    boxes = result.pred_boxes[:, :, None, :].expand(-1, -1, len(queries), -1)
                    boxes = boxes.gather(1, indices[..., None].expand(-1,-1,-1,4))
                    proposals.extend(zip(boxes.cpu().numpy(), scores.cpu().numpy()))
                return proposals

            def spatial_candidates(proposal, query_index, width, height):
                boxes, scores = proposal
                center, extent = np.split(boxes[:, query_index], 2, axis=1)
                coords = np.clip(np.concatenate((center-extent/2, center+extent/2), axis=1), 0, 1)
                scores = scores[:, query_index]
                keep = (scores >= scores[0] * .65) & (coords[:, 2:] > coords[:, :2]).all(1)
                return coords[keep] * [width,height,width,height], scores[keep]

            proposals = scan(selection, len(ids))
            frame_boxes = [boxes[0] for boxes, _ in proposals]
            frame_scores = [scores[0] for _, scores in proposals]
            rescan = []
            for j, query in enumerate(queries):
                crop = query["visual_crop"]
                w, h = crop["original_width"], crop["original_height"]
                target = ResponseTrack([BBox(b["frame_number"], b["x"], b["y"],
                                             b["x"]+b["width"], b["y"]+b["height"])
                                        for b in query["response_track"]])
                truth = {b["frame_number"] for b in query["response_track"]}
                limit = (query["query_frame"] + stride - 1) // stride
                matches = []
                order = ([ids.index(i) for i in selections[j]] if shortlist
                         else np.argsort(np.asarray(frame_scores)[:limit,j])[::-1])
                if shortlist and not order:
                    continue
                for position in order:
                    k = ids[position]
                    if any(abs(int(k)-m["index"]) <= 1 for m in matches):
                        continue
                    x, y, bw, bh = frame_boxes[position][j]
                    l,t,r,b = np.clip([x-bw/2,y-bh/2,x+bw/2,y+bh/2],0,1)*[w,h,w,h]
                    if r <= l or b <= t:
                        continue
                    start = max(0,int(k)*stride-stride//2)
                    end = min(query["query_frame"],start+stride)
                    track = ResponseTrack([BBox(f,l,t,r,b) for f in range(start,end)])
                    times = set(range(start,end))
                    matches.append(dict(index=int(k),score=frame_scores[position][j],
                        temporal_iou=len(times&truth)/len(times|truth),
                        iou=spatio_temporal_iou_response_track(track,target)))
                    if len(matches) == 3:
                        break
                rows.append(dict(video=video["video_uid"],title=query["object_title"],matches=matches))
                if not shortlist:
                    detections = []
                    for i, proposal in enumerate(proposals[:limit]):
                        detections.append((i, *spatial_candidates(proposal, j, w, h)))
                    linked = []
                    windows = []
                    for track in link(detections):
                        indices, boxes, scores = zip(*track)
                        if any(abs(indices[-1]-m["index"]) <= 1 for m in linked):
                            continue
                        start = max(0, indices[0]*stride-stride//2)
                        end = min(query["query_frame"], indices[-1]*stride-stride//2+stride)
                        times = set(range(start,end))
                        boxes = np.array(boxes)
                        coordinates = np.stack([np.interp(range(start,end), np.array(indices)*stride, boxes[:, k])
                                                for k in range(4)], axis=1)
                        response = ResponseTrack([BBox(f,*b) for f,b in zip(range(start,end),coordinates)])
                        linked.append(dict(index=indices[-1], score=float(np.mean(scores)),
                                           temporal_iou=len(times&truth)/len(times|truth),
                                           iou=spatio_temporal_iou_response_track(response,target)))
                        windows.append((start, end, np.asarray(indices)*stride, boxes))
                        if len(linked) == 3:
                            break
                    linked_rows.append(dict(video=video["video_uid"], title=query["object_title"], matches=linked))
                    rescan.append((j, query, w, h, target, truth, windows, linked))
            # Densify only the shortlisted windows. Anchor each new box to its coarse track.
            dense_stride = max(1, stride // 2)
            requested = sorted({f for _, _, _, _, _, _, windows, _ in rescan
                                for start, end, _, _ in windows
                                for f in range(start, end, dense_stride)})
            if requested:
                # Compress contiguous windows into intervals to keep FFmpeg expressions small.
                spans = []
                for frame in requested:
                    if spans and frame == spans[-1][1] + dense_stride:
                        spans[-1][1] = frame
                    else:
                        spans.append([frame, frame])
                selection = '+'.join(f'between(n\\,{a}\\,{b})*not(mod(n-{a}\\,{dense_stride}))'
                                     for a, b in spans)
                dense_proposals = dict(zip(requested, scan(selection, len(requested)), strict=True))
                for j, query, w, h, target, truth, windows, linked in rescan:
                    refined = []
                    for (start, end, sample_ids, boxes), match in zip(windows, linked, strict=True):
                        points, coords = [], []
                        for frame in range(start, end, dense_stride):
                            candidates, scores = spatial_candidates(dense_proposals[frame], j, w, h)
                            if not len(candidates):
                                continue
                            anchor = np.array([np.interp(frame, sample_ids, boxes[:, k]) for k in range(4)])
                            overlap = np.maximum(np.minimum(candidates[:, 2:], anchor[2:]) -
                                                 np.maximum(candidates[:, :2], anchor[:2]), 0).prod(1)
                            union = (candidates[:, 2:] - candidates[:, :2]).prod(1) + np.prod(anchor[2:]-anchor[:2]) - overlap
                            ious = overlap / np.maximum(union, 1e-8)
                            best = np.argmax(scores * ious)
                            if ious[best] >= .15:
                                points.append(frame)
                                coords.append(candidates[best])
                        if not points:
                            refined.append(match)
                            continue
                        # Use the span supported by dense detections and interpolate its boxes.
                        times = range(points[0], min(end, points[-1]+dense_stride))
                        interpolated = np.stack([np.interp(times, points, np.asarray(coords)[:, k])
                                                 for k in range(4)], axis=1)
                        response = ResponseTrack([BBox(f,*b) for f,b in zip(times,interpolated)])
                        refined.append(dict(match, temporal_iou=len(set(times)&truth)/len(set(times)|truth),
                                            iou=spatio_temporal_iou_response_track(response,target)))
                    dense_rows.append(dict(video=video["video_uid"], title=query["object_title"], matches=refined))
            elif rescan:
                dense_rows.extend(dict(video=video["video_uid"], title=query["object_title"], matches=linked)
                                  for _, query, _, _, _, _, _, linked in rescan)
            print(uid,len(rows),flush=True)
rows.sort(key=lambda row: row["matches"][0]["score"] if row["matches"] else 0, reverse=True)
hits=np.cumsum([bool(r["matches"]) and r["matches"][0]["iou"]>=.25 for r in rows])
print(json.dumps(dict(queries=len(rows),videos=len({r["video"] for r in rows}),
    top1_hits=int(hits[-1]),top3_hits=sum(any(m["iou"]>=.25 for m in r["matches"]) for r in rows),
    temporal_top3=sum(any(m["temporal_iou"]>=.25 for m in r["matches"]) for r in rows),
    stAP25=float(interpolated_prec_rec(hits/np.arange(1,len(rows)+1),hits/len(rows))),
    linked_top1=sum(bool(r["matches"]) and r["matches"][0]["iou"]>=.25 for r in linked_rows),
    linked_top3=sum(any(m["iou"]>=.25 for m in r["matches"]) for r in linked_rows),
    linked_temporal_top3=sum(any(m["temporal_iou"]>=.25 for m in r["matches"]) for r in linked_rows),
    dense_top1=sum(bool(r["matches"]) and r["matches"][0]["iou"]>=.25 for r in dense_rows),
    dense_top3=sum(any(m["iou"]>=.25 for m in r["matches"]) for r in dense_rows),
    dense_temporal_top3=sum(any(m["temporal_iou"]>=.25 for m in r["matches"]) for r in dense_rows),
    per_query=rows, linked_per_query=linked_rows, dense_per_query=dense_rows),indent=2))
