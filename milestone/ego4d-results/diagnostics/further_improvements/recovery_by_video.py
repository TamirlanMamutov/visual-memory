"""Count annotated frames recovered by two predictions, grouped by source video."""
import json
from pathlib import Path
import sys

import numpy as np
from vq2d.metrics.utils import spatial_iou
from vq2d.structures import BBox

ground_truth, raw, fused = [json.loads(Path(path).read_text()) for path in sys.argv[1:4]]
rows = []
for video, baseline, candidate in zip(ground_truth["videos"], raw["results"]["videos"],
                                      fused["results"]["videos"], strict=True):
    assert video["video_uid"] == baseline["video_uid"] == candidate["video_uid"]
    frames = queries = 0
    hits = [0, 0]
    for clip, a, b in zip(video["clips"], baseline["clips"], candidate["clips"], strict=True):
        assert clip["clip_uid"] == a["clip_uid"] == b["clip_uid"]
        for ann, aa, bb in zip(clip["annotations"], a["predictions"], b["predictions"], strict=True):
            for key, query in ann["query_sets"].items():
                if not query["is_valid"]:
                    continue
                queries += 1
                truth = [BBox(t["frame_number"], t["x"], t["y"], t["x"]+t["width"], t["y"]+t["height"])
                         for t in query["response_track"]]
                frames += len(truth)
                tracks = [p["query_sets"][key] for p in (aa, bb)]
                assert tracks[0]["score"] == tracks[1]["score"]
                assert [box["fno"] for box in tracks[0]["bboxes"]] == [box["fno"] for box in tracks[1]["bboxes"]]
                for index, track in enumerate(tracks):
                    boxes = {box["fno"]: BBox(box["fno"], box["x1"], box["y1"], box["x2"], box["y2"])
                             for box in track["bboxes"]}
                    hits[index] += sum(t.fno in boxes and spatial_iou(t, boxes[t.fno]) >= .5 for t in truth)
    rows.append(dict(queries=queries, frames=frames, baseline=hits[0], fusion=hits[1]))
counts = np.array([[r["frames"], r["fusion"] - r["baseline"]] for r in rows])
draws = np.random.default_rng(230).integers(len(rows), size=(2000, len(rows)))
pooled = counts[draws].sum(axis=1)
ci = 100 * np.quantile(pooled[:, 1] / pooled[:, 0], [.025, .975])
print(json.dumps(dict(by_source_video=rows, recovery_gain_ci95_percentage_points=ci.tolist()), indent=2))
