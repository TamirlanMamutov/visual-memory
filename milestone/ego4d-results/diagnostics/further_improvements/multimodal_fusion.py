"""Compare fixed score fusion and simulated remembered-time hints."""
import json
import sys
from pathlib import Path

import numpy as np

rows = json.loads(Path(sys.argv[1]).read_text())


def standardize(scores):
    return (scores - scores.mean(axis=0)) / np.maximum(scores.std(axis=0), 0.01)


def rank(row, mode):
    raw = np.asarray(row["candidate_scores"])
    scores = standardize(raw)
    if mode in ("image", "text", "speech"):
        return scores[:, ("image", "text", "speech").index(mode)]
    if mode == "image_text":
        return (scores[:, 0] + scores[:, 1]) / 2
    if mode == "image_speech":
        return (scores[:, 0] + scores[:, 2]) / 2
    if mode == "image_text_speech":
        return .5 * scores[:, 0] + .25 * (scores[:, 1] + scores[:, 2])
    fps = row["fps"]
    anchor = row["target_start"] / fps - 3 + (30 if mode == "text_wrong_time" else 0)
    start, end = np.asarray(row["intervals"]).T / fps
    # Prefer an appearance during or shortly after the remembered event.
    distance = np.maximum.reduce((anchor - end, start - (anchor + 8), np.zeros(len(end))))
    return scores[:, 1] + .75 * np.exp(-distance / 8)


results = {}
for mode in ("image", "text", "speech", "image_text", "image_speech", "image_text_speech",
             "text_nearby_time", "text_wrong_time"):
    outcomes = []
    for row in rows:
        order = np.argsort(-rank(row, mode), kind="stable")
        ious = np.asarray(row["outcomes"])[order]
        outcomes.append([float(ious[0]), bool((ious[:3] >= .25).any())])
    results[mode] = dict(top1_hits=sum(x >= .25 for x, _ in outcomes),
                         top3_hits=sum(y for _, y in outcomes),
                         mean_top1_iou=float(np.mean([x for x, _ in outcomes])))
print(json.dumps(dict(queries=len(rows), videos=len({r["video"] for r in rows}),
                      hint_design="Simulated anchor 3 s before annotated track start; wrong anchor shifted 30 s later.",
                      results=results), indent=2))
