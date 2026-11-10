"""Sweep candidate generation using cached Ego4D frame scores."""
import json
import sys
from pathlib import Path

import numpy as np


def load(path):
    return json.loads(Path(path).read_text())


def examples(gt, predictions):
    for video, result in zip(gt["videos"], predictions["results"]["videos"], strict=True):
        assert video["video_uid"] == result["video_uid"]
        for clip, output in zip(video["clips"], result["clips"], strict=True):
            assert clip["clip_uid"] == output["clip_uid"]
            for annotation, response in zip(clip["annotations"], output["predictions"], strict=True):
                for key, query in annotation["query_sets"].items():
                    if query["is_valid"]:
                        track = response["query_sets"][key]
                        truth = {box["frame_number"] for box in query["response_track"]}
                        yield video["video_uid"], np.asarray(track["frame_scores"]), truth


def evaluate(rows, margin, minimum):
    oracle = selected = missing = 0
    for _, scores, truth in rows:
        if not scores.size:
            missing += 1
            continue
        edges = np.flatnonzero(np.diff(np.r_[False, scores >= np.quantile(scores, .95) - margin, False]))
        runs = [(a, b) for a, b in zip(edges[::2], edges[1::2]) if b - a >= minimum]
        if not runs:
            missing += 1
            continue
        ranked = sorted(runs, key=lambda run: scores[slice(*run)].mean(), reverse=True)
        shortlist = list(dict.fromkeys([runs[-1], *ranked]))[:3]
        hits = []
        for a, b in shortlist:
            intersection = sum(a <= frame < b for frame in truth)
            hits.append(intersection / (len(truth) + b - a - intersection) >= .25)
        selected += hits[0]
        oracle += any(hits)
    return dict(margin=margin, minimum=minimum, selected=int(selected), top_three=oracle, missing=missing)


def main():
    tuning = list(examples(load(sys.argv[1]), load(sys.argv[2])))
    check = list(examples(load(sys.argv[3]), load(sys.argv[4])))
    assert not ({v for v, *_ in tuning} & {v for v, *_ in check})
    grid = [evaluate(tuning, margin, minimum) for margin in (.05, .10, .15, .20)
            for minimum in (2, 3, 5)]
    best = max(grid, key=lambda r: (r["top_three"], r["selected"], -r["missing"]))
    result = dict(tuning_queries=len(tuning), tuning=grid, chosen=best, check_queries=len(check),
                  check_videos=len({v for v, *_ in check}),
                  check_baseline=evaluate(check, .15, 2),
                  check_chosen=evaluate(check, best["margin"], best["minimum"]))
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
