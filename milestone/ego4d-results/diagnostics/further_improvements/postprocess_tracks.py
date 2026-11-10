"""Trim VQ2D patch tracks to their strongest window and expand their boxes."""

import argparse
import json
from pathlib import Path

import numpy as np


def adaptive_window(scores, drop, patience):
    padded = np.pad(scores, 1, mode="edge")
    smooth = np.convolve(padded, np.ones(3) / 3, mode="valid")
    peak = len(smooth) - 1 - int(np.argmax(smooth[::-1]))
    threshold = smooth[peak] - drop
    bounds = []
    for direction in (-1, 1):
        boundary = peak
        misses = 0
        for index in range(peak + direction, -1 if direction < 0 else len(smooth), direction):
            if smooth[index] >= threshold:
                boundary = index
                misses = 0
            else:
                misses += 1
                if misses >= patience:
                    break
        bounds.append(boundary)
    return slice(bounds[0], bounds[1] + 1)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("annotations", "predictions", "output"):
        parser.add_argument(name, type=Path)
    parser.add_argument("--max-frames", type=int, default=20)
    parser.add_argument("--scale", type=float, default=1.25)
    parser.add_argument("--score-drop", type=float)
    parser.add_argument("--patience", type=int, default=1)
    args = parser.parse_args()
    if args.patience < 1 or (args.score_drop is not None and args.score_drop < 0):
        parser.error("patience must be positive and score-drop must be nonnegative")
    if args.max_frames < 1 or args.scale <= 0:
        parser.error("max-frames and scale must be positive")

    annotations = json.loads(args.annotations.read_text())["videos"]
    predictions = json.loads(args.predictions.read_text())
    keys = ("x1", "y1", "x2", "y2")
    for video, result in zip(annotations, predictions["results"]["videos"], strict=True):
        if video["video_uid"] != result["video_uid"]:
            raise ValueError("Video IDs differ")
        for clip, pred in zip(video["clips"], result["clips"], strict=True):
            if clip["clip_uid"] != pred["clip_uid"]:
                raise ValueError("Clip IDs differ")
            for annotation, response in zip(clip["annotations"], pred["predictions"], strict=True):
                for key, track in response["query_sets"].items():
                    boxes = track["bboxes"]
                    if boxes and args.score_drop is not None:
                        scores = [track["frame_scores"][box["fno"]] for box in boxes]
                        boxes = track["bboxes"] = boxes[adaptive_window(scores, args.score_drop, args.patience)]
                    elif len(boxes) > args.max_frames:
                        scores = [track["frame_scores"][box["fno"]] for box in boxes]
                        sums = np.convolve(scores, np.ones(args.max_frames), mode="valid")
                        start = len(sums) - 1 - int(np.argmax(sums[::-1]))
                        boxes = track["bboxes"] = boxes[start:start + args.max_frames]
                    crop = annotation["query_sets"][key]["visual_crop"]
                    bounds = [crop["original_width"], crop["original_height"]] * 2
                    for box in boxes:
                        left, top, right, bottom = (box[k] for k in keys)
                        dx = (right - left) * (args.scale - 1) / 2
                        dy = (bottom - top) * (args.scale - 1) / 2
                        coords = np.rint([left - dx, top - dy, right + dx, bottom + dy])
                        box.update(zip(keys, map(int, np.clip(coords, 0, bounds))))
    args.output.write_text(json.dumps(predictions))


if __name__ == "__main__":
    main()
