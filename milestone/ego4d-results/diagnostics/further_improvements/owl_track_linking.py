"""Associate per-query detections across sampled video frames."""
import numpy as np


def link(detections, max_gap=2):
    tracks = []
    for frame, boxes, scores in detections:
        pairs = []
        for i, track in enumerate(tracks):
            last_frame, last_box, _ = track[-1]
            if frame - last_frame > max_gap:
                continue
            intersection = np.maximum(np.minimum(boxes[:, 2:], last_box[2:]) -
                                      np.maximum(boxes[:, :2], last_box[:2]), 0).prod(1)
            union = (boxes[:, 2:] - boxes[:, :2]).prod(1) + np.prod(last_box[2:] - last_box[:2]) - intersection
            for j in np.flatnonzero(intersection / np.maximum(union, 1e-8) >= .25):
                pairs.append((intersection[j] / union[j], i, j))
        used_tracks, used_boxes = set(), set()
        for _, i, j in sorted(pairs, reverse=True):
            if i not in used_tracks and j not in used_boxes:
                tracks[i].append((frame, boxes[j], scores[j]))
                used_tracks.add(i)
                used_boxes.add(j)
        tracks.extend([[(frame, box, score)] for j, (box, score) in enumerate(zip(boxes, scores))
                       if j not in used_boxes])
    return sorted(tracks, key=lambda t: (float(np.mean([p[2] for p in t])), t[-1][0]), reverse=True)
