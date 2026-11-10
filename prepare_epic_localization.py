"""Extract EPIC training-segment features for auxiliary noun supervision."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import cv2
import numpy as np
import torch

from experiment_paths import ROOT
from prepare_ego4d_localization import DINO_REPOSITORY, preprocess_image, read_frame


def timestamp_seconds(timestamp: str) -> float:
    hours, minutes, seconds = timestamp.split(":")
    return 3600 * int(hours) + 60 * int(minutes) + float(seconds)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--videos", type=Path, default=ROOT / "EPIC-KITCHENS")
    parser.add_argument("--annotations", type=Path, default=ROOT / "epic-kitchens-100-annotations" / "EPIC_100_train.csv")
    parser.add_argument("--frames", type=int, default=4)
    args = parser.parse_args()
    if args.frames <= 0:
        parser.error("--frames must be positive")
    if args.output_dir.exists():
        parser.error("Choose a fresh --output-dir")
    # Only canonical filenames are accepted; suffixed exports may be trimmed or concatenated.
    videos = {
        path.stem: path for path in args.videos.rglob("*.MP4")
        if path.stat().st_size > 0 and len(path.stem.split("_")) == 2
        and path.stem.split("_")[1].isdigit()
    }
    with args.annotations.open(encoding="utf-8") as file:
        records = [row for row in csv.DictReader(file) if row["video_id"] in videos]
    if not records:
        raise ValueError("No EPIC training annotations match canonical non-empty local videos")
    torch.set_num_threads(4)
    encoder = torch.hub.load(
        DINO_REPOSITORY, "dinov2_vits14", pretrained=True, trust_repo=True
    ).eval().requires_grad_(False)
    features, labels, samples = [], [], []
    for row in records:
        path = videos[row["video_id"]]
        capture = cv2.VideoCapture(str(path))
        try:
            if not capture.isOpened():
                raise RuntimeError(f"Cannot open {path}")
            fps = capture.get(cv2.CAP_PROP_FPS)
            count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
            start = timestamp_seconds(row["start_timestamp"])
            stop = timestamp_seconds(row["stop_timestamp"])
            if not np.isfinite(fps) or fps <= 0 or not 0 <= start < stop <= count / fps:
                raise ValueError(f"Segment outside media time range: {row['narration_id']}")
            times = np.linspace(start, stop, args.frames + 2)[1:-1]
            images = torch.stack([
                preprocess_image(read_frame(capture, round(time * fps), path), 224)
                for time in times
            ])
            with torch.inference_mode():
                tokens = encoder.forward_features(images)["x_norm_patchtokens"]
            features.append(tokens.mean((0, 1)).numpy())
            labels.append(int(row["noun_class"]))
            samples.append({
                "sample_id": row["narration_id"], "video_id": row["video_id"],
                "noun_class": int(row["noun_class"]), "noun": row["noun"],
                "start_seconds": start, "stop_seconds": stop,
                "sampled_seconds": times.tolist(),
            })
            print(f"EPIC {len(samples)}/{len(records)}: {row['narration_id']}", flush=True)
        finally:
            capture.release()
    args.output_dir.mkdir(parents=True)
    np.save(args.output_dir / "features.npy", np.stack(features).astype(np.float32))
    np.save(args.output_dir / "noun_labels.npy", np.array(labels, dtype=np.int64))
    (args.output_dir / "samples.json").write_text(json.dumps(samples, indent=2), encoding="utf-8")
    (args.output_dir / "preparation.json").write_text(json.dumps({
        "split": "official_train", "encoder": "dinov2_vits14",
        "encoder_repository": DINO_REPOSITORY, "image_size": 224,
        "frames_per_segment": args.frames, "segments": len(samples),
        "videos": sorted({row["video_id"] for row in samples}),
        "noun_classes": len(set(labels)), "box_supervision": False,
    }, indent=2), encoding="utf-8")
    (args.output_dir / "COMPLETE").touch()


if __name__ == "__main__":
    main()
