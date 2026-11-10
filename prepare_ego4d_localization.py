"""Build image-query DINOv2 feature bundles from locally available VQ2D clips."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image

from experiment_paths import ROOT


DINO_REPOSITORY = "facebookresearch/dinov2:7764ea0f912e53c92e82eb78a2a1631e92725fc8"


def normalized_box(annotation: dict) -> np.ndarray:
    width, height = annotation["original_width"], annotation["original_height"]
    if width <= 0 or height <= 0:
        raise ValueError("Annotation image dimensions must be positive")
    box = np.array([
        annotation["x"] / width,
        annotation["y"] / height,
        (annotation["x"] + annotation["width"]) / width,
        (annotation["y"] + annotation["height"]) / height,
    ], dtype=np.float32)
    if not np.isfinite(box).all():
        raise ValueError("Annotation box contains non-finite values")
    box = box.clip(0, 1)
    if box[2] <= box[0] or box[3] <= box[1]:
        raise ValueError("Annotation box has no area inside the image")
    return box


def padded_box(box: np.ndarray, width: int, height: int) -> np.ndarray:
    side = max(width, height)
    left, top = (side - width) // 2, (side - height) // 2
    return (
        box * np.array([width, height, width, height])
        + np.array([left, top, left, top])
    ).astype(np.float32) / side


def preprocess_image(image: np.ndarray, image_size: int) -> torch.Tensor:
    height, width = image.shape[:2]
    side = max(width, height)
    canvas = Image.new("RGB", (side, side))
    canvas.paste(Image.fromarray(image), ((side - width) // 2, (side - height) // 2))
    canvas = canvas.resize((image_size, image_size), Image.Resampling.BICUBIC)
    pixels = torch.from_numpy(np.array(canvas, copy=True)).permute(2, 0, 1).float() / 255
    mean = pixels.new_tensor([0.485, 0.456, 0.406])[:, None, None]
    std = pixels.new_tensor([0.229, 0.224, 0.225])[:, None, None]
    return (pixels - mean) / std


def sample_frame_ids(record: dict, count: int, sampling: str) -> list[int]:
    """Choose pre-query frames; balanced sampling is training-only."""
    query_frame = int(record["query_frame"])
    if query_frame <= 0 or count <= 0:
        raise ValueError("query_frame and sample count must be positive")
    if sampling == "history":
        return np.linspace(0, query_frame - 1, min(count, query_frame)).astype(int).tolist()
    if sampling != "balanced":
        raise ValueError(f"Unknown sampling policy: {sampling}")
    positives = sorted({
        int(box["frame_number"]) for box in record["response_track"]
        if 0 <= int(box["frame_number"]) < query_frame
    })
    if not positives:
        raise ValueError(f"No pre-query response frames for {record['dataset_uid']}")
    positive_ids = np.array(positives)[
        np.linspace(0, len(positives) - 1, min(max(1, count // 2), len(positives))).astype(int)
    ].tolist()
    negatives = sorted(set(range(query_frame)) - set(positives))
    remaining = min(count - len(positive_ids), len(negatives))
    negative_ids = (
        np.array(negatives)[np.linspace(0, len(negatives) - 1, remaining).astype(int)].tolist()
        if remaining else []
    )
    return sorted(positive_ids + negative_ids)


def read_frame(capture: cv2.VideoCapture, index: int, path: Path) -> np.ndarray:
    if not capture.set(cv2.CAP_PROP_POS_FRAMES, index):
        raise RuntimeError(f"Cannot seek to frame {index}: {path}")
    success, frame = capture.read()
    if not success:
        raise RuntimeError(f"Cannot decode frame {index}: {path}")
    return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)


def load_records(annotations: Path, clips: Path) -> tuple[list[dict], int]:
    records = json.loads(annotations.read_text(encoding="utf-8"))
    matched = [
        record for record in records
        if (clips / f"{record['clip_uid']}.mp4").is_file()
        and (clips / f"{record['clip_uid']}.mp4").stat().st_size > 0
    ]
    if not matched:
        raise ValueError(f"No non-empty clips match {annotations}")
    return matched, len(records)


def extract_record(
    record: dict,
    clip_path: Path,
    encoder: torch.nn.Module,
    *,
    frames: int,
    sampling: str,
    image_size: int,
    batch_size: int,
    device: torch.device,
) -> tuple[dict[str, np.ndarray], dict]:
    capture = cv2.VideoCapture(str(clip_path))
    try:
        if not capture.isOpened():
            raise RuntimeError(f"Cannot open clip: {clip_path}")
        fps = capture.get(cv2.CAP_PROP_FPS)
        frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
        annotation_fps = float(record["clip_fps"])
        if not math.isfinite(fps) or fps <= 0 or annotation_fps <= 0:
            raise ValueError(f"Invalid frame rate for {clip_path}")
        frame_ids = sample_frame_ids(record, frames, sampling)
        media_ids = [round(index * fps / annotation_fps) for index in frame_ids]
        crop = record["visual_crop"]
        query_media_id = round(int(crop["frame_number"]) * fps / annotation_fps)
        if min(media_ids + [query_media_id]) < 0 or max(media_ids + [query_media_id]) >= frame_count:
            raise ValueError(f"Annotation frame outside decoded clip: {clip_path}")

        query_frame = read_frame(capture, query_media_id, clip_path)
        height, width = query_frame.shape[:2]
        crop_box = normalized_box(crop) * np.array([width, height, width, height])
        x1, y1 = np.floor(crop_box[:2]).astype(int)
        x2, y2 = np.ceil(crop_box[2:]).astype(int)
        query_pixels = query_frame[y1:y2, x1:x2]
        if query_pixels.size == 0:
            raise ValueError(f"Empty query crop for {record['dataset_uid']}")
        query_tensor = preprocess_image(query_pixels, image_size)
        targets = {int(box["frame_number"]): box for box in record["response_track"]}
        boxes = np.zeros((frames, 4), dtype=np.float32)
        present = np.zeros(frames, dtype=bool)
        frame_mask = np.zeros(frames, dtype=bool)
        video_tokens = []
        for start in range(0, len(frame_ids), batch_size):
            tensors = []
            for position in range(start, min(start + batch_size, len(frame_ids))):
                image = read_frame(capture, media_ids[position], clip_path)
                h, w = image.shape[:2]
                if frame_ids[position] in targets:
                    boxes[position] = padded_box(
                        normalized_box(targets[frame_ids[position]]), w, h
                    )
                    present[position] = True
                frame_mask[position] = True
                tensors.append(preprocess_image(image, image_size))
            with torch.inference_mode():
                features = encoder.forward_features(torch.stack(tensors).to(device))
            video_tokens.append(features["x_norm_patchtokens"].cpu().numpy())
        with torch.inference_mode():
            query_features = encoder.forward_features(query_tensor[None].to(device))
        tokens = np.concatenate(video_tokens).astype(np.float32)
        padded_tokens = np.zeros((frames, *tokens.shape[1:]), dtype=np.float32)
        padded_tokens[:len(frame_ids)] = tokens
        arrays = {
            "video_features": padded_tokens,
            "image_query_features": query_features["x_norm_patchtokens"][0].cpu().numpy().astype(np.float32),
            "boxes_xyxy": boxes,
            "object_present": present,
            "frame_mask": frame_mask,
        }
        metadata = {
            "sample_id": record["dataset_uid"],
            "clip_uid": record["clip_uid"],
            "video_uid": record["metadata"]["video_uid"],
            "object_title": record["object_title"],
            "frame_ids": frame_ids,
            "media_frame_ids": media_ids,
            "annotation_fps": annotation_fps,
            "media_fps": fps,
            "query_media_frame": query_media_id,
            "query_frame": record["query_frame"],
            "positive_frames": int(present.sum()),
            "decoded_width": width,
            "decoded_height": height,
        }
        return arrays, metadata
    finally:
        capture.release()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--clips", type=Path, default=ROOT / "Ego4D" / "v2" / "clips")
    parser.add_argument("--annotations", type=Path, default=ROOT / "VQLoC" / "data")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--backbone", choices=("dinov2_vits14", "dinov2_vitb14"), default="dinov2_vitb14")
    parser.add_argument("--image-size", type=int, default=448)
    parser.add_argument("--train-frames", type=int, default=16)
    parser.add_argument("--val-frames", type=int, default=64)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--threads", type=int, default=4)
    args = parser.parse_args()
    if args.image_size <= 0 or args.image_size % 14:
        parser.error("--image-size must be positive and divisible by 14")
    if min(args.train_frames, args.val_frames, args.batch_size, args.threads) <= 0:
        parser.error("Frame, batch, and thread counts must be positive")
    if args.output_dir.exists():
        parser.error("--output-dir already exists; choose a fresh directory")
    splits = {}
    totals = {}
    for split in ("train", "val"):
        splits[split], totals[split] = load_records(
            args.annotations / f"{split}_anno_new.json", args.clips
        )
    train_videos = {r["metadata"]["video_uid"] for r in splits["train"]}
    val_videos = {r["metadata"]["video_uid"] for r in splits["val"]}
    if train_videos & val_videos:
        raise ValueError("Source videos overlap between training and validation")

    torch.set_num_threads(args.threads)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    encoder = torch.hub.load(
        DINO_REPOSITORY, args.backbone, pretrained=True, trust_repo=True
    ).to(device).eval().requires_grad_(False)
    args.output_dir.mkdir(parents=True)
    summary = {
        "encoder": args.backbone,
        "encoder_repository": DINO_REPOSITORY,
        "image_size": args.image_size,
        "preprocessing": "RGB; centered zero-pad square; bicubic resize; ImageNet normalize",
        "query_mode": "image_only",
        "device": str(device),
        "splits": {},
    }
    for split in ("train", "val"):
        sampling = "balanced" if split == "train" else "history"
        count = args.train_frames if split == "train" else args.val_frames
        destination = args.output_dir / split
        destination.mkdir()
        storage = {}
        samples = []
        for index, record in enumerate(splits[split]):
            arrays, metadata = extract_record(
                record, args.clips / f"{record['clip_uid']}.mp4", encoder,
                frames=count, sampling=sampling, image_size=args.image_size,
                batch_size=args.batch_size, device=device,
            )
            if not storage:
                for name, array in arrays.items():
                    storage[name] = np.lib.format.open_memmap(
                        destination / f"{name}.npy", mode="w+",
                        dtype=array.dtype, shape=(len(splits[split]), *array.shape),
                    )
            for name, array in arrays.items():
                storage[name][index] = array
            samples.append(metadata)
            print(f"{split} {index + 1}/{len(splits[split])}: {metadata['sample_id']}", flush=True)
        for array in storage.values():
            array.flush()
            array._mmap.close()
        (destination / "samples.json").write_text(json.dumps(samples, indent=2), encoding="utf-8")
        summary["splits"][split] = {
            "matched_queries": len(samples),
            "annotation_queries": totals[split],
            "matched_clips": len({r["clip_uid"] for r in samples}),
            "sampling": sampling,
            "frames_per_query": count,
            "positive_frames": sum(r["positive_frames"] for r in samples),
            "queries_with_positive_frames": sum(r["positive_frames"] > 0 for r in samples),
        }
    (args.output_dir / "preparation.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    (args.output_dir / "COMPLETE").touch()
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
