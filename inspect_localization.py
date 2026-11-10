"""Export validation predictions and an offline visual localization review."""

from __future__ import annotations

import argparse
import html
import json
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image, ImageDraw
from torch.utils.data import DataLoader

from experiment_paths import ROOT
from multimodal_localizer import MultimodalTemporalLocalizer
from spatial_localizer import SpatialLocalizer
from prepare_ego4d_localization import normalized_box, read_frame
from train_multimodal_localizer import FeatureBundleDataset, _box_iou, _model_inputs, _run_epoch


def original_box(box: np.ndarray, width: int, height: int) -> list[float]:
    side = max(width, height)
    offsets = np.array([(side - width) // 2, (side - height) // 2] * 2)
    pixels = box * side - offsets
    return (pixels / np.array([width, height] * 2)).clip(0, 1).tolist()


def draw_box(image: Image.Image, box: list[float], color: str) -> None:
    width, height = image.size
    coordinates = np.array(box) * np.array([width, height, width, height])
    ImageDraw.Draw(image).rectangle(coordinates.tolist(), outline=color, width=3)


def export_review(
    model: MultimodalTemporalLocalizer | SpatialLocalizer,
    dataset: FeatureBundleDataset,
    samples: list[dict],
    annotations: dict[str, dict],
    clips: Path,
    output: Path,
) -> dict:
    rows = []
    predictions = []
    model.eval()
    with torch.inference_mode():
        for index, metadata in enumerate(samples):
            batch = {key: value[None] for key, value in dataset[index].items()}
            result = model(**_model_inputs(batch))
            valid = batch["frame_mask"][0]
            probabilities = result.presence_logits[0].sigmoid()
            best_frame = int(probabilities.masked_fill(~valid, -1).argmax().item())
            boxes = result.boxes_xyxy[0].numpy()
            width, height = metadata["decoded_width"], metadata["decoded_height"]
            original_boxes = [original_box(box, width, height) for box in boxes]
            prediction = {
                "sample_id": metadata["sample_id"],
                "clip_uid": metadata["clip_uid"],
                "annotation_frame_ids": metadata["frame_ids"],
                "clip_timestamps_seconds": [
                    frame / metadata["annotation_fps"] for frame in metadata["frame_ids"]
                ],
                "presence_probability": probabilities[valid].tolist(),
                "boxes_xyxy_padded": boxes[valid.numpy()].tolist(),
                "boxes_xyxy_original": [
                    box for box, is_valid in zip(original_boxes, valid.tolist()) if is_valid
                ],
            }
            predictions.append(prediction)
            path = clips / f"{metadata['clip_uid']}.mp4"
            capture = cv2.VideoCapture(str(path))
            try:
                record = annotations[metadata["sample_id"]]
                query = read_frame(capture, metadata["query_media_frame"], path)
                crop_box = normalized_box(record["visual_crop"]) * np.array([width, height] * 2)
                x1, y1 = np.floor(crop_box[:2]).astype(int)
                x2, y2 = np.ceil(crop_box[2:]).astype(int)
                query_image = Image.fromarray(query[y1:y2, x1:x2])
                query_image.thumbnail((180, 180))
                query_image.save(output / f"query-{index}.jpg")
                # Show the model's highest-presence frame and, separately, a
                # labeled positive frame for diagnosing spatial predictions.
                positions = [best_frame]
                positives = torch.where(batch["object_present"][0] & valid)[0]
                if len(positives) and int(positives[0]) != best_frame:
                    positions.append(int(positives[0]))
                panels = []
                for position in positions:
                    rgb = read_frame(capture, metadata["media_frame_ids"][position], path)
                    image = Image.fromarray(rgb)
                    image.thumbnail((640, 480))
                    draw_box(image, original_boxes[position], "red")
                    positive = bool(batch["object_present"][0, position])
                    iou_text = "no annotated response at this frame"
                    if positive:
                        target = batch["boxes_xyxy"][0, position]
                        draw_box(image, original_box(target.numpy(), width, height), "lime")
                        iou = _box_iou(result.boxes_xyxy[0, position:position + 1], target[None])
                        iou_text = f"box IoU {float(iou.item()):.3f}"
                    name = f"prediction-{index}-{position}.jpg"
                    image.save(output / name)
                    label = "highest presence" if position == best_frame else "annotated-positive diagnostic"
                    panels.append(
                        f'<figure><img src="{name}"><figcaption>{label}; '
                        f'{metadata["frame_ids"][position] / metadata["annotation_fps"]:.2f}s; '
                        f'presence {float(probabilities[position]):.3f}; {iou_text}'
                        '</figcaption></figure>'
                    )
                rows.append(
                    f'<section><h2>{html.escape(metadata["sample_id"])}: '
                    f'{html.escape(metadata["object_title"])}</h2>'
                    f'<p>Query crop:</p><img src="query-{index}.jpg">'
                    + "".join(panels) + "</section>"
                )
            finally:
                capture.release()
    (output / "predictions.json").write_text(json.dumps(predictions, indent=2), encoding="utf-8")
    (output / "review.html").write_text(
        '<!doctype html><meta charset="utf-8"><title>Ego4D localization review</title>'
        '<style>body{font-family:sans-serif;max-width:1100px;margin:auto}'
        'section{border-top:1px solid #aaa;padding:12px}figure{display:inline-block;margin:8px}'
        'img{max-width:100%}figcaption{max-width:640px}</style>'
        '<h1>Ego4D validation: image-query prototype</h1>'
        '<p>Red: prediction. Green: annotated target. One highest-presence frame '
        'per query plus an explicitly ground-truth-selected diagnostic frame where available. '
        'Sparse history sampling; not official VQ2D evaluation.</p>'
        + "".join(rows), encoding="utf-8",
    )
    return {"queries": len(samples)}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--features", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--clips", type=Path, default=ROOT / "Ego4D" / "v2" / "clips")
    parser.add_argument("--annotations", type=Path, default=ROOT / "VQLoC" / "data" / "val_anno_new.json")
    args = parser.parse_args()
    if args.output_dir.exists():
        parser.error("--output-dir already exists; choose a fresh directory")
    torch.set_num_threads(4)
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    model = (
        SpatialLocalizer(**checkpoint["model_config"])
        if checkpoint.get("model_type") == "spatial"
        else MultimodalTemporalLocalizer(**checkpoint["model_config"])
    )
    model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    dataset = FeatureBundleDataset(args.features)
    try:
        samples = json.loads((args.features / "samples.json").read_text(encoding="utf-8"))
        if len(samples) != len(dataset):
            raise ValueError("Sample metadata and feature bundle lengths do not match")
        annotations = {
            record["dataset_uid"]: record
            for record in json.loads(args.annotations.read_text(encoding="utf-8"))
        }
        metrics = _run_epoch(model, DataLoader(dataset, batch_size=1), torch.device("cpu"))
        args.output_dir.mkdir(parents=True)
        export_review(model, dataset, samples, annotations, args.clips, args.output_dir)
        (args.output_dir / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
        print(json.dumps(metrics, indent=2))
        print(f"Open {args.output_dir / 'review.html'}")
    finally:
        dataset.close()


if __name__ == "__main__":
    main()
