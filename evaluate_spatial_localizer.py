"""Frame-level diagnostics and development-only presence calibration."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from torch import Tensor
from torch.utils.data import DataLoader

from spatial_localizer import SpatialLocalizer
from train_multimodal_localizer import FeatureBundleDataset, _box_iou, _model_inputs, _run_epoch


def threshold_curve(scores: Tensor, labels: Tensor) -> tuple[Tensor, Tensor, Tensor, Tensor]:
    if scores.ndim != 1 or scores.shape != labels.shape or not len(scores):
        raise ValueError("Scores and labels must be matching nonempty vectors")
    if labels.dtype != torch.bool or not torch.isfinite(scores).all():
        raise ValueError("Scores must be finite and labels boolean")
    order = scores.argsort(descending=True, stable=True)
    sorted_scores = scores[order]
    true_positives = labels[order].long().cumsum(0)
    count = torch.arange(1, len(scores) + 1, device=scores.device)
    ends = torch.cat((
        sorted_scores[:-1] != sorted_scores[1:],
        torch.tensor([True], device=scores.device),
    ))
    tp, count = true_positives[ends], count[ends]
    precision = tp / count
    recall = tp / labels.sum().clamp_min(1)
    f1 = 2 * tp / (count + labels.sum()).clamp_min(1)
    return sorted_scores[ends], precision, recall, f1


def calibrate_presence(model: SpatialLocalizer, loader: DataLoader) -> dict:
    model.eval()
    device = next(model.parameters()).device
    scores, labels = [], []
    with torch.inference_mode():
        for batch in loader:
            batch = {key: value.to(device) for key, value in batch.items()}
            result = model(**_model_inputs(batch))
            valid = batch["frame_mask"]
            scores.append(result.presence_logits[valid].cpu())
            labels.append(batch["object_present"][valid].cpu())
    scores, labels = torch.cat(scores), torch.cat(labels)
    if not labels.any():
        raise ValueError("Cannot calibrate presence without development response frames")
    thresholds, _, _, f1 = threshold_curve(scores, labels)
    index = int(f1.argmax())
    return {
        "logit_threshold": float(thresholds[index]),
        "development_f1": float(f1[index]),
        "development_frames": len(scores),
        "development_positive_frames": int(labels.sum()),
        "selection_rule": "maximum development F1; ties use higher threshold",
    }


def evaluate(model: SpatialLocalizer, loader: DataLoader) -> dict:
    device = next(model.parameters()).device
    metrics = _run_epoch(model, loader, device)
    scores, labels, overlaps = [], [], []
    top1_joint = top1_response = query_count = 0
    with torch.inference_mode():
        for batch in loader:
            batch = {key: value.to(device) for key, value in batch.items()}
            result = model(**_model_inputs(batch))
            valid = batch["frame_mask"]
            target = batch["object_present"] & valid
            iou = _box_iou(
                result.boxes_xyxy.reshape(-1, 4), batch["boxes_xyxy"].reshape(-1, 4)
            ).reshape_as(valid)
            scores.append(result.presence_logits[valid].cpu())
            labels.append(target[valid].cpu())
            overlaps.append(iou[valid].cpu())
            best = result.presence_logits.masked_fill(~valid, -float("inf")).argmax(-1)
            row = torch.arange(len(best), device=device)
            top1_response += int(target[row, best].sum())
            top1_joint += int((target[row, best] & (iou[row, best] >= 0.25)).sum())
            query_count += len(best)
    scores, labels, iou = torch.cat(scores), torch.cat(labels), torch.cat(overlaps)
    _, precision_curve, recall_curve, _ = threshold_curve(scores, labels)
    increments = torch.diff(torch.cat((torch.tensor([0.0]), recall_curve)))
    ap = float((increments * precision_curve).sum())
    prediction = scores >= 0
    tp = int((prediction & labels).sum())
    fp = int((prediction & ~labels).sum())
    fn = int((~prediction & labels).sum())
    joint_tp = int((prediction & labels & (iou >= 0.25)).sum())
    metrics.update({
        "response_frame_ap": ap,
        "presence_precision": tp / max(tp + fp, 1),
        "presence_recall": tp / max(tp + fn, 1),
        "presence_true_positive": tp,
        "presence_false_positive": fp,
        "presence_false_negative": fn,
        "positive_frame_iou25_fraction": float((iou[labels] >= 0.25).float().mean()) if labels.any() else None,
        "joint_frame_iou25_precision": joint_tp / max(tp + fp, 1),
        "joint_frame_iou25_recall": joint_tp / max(int(labels.sum()), 1),
        "top1_response_queries": top1_response,
        "top1_response_and_iou25_queries": top1_joint,
        "queries": query_count,
        "valid_frames": len(scores),
        "positive_frames": int(labels.sum()),
        "official_vq2d_metrics": False,
    })
    return metrics


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--features", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error("Refusing to overwrite an existing metric file")
    torch.set_num_threads(4)
    checkpoint = torch.load(args.checkpoint, weights_only=True, map_location="cpu")
    if checkpoint.get("model_type") != "spatial":
        raise ValueError("This evaluator requires a spatial model checkpoint")
    model = SpatialLocalizer(**checkpoint["model_config"])
    model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    dataset = FeatureBundleDataset(args.features)
    try:
        metrics = evaluate(model, DataLoader(dataset, batch_size=2))
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(metrics, indent=2, allow_nan=False), encoding="utf-8")
        print(json.dumps(metrics, indent=2))
    finally:
        dataset.close()


if __name__ == "__main__":
    main()
