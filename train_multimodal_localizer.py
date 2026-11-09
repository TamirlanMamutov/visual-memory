"""Train the feature-based multimodal temporal localizer on a feature bundle."""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import Tensor
from torch.utils.data import DataLoader, Dataset

from multimodal_localizer import MultimodalTemporalLocalizer, localization_loss


_REQUIRED_ARRAYS = ("video_features", "boxes_xyxy", "object_present")
_OPTIONAL_ARRAYS = (
    "image_query_features",
    "image_query_mask",
    "text_query_features",
    "text_query_mask",
    "frame_mask",
)


class FeatureBundleDataset(Dataset):
    """Load per-sample localization features from a directory of ``.npy`` files.

    Required arrays are ``video_features[N,T,P,C]``, ``boxes_xyxy[N,T,4]``,
    and ``object_present[N,T]``. Optional query features have shape
    ``[N,Q,C]``; corresponding masks have shape ``[N,Q]``.
    """

    def __init__(self, bundle_dir: str | Path) -> None:
        self.bundle_dir = Path(bundle_dir)
        if not self.bundle_dir.is_dir():
            raise FileNotFoundError(f"Feature bundle directory does not exist: {self.bundle_dir}")
        missing = [
            name for name in _REQUIRED_ARRAYS
            if not (self.bundle_dir / f"{name}.npy").is_file()
        ]
        if missing:
            raise FileNotFoundError(
                f"Feature bundle is missing required arrays: {', '.join(missing)}"
            )

        self.arrays: dict[str, np.ndarray] = {}
        for name in (*_REQUIRED_ARRAYS, *_OPTIONAL_ARRAYS):
            path = self.bundle_dir / f"{name}.npy"
            if path.is_file():
                array = np.load(path, mmap_mode="r", allow_pickle=False)
                if array.dtype.kind == "O":
                    raise TypeError(f"{path} must not contain object data")
                self.arrays[name] = array

        video = self.arrays["video_features"]
        boxes = self.arrays["boxes_xyxy"]
        present = self.arrays["object_present"]
        if video.ndim != 4 or min(video.shape[1:]) <= 0:
            raise ValueError("video_features must have shape [N,T,P,C] with positive dimensions")
        if boxes.shape != (video.shape[0], video.shape[1], 4):
            raise ValueError("boxes_xyxy must have shape [N,T,4] matching video_features")
        if present.shape != video.shape[:2]:
            raise ValueError("object_present must have shape [N,T] matching video_features")
        if video.shape[0] == 0:
            raise ValueError("Feature bundle must contain at least one sample")
        if present.dtype.kind not in "biu" or not np.isin(present, (0, 1)).all():
            raise ValueError("object_present must contain only boolean or 0/1 values")
        if not np.isfinite(video).all() or not np.isfinite(boxes).all():
            raise ValueError("Feature bundle video features and boxes must be finite")
        if ((boxes < 0) | (boxes > 1)).any():
            raise ValueError("boxes_xyxy coordinates must be normalized to [0,1]")
        if np.any((boxes[..., 2] < boxes[..., 0]) | (boxes[..., 3] < boxes[..., 1])):
            raise ValueError("boxes_xyxy must use ordered [x1,y1,x2,y2] coordinates")
        positive = present.astype(bool)
        if np.any(
            positive
            & (
                (boxes[..., 2] <= boxes[..., 0])
                | (boxes[..., 3] <= boxes[..., 1])
            )
        ):
            raise ValueError("Present-object targets must have positive box area")
        if not any(
            name in self.arrays
            for name in ("image_query_features", "text_query_features")
        ):
            raise ValueError("Feature bundle needs image and/or text query features")

        valid_query = np.zeros(video.shape[0], dtype=bool)
        for feature_name, mask_name in (
            ("image_query_features", "image_query_mask"),
            ("text_query_features", "text_query_mask"),
        ):
            features = self.arrays.get(feature_name)
            mask = self.arrays.get(mask_name)
            if features is None:
                if mask is not None:
                    raise ValueError(f"{mask_name} was supplied without {feature_name}")
                continue
            if features.ndim != 3 or features.shape[0] != video.shape[0]:
                raise ValueError(f"{feature_name} must have shape [N,Q,C]")
            if min(features.shape[1:]) <= 0 or not np.isfinite(features).all():
                raise ValueError(f"{feature_name} must be finite with non-empty Q and C")
            if mask is not None:
                if mask.shape != features.shape[:2] or mask.dtype.kind not in "biu":
                    raise ValueError(f"{mask_name} must be a boolean [N,Q] array")
                if not np.isin(mask, (0, 1)).all():
                    raise ValueError(f"{mask_name} must contain only boolean or 0/1 values")
                valid_query |= mask.astype(bool).any(axis=1)
            else:
                valid_query[:] = True
        if np.any(~valid_query):
            raise ValueError("Every sample needs at least one valid query token")

        frame_mask = self.arrays.get("frame_mask")
        if frame_mask is not None:
            if frame_mask.shape != video.shape[:2] or frame_mask.dtype.kind not in "biu":
                raise ValueError("frame_mask must be a boolean [N,T] array")
            if not np.isin(frame_mask, (0, 1)).all() or np.any(
                ~frame_mask.astype(bool).any(axis=1)
            ):
                raise ValueError("Each sample needs at least one valid frame")

        self.video_dim = int(video.shape[-1])
        self.image_query_dim = self._feature_dim("image_query_features")
        self.text_query_dim = self._feature_dim("text_query_features")

    def __len__(self) -> int:
        return int(self.arrays["video_features"].shape[0])

    def close(self) -> None:
        for array in self.arrays.values():
            if isinstance(array, np.memmap):
                array._mmap.close()

    def __getitem__(self, index: int) -> dict[str, Tensor]:
        sample = {
            "video_features": self._tensor("video_features", index),
            "boxes_xyxy": self._tensor("boxes_xyxy", index),
            "object_present": self._tensor("object_present", index).bool(),
        }
        for name in (
            "image_query_features",
            "image_query_mask",
            "text_query_features",
            "text_query_mask",
            "frame_mask",
        ):
            if name in self.arrays:
                sample[name] = self._tensor(name, index)
                if name.endswith("_mask"):
                    sample[name] = sample[name].bool()
        if "frame_mask" not in sample:
            sample["frame_mask"] = torch.ones(
                sample["video_features"].shape[0], dtype=torch.bool
            )
        return sample

    def _feature_dim(self, name: str) -> int | None:
        return int(self.arrays[name].shape[-1]) if name in self.arrays else None

    def _tensor(self, name: str, index: int) -> Tensor:
        array = np.array(self.arrays[name][index], copy=True)
        tensor = torch.from_numpy(array)
        if name in {"video_features", "boxes_xyxy", "image_query_features", "text_query_features"}:
            tensor = tensor.to(torch.float32)
        return tensor


def _model_inputs(batch: dict[str, Tensor]) -> dict[str, Tensor]:
    return {
        key: batch[key]
        for key in (
            "video_features",
            "image_query_features",
            "text_query_features",
            "image_query_mask",
            "text_query_mask",
            "frame_mask",
        )
        if key in batch
    }


def _run_epoch(
    model: MultimodalTemporalLocalizer,
    loader: DataLoader,
    device: torch.device,
    optimizer: torch.optim.Optimizer | None = None,
) -> dict[str, float | None]:
    training = optimizer is not None
    model.train(training)
    total_loss = 0.0
    sample_count = 0
    true_positive = false_positive = false_negative = 0
    iou_sum = 0.0
    positive_frame_count = 0
    context = torch.enable_grad if training else torch.no_grad
    with context():
        for batch in loader:
            batch = {key: value.to(device) for key, value in batch.items()}
            if optimizer is not None:
                optimizer.zero_grad(set_to_none=True)
            output = model(**_model_inputs(batch))
            losses = localization_loss(
                output,
                batch["boxes_xyxy"],
                batch["object_present"],
                batch["frame_mask"],
            )
            if optimizer is not None:
                losses["loss"].backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=20.0)
                optimizer.step()

            batch_size = batch["video_features"].shape[0]
            total_loss += losses["loss"].detach().item() * batch_size
            sample_count += batch_size
            valid = batch["frame_mask"]
            target = batch["object_present"]
            prediction = output.presence_logits >= 0
            true_positive += int((prediction & target & valid).sum().item())
            false_positive += int((prediction & ~target & valid).sum().item())
            false_negative += int((~prediction & target & valid).sum().item())
            positive = target & valid
            if positive.any():
                iou_sum += _box_iou(
                    output.boxes_xyxy[positive], batch["boxes_xyxy"][positive]
                ).sum().item()
                positive_frame_count += int(positive.sum().item())

    if sample_count == 0:
        raise ValueError("Cannot run an epoch with an empty feature bundle")
    f1_denominator = 2 * true_positive + false_positive + false_negative
    return {
        "loss": total_loss / sample_count,
        "presence_f1": 2 * true_positive / f1_denominator if f1_denominator else 0.0,
        "mean_positive_iou": (
            iou_sum / positive_frame_count if positive_frame_count else None
        ),
    }


def _box_iou(boxes1: Tensor, boxes2: Tensor) -> Tensor:
    intersection_min = torch.maximum(boxes1[:, :2], boxes2[:, :2])
    intersection_max = torch.minimum(boxes1[:, 2:], boxes2[:, 2:])
    intersection_size = (intersection_max - intersection_min).clamp_min(0)
    intersection = intersection_size[:, 0] * intersection_size[:, 1]
    area1_size = (boxes1[:, 2:] - boxes1[:, :2]).clamp_min(0)
    area2_size = (boxes2[:, 2:] - boxes2[:, :2]).clamp_min(0)
    area1 = area1_size[:, 0] * area1_size[:, 1]
    area2 = area2_size[:, 0] * area2_size[:, 1]
    return intersection / (area1 + area2 - intersection).clamp_min(1e-8)


def _seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-features", required=True, type=Path)
    parser.add_argument("--val-features", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--hidden-dim", type=int, default=256)
    parser.add_argument("--num-heads", type=int, default=8)
    parser.add_argument("--temporal-layers", type=int, default=2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="auto", choices=("auto", "cpu", "cuda"))
    args = parser.parse_args()
    if args.epochs <= 0 or args.batch_size <= 0:
        parser.error("--epochs and --batch-size must be positive")
    if args.learning_rate <= 0 or args.weight_decay < 0:
        parser.error("--learning-rate must be positive and --weight-decay nonnegative")
    if args.device == "cuda" and not torch.cuda.is_available():
        parser.error("--device cuda was requested but CUDA is unavailable")

    _seed_everything(args.seed)
    train_data = FeatureBundleDataset(args.train_features)
    val_data = FeatureBundleDataset(args.val_features)
    if (
        train_data.video_dim != val_data.video_dim
        or train_data.image_query_dim != val_data.image_query_dim
        or train_data.text_query_dim != val_data.text_query_dim
    ):
        raise ValueError("Training and validation feature dimensions/modalities must match")

    device = torch.device(
        "cuda" if args.device == "cuda" or (args.device == "auto" and torch.cuda.is_available()) else "cpu"
    )
    train_loader = DataLoader(
        train_data, batch_size=args.batch_size, shuffle=True, num_workers=0
    )
    val_loader = DataLoader(
        val_data, batch_size=args.batch_size, shuffle=False, num_workers=0
    )
    model = MultimodalTemporalLocalizer(
        video_dim=train_data.video_dim,
        image_query_dim=train_data.image_query_dim,
        text_query_dim=train_data.text_query_dim,
        hidden_dim=args.hidden_dim,
        num_heads=args.num_heads,
        num_temporal_layers=args.temporal_layers,
    ).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = args.output_dir / "metrics.jsonl"
    best_val_loss = float("inf")

    with metrics_path.open("w", encoding="utf-8") as metrics_file:
        for epoch in range(1, args.epochs + 1):
            train_metrics = _run_epoch(model, train_loader, device, optimizer)
            val_metrics = _run_epoch(model, val_loader, device)
            record: dict[str, Any] = {
                "epoch": epoch,
                "train": train_metrics,
                "validation": val_metrics,
            }
            metrics_file.write(json.dumps(record, allow_nan=False) + "\n")
            metrics_file.flush()
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "model_config": {
                        "video_dim": train_data.video_dim,
                        "image_query_dim": train_data.image_query_dim,
                        "text_query_dim": train_data.text_query_dim,
                        "hidden_dim": args.hidden_dim,
                        "num_heads": args.num_heads,
                        "num_temporal_layers": args.temporal_layers,
                    },
                    "seed": args.seed,
                },
                args.output_dir / "last.pt",
            )
            val_loss = float(val_metrics["loss"])
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                torch.save(
                    {
                        "epoch": epoch,
                        "model_state_dict": model.state_dict(),
                        "model_config": {
                            "video_dim": train_data.video_dim,
                            "image_query_dim": train_data.image_query_dim,
                            "text_query_dim": train_data.text_query_dim,
                            "hidden_dim": args.hidden_dim,
                            "num_heads": args.num_heads,
                            "num_temporal_layers": args.temporal_layers,
                        },
                        "seed": args.seed,
                        "validation": val_metrics,
                    },
                    args.output_dir / "best.pt",
                )
            print(json.dumps(record, allow_nan=False), flush=True)
    train_data.close()
    val_data.close()


if __name__ == "__main__":
    main()
