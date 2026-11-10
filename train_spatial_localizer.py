"""Compare Ego4D-only and joint EPIC/Ego4D training of a shared spatial matcher."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader, Subset, TensorDataset

from spatial_localizer import SpatialLocalizer, spatial_loss
from train_multimodal_localizer import (
    FeatureBundleDataset, _model_inputs, _run_epoch, _seed_everything,
)


def train(
    train_features: Path,
    val_features: Path,
    output: Path,
    *,
    epic_features: Path | None,
    epochs: int,
    seed: int,
    auxiliary_weight: float,
    development_features: Path | None = None,
    assigned_boxes: bool = False,
    positive_presence_weight: float = 1.0,
    patience: int = 0,
    training_box_prior: bool = False,
    device: str = "cpu",
) -> None:
    if epochs <= 0 or not np.isfinite(auxiliary_weight) or auxiliary_weight < 0:
        raise ValueError("Epochs must be positive and auxiliary weight finite and nonnegative")
    if patience < 0 or not np.isfinite(positive_presence_weight) or positive_presence_weight <= 0:
        raise ValueError("Patience must be nonnegative and presence weight finite and positive")
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite {output}")
    if device not in ("cpu", "cuda", "auto"):
        raise ValueError("device must be cpu, cuda, or auto")
    if device == "cuda" and not torch.cuda.is_available():
        raise ValueError("CUDA was requested but is unavailable")
    compute_device = torch.device(
        "cuda" if device == "cuda" or (device == "auto" and torch.cuda.is_available()) else "cpu"
    )
    if train_features.resolve() == val_features.resolve():
        raise ValueError("Train and validation bundles must differ")
    _seed_everything(seed)
    train_data = FeatureBundleDataset(train_features)
    val_data = FeatureBundleDataset(val_features)
    development_data = None
    try:
        if (
            train_data.video_dim != val_data.video_dim
            or train_data.image_query_dim != train_data.video_dim
            or val_data.image_query_dim != train_data.video_dim
        ):
            raise ValueError("Matching image/video feature dimensions are required")
        metadata_paths = [path / "samples.json" for path in (train_features, val_features)]
        if all(path.exists() for path in metadata_paths):
            source_videos = [
                {sample["video_uid"] for sample in json.loads(path.read_text(encoding="utf-8"))}
                for path in metadata_paths
            ]
            if source_videos[0] & source_videos[1]:
                raise ValueError("Source videos overlap between training and validation")
        fitting_data = train_data
        if development_features is not None:
            if development_features.resolve() in (train_features.resolve(), val_features.resolve()):
                raise ValueError("Development features must be a separate bundle")
            development_data = FeatureBundleDataset(development_features)
            if (
                development_data.video_dim != train_data.video_dim
                or development_data.image_query_dim != train_data.image_query_dim
            ):
                raise ValueError("Development feature dimensions must match training")
            dev_metadata = json.loads((development_features / "samples.json").read_text(encoding="utf-8"))
            train_metadata = json.loads((train_features / "samples.json").read_text(encoding="utf-8"))
            final_metadata = json.loads((val_features / "samples.json").read_text(encoding="utf-8"))
            dev_videos = {row["video_uid"] for row in dev_metadata}
            train_videos = {row["video_uid"] for row in train_metadata}
            if not dev_videos or not dev_videos < train_videos:
                raise ValueError("Development videos must be a strict subset of training source videos")
            if dev_videos & {row["video_uid"] for row in final_metadata}:
                raise ValueError("Development and final validation source videos overlap")
            indices = [i for i, row in enumerate(train_metadata) if row["video_uid"] not in dev_videos]
            fitting_data = Subset(train_data, indices)
        # A separate generator preserves identical Ego4D order in the ablation.
        generator = torch.Generator().manual_seed(seed)
        loader = DataLoader(fitting_data, batch_size=4, shuffle=True, generator=generator)
        validation = DataLoader(development_data if development_data is not None else val_data, batch_size=2)
        epic_loader = None
        if epic_features is not None:
            if not (epic_features / "COMPLETE").exists():
                raise ValueError("EPIC feature preparation is incomplete")
            features = torch.from_numpy(np.load(epic_features / "features.npy", allow_pickle=False)).float()
            labels = torch.from_numpy(np.load(epic_features / "noun_labels.npy", allow_pickle=False))
            if features.ndim != 2 or features.shape != (len(labels), train_data.video_dim):
                raise ValueError("EPIC features must be [N, feature_dim]")
            if labels.dtype != torch.long or labels.ndim != 1 or len(labels) == 0:
                raise ValueError("EPIC noun labels must be a nonempty int64 vector")
            if not torch.isfinite(features).all() or labels.min() < 0 or labels.max() >= 300:
                raise ValueError("Invalid EPIC feature values or noun label range")
            epic_loader = DataLoader(
                TensorDataset(features, labels), batch_size=16, shuffle=True,
                generator=torch.Generator().manual_seed(seed),
            )
        model = SpatialLocalizer(feature_dim=train_data.video_dim)
        box_prior = None
        if training_box_prior:
            indices = fitting_data.indices if isinstance(fitting_data, Subset) else np.arange(len(train_data))
            fitting_boxes = train_data.arrays["boxes_xyxy"][indices]
            fitting_present = train_data.arrays["object_present"][indices].astype(bool)
            if "frame_mask" in train_data.arrays:
                fitting_present &= train_data.arrays["frame_mask"][indices].astype(bool)
            if not fitting_present.any():
                raise ValueError("Cannot initialize box prior without fitting-set positive boxes")
            positive_boxes = fitting_boxes[fitting_present]
            median_size = np.median(positive_boxes[:, 2:] - positive_boxes[:, :2], axis=0)
            box_prior = median_size.tolist()
            with torch.no_grad():
                model.box_head[-1].bias[2:].copy_(
                    torch.from_numpy(np.log(median_size / 0.15)).clamp(-3, 3)
                )
        model.to(compute_device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-4)
        output.mkdir(parents=True)
        config = {
            "feature_dim": train_data.video_dim, "noun_classes": 300,
        }
        run = {
            "seed": seed, "epochs": epochs, "learning_rate": 1e-4,
            "batch_size": 4, "auxiliary_weight": auxiliary_weight,
            "train_features": str(train_features), "val_features": str(val_features),
            "epic_features": str(epic_features) if epic_features else None,
            "selection": (
                "maximum development mean_positive_iou, including epoch zero"
                if development_data is not None else "maximum validation mean_positive_iou"
            ),
            "development_features": str(development_features) if development_features else None,
            "fitting_queries": len(fitting_data),
            "selection_queries": len(validation.dataset),
            "assigned_box_supervision": assigned_boxes,
            "positive_presence_weight": positive_presence_weight,
            "patience": patience,
            "box_prior_from_fitting_data": box_prior,
            "device": str(compute_device),
        }
        (output / "run.json").write_text(json.dumps(run, indent=2), encoding="utf-8")
        initial = _run_epoch(model, validation, compute_device)
        (output / "initial.json").write_text(json.dumps(initial, indent=2), encoding="utf-8")
        if initial["mean_positive_iou"] is None:
            raise ValueError("Checkpoint selection requires at least one sampled positive frame")
        best = initial["mean_positive_iou"] if development_data is not None else -1.0
        if development_data is not None:
            torch.save({
                "model_type": "spatial", "model_config": config,
                "model_state_dict": model.state_dict(), "epoch": 0,
                "validation": initial, "run": run,
            }, output / "best.pt")
        stale_epochs = 0
        with (output / "metrics.jsonl").open("w", encoding="utf-8") as file:
            for epoch in range(1, epochs + 1):
                model.train()
                auxiliary = []
                localization = []
                epic_batches = iter(epic_loader) if epic_loader else None
                epic_steps = 0
                for batch in loader:
                    batch = {key: value.to(compute_device) for key, value in batch.items()}
                    optimizer.zero_grad(set_to_none=True)
                    result = model(**_model_inputs(batch))
                    losses = spatial_loss(
                        result, batch["boxes_xyxy"], batch["object_present"], batch["frame_mask"],
                        assigned_boxes=assigned_boxes,
                        positive_presence_weight=positive_presence_weight,
                    )
                    loss = losses["loss"]
                    localization.append(float(loss.detach()))
                    if epic_loader is not None:
                        try:
                            features, labels = next(epic_batches)
                        except StopIteration:
                            epic_batches = iter(epic_loader)
                            features, labels = next(epic_batches)
                        noun_loss = F.cross_entropy(
                            model.classify_epic(features.to(compute_device)), labels.to(compute_device)
                        )
                        loss = loss + auxiliary_weight * noun_loss
                        auxiliary.append(float(noun_loss.detach()))
                        epic_steps += 1
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 5)
                    optimizer.step()
                metrics = _run_epoch(model, validation, compute_device)
                record = {
                    "epoch": epoch, "train_loss": sum(localization) / len(localization),
                    "epic_noun_loss": sum(auxiliary) / len(auxiliary) if auxiliary else None,
                    "epic_steps": epic_steps, "validation": metrics,
                }
                file.write(json.dumps(record, allow_nan=False) + "\n")
                file.flush()
                checkpoint = {
                    "model_type": "spatial", "model_config": config,
                    "model_state_dict": model.state_dict(), "epoch": epoch,
                    "validation": metrics, "run": run,
                }
                torch.save(checkpoint, output / "last.pt")
                if metrics["mean_positive_iou"] > best:
                    best = metrics["mean_positive_iou"]
                    stale_epochs = 0
                    torch.save(checkpoint, output / "best.pt")
                else:
                    stale_epochs += 1
                print(json.dumps(record), flush=True)
                if patience and stale_epochs >= patience:
                    print(f"Early stopping after {epoch} epochs without further selection improvement.", flush=True)
                    break
        if development_data is not None:
            from evaluate_spatial_localizer import calibrate_presence, evaluate

            checkpoint = torch.load(output / "best.pt", map_location="cpu", weights_only=True)
            model.load_state_dict(checkpoint["model_state_dict"], strict=True)
            calibration = calibrate_presence(model, validation)
            with torch.no_grad():
                model.presence_bias.sub_(calibration["logit_threshold"])
            checkpoint["model_state_dict"] = model.state_dict()
            checkpoint["calibration"] = calibration
            checkpoint["selection_split"] = "training_source_video_disjoint_development"
            final_metrics = evaluate(model, DataLoader(val_data, batch_size=2))
            checkpoint["final_validation"] = final_metrics
            torch.save(checkpoint, output / "calibrated.pt")
            (output / "final-validation.json").write_text(
                json.dumps(final_metrics, indent=2, allow_nan=False), encoding="utf-8"
            )
            (output / "calibration.json").write_text(json.dumps(calibration, indent=2), encoding="utf-8")
            print("FINAL VALIDATION " + json.dumps(final_metrics), flush=True)
        (output / "COMPLETE").touch()
    finally:
        train_data.close()
        val_data.close()
        if development_data is not None:
            development_data.close()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-features", type=Path, required=True)
    parser.add_argument("--val-features", type=Path, required=True)
    parser.add_argument("--epic-features", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--auxiliary-weight", type=float, default=0.1)
    parser.add_argument("--development-features", type=Path)
    parser.add_argument("--assigned-boxes", action="store_true")
    parser.add_argument("--positive-presence-weight", type=float, default=1.0)
    parser.add_argument("--patience", type=int, default=0)
    parser.add_argument("--training-box-prior", action="store_true")
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="cpu")
    args = parser.parse_args()
    if args.epochs <= 0 or not np.isfinite(args.auxiliary_weight) or args.auxiliary_weight < 0:
        parser.error("Epochs must be positive and auxiliary weight finite and nonnegative")
    if args.patience < 0 or not np.isfinite(args.positive_presence_weight) or args.positive_presence_weight <= 0:
        parser.error("Patience must be nonnegative and presence weight finite and positive")
    torch.set_num_threads(4)
    train(
        args.train_features, args.val_features, args.output_dir,
        epic_features=args.epic_features, epochs=args.epochs,
        seed=args.seed, auxiliary_weight=args.auxiliary_weight,
        development_features=args.development_features, assigned_boxes=args.assigned_boxes,
        positive_presence_weight=args.positive_presence_weight, patience=args.patience,
        training_box_prior=args.training_box_prior, device=args.device,
    )


if __name__ == "__main__":
    main()
