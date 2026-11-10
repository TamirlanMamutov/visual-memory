"""Compare Ego4D-only and joint EPIC/Ego4D training of a shared spatial matcher."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader, TensorDataset

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
) -> None:
    if epochs <= 0 or not np.isfinite(auxiliary_weight) or auxiliary_weight < 0:
        raise ValueError("Epochs must be positive and auxiliary weight finite and nonnegative")
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite {output}")
    if train_features.resolve() == val_features.resolve():
        raise ValueError("Train and validation bundles must differ")
    _seed_everything(seed)
    train_data = FeatureBundleDataset(train_features)
    val_data = FeatureBundleDataset(val_features)
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
        # A separate generator preserves identical Ego4D order in the ablation.
        generator = torch.Generator().manual_seed(seed)
        loader = DataLoader(train_data, batch_size=4, shuffle=True, generator=generator)
        validation = DataLoader(val_data, batch_size=2)
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
            "selection": "maximum validation mean_positive_iou",
        }
        (output / "run.json").write_text(json.dumps(run, indent=2), encoding="utf-8")
        initial = _run_epoch(model, validation, torch.device("cpu"))
        (output / "initial.json").write_text(json.dumps(initial, indent=2), encoding="utf-8")
        best = -1.0
        with (output / "metrics.jsonl").open("w", encoding="utf-8") as file:
            for epoch in range(1, epochs + 1):
                model.train()
                auxiliary = []
                localization = []
                epic_batches = iter(epic_loader) if epic_loader else None
                epic_steps = 0
                for batch in loader:
                    optimizer.zero_grad(set_to_none=True)
                    result = model(**_model_inputs(batch))
                    losses = spatial_loss(
                        result, batch["boxes_xyxy"], batch["object_present"], batch["frame_mask"]
                    )
                    loss = losses["loss"]
                    localization.append(float(loss.detach()))
                    if epic_loader is not None:
                        try:
                            features, labels = next(epic_batches)
                        except StopIteration:
                            epic_batches = iter(epic_loader)
                            features, labels = next(epic_batches)
                        noun_loss = F.cross_entropy(model.classify_epic(features), labels)
                        loss = loss + auxiliary_weight * noun_loss
                        auxiliary.append(float(noun_loss.detach()))
                        epic_steps += 1
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 5)
                    optimizer.step()
                metrics = _run_epoch(model, validation, torch.device("cpu"))
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
                    torch.save(checkpoint, output / "best.pt")
                print(json.dumps(record), flush=True)
        (output / "COMPLETE").touch()
    finally:
        train_data.close()
        val_data.close()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-features", type=Path, required=True)
    parser.add_argument("--val-features", type=Path, required=True)
    parser.add_argument("--epic-features", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--auxiliary-weight", type=float, default=0.1)
    args = parser.parse_args()
    if args.epochs <= 0 or not np.isfinite(args.auxiliary_weight) or args.auxiliary_weight < 0:
        parser.error("Epochs must be positive and auxiliary weight finite and nonnegative")
    torch.set_num_threads(4)
    train(
        args.train_features, args.val_features, args.output_dir,
        epic_features=args.epic_features, epochs=args.epochs,
        seed=args.seed, auxiliary_weight=args.auxiliary_weight,
    )


if __name__ == "__main__":
    main()
