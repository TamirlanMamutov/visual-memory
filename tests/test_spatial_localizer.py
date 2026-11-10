import json
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
import torch
from torch.nn import functional as F

from prepare_epic_localization import timestamp_seconds
from spatial_localizer import SpatialLocalizer, spatial_loss
from train_spatial_localizer import train


class SpatialLocalizerTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(42)
        torch.set_num_threads(2)

    def test_patch_centers_and_matching_peak_are_spatially_anchored(self):
        model = SpatialLocalizer(feature_dim=4, noun_classes=3)
        video = torch.eye(4)[None, None]
        query = torch.tensor([[[0.0, 0.0, 1.0, 0.0]] * 4])
        output = model(video, query)
        self.assertEqual(int(output.patch_logits.argmax(-1)), 2)
        np.testing.assert_allclose(
            output.boxes_xyxy.detach().numpy()[0, 0],
            [0.175, 0.675, 0.325, 0.825], atol=1e-6,
        )

    def test_both_datasets_backpropagate_into_the_same_projection(self):
        model = SpatialLocalizer(feature_dim=8, noun_classes=3)
        epic_loss = F.cross_entropy(
            model.classify_epic(torch.randn(4, 8)), torch.tensor([0, 1, 2, 1])
        )
        epic_loss.backward()
        self.assertGreater(float(model.projection.weight.grad.norm()), 0)
        model.zero_grad(set_to_none=True)
        output = model(torch.randn(2, 3, 4, 8), torch.randn(2, 4, 8))
        loss = spatial_loss(
            output, torch.tensor([[[0.1, 0.2, 0.5, 0.6]] * 3] * 2),
            torch.ones(2, 3, dtype=torch.bool), torch.ones(2, 3, dtype=torch.bool),
        )["loss"]
        loss.backward()
        self.assertGreater(float(model.projection.weight.grad.norm()), 0)
        self.assertGreater(float(model.box_head[-1].weight.grad.norm()), 0)

    def test_spatial_loss_handles_no_response_frames(self):
        model = SpatialLocalizer(feature_dim=8)
        output = model(torch.randn(1, 2, 4, 8), torch.randn(1, 4, 8))
        losses = spatial_loss(
            output, torch.zeros(1, 2, 4), torch.zeros(1, 2, dtype=torch.bool),
            torch.ones(1, 2, dtype=torch.bool),
        )
        self.assertTrue(torch.isfinite(losses["loss"]))
        self.assertEqual(float(losses["attention_loss"].detach()), 0)
        losses["loss"].backward()

    def test_rejects_nonspatial_patch_layout(self):
        model = SpatialLocalizer(feature_dim=8)
        with self.assertRaisesRegex(ValueError, "square patch"):
            model(torch.randn(1, 2, 5, 8), torch.randn(1, 4, 8))

    def test_epic_timestamps(self):
        self.assertAlmostEqual(timestamp_seconds("01:02:03.5"), 3723.5)

    def test_joint_training_writes_reloadable_checkpoint_and_epic_steps(self):
        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            for split in ("train", "val"):
                bundle = root / split
                bundle.mkdir()
                np.save(bundle / "video_features.npy", np.random.default_rng(42).normal(size=(2, 2, 4, 8)).astype(np.float32))
                np.save(bundle / "image_query_features.npy", np.ones((2, 4, 8), dtype=np.float32))
                np.save(bundle / "boxes_xyxy.npy", np.tile(np.array([0.1, 0.2, 0.5, 0.6], dtype=np.float32), (2, 2, 1)))
                np.save(bundle / "object_present.npy", np.ones((2, 2), dtype=bool))
            epic = root / "epic"
            epic.mkdir()
            np.save(epic / "features.npy", np.ones((3, 8), dtype=np.float32))
            np.save(epic / "noun_labels.npy", np.array([0, 1, 2], dtype=np.int64))
            (epic / "COMPLETE").touch()
            train(
                root / "train", root / "val", root / "out",
                epic_features=epic, epochs=1, seed=42, auxiliary_weight=0.1,
            )
            metrics = json.loads((root / "out" / "metrics.jsonl").read_text())
            self.assertEqual(metrics["epic_steps"], 1)
            self.assertTrue(np.isfinite(metrics["epic_noun_loss"]))
            checkpoint = torch.load(root / "out" / "best.pt", weights_only=True)
            reloaded = SpatialLocalizer(**checkpoint["model_config"])
            reloaded.load_state_dict(checkpoint["model_state_dict"], strict=True)
            self.assertTrue((root / "out" / "COMPLETE").exists())


if __name__ == "__main__":
    unittest.main()
