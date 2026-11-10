import json
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

import numpy as np
import torch
from torch.nn import functional as F

from prepare_epic_localization import timestamp_seconds
from spatial_localizer import SpatialLocalizer, spatial_loss
from train_spatial_localizer import train
from evaluate_spatial_localizer import threshold_curve


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

    def test_assigned_box_loss_trains_target_patch_not_wrong_peak(self):
        model = SpatialLocalizer(feature_dim=4)
        output = model(torch.eye(4)[None, None], torch.tensor([[[1., 0., 0., 0.]] * 4]))
        output.patch_boxes_xyxy.retain_grad()
        losses = spatial_loss(
            output, torch.tensor([[[0.6, 0.6, 0.9, 0.9]]]),
            torch.ones(1, 1, dtype=torch.bool), torch.ones(1, 1, dtype=torch.bool),
            assigned_boxes=True,
        )
        losses["loss"].backward()
        gradients = output.patch_boxes_xyxy.grad[0, 0].abs().sum(-1)
        self.assertEqual(int(gradients.argmax()), 3)
        self.assertEqual(float(gradients[0]), 0)
        self.assertGreater(float(gradients[3]), 0)

    def test_threshold_curve_groups_ties_and_maximizes_f1(self):
        thresholds, precision, recall, f1 = threshold_curve(
            torch.tensor([2., 2., 1., 0.]), torch.tensor([True, False, True, False])
        )
        self.assertEqual(thresholds.tolist(), [2., 1., 0.])
        self.assertAlmostEqual(float(f1.max()), 0.8, places=6)
        self.assertEqual(float(thresholds[f1.argmax()]), 1.)
        self.assertAlmostEqual(float(precision[0]), 0.5)
        self.assertAlmostEqual(float(recall[-1]), 1.)

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

    def test_development_selection_excludes_its_videos_and_saves_final_evaluation(self):
        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            for split, videos in (
                ("train", ["fit-video", "dev-video"]),
                ("development", ["dev-video", "dev-video"]),
                ("val", ["final-video", "final-video"]),
            ):
                bundle = root / split
                bundle.mkdir()
                rng = np.random.default_rng(42)
                np.save(bundle / "video_features.npy", rng.normal(size=(2, 2, 4, 8)).astype(np.float32))
                np.save(bundle / "image_query_features.npy", rng.normal(size=(2, 4, 8)).astype(np.float32))
                boxes = np.tile(np.array([0.1, 0.2, 0.5, 0.6], dtype=np.float32), (2, 2, 1))
                if split == "train":
                    boxes[1] = [0.0, 0.0, 0.9, 0.9]
                np.save(bundle / "boxes_xyxy.npy", boxes)
                np.save(bundle / "object_present.npy", np.array([[True, False], [False, True]]))
                (bundle / "samples.json").write_text(json.dumps([
                    {"video_uid": video} for video in videos
                ]), encoding="utf-8")
            with patch("train_spatial_localizer._run_epoch", side_effect=[
                {"loss": 1.0, "presence_f1": 0.1, "mean_positive_iou": 0.5},
                {"loss": 0.9, "presence_f1": 0.2, "mean_positive_iou": 0.1},
            ]):
                train(
                    root / "train", root / "val", root / "out",
                    epic_features=None, epochs=1, seed=42, auxiliary_weight=0.1,
                    development_features=root / "development", assigned_boxes=True,
                    positive_presence_weight=0.1, patience=1, training_box_prior=True,
                )
            checkpoint = torch.load(root / "out" / "calibrated.pt", weights_only=True)
            self.assertEqual(checkpoint["epoch"], 0)
            self.assertEqual(checkpoint["run"]["fitting_queries"], 1)
            self.assertEqual(checkpoint["calibration"]["development_frames"], 4)
            self.assertEqual(checkpoint["final_validation"]["queries"], 2)
            self.assertEqual(
                checkpoint["selection_split"], "training_source_video_disjoint_development"
            )
            self.assertTrue((root / "out" / "final-validation.json").exists())
            np.testing.assert_allclose(
                checkpoint["run"]["box_prior_from_fitting_data"], [0.4, 0.4], atol=1e-6,
            )
            self.assertIn("epoch zero", checkpoint["run"]["selection"])


if __name__ == "__main__":
    unittest.main()
