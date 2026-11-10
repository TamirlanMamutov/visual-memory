import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
import torch
from torch.utils.data import DataLoader

from multimodal_localizer import MultimodalTemporalLocalizer, localization_loss
from train_multimodal_localizer import FeatureBundleDataset, _run_epoch


class MultimodalLocalizerTests(unittest.TestCase):
    def make_model(self):
        return MultimodalTemporalLocalizer(
            video_dim=12,
            image_query_dim=10,
            text_query_dim=8,
            hidden_dim=32,
            num_heads=4,
            num_temporal_layers=1,
            dropout=0,
        )

    def test_localizes_with_image_text_and_combined_queries(self):
        model = self.make_model()
        video = torch.randn(2, 3, 5, 12)
        cases = (
            (torch.randn(2, 4, 10), None),
            (None, torch.randn(2, 6, 8)),
            (torch.randn(2, 4, 10), torch.randn(2, 6, 8)),
        )
        for image, text in cases:
            with self.subTest(image=image is not None, text=text is not None):
                output = model(video, image, text)
                self.assertEqual(output.boxes_xyxy.shape, (2, 3, 4))
                self.assertEqual(output.presence_logits.shape, (2, 3))
                self.assertEqual(output.patch_logits.shape, (2, 3, 5))
                self.assertEqual(output.patch_boxes_xyxy.shape, (2, 3, 5, 4))
                self.assertTrue(torch.all(output.boxes_xyxy >= 0))
                self.assertTrue(torch.all(output.boxes_xyxy <= 1))

    def test_loss_backpropagates_through_temporal_and_box_heads(self):
        model = self.make_model()
        frame_mask = torch.tensor([[True, True, False], [True, True, True]])
        output = model(
            torch.randn(2, 3, 5, 12),
            image_query_features=torch.randn(2, 4, 10),
            text_query_features=torch.randn(2, 6, 8),
            frame_mask=frame_mask,
        )
        target_boxes = torch.tensor(
            [
                [[0.1, 0.2, 0.4, 0.6], [0.2, 0.2, 0.5, 0.6], [0, 0, 0, 0]],
                [[0, 0, 0, 0], [0.3, 0.1, 0.7, 0.8], [0.4, 0.2, 0.8, 0.9]],
            ]
        )
        target_present = torch.tensor(
            [[True, True, False], [False, True, True]]
        )
        losses = localization_loss(output, target_boxes, target_present, frame_mask)

        losses["loss"].backward()

        self.assertTrue(torch.isfinite(losses["loss"]))
        self.assertIsNotNone(model.patch_box.weight.grad)
        self.assertIsNotNone(model.presence_score.weight.grad)
        self.assertIsNotNone(model.query_attention.in_proj_weight.grad)

    def test_patch_averaging_preserves_normalized_bounds_at_saturation(self):
        torch.manual_seed(0)
        model = self.make_model()
        with torch.no_grad():
            model.patch_box.weight.zero_()
            model.patch_box.bias.fill_(100)
            output = model(
                torch.randn(2, 3, 5, 12),
                text_query_features=torch.randn(2, 6, 8),
            )
        self.assertTrue(torch.all(output.boxes_xyxy <= 1))
        self.assertTrue(torch.all(output.boxes_xyxy >= 0))

    def test_query_mask_can_disable_a_modality_for_a_sample(self):
        model = self.make_model()
        output = model(
            torch.randn(2, 2, 3, 12),
            image_query_features=torch.randn(2, 2, 10),
            text_query_features=torch.randn(2, 2, 8),
            image_query_mask=torch.tensor([[True, True], [False, False]]),
            text_query_mask=torch.tensor([[False, False], [True, True]]),
        )

        self.assertEqual(output.boxes_xyxy.shape, (2, 2, 4))

    def test_rejects_samples_without_any_valid_query_token(self):
        model = self.make_model()
        with self.assertRaisesRegex(ValueError, "at least one valid query token"):
            model(
                torch.randn(1, 2, 3, 12),
                image_query_features=torch.randn(1, 2, 10),
                image_query_mask=torch.zeros(1, 2, dtype=torch.bool),
            )

    def test_empty_object_frames_keep_presence_loss_trainable(self):
        model = self.make_model()
        output = model(
            torch.randn(1, 2, 3, 12), text_query_features=torch.randn(1, 2, 8)
        )
        losses = localization_loss(
            output,
            torch.zeros(1, 2, 4),
            torch.zeros(1, 2, dtype=torch.bool),
        )

        losses["loss"].backward()

        self.assertEqual(losses["box_l1_loss"].item(), 0)
        self.assertIsNotNone(model.presence_score.weight.grad)

    def test_feature_bundle_supports_a_train_and_validation_pass(self):
        with TemporaryDirectory() as temporary_directory:
            bundle = Path(temporary_directory)
            np.save(
                bundle / "video_features.npy",
                np.random.randn(2, 2, 3, 12).astype(np.float32),
            )
            np.save(
                bundle / "boxes_xyxy.npy",
                np.tile(
                    np.array([0.1, 0.2, 0.7, 0.8], dtype=np.float32),
                    (2, 2, 1),
                ),
            )
            np.save(
                bundle / "object_present.npy",
                np.ones((2, 2), dtype=np.bool_),
            )
            np.save(
                bundle / "text_query_features.npy",
                np.random.randn(2, 3, 8).astype(np.float32),
            )
            np.save(
                bundle / "image_query_features.npy",
                np.random.randn(2, 2, 10).astype(np.float32),
            )
            np.save(
                bundle / "text_query_mask.npy",
                np.array([[False, False, False], [True, True, True]]),
            )
            np.save(
                bundle / "image_query_mask.npy",
                np.array([[True, True], [False, False]]),
            )
            dataset = FeatureBundleDataset(bundle)
            loader = DataLoader(dataset, batch_size=2)
            model = self.make_model()
            optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)

            train_metrics = _run_epoch(
                model, loader, torch.device("cpu"), optimizer
            )
            validation_metrics = _run_epoch(model, loader, torch.device("cpu"))

            dataset.close()
            self.assertTrue(np.isfinite(train_metrics["loss"]))
            self.assertTrue(np.isfinite(validation_metrics["loss"]))
            self.assertIsNotNone(validation_metrics["mean_positive_iou"])

    def test_validation_loss_is_independent_of_batch_size(self):
        torch.manual_seed(42)
        model = self.make_model()
        samples = []
        for index in range(3):
            samples.append({
                "video_features": torch.randn(3, 5, 12),
                "text_query_features": torch.randn(2, 8),
                "boxes_xyxy": torch.tensor([[0.1, 0.2, 0.6, 0.7]] * 3),
                "object_present": torch.arange(3) <= index,
                "frame_mask": torch.arange(3) >= index,
            })
        individual = _run_epoch(
            model, DataLoader(samples, batch_size=1), torch.device("cpu")
        )
        batched = _run_epoch(
            model, DataLoader(samples, batch_size=3), torch.device("cpu")
        )
        for key in ("loss", "presence_f1", "mean_positive_iou"):
            self.assertAlmostEqual(individual[key], batched[key], places=6)
