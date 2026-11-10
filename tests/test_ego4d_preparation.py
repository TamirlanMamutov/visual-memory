import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

import cv2
import numpy as np
import torch

from prepare_ego4d_localization import (
    extract_record,
    normalized_box,
    padded_box,
    sample_frame_ids,
)
from inspect_localization import original_box


class FakeEncoder(torch.nn.Module):
    def forward_features(self, images):
        return {"x_norm_patchtokens": images.mean(dim=(-2, -1))[:, None, :]}


class Ego4DPreparationTests(unittest.TestCase):
    def record(self):
        box = {
            "x": 20, "y": 10, "width": 40, "height": 20,
            "original_width": 100, "original_height": 50,
        }
        return {
            "dataset_uid": "train_1", "clip_uid": "clip1",
            "metadata": {"video_uid": "video1"}, "object_title": "cup",
            "clip_fps": 5, "query_frame": 8,
            "visual_crop": dict(box, frame_number=9),
            "response_track": [dict(box, frame_number=i) for i in (2, 3)],
        }

    def test_padding_and_annotation_scaling(self):
        box = normalized_box(self.record()["visual_crop"])
        np.testing.assert_allclose(box, [0.2, 0.2, 0.6, 0.6])
        np.testing.assert_allclose(padded_box(box, 100, 50), [0.2, 0.35, 0.6, 0.55])
        for width, height in ((100, 50), (51, 100), (100, 51)):
            np.testing.assert_allclose(
                original_box(padded_box(box, width, height), width, height), box,
                atol=1e-6,
            )

    def test_balanced_frames_include_positive_and_negative_without_query_leak(self):
        record = self.record()
        frames = sample_frame_ids(record, 6, "balanced")
        self.assertEqual(len(frames), len(set(frames)))
        self.assertIn(2, frames)
        self.assertIn(3, frames)
        self.assertTrue(all(0 <= frame < record["query_frame"] for frame in frames))

    def test_validation_sampling_does_not_depend_on_response_track(self):
        first = self.record()
        second = dict(first, response_track=[])
        self.assertEqual(
            sample_frame_ids(first, 4, "history"),
            sample_frame_ids(second, 4, "history"),
        )

    def test_decoder_maps_five_fps_annotations_to_ten_fps_media(self):
        with TemporaryDirectory() as temporary:
            path = Path(temporary) / "clip1.avi"
            writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"MJPG"), 10, (100, 50))
            self.assertTrue(writer.isOpened())
            for index in range(20):
                writer.write(np.full((50, 100, 3), index * 10, dtype=np.uint8))
            writer.release()
            arrays, metadata = extract_record(
                self.record(), path, FakeEncoder(), frames=6, sampling="balanced",
                image_size=28, batch_size=2, device=torch.device("cpu"),
            )
            self.assertEqual(arrays["video_features"].shape, (6, 1, 3))
            self.assertEqual(arrays["image_query_features"].shape, (1, 3))
            self.assertEqual(int(arrays["object_present"].sum()), 2)
            self.assertEqual(metadata["query_media_frame"], 18)
            self.assertEqual(
                metadata["media_frame_ids"], [2 * index for index in metadata["frame_ids"]]
            )
            self.assertTrue(np.all(arrays["frame_mask"]))


if __name__ == "__main__":
    unittest.main()
