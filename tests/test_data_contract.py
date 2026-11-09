import unittest

import torch

from data_contract import (
    ModelSample,
    adapt_vqloc_sample,
    collate_model_samples,
    make_retrieval_sample,
    make_video_only_sample,
    validate_sample,
)


class DataContractTests(unittest.TestCase):
    def make_vqloc_output(self, frames: int) -> dict[str, torch.Tensor]:
        boxes = torch.tensor(
            [[0.1, 0.2, 0.6, 0.8]] * frames,
            dtype=torch.float32,
        )
        return {
            "clip": torch.zeros((frames, 3, 8, 8), dtype=torch.float32),
            "clip_bbox": boxes,
            "clip_with_bbox": torch.ones(frames, dtype=torch.float32),
            "before_query": torch.arange(frames) < 1,
            "query": torch.ones((3, 8, 8), dtype=torch.float32),
        }

    def test_vqloc_adapter_normalizes_boxes_and_mask_types(self) -> None:
        sample = adapt_vqloc_sample("clip-1", self.make_vqloc_output(2))

        self.assertEqual(sample.task, "localization")
        self.assertEqual(sample.inputs["video_frames"].dtype, torch.float32)
        self.assertEqual(sample.inputs["query_image"].dtype, torch.float32)
        self.assertTrue(torch.equal(
            sample.targets["boxes_xyxy"][0],
            torch.tensor([0.2, 0.1, 0.8, 0.6]),
        ))
        self.assertEqual(sample.masks["object_present"].dtype, torch.bool)
        self.assertTrue(sample.masks["object_present"].all())

    def test_retrieval_features_are_float32_and_batch(self) -> None:
        samples = [
            make_retrieval_sample(
                f"pair-{index}",
                torch.ones(4, dtype=torch.float64),
                torch.zeros(3, dtype=torch.float64),
                relevance=1,
            )
            for index in range(2)
        ]

        batch = collate_model_samples(samples)

        self.assertEqual(batch.task, "retrieval")
        self.assertEqual(batch.inputs["video_features"].dtype, torch.float32)
        self.assertEqual(batch.inputs["query_features"].shape, (2, 3))
        self.assertEqual(batch.targets["relevance"].dtype, torch.float32)

    def test_collation_pads_localization_sequences_and_masks(self) -> None:
        first = adapt_vqloc_sample("short", self.make_vqloc_output(2))
        second = adapt_vqloc_sample("long", self.make_vqloc_output(3))

        batch = collate_model_samples([first, second])

        self.assertEqual(batch.inputs["video_frames"].shape, (2, 3, 3, 8, 8))
        self.assertEqual(
            batch.targets["boxes_xyxy"].shape,
            (2, 3, 4),
        )
        self.assertTrue(torch.equal(
            batch.masks["frame_valid"],
            torch.tensor([[True, True, False], [True, True, True]]),
        ))
        self.assertFalse(batch.masks["object_present"][0, 2])

    def test_contract_rejects_noncanonical_boxes(self) -> None:
        sample = adapt_vqloc_sample("clip-1", self.make_vqloc_output(1))
        invalid = ModelSample(
            sample_id=sample.sample_id,
            dataset=sample.dataset,
            task=sample.task,
            inputs=sample.inputs,
            targets={"boxes_xyxy": torch.tensor([[0.8, 0.1, 0.2, 0.6]])},
            masks=sample.masks,
        )

        with self.assertRaisesRegex(ValueError, "ordered"):
            validate_sample(invalid)

    def test_video_only_sample_is_valid_for_unlabelled_clips(self) -> None:
        sample = make_video_only_sample(
            "take-1",
            torch.zeros((2, 3, 8, 8), dtype=torch.uint8),
            dataset="egoexo4d",
        )

        self.assertEqual(sample.task, "video_only")
        self.assertEqual(sample.inputs["video_frames"].dtype, torch.float32)
        self.assertEqual(sample.inputs["video_frames"].max().item(), 0.0)

    def test_collation_pads_feature_sequences_and_text_tokens(self) -> None:
        samples = [
            ModelSample(
                sample_id=f"query-{index}",
                dataset="epic_kitchens_jpose",
                task="retrieval",
                inputs={
                    "video_features": torch.ones((index + 1, 4), dtype=torch.float32),
                    "query_token_ids": torch.arange(index + 2, dtype=torch.int64),
                },
            )
            for index in range(2)
        ]

        batch = collate_model_samples(samples)

        self.assertEqual(batch.inputs["video_features"].shape, (2, 2, 4))
        self.assertTrue(torch.equal(
            batch.masks["video_feature_valid"],
            torch.tensor([[True, False], [True, True]]),
        ))
        self.assertEqual(batch.inputs["query_token_ids"].shape, (2, 3))
        self.assertTrue(torch.equal(
            batch.masks["query_attention_mask"],
            torch.tensor([[True, True, False], [True, True, True]]),
        ))

    def test_batch_does_not_mix_task_types(self) -> None:
        retrieval = make_retrieval_sample("retrieval", torch.ones(4), torch.ones(3))
        video = make_video_only_sample(
            "video",
            torch.zeros((1, 3, 8, 8)),
            dataset="egoexo4d",
        )

        with self.assertRaisesRegex(ValueError, "cannot mix"):
            collate_model_samples([retrieval, video])


if __name__ == "__main__":
    unittest.main()
