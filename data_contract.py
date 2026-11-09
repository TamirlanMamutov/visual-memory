"""Shared, validated sample types for retrieval and visual localization."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal, Mapping, TypeAlias

import torch


Task: TypeAlias = Literal["retrieval", "localization", "video_only"]
TensorMap: TypeAlias = Mapping[str, torch.Tensor]
MetadataValue: TypeAlias = str | int | float | bool | None

_INPUT_KEYS = {
    "video_frames",
    "video_features",
    "query_image",
    "query_features",
    "query_token_ids",
}
_TARGET_KEYS = {"boxes_xyxy", "relevance"}
_MASK_KEYS = {
    "object_present",
    "frame_valid",
    "video_feature_valid",
    "before_query",
    "query_attention_mask",
}
_FLOAT_INPUT_KEYS = {
    "video_frames",
    "video_features",
    "query_image",
    "query_features",
}
_FLOAT_TARGET_KEYS = {"boxes_xyxy", "relevance"}
_TEMPORAL_MASK_KEYS = {"object_present", "frame_valid", "before_query"}


@dataclass(frozen=True)
class ModelSample:
    """One source-independent example at a model boundary.

    Retrieval and localization samples share this container, but retain distinct
    ``task`` values and task-specific fields.
    """

    sample_id: str
    dataset: str
    task: Task
    inputs: TensorMap
    targets: TensorMap = field(default_factory=dict)
    masks: TensorMap = field(default_factory=dict)
    metadata: Mapping[str, MetadataValue] = field(default_factory=dict)


@dataclass(frozen=True)
class ModelBatch:
    """A homogeneous batch produced by :func:`collate_model_samples`."""

    task: Task
    inputs: Mapping[str, torch.Tensor]
    targets: Mapping[str, torch.Tensor]
    masks: Mapping[str, torch.Tensor]
    sample_ids: tuple[str, ...]
    datasets: tuple[str, ...]
    metadata: tuple[Mapping[str, MetadataValue], ...]


def _require_tensor(name: str, value: torch.Tensor) -> None:
    if not isinstance(value, torch.Tensor):
        raise TypeError(f"{name} must be a torch.Tensor, got {type(value).__name__}.")


def _require_finite(name: str, value: torch.Tensor) -> None:
    if not torch.isfinite(value).all().item():
        raise ValueError(f"{name} contains non-finite values.")


def _require_float32(name: str, value: torch.Tensor) -> None:
    _require_tensor(name, value)
    if value.dtype != torch.float32:
        raise TypeError(f"{name} must use torch.float32, got {value.dtype}.")
    _require_finite(name, value)


def _require_image(name: str, value: torch.Tensor, *, sequence: bool = False) -> None:
    _require_float32(name, value)
    expected_rank = 4 if sequence else 3
    if value.ndim != expected_rank:
        shape = "[T, 3, H, W]" if sequence else "[3, H, W]"
        raise ValueError(f"{name} must have shape {shape}, got {tuple(value.shape)}.")
    if value.shape[-3] != 3 or any(size <= 0 for size in value.shape[-2:]):
        raise ValueError(f"{name} must contain RGB images with nonempty dimensions.")
    if value.numel() and (value.min().item() < 0 or value.max().item() > 1):
        raise ValueError(f"{name} pixel values must be in [0, 1].")
    if sequence and value.shape[0] == 0:
        raise ValueError(f"{name} must contain at least one frame.")


def validate_sample(sample: ModelSample, *, require_targets: bool = False) -> None:
    """Raise a precise error when a sample violates the shared tensor contract."""
    if not isinstance(sample, ModelSample):
        raise TypeError(f"sample must be ModelSample, got {type(sample).__name__}.")
    if not isinstance(sample.sample_id, str):
        raise TypeError("sample_id must be a string.")
    if not sample.sample_id.strip():
        raise ValueError("sample_id must not be empty.")
    if not isinstance(sample.dataset, str):
        raise TypeError("dataset must be a string.")
    if not sample.dataset.strip():
        raise ValueError("dataset must not be empty.")
    if sample.task not in {"retrieval", "localization", "video_only"}:
        raise ValueError(f"Unsupported task: {sample.task!r}.")

    for label, mapping, allowed in (
        ("input", sample.inputs, _INPUT_KEYS),
        ("target", sample.targets, _TARGET_KEYS),
        ("mask", sample.masks, _MASK_KEYS),
    ):
        if not isinstance(mapping, Mapping):
            raise TypeError(f"{label} fields must be a mapping.")
        unknown = set(mapping) - allowed
        if unknown:
            raise ValueError(f"Unknown {label} field(s): {sorted(unknown)}.")
        if any(not isinstance(key, str) for key in mapping):
            raise TypeError(f"{label} field names must be strings.")

    video_keys = {"video_frames", "video_features"} & sample.inputs.keys()
    query_keys = {
        "query_image",
        "query_features",
        "query_token_ids",
    } & sample.inputs.keys()
    if not video_keys:
        raise ValueError("A sample needs video_frames or video_features.")
    if sample.task in {"retrieval", "localization"} and not query_keys:
        raise ValueError(f"{sample.task} samples need at least one query input.")
    if sample.task == "video_only" and (query_keys or sample.targets):
        raise ValueError("video_only samples cannot contain query inputs or targets.")

    for name in sample.inputs.keys() & _FLOAT_INPUT_KEYS:
        value = sample.inputs[name]
        _require_float32(name, value)
        if name == "video_frames":
            _require_image(name, value, sequence=True)
        elif name == "query_image":
            _require_image(name, value)
        elif name in {"video_features", "query_features"}:
            if value.ndim not in {1, 2} or any(size <= 0 for size in value.shape):
                raise ValueError(
                    f"{name} must have shape [D] or [N, D], got {tuple(value.shape)}."
                )

    if "query_token_ids" in sample.inputs:
        tokens = sample.inputs["query_token_ids"]
        _require_tensor("query_token_ids", tokens)
        if tokens.dtype != torch.int64:
            raise TypeError(f"query_token_ids must use torch.int64, got {tokens.dtype}.")
        if tokens.ndim != 1 or tokens.numel() == 0:
            raise ValueError("query_token_ids must be a nonempty [L] tensor.")
        if tokens.min().item() < 0:
            raise ValueError("query_token_ids cannot contain negative values.")

    if "boxes_xyxy" in sample.targets:
        boxes = sample.targets["boxes_xyxy"]
        _require_float32("boxes_xyxy", boxes)
        if boxes.ndim != 2 or boxes.shape[1] != 4:
            raise ValueError(f"boxes_xyxy must have shape [T, 4], got {tuple(boxes.shape)}.")
        if boxes.numel() and (boxes.min().item() < 0 or boxes.max().item() > 1):
            raise ValueError("boxes_xyxy coordinates must be normalized to [0, 1].")
        if boxes.numel() and (
            (boxes[:, 2] < boxes[:, 0]).any().item()
            or (boxes[:, 3] < boxes[:, 1]).any().item()
        ):
            raise ValueError("boxes_xyxy must use ordered [x1, y1, x2, y2] coordinates.")
        if "object_present" not in sample.masks:
            raise ValueError("boxes_xyxy requires an object_present mask.")

    for name in sample.targets.keys() & _FLOAT_TARGET_KEYS:
        _require_float32(name, sample.targets[name])

    for name, mask in sample.masks.items():
        _require_tensor(name, mask)
        if mask.dtype != torch.bool:
            raise TypeError(f"{name} must use torch.bool, got {mask.dtype}.")
        if name == "video_feature_valid":
            features = sample.inputs.get("video_features")
            if features is None or features.ndim != 2 or mask.shape != features.shape[:1]:
                raise ValueError(
                    "video_feature_valid must match the sequence dimension of "
                    "two-dimensional video_features."
                )
        if name == "query_attention_mask":
            tokens = sample.inputs.get("query_token_ids")
            if tokens is None or mask.shape != tokens.shape:
                raise ValueError(
                    "query_attention_mask must match query_token_ids; tokens are required."
                )
        elif name in _TEMPORAL_MASK_KEYS and (
            mask.ndim != 1 or mask.numel() == 0
        ):
            raise ValueError(f"{name} must be a nonempty [T] boolean mask.")

    sequence_lengths = []
    frames = sample.inputs.get("video_frames")
    if frames is not None:
        sequence_lengths.append(("video_frames", frames.shape[0]))
    boxes = sample.targets.get("boxes_xyxy")
    if boxes is not None:
        sequence_lengths.append(("boxes_xyxy", boxes.shape[0]))
    for name in _TEMPORAL_MASK_KEYS & sample.masks.keys():
        sequence_lengths.append((name, sample.masks[name].shape[0]))
    if sequence_lengths:
        expected_length = sequence_lengths[0][1]
        mismatched = [
            f"{name}={length}"
            for name, length in sequence_lengths
            if length != expected_length
        ]
        if mismatched:
            raise ValueError(
                f"Temporal fields must align with video_frames length "
                f"{expected_length}; mismatched: {', '.join(mismatched)}."
            )

    if require_targets and sample.task == "localization":
        if "boxes_xyxy" not in sample.targets or "object_present" not in sample.masks:
            raise ValueError("Supervised localization needs boxes_xyxy and object_present.")

    if not isinstance(sample.metadata, Mapping):
        raise TypeError("metadata must be a mapping.")
    for key, value in sample.metadata.items():
        if not isinstance(key, str):
            raise TypeError("Metadata keys must be strings.")
        if not isinstance(value, (str, int, float, bool, type(None))):
            raise TypeError(f"Metadata value {key!r} must be a scalar.")


def _as_float32(value: object, name: str) -> torch.Tensor:
    tensor = torch.as_tensor(value)
    if tensor.is_complex():
        raise TypeError(f"{name} cannot be complex.")
    tensor = tensor.to(dtype=torch.float32)
    _require_finite(name, tensor)
    return tensor


def _as_image_float32(value: object, name: str) -> torch.Tensor:
    tensor = torch.as_tensor(value)
    if tensor.dtype == torch.uint8:
        tensor = tensor.to(dtype=torch.float32).div(255)
    elif tensor.is_floating_point():
        tensor = tensor.to(dtype=torch.float32)
    else:
        raise TypeError(f"{name} must be uint8 pixels or floating-point values in [0, 1].")
    _require_image(name, tensor, sequence=(name == "video_frames"))
    return tensor


def _as_bool_mask(value: object, name: str) -> torch.Tensor:
    tensor = torch.as_tensor(value)
    if tensor.dtype == torch.bool:
        return tensor
    if tensor.is_floating_point() or tensor.dtype in {
        torch.uint8,
        torch.int8,
        torch.int16,
        torch.int32,
        torch.int64,
    }:
        if not torch.logical_or(tensor == 0, tensor == 1).all().item():
            raise ValueError(f"{name} can only contain 0/1 values.")
        return tensor.to(dtype=torch.bool)
    raise TypeError(f"{name} must be boolean or contain only 0/1 values.")


def make_retrieval_sample(
    sample_id: str,
    video_features: object,
    query_features: object,
    *,
    dataset: str = "epic_kitchens_jpose",
    relevance: object | None = None,
    metadata: Mapping[str, MetadataValue] | None = None,
) -> ModelSample:
    """Adapt one video/query feature pair from JPoSE/MMEN to the shared type."""
    targets = {}
    if relevance is not None:
        targets["relevance"] = _as_float32(relevance, "relevance")
    sample = ModelSample(
        sample_id=sample_id,
        dataset=dataset,
        task="retrieval",
        inputs={
            "video_features": _as_float32(video_features, "video_features"),
            "query_features": _as_float32(query_features, "query_features"),
        },
        targets=targets,
        metadata=metadata or {},
    )
    validate_sample(sample)
    return sample


def make_video_only_sample(
    sample_id: str,
    video_frames: object,
    *,
    dataset: str,
    metadata: Mapping[str, MetadataValue] | None = None,
) -> ModelSample:
    """Adapt decoded but unlabelled video frames for inference or review."""
    sample = ModelSample(
        sample_id=sample_id,
        dataset=dataset,
        task="video_only",
        inputs={"video_frames": _as_image_float32(video_frames, "video_frames")},
        metadata=metadata or {},
    )
    validate_sample(sample)
    return sample


def adapt_vqloc_sample(
    sample_id: str,
    output: Mapping[str, object],
    *,
    dataset: str = "ego4d_vq2d_vqloc",
) -> ModelSample:
    """Adapt a VQLoC ``QueryVideoDataset`` item without changing upstream code.

    VQLoC stores boxes as normalized ``[y1, x1, y2, x2]``; the common contract
    uses normalized ``[x1, y1, x2, y2]``.
    """
    required = {"clip", "clip_bbox", "clip_with_bbox", "query"}
    missing = required - output.keys()
    if missing:
        raise ValueError(f"VQLoC sample is missing fields: {sorted(missing)}.")

    clip = _as_image_float32(output["clip"], "video_frames")
    query = _as_image_float32(output["query"], "query_image")
    vqloc_boxes = _as_float32(output["clip_bbox"], "clip_bbox")
    if vqloc_boxes.ndim != 2 or vqloc_boxes.shape[1] != 4:
        raise ValueError(
            f"clip_bbox must have shape [T, 4], got {tuple(vqloc_boxes.shape)}."
        )
    boxes_xyxy = vqloc_boxes[:, [1, 0, 3, 2]].contiguous()
    masks = {
        "object_present": _as_bool_mask(
            output["clip_with_bbox"], "object_present"
        ),
    }
    if "before_query" in output:
        masks["before_query"] = _as_bool_mask(output["before_query"], "before_query")

    sample_metadata: dict[str, MetadataValue] = {"adapter": "VQLoC"}
    for source_key, target_key in (("clip_h", "original_clip_height"),
                                   ("clip_w", "original_clip_width")):
        if source_key in output:
            value = torch.as_tensor(output[source_key])
            if value.numel() != 1:
                raise ValueError(f"{source_key} must be a scalar.")
            sample_metadata[target_key] = value.item()

    adapted = ModelSample(
        sample_id=sample_id,
        dataset=dataset,
        task="localization",
        inputs={"video_frames": clip, "query_image": query},
        targets={"boxes_xyxy": boxes_xyxy},
        masks=masks,
        metadata=sample_metadata,
    )
    validate_sample(adapted, require_targets=True)
    return adapted


def _pad_first_dimension(
    values: list[torch.Tensor], *, pad_value: int | float | bool = 0
) -> tuple[torch.Tensor, torch.Tensor]:
    trailing_shape = values[0].shape[1:]
    if any(value.shape[1:] != trailing_shape for value in values):
        raise ValueError("Cannot pad tensors with different trailing dimensions.")
    max_length = max(value.shape[0] for value in values)
    result = values[0].new_full((len(values), max_length, *trailing_shape), pad_value)
    valid = torch.zeros(
        (len(values), max_length), dtype=torch.bool, device=values[0].device
    )
    for index, value in enumerate(values):
        length = value.shape[0]
        result[index, :length] = value
        valid[index, :length] = True
    return result, valid


def _stack(values: list[torch.Tensor], name: str) -> torch.Tensor:
    try:
        return torch.stack(values)
    except RuntimeError as exc:
        raise ValueError(
            f"{name} shapes must match within a batch; resize/tokenize them "
            "during dataset preprocessing."
        ) from exc


def collate_model_samples(samples: list[ModelSample]) -> ModelBatch:
    """Batch same-task samples, padding temporal sequences and token IDs."""
    if not samples:
        raise ValueError("Cannot collate an empty sample list.")
    for sample in samples:
        validate_sample(sample)
    task = samples[0].task
    if any(sample.task != task for sample in samples):
        raise ValueError("A batch cannot mix retrieval, localization, and video-only tasks.")

    for label, get_mapping in (
        ("inputs", lambda sample: sample.inputs),
        ("targets", lambda sample: sample.targets),
        ("masks", lambda sample: sample.masks),
    ):
        key_sets = [set(get_mapping(sample)) for sample in samples]
        if any(keys != key_sets[0] for keys in key_sets[1:]):
            raise ValueError(f"All samples in a batch must have the same {label} fields.")

    input_keys = set(samples[0].inputs)
    target_keys = set(samples[0].targets)
    mask_keys = set(samples[0].masks)
    batched_inputs: dict[str, torch.Tensor] = {}
    batched_targets: dict[str, torch.Tensor] = {}
    batched_masks: dict[str, torch.Tensor] = {}

    if "video_frames" in input_keys:
        frames, frame_valid = _pad_first_dimension(
            [sample.inputs["video_frames"] for sample in samples]
        )
        batched_inputs["video_frames"] = frames
        batched_masks["frame_valid"] = frame_valid

    for key in input_keys - {"video_frames"}:
        values = [sample.inputs[key] for sample in samples]
        if key == "query_token_ids":
            tokens, token_valid = _pad_first_dimension(values)
            batched_inputs[key] = tokens
            batched_masks["query_attention_mask"] = token_valid
        elif key == "video_features" and values[0].ndim == 2:
            features, feature_valid = _pad_first_dimension(values)
            batched_inputs[key] = features
            batched_masks["video_feature_valid"] = feature_valid
        else:
            batched_inputs[key] = _stack(values, key)

    for key in target_keys:
        values = [sample.targets[key] for sample in samples]
        if key == "boxes_xyxy":
            batched_targets[key], _ = _pad_first_dimension(values)
        else:
            batched_targets[key] = _stack(values, key)

    for key in mask_keys:
        values = [sample.masks[key] for sample in samples]
        if key in _TEMPORAL_MASK_KEYS or key == "video_feature_valid":
            batched_masks[key], _ = _pad_first_dimension(values)
        elif key != "query_attention_mask":
            batched_masks[key] = _stack(values, key)

    return ModelBatch(
        task=task,
        inputs=batched_inputs,
        targets=batched_targets,
        masks=batched_masks,
        sample_ids=tuple(sample.sample_id for sample in samples),
        datasets=tuple(sample.dataset for sample in samples),
        metadata=tuple(sample.metadata for sample in samples),
    )
