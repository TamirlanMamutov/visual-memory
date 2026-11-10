"""Feature-based temporal object localization for image and text queries."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional

import torch
from torch import Tensor, nn
from torch.nn import functional as F


@dataclass
class LocalizationOutput:
    boxes_xyxy: Tensor
    presence_logits: Tensor
    patch_logits: Tensor
    patch_boxes_xyxy: Tensor


class MultimodalTemporalLocalizer(nn.Module):
    """Localize a queried object in video feature sequences.

    Inputs are frozen-backbone features, not RGB images or raw text. Video
    features have shape [batch, time, patches, video_dim]. Image and text
    query features have shape [batch, tokens, modality_dim].
    """

    def __init__(
        self,
        video_dim: int,
        image_query_dim: Optional[int] = None,
        text_query_dim: Optional[int] = None,
        hidden_dim: int = 256,
        num_heads: int = 8,
        num_temporal_layers: int = 2,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        if video_dim <= 0 or hidden_dim <= 0:
            raise ValueError("video_dim and hidden_dim must be positive")
        if hidden_dim % num_heads or hidden_dim % 2:
            raise ValueError("hidden_dim must be even and divisible by num_heads")
        if num_temporal_layers <= 0:
            raise ValueError("num_temporal_layers must be positive")
        if image_query_dim is None and text_query_dim is None:
            raise ValueError("At least one query modality must be configured")

        self.video_dim = video_dim
        self.image_query_dim = image_query_dim
        self.text_query_dim = text_query_dim
        self.hidden_dim = hidden_dim
        self.video_projection = nn.Linear(video_dim, hidden_dim)
        self.image_projection = (
            nn.Linear(image_query_dim, hidden_dim) if image_query_dim is not None else None
        )
        self.text_projection = (
            nn.Linear(text_query_dim, hidden_dim) if text_query_dim is not None else None
        )
        self.image_modality = nn.Parameter(torch.zeros(1, 1, hidden_dim))
        self.text_modality = nn.Parameter(torch.zeros(1, 1, hidden_dim))
        self.query_attention = nn.MultiheadAttention(
            hidden_dim, num_heads, dropout=dropout, batch_first=True
        )
        self.cross_attention_norm = nn.LayerNorm(hidden_dim)
        self.temporal_encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=hidden_dim,
                nhead=num_heads,
                dim_feedforward=hidden_dim * 4,
                dropout=dropout,
                activation="gelu",
                batch_first=True,
                norm_first=False,
            ),
            num_layers=num_temporal_layers,
            enable_nested_tensor=False,
        )
        self.temporal_norm = nn.LayerNorm(hidden_dim)
        self.patch_score = nn.Linear(hidden_dim, 1)
        self.patch_box = nn.Linear(hidden_dim, 4)
        self.presence_score = nn.Linear(hidden_dim, 1)

    def forward(
        self,
        video_features: Tensor,
        image_query_features: Optional[Tensor] = None,
        text_query_features: Optional[Tensor] = None,
        image_query_mask: Optional[Tensor] = None,
        text_query_mask: Optional[Tensor] = None,
        frame_mask: Optional[Tensor] = None,
    ) -> LocalizationOutput:
        self._validate_features(
            video_features, image_query_features, text_query_features
        )
        batch, frames, patches, _ = video_features.shape
        if frame_mask is None:
            frame_mask = torch.ones(
                batch, frames, dtype=torch.bool, device=video_features.device
            )
        elif frame_mask.shape != (batch, frames) or frame_mask.dtype != torch.bool:
            raise ValueError("frame_mask must be boolean with shape [batch, time]")
        elif frame_mask.device != video_features.device:
            raise ValueError("frame_mask and video_features must be on the same device")
        if not frame_mask.any(dim=1).all():
            raise ValueError("Each sample must contain at least one valid frame")

        query_tokens = []
        query_masks = []
        if image_query_features is not None:
            image_query_mask = self._query_mask(
                image_query_mask, image_query_features, "image_query_mask"
            )
            query_tokens.append(
                self.image_projection(image_query_features) + self.image_modality
            )
            query_masks.append(image_query_mask)
        elif image_query_mask is not None:
            raise ValueError("image_query_mask was provided without image query features")

        if text_query_features is not None:
            text_query_mask = self._query_mask(
                text_query_mask, text_query_features, "text_query_mask"
            )
            query_tokens.append(
                self.text_projection(text_query_features) + self.text_modality
            )
            query_masks.append(text_query_mask)
        elif text_query_mask is not None:
            raise ValueError("text_query_mask was provided without text query features")

        query = torch.cat(query_tokens, dim=1)
        query_mask = torch.cat(query_masks, dim=1)
        if not query_mask.any(dim=1).all():
            raise ValueError("Each sample must have at least one valid query token")

        video = self.video_projection(video_features)
        video_flat = video.reshape(batch * frames, patches, self.hidden_dim)
        query_flat = query[:, None].expand(-1, frames, -1, -1).reshape(
            batch * frames, query.shape[1], self.hidden_dim
        )
        query_padding_mask = (~query_mask[:, None, :].expand(-1, frames, -1)).reshape(
            batch * frames, query.shape[1]
        )
        attended, _ = self.query_attention(
            video_flat,
            query_flat,
            query_flat,
            key_padding_mask=query_padding_mask,
            need_weights=False,
        )
        video = self.cross_attention_norm(video_flat + attended).reshape(
            batch, frames, patches, self.hidden_dim
        )

        frame_tokens = video.mean(dim=2)
        frame_tokens = self.temporal_norm(
            frame_tokens + self._temporal_encoding(frames, video)
        )
        frame_context = self.temporal_encoder(
            frame_tokens, src_key_padding_mask=~frame_mask
        )
        patch_features = video + frame_context.unsqueeze(2)

        patch_logits = self.patch_score(patch_features).squeeze(-1)
        patch_boxes = self._decode_boxes(self.patch_box(patch_features))
        patch_weights = patch_logits.softmax(dim=-1).unsqueeze(-1)
        boxes = (patch_weights * patch_boxes).sum(dim=2).clamp(0, 1)
        presence_logits = self.presence_score(frame_context).squeeze(-1)
        return LocalizationOutput(
            boxes_xyxy=boxes,
            presence_logits=presence_logits,
            patch_logits=patch_logits,
            patch_boxes_xyxy=patch_boxes,
        )

    def _validate_features(
        self,
        video_features: Tensor,
        image_query_features: Optional[Tensor],
        text_query_features: Optional[Tensor],
    ) -> None:
        if video_features.ndim != 4 or video_features.shape[-1] != self.video_dim:
            raise ValueError(
                f"video_features must have shape [B,T,P,{self.video_dim}]"
            )
        if min(video_features.shape[:3]) <= 0:
            raise ValueError("video_features must have non-empty batch, time, and patches")
        for name, features in (
            ("video_features", video_features),
            ("image_query_features", image_query_features),
            ("text_query_features", text_query_features),
        ):
            if features is not None:
                if not features.is_floating_point():
                    raise TypeError(f"{name} must be a floating-point tensor")
                if not torch.isfinite(features).all():
                    raise ValueError(f"{name} must contain only finite values")
        batch = video_features.shape[0]
        for name, features, dim in (
            ("image_query_features", image_query_features, self.image_query_dim),
            ("text_query_features", text_query_features, self.text_query_dim),
        ):
            if features is None:
                continue
            if dim is None:
                raise ValueError(f"{name} is not configured for this model")
            if features.ndim != 3 or features.shape[0] != batch or features.shape[-1] != dim:
                raise ValueError(f"{name} must have shape [B,Q,{dim}]")
            if features.shape[1] == 0:
                raise ValueError(f"{name} must contain at least one token")
        if image_query_features is None and text_query_features is None:
            raise ValueError("At least one query modality must be provided")

    @staticmethod
    def _query_mask(mask: Optional[Tensor], features: Tensor, name: str) -> Tensor:
        expected = features.shape[:2]
        if mask is None:
            return torch.ones(expected, dtype=torch.bool, device=features.device)
        if mask.shape != expected or mask.dtype != torch.bool:
            raise ValueError(f"{name} must be boolean with shape {tuple(expected)}")
        if mask.device != features.device:
            raise ValueError(f"{name} and query features must be on the same device")
        return mask

    def _temporal_encoding(self, frames: int, reference: Tensor) -> Tensor:
        positions = torch.arange(
            frames, device=reference.device, dtype=reference.dtype
        ).unsqueeze(1)
        frequencies = torch.exp(
            torch.arange(
                0, self.hidden_dim, 2, device=reference.device, dtype=reference.dtype
            )
            * (-math.log(10000.0) / self.hidden_dim)
        )
        encoding = reference.new_zeros(1, frames, self.hidden_dim)
        encoding[0, :, 0::2] = torch.sin(positions * frequencies)
        encoding[0, :, 1::2] = torch.cos(positions * frequencies)
        return encoding

    @staticmethod
    def _decode_boxes(raw_boxes: Tensor) -> Tensor:
        center_x, center_y, width, height = raw_boxes.sigmoid().unbind(dim=-1)
        x1 = (center_x - width / 2).clamp(0, 1)
        y1 = (center_y - height / 2).clamp(0, 1)
        x2 = (center_x + width / 2).clamp(0, 1)
        y2 = (center_y + height / 2).clamp(0, 1)
        return torch.stack((x1, y1, x2, y2), dim=-1)


def localization_loss(
    output: LocalizationOutput,
    target_boxes_xyxy: Tensor,
    target_present: Tensor,
    frame_mask: Optional[Tensor] = None,
    presence_weight: float = 1.0,
    box_l1_weight: float = 1.0,
    giou_weight: float = 1.0,
) -> dict[str, Tensor]:
    """Compute temporal presence and positive-frame box regression losses."""
    if target_boxes_xyxy.shape != output.boxes_xyxy.shape:
        raise ValueError("target_boxes_xyxy must have shape [B,T,4]")
    if target_present.shape != output.presence_logits.shape:
        raise ValueError("target_present must have shape [B,T]")
    if frame_mask is None:
        frame_mask = torch.ones_like(target_present, dtype=torch.bool)
    if frame_mask.shape != target_present.shape or frame_mask.dtype != torch.bool:
        raise ValueError("frame_mask must be boolean with shape [B,T]")
    if target_present.dtype != torch.bool:
        raise ValueError("target_present must be boolean")
    if not torch.isfinite(target_boxes_xyxy).all():
        raise ValueError("target_boxes_xyxy must contain only finite coordinates")
    if ((target_boxes_xyxy < 0) | (target_boxes_xyxy > 1)).any():
        raise ValueError("target boxes must be normalized to [0,1]")
    positive = frame_mask & target_present
    if (positive & (
        (target_boxes_xyxy[..., 2] <= target_boxes_xyxy[..., 0])
        | (target_boxes_xyxy[..., 3] <= target_boxes_xyxy[..., 1])
    )).any():
        raise ValueError("Present-object targets must have positive box area")

    valid_count = frame_mask.sum().clamp_min(1)
    presence_loss = (
        F.binary_cross_entropy_with_logits(
            output.presence_logits, target_present.to(output.presence_logits.dtype), reduction="none"
        )
        * frame_mask
    ).sum() / valid_count
    if positive.any():
        predicted = output.boxes_xyxy[positive]
        target = target_boxes_xyxy[positive]
        box_l1 = F.l1_loss(predicted, target)
        giou = _generalized_iou(predicted, target)
        giou_loss = (1 - giou).mean()
    else:
        box_l1 = output.boxes_xyxy.sum() * 0
        giou_loss = output.boxes_xyxy.sum() * 0
    total = (
        presence_weight * presence_loss
        + box_l1_weight * box_l1
        + giou_weight * giou_loss
    )
    return {
        "loss": total,
        "presence_loss": presence_loss,
        "box_l1_loss": box_l1,
        "giou_loss": giou_loss,
    }


def _generalized_iou(boxes1: Tensor, boxes2: Tensor) -> Tensor:
    intersection_min = torch.maximum(boxes1[:, :2], boxes2[:, :2])
    intersection_max = torch.minimum(boxes1[:, 2:], boxes2[:, 2:])
    intersection_size = (intersection_max - intersection_min).clamp_min(0)
    intersection = intersection_size[:, 0] * intersection_size[:, 1]
    area1_size = (boxes1[:, 2:] - boxes1[:, :2]).clamp_min(0)
    area2_size = (boxes2[:, 2:] - boxes2[:, :2]).clamp_min(0)
    area1 = area1_size[:, 0] * area1_size[:, 1]
    area2 = area2_size[:, 0] * area2_size[:, 1]
    union = area1 + area2 - intersection
    iou = intersection / union.clamp_min(torch.finfo(union.dtype).eps)
    enclosing_min = torch.minimum(boxes1[:, :2], boxes2[:, :2])
    enclosing_max = torch.maximum(boxes1[:, 2:], boxes2[:, 2:])
    enclosing_size = (enclosing_max - enclosing_min).clamp_min(0)
    enclosing_area = enclosing_size[:, 0] * enclosing_size[:, 1]
    return iou - (enclosing_area - union) / enclosing_area.clamp_min(
        torch.finfo(enclosing_area.dtype).eps
    )
