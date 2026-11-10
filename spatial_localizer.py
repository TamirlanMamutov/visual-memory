"""Spatially anchored image-query localization with a shared EPIC noun head."""

from __future__ import annotations

import math

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from multimodal_localizer import LocalizationOutput, localization_loss


class SpatialLocalizer(nn.Module):
    def __init__(self, feature_dim: int = 384, noun_classes: int = 300) -> None:
        super().__init__()
        if feature_dim <= 0 or noun_classes <= 0:
            raise ValueError("Feature and noun class dimensions must be positive")
        self.feature_dim = feature_dim
        self.projection = nn.Linear(feature_dim, feature_dim, bias=False)
        nn.init.zeros_(self.projection.weight)
        self.noun_head = nn.Linear(feature_dim, noun_classes)
        self.box_head = nn.Sequential(
            nn.Linear(feature_dim * 2, 64), nn.GELU(), nn.Linear(64, 4)
        )
        nn.init.zeros_(self.box_head[-1].weight)
        nn.init.zeros_(self.box_head[-1].bias)
        self.log_temperature = nn.Parameter(torch.tensor(math.log(20.0)))
        self.presence_bias = nn.Parameter(torch.tensor(-10.0))

    def embed(self, features: Tensor) -> Tensor:
        return F.normalize(features + self.projection(features), dim=-1)

    def classify_epic(self, features: Tensor) -> Tensor:
        return self.noun_head(self.embed(features))

    @staticmethod
    def patch_centers(patches: int, reference: Tensor) -> Tensor:
        side = math.isqrt(patches)
        if side * side != patches:
            raise ValueError("Spatial features must use a square patch grid")
        y, x = torch.meshgrid(
            torch.arange(side, device=reference.device, dtype=reference.dtype),
            torch.arange(side, device=reference.device, dtype=reference.dtype),
            indexing="ij",
        )
        return torch.stack(((x.flatten() + 0.5) / side, (y.flatten() + 0.5) / side), -1)

    def forward(
        self,
        video_features: Tensor,
        image_query_features: Tensor,
        frame_mask: Tensor | None = None,
    ) -> LocalizationOutput:
        if video_features.ndim != 4 or video_features.shape[-1] != self.feature_dim:
            raise ValueError("video_features must have shape [B,T,P,feature_dim]")
        if image_query_features.ndim != 3 or image_query_features.shape[-1] != self.feature_dim:
            raise ValueError("image_query_features must have shape [B,Q,feature_dim]")
        if image_query_features.shape[0] != video_features.shape[0]:
            raise ValueError("Video and query batch dimensions must match")
        if min(video_features.shape) <= 0 or min(image_query_features.shape) <= 0:
            raise ValueError("Feature dimensions must be nonempty")
        for features in (video_features, image_query_features):
            if not features.is_floating_point() or not torch.isfinite(features).all():
                raise ValueError("Features must be finite floating-point tensors")
        if frame_mask is not None and (
            frame_mask.dtype != torch.bool or frame_mask.shape != video_features.shape[:2]
            or not frame_mask.any(-1).all()
        ):
            raise ValueError("frame_mask must be boolean [B,T] with valid frames per sample")
        centers = self.patch_centers(video_features.shape[2], video_features)
        query_side = math.isqrt(image_query_features.shape[1])
        if query_side * query_side != image_query_features.shape[1] or query_side < 2:
            raise ValueError("Image query features must use a square grid of at least 2x2")
        # Center pooling limits the influence of query-crop square padding.
        border = query_side // 4
        query_grid = image_query_features.reshape(-1, query_side, query_side, self.feature_dim)
        query = query_grid[:, border:query_side - border, border:query_side - border].mean((1, 2))
        query = self.embed(query)
        video = self.embed(video_features)
        scale = self.log_temperature.exp().clamp(1, 100)
        similarity = (video * query[:, None, None]).sum(-1)
        logits = scale * similarity
        query_expanded = query[:, None, None].expand_as(video)
        raw_boxes = self.box_head(torch.cat((video, query_expanded), -1))
        side = math.isqrt(video.shape[2])
        xy = centers + raw_boxes[..., :2].tanh() / side
        wh = (raw_boxes[..., 2:].clamp(-3, 3).exp() * 0.15).clamp(max=1)
        patch_boxes = torch.cat((xy - wh / 2, xy + wh / 2), -1).clamp(0, 1)
        winner = logits.argmax(-1)
        boxes = patch_boxes.gather(
            2, winner[..., None, None].expand(-1, -1, 1, 4)
        ).squeeze(2)
        return LocalizationOutput(
            boxes_xyxy=boxes,
            presence_logits=logits.max(-1).values + self.presence_bias,
            patch_logits=logits,
            patch_boxes_xyxy=patch_boxes,
        )


def spatial_loss(
    output: LocalizationOutput, boxes: Tensor, present: Tensor, frame_mask: Tensor
) -> dict[str, Tensor]:
    losses = localization_loss(output, boxes, present, frame_mask)
    positive = present & frame_mask
    centers = SpatialLocalizer.patch_centers(output.patch_logits.shape[-1], boxes)
    if positive.any():
        positive_boxes = boxes[positive]
        target_center = (positive_boxes[:, :2] + positive_boxes[:, 2:]) / 2
        distance = (target_center[:, None] - centers[None]).square().sum(-1)
        # A soft spatial target supplies gradients even for initially wrong peaks.
        target_distribution = (-distance * output.patch_logits.shape[-1] / 2).softmax(-1)
        attention_loss = -(
            target_distribution * output.patch_logits[positive].log_softmax(-1)
        ).sum(-1).mean()
    else:
        attention_loss = output.patch_logits.sum() * 0
    losses["attention_loss"] = attention_loss
    losses["loss"] = losses["loss"] + attention_loss
    return losses
