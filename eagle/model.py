"""EAGLE (arXiv 2511.08007) 2D branch: frozen DINOv2 backbone, appearance-aware
meta-learning memory (AMM, Eq. 1-2), geometry-aware localization memory with a
discriminative correlation filter (GLM, Eq. 3-4), fusion F_N = F_A + kappa(H_J)
and a mask decoder.

Unspecified details are filled with the closest published designs the paper
builds on (LWL for the AMM learner, DiMP for the DCF); see REPRODUCTION.md."""
import math

import torch
import torch.nn as nn
import torch.nn.functional as F

IMNET_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
IMNET_STD = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)


def conv_bn_relu(i, o, k=3):
    return nn.Sequential(nn.Conv2d(i, o, k, padding=k // 2, bias=False), nn.BatchNorm2d(o), nn.ReLU(inplace=True))


class Backbone(nn.Module):
    """DINOv2 ViT (frozen). 448x448 input -> 32x32 patch features."""

    def __init__(self, name='facebook/dinov2-base'):
        super().__init__()
        from transformers import Dinov2Model
        self.vit = Dinov2Model.from_pretrained(name)
        self.vit.requires_grad_(False).eval()
        self.dim = self.vit.config.hidden_size
        self.patch = self.vit.config.patch_size

    def train(self, mode=True):
        super().train(mode); self.vit.eval(); return self

    @torch.no_grad()
    def forward(self, images):
        """images: uint8 [B, H, W, 3] or float [B, 3, H, W] in 0..1."""
        if images.dtype == torch.uint8:
            images = images.permute(0, 3, 1, 2).float() / 255
        x = (images - IMNET_MEAN.to(images.device)) / IMNET_STD.to(images.device)
        h, w = x.shape[2] // self.patch, x.shape[3] // self.patch
        tok = self.vit(pixel_values=x).last_hidden_state[:, 1:]          # drop CLS
        return tok.transpose(1, 2).reshape(x.shape[0], self.dim, h, w)


# ------------------------------------------------------------------ AMM -----

class LabelEncoder(nn.Module):
    """P_theta / W_theta: mask (full resolution) -> D-channel map at feature resolution."""

    def __init__(self, out, feat_stride=14, positive=False):
        super().__init__()
        self.net = nn.Sequential(conv_bn_relu(1, 16), nn.Conv2d(16, 32, 3, padding=1), nn.ReLU(inplace=True),
                                 nn.Conv2d(32, out, 3, padding=1))
        self.positive = positive

    def forward(self, mask, size):
        m = F.adaptive_avg_pool2d(mask, size)            # soft occupancy per feature cell
        y = self.net(m)
        return F.softplus(y) + 1e-3 if self.positive else y


class AppearanceMemory(nn.Module):
    """Few-shot learner G_theta: optimises the query segment kernel sigma over
    the memory O_AMM by steepest descent (Eq. 1-2), then A_sigma(F) = F * sigma."""

    def __init__(self, feat_dim, enc_dim=16, k=3):
        super().__init__()
        self.k, self.enc_dim, self.feat_dim = k, enc_dim, feat_dim
        self.P = LabelEncoder(enc_dim)
        self.W = LabelEncoder(enc_dim, positive=True)
        self.log_delta = nn.Parameter(torch.tensor(math.log(0.1)))      # learnable regulariser delta

    def init_kernel(self, feats):
        return feats.new_zeros(self.enc_dim, self.feat_dim, self.k, self.k)

    def apply(self, feats, sigma):
        return F.conv2d(feats, sigma, padding=self.k // 2)

    def optimise(self, sigma, feats, masks, iters, create_graph):
        """feats: [M, C, h, w] memory features, masks: [M, 1, H, W] memory masks.
        With create_graph the update is differentiable (meta-learning of P, W,
        delta and the feature reducer); otherwise it runs as a plain optimiser."""
        with torch.enable_grad():
            return self._optimise(sigma, feats, masks, iters, create_graph)

    def _optimise(self, sigma, feats, masks, iters, create_graph):
        if not create_graph:
            feats, masks = feats.detach(), masks.detach()
        delta = self.log_delta.exp()
        size = feats.shape[-2:]
        target, weight = self.P(masks, size), self.W(masks, size)
        m = feats.shape[0]
        for _ in range(iters):
            if not create_graph:
                sigma = sigma.detach().requires_grad_(True)
            elif not sigma.requires_grad:
                sigma = sigma.requires_grad_(True)
            resid = weight * (self.apply(feats, sigma) - target)
            loss = 0.5 * (resid ** 2).sum() / m + 0.5 * delta * (sigma ** 2).sum()
            g, = torch.autograd.grad(loss, sigma, create_graph=create_graph)
            num = (g ** 2).sum()
            den = ((weight * self.apply(feats, g)) ** 2).sum() / m + delta * num
            sigma = sigma - (num / den.clamp(min=1e-8)) * g
        return sigma


# ------------------------------------------------------------------ GLM -----

def gaussian_label(boxes, size, stride, sigma_factor=0.25):
    """boxes: [N, 4] input-pixel xyxy (nan for none) -> [N, 1, h, w] Gaussian G and
    target-region indicator S (1 inside box)."""
    n, (h, w) = boxes.shape[0], size
    ys = (torch.arange(h, device=boxes.device).float() + 0.5) * stride
    xs = (torch.arange(w, device=boxes.device).float() + 0.5) * stride
    g = boxes.new_zeros(n, 1, h, w); s = boxes.new_zeros(n, 1, h, w)
    for i, b in enumerate(boxes):
        if torch.isnan(b).any():
            continue
        cx, cy = (b[0] + b[2]) / 2, (b[1] + b[3]) / 2
        sd = sigma_factor * torch.clamp(torch.stack([b[2] - b[0], b[3] - b[1]]), min=stride)
        g[i, 0] = torch.exp(-0.5 * (((xs[None] - cx) / sd[0]) ** 2 + ((ys[:, None] - cy) / sd[1]) ** 2))
        s[i, 0] = ((xs[None] >= b[0]) & (xs[None] <= b[2]) & (ys[:, None] >= b[1]) & (ys[:, None] <= b[3])).float()
    return g, s


class GeometryMemory(nn.Module):
    """Model updater U: DCF kernel c optimised over O_GLM with the residual of
    Eq. 4 (least squares inside the target, hinge outside) using steepest descent
    with a Gauss-Newton step length (beta = g'g / g'J'Jg)."""

    def __init__(self, feat_dim, k=5):
        super().__init__()
        self.k = k
        self.log_lambda = nn.Parameter(torch.tensor(math.log(0.1)))

    def response(self, feats, c):
        return F.conv2d(feats, c, padding=self.k // 2)

    def optimise(self, c, feats, gauss, region, iters, weight=None):
        with torch.enable_grad():
            return self._optimise(c, feats.detach(), gauss, region, iters, weight)

    def _optimise(self, c, feats, gauss, region, iters, weight):
        lam2 = self.log_lambda.exp().detach() ** 2
        sw = torch.ones_like(gauss) if weight is None else weight
        if c is None:
            c = feats.new_zeros(1, feats.shape[1], self.k, self.k)
        m = feats.shape[0]
        for _ in range(iters):
            c = c.detach().requires_grad_(True)
            hj = self.response(feats, c)
            resid = sw * (region * hj + (1 - region) * F.relu(hj) - gauss)
            loss = (resid ** 2).sum() / m + lam2 * (c ** 2).sum()
            g, = torch.autograd.grad(loss, c)
            with torch.no_grad():
                active = region + (1 - region) * (hj > 0).float()
                jg = sw * active * self.response(feats, g)
                beta = (g ** 2).sum() / (2 * ((jg ** 2).sum() / m + lam2 * (g ** 2).sum())).clamp(min=1e-8)
            c = (c - beta * g).detach()
        return c


# --------------------------------------------------------------- decoder ----

class Decoder(nn.Module):
    """D: fused encoding + reduced backbone features (+ a shallow RGB stem for
    boundaries) -> mask logits at input resolution."""

    def __init__(self, enc_dim, feat_dim, width=128):
        super().__init__()
        self.inp = nn.Sequential(conv_bn_relu(enc_dim + feat_dim, width), conv_bn_relu(width, width))
        self.stem = nn.Sequential(conv_bn_relu(3, 32), nn.MaxPool2d(2), conv_bn_relu(32, 32), nn.MaxPool2d(2),
                                  conv_bn_relu(32, 32))                    # 1/4 resolution
        self.up1 = conv_bn_relu(width, 64)
        self.up2 = conv_bn_relu(64 + 32, 64)
        self.out = nn.Conv2d(64, 1, 3, padding=1)
        # Start from "mostly background" (VQ2D objects cover ~0.5 % of a frame) rather than logit 0,
        # the flat point where a Lovasz-only decoder got stuck.
        nn.init.constant_(self.out.bias, -4.0)

    def forward(self, enc, feats, image):
        x = self.inp(torch.cat([enc, feats], 1))
        x = F.interpolate(x, scale_factor=2, mode='bilinear', align_corners=False)
        x = self.up1(x)
        low = self.stem(image)
        x = F.interpolate(x, size=low.shape[-2:], mode='bilinear', align_corners=False)
        x = self.up2(torch.cat([x, low], 1))
        x = self.out(x)
        return F.interpolate(x, size=image.shape[-2:], mode='bilinear', align_corners=False)


class EAGLE(nn.Module):
    def __init__(self, backbone='facebook/dinov2-base', feat_dim=256, enc_dim=16, use_glm=True, use_amm=True):
        super().__init__()
        self.backbone = Backbone(backbone)
        self.stride = self.backbone.patch
        self.seg_reduce = conv_bn_relu(self.backbone.dim, feat_dim, 1)
        self.tck_reduce = conv_bn_relu(self.backbone.dim, feat_dim, 1)
        self.amm = AppearanceMemory(feat_dim, enc_dim)
        self.glm = GeometryMemory(feat_dim)
        self.kappa = conv_bn_relu(1, enc_dim)                      # conv-bn-relu on H_J
        self.decoder = Decoder(enc_dim, feat_dim)
        self.use_glm, self.use_amm = use_glm, use_amm

    def features(self, images_uint8):
        raw = self.backbone(images_uint8)
        image = images_uint8.permute(0, 3, 1, 2).float() / 255
        return dict(seg=self.seg_reduce(raw), tck=self.tck_reduce(raw), image=image)

    def segment(self, f, sigma, c):
        enc = self.amm.apply(f['seg'], sigma) if self.use_amm else f['seg'].new_zeros(
            f['seg'].shape[0], self.amm.enc_dim, *f['seg'].shape[-2:])
        hj = self.glm.response(f['tck'], c) if self.use_glm else enc[:, :1] * 0
        fused = enc + self.kappa(hj)                               # F_N = F_A + F_J
        return self.decoder(fused, f['seg'], f['image']), hj


def use_batch_norm_stats(model):
    """Normalise with each forward pass's own statistics at inference, as in training.
    The decoder trains on one frame at a time, and its BatchNorm running averages do not
    match those per-frame statistics: with them every logit on the val set was negative
    (no mask at all), while per-frame statistics gave box IoU 0.6-0.8 on the same frames.
    momentum 0 leaves the stored averages untouched."""
    for m in model.modules():
        if isinstance(m, nn.BatchNorm2d):
            m.train(); m.momentum = 0.0
    return model


def confidence(logits):
    """s_conf: mean foreground probability inside the predicted mask (0 if empty)."""
    p = logits.sigmoid().flatten(1)
    fg = p > 0.5
    return torch.where(fg.any(1), (p * fg).sum(1) / fg.sum(1).clamp(min=1), torch.zeros_like(p[:, 0]))


# ---------------------------------------------------------------- losses ----

def lovasz_hinge(logits, labels):
    """Binary Lovasz hinge (Berman et al. 2018), averaged over images."""
    losses = []
    for lg, lb in zip(logits.flatten(1), labels.flatten(1).float()):
        signs = 2 * lb - 1
        errors = 1 - lg * signs
        errors_sorted, perm = torch.sort(errors, descending=True)
        gt_sorted = lb[perm]
        gts = gt_sorted.sum()
        inter = gts - gt_sorted.cumsum(0)
        union = gts + (1 - gt_sorted).cumsum(0)
        jac = 1 - inter / union
        jac[1:] = jac[1:] - jac[:-1]
        losses.append(torch.dot(F.relu(errors_sorted), jac))
    return torch.stack(losses).mean()


def pixel_losses(logits, labels, pos_weight=10.0):
    """Per-pixel BCE (foreground up-weighted) + soft Dice, averaged over images.
    Dice is skipped on object-free frames, where BCE alone pushes the mask to empty."""
    lg, lb = logits.flatten(1), labels.flatten(1).float()
    bce = F.binary_cross_entropy_with_logits(lg, lb, pos_weight=lg.new_tensor(pos_weight))
    p = lg.sigmoid()
    has = lb.sum(1) > 0
    dice = 1 - (2 * (p * lb).sum(1) + 1) / (p.sum(1) + lb.sum(1) + 1)
    dice = dice[has].mean() if has.any() else lg.new_zeros(())
    return bce, dice


def dcf_hinge(response, gauss, region, threshold=0.05):
    """DiMP-style hinge: least squares on target cells, hinge on background."""
    neg = gauss < threshold
    pred = torch.where(neg, F.relu(response), response)
    return F.mse_loss(pred, torch.where(neg, torch.zeros_like(gauss), gauss))
