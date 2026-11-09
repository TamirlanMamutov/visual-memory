"""TempRet model: CLIP dual encoder + video-side temporal transformer, and a
cross-encoder ITM reranker (arXiv 2605.24470, Sec. 2)."""
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms.v2.functional as TF

CLIP_MEAN = (0.48145466, 0.4578275, 0.40821073)
CLIP_STD = (0.26862954, 0.26130258, 0.27577711)


def preprocess_tensor(rgb, train=False, size=224):
    """uint8 HxWx3 (224 short side) -> normalised CHW float tensor."""
    x = torch.from_numpy(rgb).permute(2, 0, 1)
    h, w = x.shape[1:]
    if train:   # random crop along the long side + light colour jitter
        top = torch.randint(0, h - size + 1, ()).item(); left = torch.randint(0, w - size + 1, ()).item()
    else:
        top, left = (h - size) // 2, (w - size) // 2
    x = x[:, top:top + size, left:left + size].float() / 255
    return TF.normalize(x, CLIP_MEAN, CLIP_STD)


class TemporalTransformer(nn.Module):
    """Eq. 2-5: learnable positional encodings, L pre-LN blocks (MHA + 4x GELU
    MLP), a residual to the CLIP frame features and masked mean pooling."""

    def __init__(self, dim, layers=4, heads=8, max_frames=32):
        super().__init__()
        self.pos = nn.Parameter(torch.zeros(1, max_frames, dim))
        nn.init.normal_(self.pos, std=0.01)
        layer = nn.TransformerEncoderLayer(dim, heads, 4 * dim, dropout=0.1, activation='gelu',
                                           batch_first=True, norm_first=True)
        self.blocks = nn.TransformerEncoder(layer, layers, enable_nested_tensor=False)

    def forward(self, x, mask=None):
        """x: [B, T, D] frame embeddings; mask: [B, T] bool (True = valid)."""
        if mask is None:
            mask = torch.ones(x.shape[:2], dtype=torch.bool, device=x.device)
        h = self.blocks(x + self.pos[:, :x.shape[1]], src_key_padding_mask=~mask)
        return h + x                       # Eq. 4: X~ = Transformer(X) + X


def masked_mean(x, mask):
    m = mask.float().unsqueeze(-1)
    return (x * m).sum(1) / m.sum(1).clamp(min=1)


class TempRet(nn.Module):
    def __init__(self, clip_model, layers=4, heads=8, use_temporal=True):
        super().__init__()
        self.clip = clip_model
        dim = clip_model.text_projection.shape[1]
        self.dim = dim
        self.temporal = TemporalTransformer(dim, layers, heads) if use_temporal else None

    # ---- video
    def encode_frames(self, frames):
        """frames: [B, T, 3, H, W] -> per-frame CLIP embeddings [B, T, D]."""
        b, t = frames.shape[:2]
        return self.clip.encode_image(frames.flatten(0, 1)).view(b, t, -1)

    def video_tokens(self, frame_feats, mask=None):
        x = frame_feats.float()
        return self.temporal(x, mask) if self.temporal is not None else x

    def encode_video(self, frame_feats, mask=None):
        if mask is None:
            mask = torch.ones(frame_feats.shape[:2], dtype=torch.bool, device=frame_feats.device)
        tokens = self.video_tokens(frame_feats, mask)
        return F.normalize(masked_mean(tokens, mask), dim=-1), tokens

    # ---- text
    def encode_text(self, text):
        """Returns (normalised EOS embedding [B, D], token states [B, L, W])."""
        c = self.clip
        cast = c.transformer.get_cast_dtype() if hasattr(c.transformer, 'get_cast_dtype') else torch.float32
        x = c.token_embedding(text).to(cast)
        x = x + c.positional_embedding.to(cast)[:x.shape[1]]
        attn = c.attn_mask[:x.shape[1], :x.shape[1]] if c.attn_mask is not None else None
        if getattr(c.transformer, 'batch_first', True):
            x = c.transformer(x, attn_mask=attn)
        else:
            x = c.transformer(x.permute(1, 0, 2), attn_mask=attn).permute(1, 0, 2)
        x = c.ln_final(x)
        eos = x[torch.arange(x.shape[0], device=x.device), text.argmax(dim=-1)]
        pooled = eos @ c.text_projection
        return F.normalize(pooled.float(), dim=-1), x


class CrossEncoder(nn.Module):
    """Joint transformer over [CLS; video tokens; text tokens] with a binary
    ITM head (Eq. 8-9). The paper does not specify its architecture; this is
    a 4-layer, 512-wide encoder on top of TempRet's (detached) token states."""

    def __init__(self, video_dim, text_dim, width=512, layers=4, heads=8, max_frames=32, max_tokens=77):
        super().__init__()
        self.v_proj = nn.Linear(video_dim, width)
        self.t_proj = nn.Linear(text_dim, width)
        self.cls = nn.Parameter(torch.zeros(1, 1, width))
        self.v_pos = nn.Parameter(torch.zeros(1, max_frames, width))
        self.t_pos = nn.Parameter(torch.zeros(1, max_tokens, width))
        self.type_emb = nn.Parameter(torch.zeros(2, width))
        for p in (self.cls, self.v_pos, self.t_pos, self.type_emb):
            nn.init.normal_(p, std=0.02)
        layer = nn.TransformerEncoderLayer(width, heads, 4 * width, dropout=0.1, activation='gelu',
                                           batch_first=True, norm_first=True)
        self.encoder = nn.TransformerEncoder(layer, layers, enable_nested_tensor=False)
        self.norm = nn.LayerNorm(width)
        self.itm = nn.Linear(width, 2)

    def forward(self, v, v_mask, t, t_mask):
        b = v.shape[0]
        v = self.v_proj(v.float()) + self.v_pos[:, :v.shape[1]] + self.type_emb[0]
        t = self.t_proj(t.float()) + self.t_pos[:, :t.shape[1]] + self.type_emb[1]
        x = torch.cat([self.cls.expand(b, -1, -1), v, t], 1)
        mask = torch.cat([torch.ones(b, 1, dtype=torch.bool, device=v.device), v_mask, t_mask], 1)
        x = self.encoder(x, src_key_padding_mask=~mask)
        return self.itm(self.norm(x[:, 0]))

    def score(self, *args):
        return self(*args).float().softmax(-1)[:, 1]     # P(match)
