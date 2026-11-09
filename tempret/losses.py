"""Symmetric Multi-Similarity loss (Wang et al. 2024; TempRet Eq. 7) and ITM loss."""
import torch
import torch.nn.functional as F


def sms_loss(sim, rel, margin=0.2, tau=0.1, eps=0.1):
    """sim: [N, N] cosine similarity (row i = video i, column j = text j);
    rel: [N, N] soft relevance. Pairs (i, j != i) are weighted by
    w_ij = R_ii - R_ij: clearly-less-relevant pairs must trail the positive by
    w*m, near-equally relevant pairs must stay within tau of w*m, and
    more-relevant ones must not trail it."""
    def one_side(s, r):
        n = s.shape[0]
        pos = s.diag()[:, None]
        delta = pos - s                              # s_ii - s_ij
        w = r.diag()[:, None] - r
        target = w * margin - delta
        loss = torch.where(w > eps, F.relu(target),
               torch.where(w < -eps, F.relu(-target), F.relu(target.abs() - tau)))
        off = ~torch.eye(n, dtype=torch.bool, device=s.device)
        return loss[off].mean()
    return one_side(sim, rel) + one_side(sim.T, rel.T)


def itm_batch(sim, rel, generator=None):
    """Hard-negative pairs for ITM training. For each video choose a text
    (and for each text a video) sampled by softmax(sim) among items that are
    not fully relevant. Returns index tensors (video_idx, text_idx, label)."""
    n = sim.shape[0]
    dev = sim.device
    idx = torch.arange(n, device=dev)
    not_pos = rel < 1
    def sample(s, allowed):
        w = (s.float() * 20).softmax(-1) * allowed
        ok = allowed.any(1)
        w[~ok] = 1                        # degenerate row: uniform (label computed below)
        return torch.multinomial(w, 1, generator=generator).squeeze(1)
    neg_t = sample(sim.detach(), not_pos)
    neg_v = sample(sim.detach().T, not_pos.T)
    v = torch.cat([idx, idx, neg_v]); t = torch.cat([idx, neg_t, idx])
    label = (rel[v, t] >= 1).long()
    return v, t, label
