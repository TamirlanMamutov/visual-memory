"""Two-stage reranking for TempRet (Eq. 8-11): embedding helpers, ITM scoring
with the row-minimum miss penalty, score fusion, and a dedicated training
stage for the cross-encoder.

Why a separate stage: trained jointly inside the dual-encoder loop the
cross-encoder only sees ~10 short epochs of in-batch pairs with binary
labels and stays near chance, and with the paper's alpha = 0.002 its scores
then override a good first-stage ranking. Here it is trained on top of the
frozen first stage with global hard negatives, soft EK-100 relevance targets
(optional), and the fusion weight alpha is chosen on held-out training videos.
"""
import math
import time

import numpy as np
import torch
import torch.nn.functional as F

from tempret import data as D
from tempret.metrics import evaluate, fmt

ALPHAS = [0.002, 0.01, 0.03, 0.1, 0.3, 1.0, 3.0, 10.0]


@torch.no_grad()
def embed(model, loader, dev, amp):
    model.eval()
    V, VT = [], []
    for x, _, _, _ in loader:
        x = x.to(dev, non_blocking=True)
        with amp():
            # raw frames (--mode full): encode 16 clips at a time to bound activation memory
            feats = x if x.dim() == 3 else torch.cat([model.encode_frames(x[i:i + 16]) for i in range(0, len(x), 16)])
            v, vt = model.encode_video(feats)
        V.append(v.float()); VT.append(vt.half())
    return torch.cat(V), torch.cat(VT)


@torch.no_grad()
def embed_text(model, tokens, dev, amp, bs=512):
    model.eval()
    E, H = [], []
    for i in range(0, len(tokens), bs):
        with amp():
            e, h = model.encode_text(tokens[i:i + bs].to(dev))
        E.append(e.float()); H.append(h.half())
    return torch.cat(E), torch.cat(H)


def itm_logit(cross, v, vm, t, tm):
    out = cross(v, vm, t, tm).float()
    return out[:, 1] - out[:, 0]                 # log-odds of a match; sigmoid == cross.score


@torch.no_grad()
def itm_matrix(cross, sim, q_tok, q_mask, c_tok, c_mask, k, amp, query_is_video, bs=2048):
    """ITM probabilities for each query's top-k candidates; candidates outside
    the top-k get the row minimum (miss penalty, Eq. 10)."""
    cross.eval()
    k = min(k, sim.shape[1])
    top = sim.topk(k, dim=1).indices
    itm = torch.empty_like(top, dtype=torch.float32)
    qi = torch.arange(sim.shape[0], device=sim.device)[:, None].expand(-1, k).reshape(-1)
    ci = top.reshape(-1); flat = itm.view(-1)
    for i in range(0, len(qi), bs):
        a, b = qi[i:i + bs], ci[i:i + bs]
        with amp():
            if query_is_video:
                s = cross.score(q_tok[a], q_mask[a], c_tok[b], c_mask[b])
            else:
                s = cross.score(c_tok[b], c_mask[b], q_tok[a], q_mask[a])
        flat[i:i + bs] = s.float()
    full = itm.min(1, keepdim=True).values.expand(-1, sim.shape[1]).clone()
    full.scatter_(1, top, itm)
    return full


def minmax(x):
    lo, hi = x.min(1, keepdim=True).values, x.max(1, keepdim=True).values
    return (x - lo) / (hi - lo).clamp(min=1e-8)


def fuse(itm, sim, alpha):
    """Eq. 11: S = norm(ITM) + alpha * norm(S_init)."""
    return minmax(itm) + alpha * minmax(sim)


def rerank_sweep(cross, v, vtok, t, ttok, tmask, rel, k, alphas, amp):
    """Returns ({'dual': m, alpha: m, ...}, seconds). ITM is computed once per
    direction and fused with every alpha."""
    start = time.time()
    sim = v @ t.T
    vmask = torch.ones(vtok.shape[:2], dtype=torch.bool, device=v.device)
    i_vt = itm_matrix(cross, sim, vtok, vmask, ttok, tmask, k, amp, True)
    i_tv = itm_matrix(cross, sim.T.contiguous(), ttok, tmask, vtok, vmask, k, amp, False)
    out = {'dual': evaluate(sim.cpu().numpy(), rel)}
    for a in alphas:
        out[a] = evaluate(fuse(i_vt, sim, a).cpu().numpy(), rel, fuse(i_tv, sim.T, a).cpu().numpy())
    return out, round(time.time() - start, 1)


def score_of(m):
    return m['mAP']['AVG'] + m['nDCG']['AVG']


# ------------------------------------------------------------------ training

def split_holdout(df, frac, seed):
    vids = np.array(sorted(df.video_id.unique()))
    rng = np.random.default_rng(seed)
    n = max(1, int(round(frac * len(vids)))) if frac > 0 else 0
    held = set(rng.choice(vids, n, replace=False)) if n else set()
    return df[~df.video_id.isin(held)], df[df.video_id.isin(held)], sorted(str(v) for v in held)


def train_reranker(model, cross, fit_df, held_df, make_loader, tokenize_fn, args, dev, amp, log):
    """Train `cross` on top of the frozen first stage. make_loader(df, train)
    returns a DataLoader of (features, tokens, verbs, nouns) in df order.
    Returns (best cross state, chosen alpha, validation history)."""
    model.eval()
    for p in model.parameters(): p.requires_grad_(False)
    fit_tok = tokenize_fn(fit_df.narration)
    t_emb, t_tok = embed_text(model, fit_tok, dev, amp)
    t_mask = (fit_tok > 0).to(dev)
    n = len(fit_df)
    rel = torch.from_numpy(D.relevance(fit_df.verb_class.values, D.multi_hot(fit_df.all_noun_classes),
                                       fit_df.verb_class.values, D.multi_hot(fit_df.all_noun_classes))).to(dev)

    # held-out validation set: held clips x their distinct narrations
    val = None
    if len(held_df):
        sents = held_df.drop_duplicates('narration')
        vrel = D.relevance(held_df.verb_class.values, D.multi_hot(held_df.all_noun_classes),
                           sents.verb_class.values, D.multi_hot(sents.all_noun_classes))
        stok = tokenize_fn(sents.narration)
        sv, svt = embed(model, make_loader(held_df, False), dev, amp)
        se, sh = embed_text(model, stok, dev, amp)
        val = (sv, svt, se, sh, (stok > 0).to(dev), vrel)
        log(f'Reranker validation: {len(held_df)} held-out clips x {len(sents)} sentences '
            f'({held_df.video_id.nunique()} videos not used to train the reranker)')

    params = [p for p in cross.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=args.reranker_lr, weight_decay=0.05)
    bs = min(args.reranker_batch, n)
    steps_per_epoch = max(1, n // bs)
    total = args.reranker_epochs * steps_per_epoch; warm = max(1, total // 20)
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: min(1, (s + 1) / warm) * 0.5 * (1 + math.cos(math.pi * min(1, s / total))))
    hard, nh = args.reranker_hard_pool, args.reranker_hard
    log(f'Training reranker: {args.reranker_epochs} epochs x {steps_per_epoch} steps, {bs} anchors/step, '
        f'{nh} hard negatives per side from the top {hard}, target {args.itm_target}')
    best = (-1, None, None, None); hist = []
    for epoch in range(1, args.reranker_epochs + 1):
        print(f'Starting reranker epoch {epoch}', flush=True)
        t0 = time.time()
        v_emb, v_tok = embed(model, make_loader(fit_df, True), dev, amp)    # fresh random frame slots
        v_mask = torch.ones(v_tok.shape[:2], dtype=torch.bool, device=dev)
        sim = v_emb @ t_emb.T
        top_t = sim.topk(min(hard, n), dim=1).indices                       # hard texts per video
        top_v = sim.T.topk(min(hard, n), dim=1).indices                     # hard videos per text
        cross.train(); run = {'loss': 0.0, 'acc': 0.0}
        perm = torch.randperm(n, device=dev)
        for s in range(steps_per_epoch):
            a = perm[s * bs:(s + 1) * bs]; m = len(a)
            pick = lambda pool: pool[a][torch.arange(m, device=dev)[:, None],
                                       torch.randint(0, pool.shape[1], (m, nh), device=dev)]
            ht, hv = pick(top_t), pick(top_v)
            rand_t = torch.randint(0, n, (m, 1), device=dev)
            vi = torch.cat([a[:, None], a[:, None].expand(-1, nh), a[:, None], hv], 1).reshape(-1)
            ti = torch.cat([a[:, None], ht, rand_t, a[:, None].expand(-1, nh)], 1).reshape(-1)
            target = rel[vi, ti]
            with amp():
                logit = itm_logit(cross, v_tok[vi], v_mask[vi], t_tok[ti], t_mask[ti])
            if args.itm_target == 'soft':
                loss = F.binary_cross_entropy_with_logits(logit, target.float())
            else:
                loss = F.binary_cross_entropy_with_logits(logit, (target >= 1).float())
            opt.zero_grad(set_to_none=True); loss.backward()
            torch.nn.utils.clip_grad_norm_(params, 1.0); opt.step(); sched.step()
            run['loss'] += loss.item()
            run['acc'] += ((logit > 0) == (target >= 0.5)).float().mean().item()
        rec = dict(epoch=epoch, loss=run['loss'] / steps_per_epoch, acc=run['acc'] / steps_per_epoch,
                   seconds=round(time.time() - t0, 1))
        line = f'  reranker epoch {epoch} | loss {rec["loss"]:.4f} acc {rec["acc"]:.3f} | {rec["seconds"]}s'
        if val is not None and (epoch % args.reranker_eval_every == 0 or epoch == args.reranker_epochs):
            sv, svt, se, sh, smask, vrel = val
            res, _ = rerank_sweep(cross, sv, svt, se, sh, smask, vrel, args.rerank_k, ALPHAS, amp)
            alpha = max(ALPHAS, key=lambda x: score_of(res[x]))
            rec.update(val_dual=score_of(res['dual']) / 2, val_best=score_of(res[alpha]) / 2, val_alpha=alpha,
                       val_paper_alpha=score_of(res[0.002]) / 2)
            line += (f' | held-out (mAP+nDCG)/2: dual {rec["val_dual"]:.2f}, rerank a={alpha} {rec["val_best"]:.2f}, '
                     f'a=0.002 {rec["val_paper_alpha"]:.2f}')
            if score_of(res[alpha]) > best[0]:
                best = (score_of(res[alpha]), {k: v.detach().clone() for k, v in cross.state_dict().items()},
                        alpha, epoch)
        log(line); hist.append(rec)
    if best[1] is None:   # no validation split: keep the final weights, paper alpha
        return cross.state_dict(), 0.002, hist, args.reranker_epochs
    log(f'Reranker: best held-out epoch {best[3]}, alpha {best[2]}')
    return best[1], best[2], hist, best[3]
