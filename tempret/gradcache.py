"""Gradient caching (Gao et al., 2021, "Scaling Deep Contrastive Learning
Batch Size under Memory Limited Setup") for the SMS loss.

The SMS loss couples every clip in the batch with every caption, so the
batch cannot simply be split into gradient-accumulation steps: that would
shrink the set of in-batch negatives. Gradient caching gets the exact
large-batch gradient with chunk-sized activation memory:

1. Embed the whole batch chunk by chunk without autograd (only the [B, D]
   embeddings are kept).
2. Compute the loss on those embeddings and backpropagate to them only,
   which gives dL/dv and dL/dt for every clip and caption.
3. Re-run each chunk with autograd and backpropagate its cached embedding
   gradient into the encoders. Parameter gradients add up over the chunks.

Dropout (temporal transformer) uses the same random numbers in steps 1 and
3, so the result equals one full-batch backward pass up to floating-point
rounding.
"""
import contextlib

import torch

from tempret import data as D
from tempret.losses import sms_loss


class _RandState:
    """Captures the CPU / CUDA RNG so a chunk's dropout masks can be replayed."""

    def __init__(self, dev):
        self.dev = dev
        self.cpu = torch.get_rng_state()
        self.cuda = torch.cuda.get_rng_state(dev) if dev.type == 'cuda' else None

    @contextlib.contextmanager
    def replay(self):
        devices = [self.dev] if self.dev.type == 'cuda' else []
        with torch.random.fork_rng(devices=devices):
            torch.set_rng_state(self.cpu)
            if self.cuda is not None:
                torch.cuda.set_rng_state(self.cuda, self.dev)
            yield


def _embed(model, x, tok, amp):
    with amp():
        feats = x if x.dim() == 3 else model.encode_frames(x)
        v, _ = model.encode_video(feats)
        t, _ = model.encode_text(tok)
    return v, t


def _loss(v, t, verbs, nouns, args, amp):
    with amp():
        sim = v.float() @ t.float().T
        return sms_loss(sim, D.torch_relevance(verbs, nouns), args.margin, args.tau, args.eps)


def sms_step(model, x, tok, verbs, nouns, args, amp, chunk):
    """Backpropagates the full-batch SMS loss into model's .grad and returns
    the loss. chunk >= batch size falls back to one ordinary backward pass."""
    n = x.shape[0]
    if chunk <= 0 or chunk >= n:
        v, t = _embed(model, x, tok, amp)
        loss = _loss(v, t, verbs, nouns, args, amp)
        loss.backward()
        return loss.detach()

    dev = x.device
    spans = [(i, min(i + chunk, n)) for i in range(0, n, chunk)]
    states, vs, ts = [], [], []
    with torch.no_grad():
        for a, b in spans:
            states.append(_RandState(dev))
            v, t = _embed(model, x[a:b], tok[a:b], amp)
            vs.append(v.float()); ts.append(t.float())
    v_all = torch.cat(vs).requires_grad_(True)
    t_all = torch.cat(ts).requires_grad_(True)
    loss = _loss(v_all, t_all, verbs, nouns, args, amp)
    g_v, g_t = torch.autograd.grad(loss, [v_all, t_all])
    for (a, b), state in zip(spans, states):
        with state.replay():
            v, t = _embed(model, x[a:b], tok[a:b], amp)
        surrogate = (v.float() * g_v[a:b]).sum() + (t.float() * g_t[a:b]).sum()
        surrogate.backward()
    return loss.detach()
