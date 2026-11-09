"""Train / evaluate the EAGLE re-implementation (2D VQ localization) on Ego4D VQ2D.

Stages (run in order, each resumable):
  prepare  - decode query/response-track/negative frames from local VQ2D clips
             and generate SAM pseudo-masks (box + centre-point prompt)
  train    - meta-train AMM + GLM + decoder on VQ2D (and VISOR when present)
  evaluate - online inference on VQ2D val clips; tAP25 / stAP25 / Rec / Succ
Use --stage all for prepare -> train -> evaluate.
"""
import argparse
import json
import math
import pickle
import random
import time

import numpy as np
import torch
import torch.nn.functional as F

from common import CACHE, Logger, device, make_run_dir, seed_everything, write_json
from eagle import data as D
from eagle.infer import UPDATE_CONF, run_inference, vq_metrics
from eagle.model import EAGLE, confidence, dcf_hinge, gaussian_label, lovasz_hinge


def parse():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--stage', choices=['prepare', 'train', 'evaluate', 'all'], default='all')
    p.add_argument('--backbone', default='facebook/dinov2-base', help='paper: DINOv2 ViT-B/14 at 448')
    p.add_argument('--sam', default='facebook/sam-vit-base', help='SAM checkpoint for pseudo-masks')
    p.add_argument('--iters', type=int, default=25000, help='paper: 25k iterations')
    p.add_argument('--warmup', type=int, default=2500)
    p.add_argument('--lr', type=float, default=2.5e-3); p.add_argument('--weight-decay', type=float, default=0.05)
    p.add_argument('--batch-size', type=int, default=4, help='sequences per step (not given in the paper)')
    p.add_argument('--seq-len', type=int, default=4, help='frames per training sequence incl. the query (N)')
    p.add_argument('--rho', type=float, default=1.0, help='weight of L_tck')
    p.add_argument('--init-iters', type=int, default=10); p.add_argument('--update-iters', type=int, default=3)
    p.add_argument('--no-amm', action='store_true'); p.add_argument('--no-glm', action='store_true')
    p.add_argument('--no-visor', action='store_true')
    p.add_argument('--max-train-queries', type=int, default=0); p.add_argument('--max-eval-queries', type=int, default=0)
    p.add_argument('--eval-split', default='val')
    p.add_argument('--checkpoint', help='checkpoint to evaluate (default: the one trained in this run)')
    p.add_argument('--log-every', type=int, default=50); p.add_argument('--save-every', type=int, default=2500)
    p.add_argument('--seed', type=int, default=230)
    return p.parse_args()


# ------------------------------------------------------------- sampling -----

class Sampler:
    def __init__(self, vq_keys, visor_seqs, seq_len, rng):
        self.vq, self.visor, self.n, self.rng = vq_keys, visor_seqs, seq_len, rng
        self.visor_frames = D.VisorFrames() if visor_seqs else None

    def sample(self):
        if self.visor and (not self.vq or self.rng.random() < 0.5):
            return self.sample_visor()
        return self.sample_vq()

    def sample_vq(self):
        with D.vq_cache_path(self.rng.choice(self.vq)).open('rb') as f:
            items = pickle.load(f)['items']
        query = [i for i in items if i['role'] == 'query']
        track = [i for i in items if i['role'] == 'track']
        neg = [i for i in items if i['role'] == 'neg']
        if not query or not track:
            return self.sample_vq()
        k = self.n - 1
        n_neg = 1 if neg and k > 1 else 0
        tests = self.rng.sample(track, min(len(track), k - n_neg)) + self.rng.sample(neg, n_neg)
        tests.sort(key=lambda i: i['frame'])
        out = []
        for it in query[:1] + tests:
            img, mask = D.decode_item(it)
            out.append((img, mask))
        return out

    def sample_visor(self):
        import cv2
        video, obj, frames = self.rng.choice(self.visor)
        pick = sorted(self.rng.sample(range(len(frames)), min(self.n, len(frames))))
        out = []
        for j in pick:
            name, polys = frames[j]
            img = self.visor_frames.image(video, name)
            mask = D.polygons_to_mask(polys, img.shape)
            img, s = D.letterbox(img)
            m = cv2.resize(mask, (round(mask.shape[1] * s), round(mask.shape[0] * s)), interpolation=cv2.INTER_NEAREST)
            mm = np.zeros(img.shape[:2], np.uint8); mm[:m.shape[0], :m.shape[1]] = m
            out.append((img, mm))
        return out


def boxes_of(masks_np):
    return [D.mask_to_box(m) or [float('nan')] * 4 for m in masks_np]


# --------------------------------------------------------------- training ---

def train_step(model, batch, args, dev):
    """batch: list of sequences [(img, mask), ...]; first frame = query/reference."""
    loss_seg, loss_tck, n_frames, hits = 0.0, 0.0, 0, 0
    flip = random.random() < 0.5
    for seq in batch:
        imgs = np.stack([s[0] for s in seq]); masks = np.stack([s[1] for s in seq])
        if flip:
            imgs, masks = imgs[:, :, ::-1].copy(), masks[:, :, ::-1].copy()
        boxes = torch.tensor(boxes_of(masks), dtype=torch.float32, device=dev)
        with torch.autocast(dev.type, dtype=torch.bfloat16, enabled=dev.type == 'cuda'):
            f = model.features(torch.from_numpy(imgs).to(dev))
        f = {k: v.float() for k, v in f.items()}
        mt = torch.from_numpy(masks).float()[:, None].to(dev)
        size = f['seg'].shape[-2:]
        gauss, region = gaussian_label(boxes, size, model.stride)
        sigma = model.amm.optimise(model.amm.init_kernel(f['seg'][:1]), f['seg'][:1], mt[:1], args.init_iters, True)
        c = model.glm.optimise(None, f['tck'][:1], gauss[:1], region[:1], iters=args.init_iters)
        mem_f, mem_m = [f['seg'][:1]], [mt[:1]]
        for t in range(1, len(seq)):
            ft = {k: v[t:t + 1] for k, v in f.items()}
            logits, hj = model.segment(ft, sigma, c)
            loss_seg = loss_seg + lovasz_hinge(logits, mt[t:t + 1])
            loss_tck = loss_tck + dcf_hinge(hj, gauss[t:t + 1], region[t:t + 1])
            n_frames += 1
            pred = (logits.detach() > 0).float()
            hits += int(((pred * mt[t:t + 1]).sum() / ((pred + mt[t:t + 1]).clamp(max=1).sum()).clamp(min=1)).item() > 0.5) \
                if mt[t].sum() > 0 else int(pred.sum() == 0)
            if t < len(seq) - 1 and confidence(logits.detach())[0] >= UPDATE_CONF:
                mem_f.append(ft['seg']); mem_m.append(logits.detach().sigmoid())
                sigma = model.amm.optimise(sigma, torch.cat(mem_f), torch.cat(mem_m), args.update_iters, True)
    loss_seg, loss_tck = loss_seg / n_frames, loss_tck / n_frames
    return loss_seg, loss_tck, hits / n_frames


def train(model, args, out, dev, log):
    # Train on every query already prepared (the local clip set can grow between stages).
    vq_keys = sorted(r['key'] for r in D.load_vq('train', local_only=False) if D.vq_cache_path(r['key']).exists())
    if args.max_train_queries: vq_keys = vq_keys[:args.max_train_queries]
    visor = [] if args.no_visor or not D.visor_available() else D.load_visor_sequences('train')
    log(f'Training data: {len(vq_keys)} VQ2D queries with cached frames/masks, {len(visor)} VISOR object sequences'
        + ('' if D.visor_available() else ' (VISOR not found at ../EPIC-KITCHENS-VISOR)'))
    if not vq_keys and not visor:
        raise SystemExit('No training data. Download VQ2D train clips (and/or VISOR), then run --stage prepare.')
    params = [p for n, p in model.named_parameters() if p.requires_grad and not n.startswith('backbone.')]
    log(f'Trainable parameters: {sum(p.numel() for p in params) / 1e6:.2f} M (backbone frozen)')
    opt = torch.optim.AdamW(params, lr=args.lr, weight_decay=args.weight_decay)
    warm = min(args.warmup, max(1, args.iters // 10))
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: min((s + 1) / warm, max(0.0, (args.iters - s) / max(1, args.iters - warm))))
    sampler = Sampler(vq_keys, visor, args.seq_len, random.Random(args.seed))
    ck_dir = CACHE / 'eagle' / 'checkpoints' / out.name; ck_dir.mkdir(parents=True, exist_ok=True)
    log(f'Training {args.iters} iterations, batch {args.batch_size} x {args.seq_len} frames; checkpoints in {ck_dir}')
    model.train()
    t0, acc = time.time(), dict(seg=0.0, tck=0.0, hit=0.0, n=0)
    for it in range(1, args.iters + 1):
        batch = [sampler.sample() for _ in range(args.batch_size)]
        seg, tck, hit = train_step(model, batch, args, dev)
        loss = seg + args.rho * tck
        opt.zero_grad(set_to_none=True); loss.backward()
        torch.nn.utils.clip_grad_norm_(params, 1.0)
        opt.step(); sched.step()
        acc['seg'] += seg.item(); acc['tck'] += tck.item(); acc['hit'] += hit; acc['n'] += 1
        if it % args.log_every == 0 or it == args.iters:
            n = acc['n']
            row = dict(iter=it, seg=acc['seg'] / n, tck=acc['tck'] / n, frame_iou50=acc['hit'] / n,
                       lr=sched.get_last_lr()[0], seconds=round(time.time() - t0, 1))
            log(f"  iter {it}/{args.iters} | L_seg {row['seg']:.4f} | L_tck {row['tck']:.5f} | "
                f"frames IoU>0.5 {row['frame_iou50']:.3f} | lr {row['lr']:.2e} | {row['seconds']:.0f}s")
            with (out / 'history.jsonl').open('a') as f: f.write(json.dumps(row) + '\n')
            acc = dict(seg=0.0, tck=0.0, hit=0.0, n=0)
        if it % args.save_every == 0 or it == args.iters:
            state = {k: v for k, v in model.state_dict().items() if not k.startswith('backbone.')}
            torch.save(dict(model=state, args=vars(args), iter=it), ck_dir / 'last.pt')
    return ck_dir / 'last.pt'


def main():
    args = parse()
    seed_everything(args.seed)
    out = make_run_dir('eagle')
    log = Logger(out)
    dev = device()
    write_json(out / 'config.json', dict(vars(args), cache=str(CACHE), torch=torch.__version__,
               paper='EAGLE, arXiv 2511.08007: VQ2D val tAP25 0.47 / stAP25 0.42 / Rec 52.09 / Succ 61.29'))
    stages = ['prepare', 'train', 'evaluate'] if args.stage == 'all' else [args.stage]
    sam = None
    if 'prepare' in stages:
        print('Starting prepare', flush=True)
        records = D.load_vq('train')
        if args.max_train_queries: records = random.Random(args.seed).sample(records, min(args.max_train_queries, len(records)))
        log(f'Local VQ2D clips: {len(D.local_clip_uids())} | train queries on local clips: {len(records)}')
        if records:
            sam = D.SamMasker(dev, args.sam)
            D.build_vq_training_cache(records, sam, log)
    checkpoint = args.checkpoint
    if checkpoint:        # architecture comes from the checkpoint, not the current form settings
        saved = torch.load(checkpoint, map_location='cpu').get('args', {})
        for k in ('backbone', 'no_amm', 'no_glm'):
            if k in saved and getattr(args, k) != saved[k]:
                log(f'Using checkpoint setting {k}={saved[k]}'); setattr(args, k, saved[k])
    model = EAGLE(args.backbone, use_glm=not args.no_glm, use_amm=not args.no_amm).to(dev)
    if 'train' in stages:
        print('Starting train', flush=True)
        if checkpoint:
            model.load_state_dict(torch.load(checkpoint, map_location=dev)['model'], strict=False)
        checkpoint = train(model, args, out, dev, log)
    if 'evaluate' in stages:
        print('Starting evaluate', flush=True)
        if checkpoint:
            model.load_state_dict(torch.load(checkpoint, map_location=dev)['model'], strict=False)
            log('Evaluating', checkpoint)
        else:
            log('WARNING: no checkpoint - evaluating an untrained model (pipeline check only).')
        records = [r for r in D.load_vq(args.eval_split) if r['response_track']]
        if args.max_eval_queries: records = random.Random(args.seed).sample(records, min(args.max_eval_queries, len(records)))
        log(f'Evaluation: {len(records)} {args.eval_split} queries on local clips')
        if not records:
            raise SystemExit(f'No local {args.eval_split} clips. Download VQ2D {args.eval_split} clips into ../Ego4D/v2/clips/.')
        if sam is None:
            sam = D.SamMasker(dev, args.sam)
        preds = run_inference(model, records, sam, dev, args, log)
        metrics = vq_metrics(records, preds)
        log('VQ2D metrics:', json.dumps(metrics))
        with (out / 'predictions.json').open('w') as f:
            json.dump({k: [b.to_json() for b in v.bboxes] + [{'score': v.score}] for k, v in preds.items()}, f)
        write_json(out / 'metrics.json', dict(final=metrics, checkpoint=checkpoint, split=args.eval_split))
    (out / 'COMPLETE').write_text('done\n')


if __name__ == '__main__':
    main()
