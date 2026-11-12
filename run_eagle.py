"""Train / evaluate the EAGLE re-implementation (2D VQ localization) on Ego4D VQ2D.

Stages (run in order, each resumable):
  prepare  - decode query/response-track/negative frames from local VQ2D clips
             and generate SAM pseudo-masks (box + centre-point prompt)
  train    - meta-train AMM + GLM + decoder on VQ2D (and EgoTracks / VISOR when present)
  evaluate - online inference on VQ2D val clips; tAP25 / stAP25 / Rec / Succ
Use --stage all for prepare -> train -> evaluate.

Interrupted? Rerun the same command with --resume output/eagle-<run>: prepare
skips finished queries, training continues from the last save (every
--save-every iterations) with the same file list, optimiser, schedule and RNG,
and evaluation skips clips whose predictions are already saved.
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

from common import CACHE, Logger, device, make_run_dir, portable, seed_everything, torch_save_atomic, write_json
from eagle import data as D
from eagle.infer import POSTPROC_DEFAULTS, UPDATE_CONF, postproc_params, run_inference, vq_metrics
from eagle.model import EAGLE, confidence, dcf_hinge, gaussian_label, lovasz_hinge, pixel_losses


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
    p.add_argument('--seg-loss', choices=['lovasz+pixel', 'lovasz'], default='lovasz+pixel',
                   help='paper: Lovasz only, which collapsed to an empty mask here; default adds BCE + Dice')
    p.add_argument('--init-iters', type=int, default=10); p.add_argument('--update-iters', type=int, default=3)
    p.add_argument('--no-amm', action='store_true'); p.add_argument('--no-glm', action='store_true')
    p.add_argument('--no-visor', action='store_true')
    p.add_argument('--no-egotracks', action='store_true', help='leave EgoTracks long-term tracks out of prepare and training')
    p.add_argument('--max-train-queries', type=int, default=0); p.add_argument('--max-eval-queries', type=int, default=0)
    p.add_argument('--eval-split', default='val')
    p.add_argument('--eval-clips', metavar='A:B', help='evaluate only clips A..B-1 in annotation order, e.g. 351:551')
    p.add_argument('--pp-window', type=int, default=5, help='post-processing: median filter length (odd)')
    p.add_argument('--pp-peak', type=float, default=0.8, help='post-processing: threshold as a fraction of the max peak')
    p.add_argument('--pp-extend', type=float, default=None,
                   help='post-processing: extend the interval while above this fraction of the max peak (default: --pp-peak)')
    p.add_argument('--pp-gap', type=int, default=0, help='post-processing: bridge dips of up to this many frames')
    p.add_argument('--pp-pick', choices=['last', 'best'], default='last', help='post-processing: which peak to keep')
    p.add_argument('--checkpoint', help='checkpoint to evaluate (default: the one trained in this run)')
    p.add_argument('--resume', metavar='RUN_DIR',
                   help='continue an interrupted run (output/eagle-...): settings, training file list, weights, '
                        'optimiser, schedule and RNG come from it; finished evaluation clips are skipped')
    p.add_argument('--log-every', type=int, default=50)
    p.add_argument('--save-every', type=int, default=500, help='iterations between resumable last.pt saves')
    p.add_argument('--keep-every', type=int, default=2500, help='also keep a permanent iter_N.pt copy this often')
    p.add_argument('--seed', type=int, default=230)
    return p.parse_args()


# ------------------------------------------------------------- sampling -----

class Sampler:
    """Each sequence comes from one source, chosen uniformly among the sources present
    (VQ2D, EgoTracks, VISOR); the paper does not give mixing ratios."""

    def __init__(self, vq_keys, visor_seqs, seq_len, rng, et_keys=()):
        self.vq, self.visor, self.n, self.rng = vq_keys, visor_seqs, seq_len, rng
        self.et = list(et_keys)
        self.visor_frames = D.VisorFrames() if visor_seqs else None

    def sample(self):
        if not self.et:            # the original two-source rule, unchanged so older runs resume identically
            if self.visor and (not self.vq or self.rng.random() < 0.5):
                return self.sample_visor()
            return self.sample_vq()
        sources = [s for s, ok in (('vq', self.vq), ('et', self.et), ('visor', self.visor)) if ok]
        s = self.rng.choice(sources)
        return self.sample_visor() if s == 'visor' else self.sample_vq(s == 'et')

    def sample_vq(self, egotracks=False):
        """A cached query (VQ2D or EgoTracks): the visual crop, then track frames plus one object-free frame."""
        keys, path = (self.et, D.egotracks_cache_path) if egotracks else (self.vq, D.vq_cache_path)
        with path(self.rng.choice(keys)).open('rb') as f:
            items = pickle.load(f)['items']
        query = [i for i in items if i['role'] == 'query']
        track = [i for i in items if i['role'] == 'track']
        neg = [i for i in items if i['role'] == 'neg']
        if not query or not track:
            return self.sample_vq(egotracks)
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
    loss_seg, loss_tck, n_frames = 0.0, 0.0, 0
    st = dict(pos=0, pos_hit=0, pos_iou=0.0, neg=0, neg_ok=0)
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
            if args.seg_loss == 'lovasz+pixel':
                bce, dice = pixel_losses(logits, mt[t:t + 1])
                loss_seg = loss_seg + bce + dice
            loss_tck = loss_tck + dcf_hinge(hj, gauss[t:t + 1], region[t:t + 1])
            n_frames += 1
            pred = (logits.detach() > 0).float()
            if mt[t].sum() > 0:    # object visible: IoU of the predicted mask
                iou = ((pred * mt[t:t + 1]).sum() / (pred + mt[t:t + 1]).clamp(max=1).sum()).item()
                st['pos'] += 1; st['pos_iou'] += iou; st['pos_hit'] += int(iou > 0.5)
            else:                  # object-free frame: correct if the mask is empty
                st['neg'] += 1; st['neg_ok'] += int(pred.sum() == 0)
            if t < len(seq) - 1 and confidence(logits.detach())[0] >= UPDATE_CONF:
                mem_f.append(ft['seg']); mem_m.append(logits.detach().sigmoid())
                sigma = model.amm.optimise(sigma, torch.cat(mem_f), torch.cat(mem_m), args.update_iters, True)
    loss_seg, loss_tck = loss_seg / n_frames, loss_tck / n_frames
    return loss_seg, loss_tck, st


def rng_state(sampler):
    s = dict(py=random.getstate(), np=np.random.get_state(), torch=torch.get_rng_state(), sampler=sampler.rng.getstate())
    if torch.cuda.is_available(): s['cuda'] = torch.cuda.get_rng_state_all()
    return s


def set_rng_state(s, sampler):
    random.setstate(s['py']); np.random.set_state(s['np']); torch.set_rng_state(s['torch'])
    sampler.rng.setstate(s['sampler'])
    if 'cuda' in s and torch.cuda.is_available(): torch.cuda.set_rng_state_all(s['cuda'])


def train(model, args, out, dev, log):
    ck_dir = CACHE / 'eagle' / 'checkpoints' / out.name; ck_dir.mkdir(parents=True, exist_ok=True)
    last = ck_dir / 'last.pt'
    resume = torch.load(last, map_location='cpu', weights_only=False) if args.resume and last.exists() else None
    if resume is not None and 'opt' not in resume:
        raise SystemExit(f'{last} predates resumable checkpoints (no optimiser state); start a new run.')
    if resume is not None and resume['iter'] >= args.iters:
        log(f'Training already finished ({resume["iter"]} iterations): {last}')
        return last

    if resume is not None:
        # The exact file list of the interrupted run, even if more queries were prepared since.
        vq_keys = resume['train_files']
        et_keys = resume.get('egotracks_files', [])     # runs from before EgoTracks support used none
        missing = [k for k in vq_keys if not D.vq_cache_path(k).exists()] + \
                  [k for k in et_keys if not D.egotracks_cache_path(k).exists()]
        if missing:
            raise SystemExit(f'{len(missing)} training files of this run are gone from the cache, e.g. {missing[0]}')
    else:
        # Train on every query already prepared (the local clip set can grow between stages).
        vq_keys = sorted(r['key'] for r in D.load_vq('train', local_only=False) if D.vq_cache_path(r['key']).exists())
        if args.max_train_queries: vq_keys = vq_keys[:args.max_train_queries]
        et_dir = D.EAGLE_CACHE / 'egotracks'
        et_keys = [] if getattr(args, 'no_egotracks', False) or not et_dir.is_dir() else sorted(p.stem for p in et_dir.glob('*.pkl'))
        if args.max_train_queries: et_keys = et_keys[:args.max_train_queries]
        write_json(out / 'train_files.json', vq_keys)
        if et_keys: write_json(out / 'train_files_egotracks.json', et_keys)
    visor = [] if args.no_visor or not D.visor_available() else D.load_visor_sequences('train')
    if resume is not None and len(visor) != resume['visor_sequences']:
        raise SystemExit(f'VISOR changed since the run started ({resume["visor_sequences"]} -> {len(visor)} sequences); '
                         'resume needs the same training data.')
    log(f'Training data: {len(vq_keys)} VQ2D queries with cached frames/masks, {len(et_keys)} EgoTracks long-term tracks, '
        f'{len(visor)} VISOR object sequences'
        + ('' if D.visor_available() else ' (VISOR not found at ../EPIC-KITCHENS-VISOR)') + f'; list in {out / "train_files.json"}')
    if not vq_keys and not visor and not et_keys:
        raise SystemExit('No training data. Download VQ2D train clips (and/or VISOR), then run --stage prepare.')
    params = [p for n, p in model.named_parameters() if p.requires_grad and not n.startswith('backbone.')]
    log(f'Trainable parameters: {sum(p.numel() for p in params) / 1e6:.2f} M (backbone frozen)')
    opt = torch.optim.AdamW(params, lr=args.lr, weight_decay=args.weight_decay)
    warm = min(args.warmup, max(1, args.iters // 10))
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: min((s + 1) / warm, max(0.0, (args.iters - s) / max(1, args.iters - warm))))
    sampler = Sampler(vq_keys, visor, args.seq_len, random.Random(args.seed), et_keys)
    start = 1
    if resume is not None:
        model.load_state_dict(resume['model'], strict=False)
        opt.load_state_dict(resume['opt']); sched.load_state_dict(resume['sched'])
        set_rng_state(resume['rng'], sampler)
        start = resume['iter'] + 1
        log(f'Resuming training at iteration {start} from {last}')

    def save(it):
        state = {k: v for k, v in model.state_dict().items() if not k.startswith('backbone.')}
        ck = dict(model=state, args=vars(args), iter=it, opt=opt.state_dict(), sched=sched.state_dict(),
                  rng=rng_state(sampler), train_files=vq_keys, egotracks_files=et_keys, visor_sequences=len(visor))
        torch_save_atomic(ck, last)
        if args.keep_every and it % args.keep_every == 0 and it != args.iters:
            torch_save_atomic(dict(model=state, args=vars(args), iter=it), ck_dir / f'iter_{it}.pt')

    log(f'Training {args.iters} iterations, batch {args.batch_size} x {args.seq_len} frames; '
        f'resumable save every {args.save_every}; checkpoints in {ck_dir}')
    model.train()
    fresh = lambda: dict(seg=0.0, tck=0.0, n=0, pos=0, pos_hit=0, pos_iou=0.0, neg=0, neg_ok=0)
    t0, acc = time.time(), fresh()
    for it in range(start, args.iters + 1):
        batch = [sampler.sample() for _ in range(args.batch_size)]
        seg, tck, st = train_step(model, batch, args, dev)
        loss = seg + args.rho * tck
        opt.zero_grad(set_to_none=True); loss.backward()
        torch.nn.utils.clip_grad_norm_(params, 1.0)
        opt.step(); sched.step()
        acc['seg'] += seg.item(); acc['tck'] += tck.item(); acc['n'] += 1
        for k, v in st.items(): acc[k] += v
        if it % args.log_every == 0 or it == args.iters:
            n = acc['n']
            row = dict(iter=it, seg=acc['seg'] / n, tck=acc['tck'] / n,
                       pos_iou50=acc['pos_hit'] / max(1, acc['pos']), pos_miou=acc['pos_iou'] / max(1, acc['pos']),
                       neg_empty=acc['neg_ok'] / max(1, acc['neg']),
                       lr=sched.get_last_lr()[0], seconds=round(time.time() - t0, 1))
            log(f"  iter {it}/{args.iters} | L_seg {row['seg']:.4f} | L_tck {row['tck']:.5f} | "
                f"object frames: IoU>0.5 {row['pos_iou50']:.3f}, mIoU {row['pos_miou']:.3f} | "
                f"empty frames correct {row['neg_empty']:.3f} | lr {row['lr']:.2e} | {row['seconds']:.0f}s")
            with (out / 'history.jsonl').open('a') as f: f.write(json.dumps(row) + '\n')
            acc = fresh()
        if it % args.save_every == 0 or it == args.iters:
            save(it)
    return last


def main():
    args = parse()
    out = make_run_dir('eagle', args.resume)
    log = Logger(out)
    if args.resume:
        saved = json.loads((out / 'config.json').read_text(encoding='utf-8')) if (out / 'config.json').exists() else None
        if saved is None:
            raise SystemExit(f'{out} is not an EAGLE run folder (no config.json).')
        # The interrupted run's settings win; only the stage selection comes from this command.
        changed = [k for k in vars(args) if k in saved and k not in ('stage', 'resume') and saved[k] != getattr(args, k)]
        for k in changed: setattr(args, k, saved[k])
        log(f'Resuming {out.name} (stage {args.stage})' + (f'; kept the run\'s settings for {", ".join(changed)}' if changed else ''))
    seed_everything(args.seed)
    dev = device()
    if not args.resume:
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
        if not getattr(args, 'no_egotracks', False) and D.egotracks_available():
            et = D.load_egotracks('train')
            if args.max_train_queries: et = random.Random(args.seed).sample(et, min(args.max_train_queries, len(et)))
            log(f'EgoTracks train long-term tracks on local clips: {len(et)}')
            if et:
                sam = sam or D.SamMasker(dev, args.sam)
                D.build_egotracks_training_cache(et, sam, log)
    checkpoint = args.checkpoint
    own_ck = CACHE / 'eagle' / 'checkpoints' / out.name / 'last.pt'
    if args.resume and not checkpoint and 'train' not in stages and own_ck.exists():
        checkpoint = str(own_ck)           # evaluate-only resume: use the run's own weights
    if checkpoint:        # architecture comes from the checkpoint, not the current form settings
        saved = torch.load(checkpoint, map_location='cpu', weights_only=False).get('args', {})
        for k in ('backbone', 'no_amm', 'no_glm'):
            if k in saved and getattr(args, k) != saved[k]:
                log(f'Using checkpoint setting {k}={saved[k]}'); setattr(args, k, saved[k])
    model = EAGLE(args.backbone, use_glm=not args.no_glm, use_amm=not args.no_amm).to(dev)
    if 'train' in stages:
        print('Starting train', flush=True)
        if checkpoint and not (args.resume and own_ck.exists()):   # initial weights for a fresh run
            model.load_state_dict(torch.load(checkpoint, map_location=dev, weights_only=False)['model'], strict=False)
        checkpoint = str(train(model, args, out, dev, log))
    if 'evaluate' in stages:
        print('Starting evaluate', flush=True)
        if checkpoint:
            model.load_state_dict(torch.load(checkpoint, map_location=dev, weights_only=False)['model'], strict=False)
            log('Evaluating', checkpoint)
        else:
            log('WARNING: no checkpoint - evaluating an untrained model (pipeline check only).')
        records = [r for r in D.load_vq(args.eval_split) if r['response_track']]
        if getattr(args, 'eval_clips', None):
            a, b = (int(x) if x else None for x in args.eval_clips.split(':'))
            keep = set(list(dict.fromkeys(r['clip_uid'] for r in records))[a:b])
            records = [r for r in records if r['clip_uid'] in keep]
        if args.max_eval_queries: records = random.Random(args.seed).sample(records, min(args.max_eval_queries, len(records)))
        log(f'Evaluation: {len(records)} {args.eval_split} queries on local clips')
        if not records:
            raise SystemExit(f'No local {args.eval_split} clips. Download VQ2D {args.eval_split} clips into ../Ego4D/v2/clips/.')
        if sam is None:
            sam = D.SamMasker(dev, args.sam)
        # Per-clip predictions are saved as they finish so an interrupted evaluation resumes.
        parts = out / 'eval_parts'; parts.mkdir(exist_ok=True)
        meta = dict(checkpoint=portable(checkpoint) if checkpoint else None, split=args.eval_split, queries=len(records))
        if postproc_params(args) != POSTPROC_DEFAULTS:
            meta['postproc'] = postproc_params(args)
        if (parts / 'meta.json').exists():
            old = json.loads((parts / 'meta.json').read_text(encoding='utf-8'))
            if old.get('checkpoint'): old['checkpoint'] = portable(old['checkpoint'])   # older runs stored absolute paths
            if old != meta:
                raise SystemExit(f'{parts} holds predictions for {old}, not {meta}; delete it or use a new run.')
            log(f'Evaluation resume: {len(list(parts.glob("*.json"))) - 1} clips already done')
        else:
            write_json(parts / 'meta.json', meta)
        preds = run_inference(model, records, sam, dev, args, log, parts_dir=parts)
        metrics = vq_metrics(records, preds)
        log('VQ2D metrics:', json.dumps(metrics))
        with (out / 'predictions.json').open('w') as f:
            json.dump({k: [b.to_json() for b in v.bboxes] + [{'score': v.score}] for k, v in preds.items()}, f)
        write_json(out / 'metrics.json', dict(final=metrics, checkpoint=checkpoint, split=args.eval_split))
    (out / 'COMPLETE').write_text('done\n')


if __name__ == '__main__':
    main()
