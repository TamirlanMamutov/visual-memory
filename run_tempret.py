"""Train / evaluate the TempRet re-implementation on EK-100 Multi-Instance Retrieval.

Stages: decode frames from the raw MP4s -> (frozen mode) cache CLIP frame
features -> train with SMS loss (+ ITM reranker) -> evaluate dual-encoder and
two-stage reranking with the official mAP / nDCG. Only locally available videos
are used; the report states the subset size so numbers are not mistaken for
full-benchmark results.
"""
import argparse
import json
import math
import time

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

from common import CACHE, Logger, device, make_run_dir, seed_everything, write_json
from tempret import data as D
from tempret.losses import itm_batch, sms_loss
from tempret.metrics import evaluate, fmt
from tempret.model import CrossEncoder, TempRet, preprocess_tensor
from tempret.reranker import ALPHAS, embed, embed_text, rerank_sweep, split_holdout, train_reranker


def parse():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--mode', choices=['frozen', 'full'], default='frozen',
                   help='frozen: cached CLIP image features, train text tower + temporal + reranker (fits 12 GB). '
                        'full: end-to-end fine-tuning of both CLIP towers (paper setting; needs much more memory).')
    p.add_argument('--backbone', default='ViT-L-14', help='open_clip model name (paper: ViT-L-14)')
    p.add_argument('--pretrained', default='openai')
    p.add_argument('--frames', type=int, default=8, help='frames per clip at train/test time (<= 16)')
    p.add_argument('--epochs', type=int, default=10, help='0 = evaluate only (zero-shot CLIP or --checkpoint)')
    p.add_argument('--batch-size', type=int, default=64)
    p.add_argument('--lr', type=float, default=1.8e-5)
    p.add_argument('--temporal-lr-mult', type=float, default=2.0)
    p.add_argument('--itm-lr', type=float, default=1e-4)
    p.add_argument('--weight-decay', type=float, default=0.01)
    p.add_argument('--warmup', type=float, default=0.05, help='fraction of steps')
    p.add_argument('--layers', type=int, default=4); p.add_argument('--heads', type=int, default=8)
    p.add_argument('--margin', type=float, default=0.2); p.add_argument('--tau', type=float, default=0.1)
    p.add_argument('--eps', type=float, default=0.1, help='SMS neutral-pair threshold (not given in the paper)')
    p.add_argument('--no-temporal', action='store_true', help='ablation: mean-pool CLIP frame features')
    p.add_argument('--no-rerank', action='store_true', help='ablation: skip the ITM cross-encoder')
    p.add_argument('--itm-weight', type=float, default=1.0, help='joint ITM loss weight (only with --reranker-epochs 0)')
    p.add_argument('--rerank-k', type=int, default=1000)
    p.add_argument('--alpha', default='auto',
                   help="fusion weight of the first-stage score (paper: 0.002); 'auto' = chosen on held-out training videos")
    p.add_argument('--reranker-epochs', type=int, default=40,
                   help='separate cross-encoder training stage on the frozen first stage (0 = train it jointly, old behaviour)')
    p.add_argument('--reranker-lr', type=float, default=3e-4)
    p.add_argument('--reranker-batch', type=int, default=128, help='anchor clips per step (8 pairs each)')
    p.add_argument('--reranker-hard', type=int, default=3, help='hard negatives per anchor and direction')
    p.add_argument('--reranker-hard-pool', type=int, default=32, help='hard negatives are drawn from the top-N first-stage hits')
    p.add_argument('--reranker-eval-every', type=int, default=5)
    p.add_argument('--itm-target', choices=['soft', 'binary'], default='soft',
                   help='soft: regress EK-100 relevance; binary: match iff relevance == 1 (paper ITM)')
    p.add_argument('--holdout', type=float, default=0.15, help='fraction of training videos held out to pick alpha')
    p.add_argument('--rerank-only', action='store_true', help='train only the reranker on top of --checkpoint')
    p.add_argument('--max-train-clips', type=int, default=0, help='limit for quick checks (0 = all local)')
    p.add_argument('--max-test-clips', type=int, default=0)
    p.add_argument('--extract-only', action='store_true', help='only decode frames (and cache features)')
    p.add_argument('--checkpoint', help='evaluate / continue from a saved checkpoint')
    p.add_argument('--workers', type=int, default=4)
    p.add_argument('--seed', type=int, default=230)
    return p.parse_args()


class Clips(Dataset):
    def __init__(self, df, tokens, frames, train, feats=None):
        self.ids = list(df.index); self.tokens = tokens; self.train = train
        self.verbs = torch.tensor(df.verb_class.values); self.nouns = torch.from_numpy(D.multi_hot(df.all_noun_classes))
        self.feats = feats; self.frames = frames

    def __len__(self):
        return len(self.ids)

    def slots(self):
        n, t = D.STORED_FRAMES, self.frames
        edges = np.linspace(0, n, t + 1).astype(int)
        if self.train:
            return [np.random.randint(a, max(a + 1, b)) for a, b in zip(edges[:-1], edges[1:])]
        return [(a + max(a + 1, b) - 1) // 2 for a, b in zip(edges[:-1], edges[1:])]

    def __getitem__(self, i):
        nid = self.ids[i]; s = self.slots()
        if self.feats is not None:
            x = self.feats[nid][s].float()
        else:
            frames = D.load_frames(nid)
            x = torch.stack([preprocess_tensor(frames[j], train=self.train) for j in s])
        return x, self.tokens[i], self.verbs[i], self.nouns[i]


def build(args, dev, log):
    import open_clip
    if args.checkpoint:   # architecture comes from the checkpoint, not the current form settings
        saved = torch.load(args.checkpoint, map_location='cpu').get('args', {})
        for k in ('backbone', 'pretrained', 'layers', 'heads', 'no_temporal'):
            if k in saved and getattr(args, k) != saved[k]:
                log(f'Using checkpoint setting {k}={saved[k]} (requested {getattr(args, k)})'); setattr(args, k, saved[k])
    # OpenAI CLIP weights were trained with QuickGELU; open_clip needs it forced.
    clip_model, _, _ = open_clip.create_model_and_transforms(args.backbone, pretrained=args.pretrained,
                                                             force_quick_gelu=args.pretrained == 'openai')
    tokenizer = open_clip.get_tokenizer(args.backbone)
    clip_model.to(dev)
    model = TempRet(clip_model, args.layers, args.heads, use_temporal=not args.no_temporal).to(dev)
    width = clip_model.ln_final.weight.shape[0]
    cross = None if args.no_rerank else CrossEncoder(model.dim, width).to(dev)
    if args.checkpoint:
        ck = torch.load(args.checkpoint, map_location=dev)
        missing = model.load_state_dict(ck['model'], strict=False)
        log('Loaded', args.checkpoint, '| params not in checkpoint (kept pretrained):', len(missing.missing_keys))
        if args.rerank_only:
            log('Reranker is trained from scratch on top of the checkpoint.')
        elif cross is not None and ck.get('cross'):
            cross.load_state_dict(ck['cross'])
        elif cross is not None:
            cross = None; log('Checkpoint has no reranker; reranking disabled.')
    return model, cross, tokenizer


def tokenize(tokenizer, texts):
    tok = tokenizer(list(texts))
    return tok[:, :int(tok.argmax(-1).max()) + 1]      # trim padding after the longest EOS


def resolve_alpha(args, ckpt_alpha=None):
    if str(args.alpha) != 'auto':
        return float(args.alpha)
    return ckpt_alpha if ckpt_alpha is not None else 0.002


def run_eval(model, cross, args, test_loader, sent_tokens, rel, dev, amp, log, rerank_on, alpha=0.002):
    v, vtok = embed(model, test_loader, dev, amp)
    t, ttok = embed_text(model, sent_tokens, dev, amp)
    if not (rerank_on and cross is not None):
        result = {'dual': evaluate((v @ t.T).cpu().numpy(), rel)}
        log('  dual-encoder :', fmt(result['dual'])); return result
    alphas = sorted(set(ALPHAS) | {alpha})
    sweep, secs = rerank_sweep(cross, v, vtok, t, ttok, (sent_tokens > 0).to(dev), rel, args.rerank_k, alphas, amp)
    result = {'dual': sweep['dual'], 'rerank': sweep[alpha], 'rerank_alpha': alpha, 'rerank_seconds': secs,
              'rerank_sweep_test_diagnostic': {str(a): sweep[a] for a in alphas}}
    log('  dual-encoder :', fmt(result['dual']))
    log(f'  + reranking  : {fmt(result["rerank"])}  (alpha {alpha})')
    log('  test-set alpha sweep (diagnostic only, not used for selection): ' +
        ', '.join(f'{a}: {(sweep[a]["mAP"]["AVG"] + sweep[a]["nDCG"]["AVG"]) / 2:.2f}' for a in alphas))
    return result


def main():
    args = parse()
    seed_everything(args.seed)
    out = make_run_dir('tempret')
    log = Logger(out)
    dev = device()
    amp = lambda: torch.autocast(dev.type, dtype=torch.bfloat16, enabled=dev.type == 'cuda')

    print('Starting data', flush=True)
    train_df = D.load_split('train')
    test_full = D.load_split('test', local_only=False)
    test_df = test_full[test_full.video_id.isin(set(D.local_video_ids()))]
    if args.max_train_clips: train_df = train_df.sample(min(args.max_train_clips, len(train_df)), random_state=args.seed)
    if args.max_test_clips: test_df = test_df.sample(min(args.max_test_clips, len(test_df)), random_state=args.seed).sort_index()
    log(f'Local videos: {len(D.local_video_ids())} | train clips {len(train_df)} / 67,217 | '
        f'test clips {len(test_df)} / {len(test_full)}')
    if args.rerank_only and not args.checkpoint:
        raise SystemExit('--rerank-only needs --checkpoint (a trained TempRet checkpoint).')
    needs_train = bool(args.epochs or args.rerank_only)
    if needs_train and len(train_df) == 0:
        raise SystemExit('No training clips: download EPIC-KITCHENS videos into ../EPIC-KITCHENS/Pxx/videos/.')
    D.extract_frames(train_df if needs_train else train_df.iloc[:0], log)
    D.extract_frames(test_df, log)
    train_df, test_df = D.cached(train_df), D.cached(test_df)

    model, cross, tokenizer = build(args, dev, log)
    feats = None
    if args.mode == 'frozen':
        tag = f'{args.backbone}-{args.pretrained}'.replace('/', '_')
        feats = D.build_feature_cache(D.pd.concat([train_df, test_df]), model.clip, preprocess_tensor, tag, dev, log)
        for p in model.clip.visual.parameters(): p.requires_grad = False
    if args.extract_only:
        log('Extraction finished.'); (out / 'COMPLETE').write_text('extract only\n'); return

    # ---- evaluation set (only sentences with >=1 fully relevant local video)
    sents = D.load_test_sentences(test_full)
    s_nouns = D.multi_hot(sents.noun_classes)
    rel = D.relevance(test_df.verb_class.values, D.multi_hot(test_df.all_noun_classes), sents.verb_class.values, s_nouns)
    keep = rel.max(0) >= 1
    sents, rel = sents[keep], rel[:, keep]
    log(f'Evaluation: {rel.shape[0]} videos x {rel.shape[1]} sentences '
        f'({"FULL benchmark" if len(test_df) == len(test_full) else "LOCAL SUBSET - not comparable to the leaderboard"})')
    sent_tokens = tokenize(tokenizer, sents.narration)
    test_tokens = tokenize(tokenizer, test_df.narration)
    test_loader = DataLoader(Clips(test_df, test_tokens, args.frames, False, feats), batch_size=128,
                             num_workers=0 if feats is not None else args.workers)

    config = dict(vars(args), train_clips=len(train_df), test_clips=len(test_df), eval_sentences=int(rel.shape[1]),
                  full_benchmark=len(test_df) == len(test_full), cache=str(CACHE), torch=torch.__version__,
                  paper='TempRet, arXiv 2605.24470: 67.97 mAP / 82.92 nDCG (EK-100 MIR test server)')
    write_json(out / 'config.json', config)
    history = []
    ck_alpha = torch.load(args.checkpoint, map_location='cpu').get('rerank_alpha') if args.checkpoint else None
    make_loader = lambda df, train: DataLoader(Clips(df, tokenize(tokenizer, df.narration), args.frames, train, feats),
                                               batch_size=256, num_workers=0 if feats is not None else args.workers)

    def reranker_stage(ck_dir, state):
        """Train the cross-encoder on top of the (frozen) first stage; store it in state."""
        fit_df, held_df, held = split_holdout(train_df, args.holdout, args.seed)
        log(f'Reranker stage: {len(fit_df)} training clips, {len(held_df)} held-out clips from videos {held}')
        cross_state, alpha, rhist, repoch = train_reranker(model, cross, fit_df, held_df, make_loader,
                                                           lambda texts: tokenize(tokenizer, texts), args, dev, amp, log)
        cross.load_state_dict(cross_state)
        state.update(cross=cross_state, rerank_alpha=alpha, reranker=dict(
            epoch=repoch, history=rhist, held_out_videos=held, target=args.itm_target))
        torch.save(state, ck_dir / 'best.pt')
        with (out / 'reranker_history.jsonl').open('w') as f:
            for r in rhist: f.write(json.dumps(r) + '\n')
        return alpha

    if args.epochs == 0 and not args.rerank_only:
        print('Starting evaluation', flush=True)
        if not args.checkpoint and model.temporal is not None:
            log('No checkpoint: evaluating zero-shot CLIP with mean pooling (temporal transformer disabled).')
            model.temporal = None
        res = run_eval(model, cross if args.checkpoint else None, args, test_loader, sent_tokens, rel, dev, amp, log,
                       True, resolve_alpha(args, ck_alpha))
        write_json(out / 'metrics.json', dict(final=res, config=config))
        (out / 'COMPLETE').write_text('done\n'); return

    ck_dir = CACHE / 'tempret' / 'checkpoints' / out.name; ck_dir.mkdir(parents=True, exist_ok=True)
    if args.rerank_only:
        if cross is None: raise SystemExit('--rerank-only cannot be combined with --no-rerank.')
        state = torch.load(args.checkpoint, map_location='cpu')
        state['args'] = dict(state.get('args', {}), **{k: v for k, v in vars(args).items() if k.startswith(('reranker', 'itm', 'holdout', 'rerank_k'))})
        state['first_stage_checkpoint'] = args.checkpoint
        print('Starting reranker', flush=True)
        alpha = resolve_alpha(args, reranker_stage(ck_dir, state))
        print('Starting final evaluation', flush=True)
        res = run_eval(model, cross, args, test_loader, sent_tokens, rel, dev, amp, log, True, alpha)
        write_json(out / 'metrics.json', dict(final=res, best_epoch=state.get('epoch'), config=config,
                                              reranker=state['reranker'], checkpoint=str(ck_dir / 'best.pt')))
        (out / 'COMPLETE').write_text('done\n'); return

    # ---- training
    joint_itm = cross is not None and args.reranker_epochs == 0
    train_tokens = tokenize(tokenizer, train_df.narration)
    train_loader = DataLoader(Clips(train_df, train_tokens, args.frames, True, feats), batch_size=args.batch_size,
                              shuffle=True, drop_last=True, num_workers=0 if feats is not None else args.workers,
                              persistent_workers=feats is None and args.workers > 0)
    if args.mode == 'full':
        model.clip.set_grad_checkpointing(True)
    temporal = list(model.temporal.parameters()) if model.temporal is not None else []
    backbone = [p for p in model.clip.parameters() if p.requires_grad]
    groups = [dict(params=backbone, lr=args.lr)]
    if temporal: groups.append(dict(params=temporal, lr=args.lr * args.temporal_lr_mult))
    if joint_itm: groups.append(dict(params=list(cross.parameters()), lr=args.itm_lr))
    opt = torch.optim.AdamW(groups, weight_decay=args.weight_decay)
    total = args.epochs * len(train_loader); warm = max(1, int(args.warmup * total))
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: min(1, (s + 1) / warm) * 0.5 * (1 + math.cos(math.pi * min(1, s / total))))
    log(f'Training {args.epochs} epochs x {len(train_loader)} steps, batch {args.batch_size}, mode {args.mode}; '
        f'checkpoints in {ck_dir}')
    best, step = -1, 0
    for epoch in range(1, args.epochs + 1):
        print(f'Starting epoch {epoch}', flush=True)
        model.train(); joint_itm and cross.train()
        if args.mode == 'frozen': model.clip.visual.eval()
        t0, run = time.time(), {'sms': 0.0, 'itm': 0.0, 'itm_acc': 0.0}
        for it, (x, tok, verbs, nouns) in enumerate(train_loader, 1):
            x, tok = x.to(dev, non_blocking=True), tok.to(dev)
            verbs, nouns = verbs.to(dev), nouns.to(dev)
            with amp():
                feats_b = x if x.dim() == 3 else model.encode_frames(x)
                v, vtok = model.encode_video(feats_b)
                t, ttok = model.encode_text(tok)
                sim = v.float() @ t.float().T
                r = D.torch_relevance(verbs, nouns)
                loss_sms = sms_loss(sim, r, args.margin, args.tau, args.eps)
                loss = loss_sms
                if joint_itm:
                    vi, ti, lab = itm_batch(sim, r)
                    vmask = torch.ones(vtok.shape[:2], dtype=torch.bool, device=dev)
                    logits = cross(vtok[vi].detach(), vmask[vi], ttok[ti].detach(), (tok > 0)[ti])
                    loss_itm = F.cross_entropy(logits.float(), lab)
                    loss = loss + args.itm_weight * loss_itm
                    run['itm'] += loss_itm.item(); run['itm_acc'] += (logits.argmax(1) == lab).float().mean().item()
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_([p for g in groups for p in g['params']], 1.0)
            opt.step(); sched.step(); step += 1
            run['sms'] += loss_sms.item()
            if it % 50 == 0 or it == len(train_loader):
                log(f'  epoch {epoch} step {it}/{len(train_loader)} | SMS {run["sms"]/it:.4f} | '
                    f'ITM {run["itm"]/it:.4f} acc {run["itm_acc"]/it:.3f} | {time.time()-t0:.0f}s')
        res = run_eval(model, cross, args, test_loader, sent_tokens, rel, dev, amp, log, rerank_on=False)
        score = res['dual']['mAP']['AVG'] + res['dual']['nDCG']['AVG']
        state = dict(model={k: v for k, v in model.state_dict().items()
                            if args.mode == 'full' or not k.startswith('clip.visual.')},
                     cross=cross.state_dict() if cross is not None else None, args=vars(args), epoch=epoch)
        torch.save(state, ck_dir / 'last.pt')
        if score > best:
            best = score; torch.save(state, ck_dir / 'best.pt')
        history.append(dict(epoch=epoch, sms=run['sms'] / len(train_loader), itm=run['itm'] / len(train_loader),
                            seconds=round(time.time() - t0, 1), **{f'{m}_{d}': res['dual'][m][d]
                            for m in ('mAP', 'nDCG') for d in ('V2T', 'T2V', 'AVG')}))
        with (out / 'history.jsonl').open('a') as f: f.write(json.dumps(history[-1]) + '\n')

    print('Starting final evaluation', flush=True)
    ck = torch.load(ck_dir / 'best.pt', map_location=dev)
    model.load_state_dict(ck['model'], strict=False)
    log(f'Best epoch {ck["epoch"]}:')
    alpha = resolve_alpha(args)
    if joint_itm:
        cross.load_state_dict(ck['cross'])
    elif cross is not None:
        print('Starting reranker', flush=True)
        alpha = resolve_alpha(args, reranker_stage(ck_dir, ck))
        print('Starting final evaluation', flush=True)
    res = run_eval(model, cross, args, test_loader, sent_tokens, rel, dev, amp, log, True, alpha)
    write_json(out / 'metrics.json', dict(final=res, best_epoch=ck['epoch'], history=history, config=config,
                                          reranker=ck.get('reranker'), checkpoint=str(ck_dir / 'best.pt')))
    (out / 'COMPLETE').write_text('done\n')


if __name__ == '__main__':
    main()
