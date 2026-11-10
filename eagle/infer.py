"""EAGLE VQ2D inference: memory initialisation with an augmented query, online
memory updates, mask-confidence temporal scores and response-track extraction."""
import json
import sys

import numpy as np
import torch
import torch.nn.functional as F

from common import ROOT, write_json
from eagle import data as D
from eagle.model import confidence, gaussian_label

UPDATE_CONF = 0.6      # s_conf needed to enter memory
RESET_CONF = 0.4       # revert to the initial query when confidence stays below this
MAX_MEMORY = 50


def augment_query(img, mask, box):
    """Original, vertical flip, translation and blur (paper Sec. 3.3 / appendix)."""
    import cv2
    out = [(img, mask, box)]
    h = img.shape[0]
    out.append((img[::-1].copy(), mask[::-1].copy(), [box[0], h - box[3], box[2], h - box[1]]))
    dx, dy = int(0.06 * h), int(0.04 * h)
    M = np.float32([[1, 0, dx], [0, 1, dy]])
    out.append((cv2.warpAffine(img, M, (h, h)), cv2.warpAffine(mask, M, (h, h), flags=cv2.INTER_NEAREST),
                [box[0] + dx, box[1] + dy, box[2] + dx, box[3] + dy]))
    out.append((cv2.GaussianBlur(img, (7, 7), 2.0), mask, box))
    return out


class QueryState:
    def __init__(self, model, record, init_feats, init_masks, init_boxes, dev, args):
        self.model, self.r, self.args, self.dev = model, record, args, dev
        self.stride = model.stride
        size = init_feats['seg'].shape[-2:]
        self.size = size
        g, s = gaussian_label(torch.tensor(init_boxes, dtype=torch.float32, device=dev), size, self.stride)
        self.init_amm = (init_feats['seg'], init_masks)
        self.init_glm = (init_feats['tck'], g, s)
        self.reset()
        with torch.no_grad():
            self.ref_peak = model.glm.response(init_feats['tck'][:1], self.c).max().item()
        self.scores, self.boxes = {}, {}
        self.low_run, self.glm_hits, self.seen = 0, 0, 0

    def reset(self):
        m = self.model
        feats, masks = self.init_amm
        self.amm_feats, self.amm_masks = [], []
        self.glm_dyn = []
        with torch.enable_grad():
            self.sigma = m.amm.optimise(m.amm.init_kernel(feats), feats, masks, self.args.init_iters, False).detach()
        self.c = m.glm.optimise(None, *self.init_glm, iters=self.args.init_iters)

    def step(self, frame_no, f):
        """f: features dict for ONE frame (batch dim 1)."""
        m = self.model
        with torch.no_grad():
            logits, hj = m.segment(f, self.sigma, self.c)
        conf = float(confidence(logits)[0])
        mask = (logits[0, 0] > 0).cpu().numpy()
        box = D.mask_to_box(mask)
        self.scores[frame_no] = conf if box is not None else 0.0
        if box is not None:
            self.boxes[frame_no] = box
        self.seen += 1
        self.glm_hits += int(hj.max().item() > 0.5 * self.ref_peak)
        # ---- memory management
        self.low_run = self.low_run + 1 if conf < RESET_CONF else 0
        if self.low_run >= 25 and (self.amm_feats or self.glm_dyn):
            self.reset(); self.low_run = 0
            return
        scheduled = frame_no < 100 or frame_no % 25 == 0
        if not scheduled or conf < UPDATE_CONF or box is None:
            return
        prob = logits.sigmoid().detach()
        self.amm_feats.append(f['seg']); self.amm_masks.append(prob)
        if len(self.amm_feats) > MAX_MEMORY - self.init_amm[0].shape[0]:
            self.amm_feats.pop(0); self.amm_masks.pop(0)                     # FIFO, keep the initial query
        feats = torch.cat([self.init_amm[0]] + self.amm_feats)
        masks = torch.cat([self.init_amm[1]] + self.amm_masks)
        with torch.enable_grad():
            self.sigma = m.amm.optimise(self.sigma, feats, masks, self.args.update_iters, False).detach()
        # GLM: dynamic snapshot with the box enlarged 1.5x
        x1, y1, x2, y2 = box; cx, cy, w, h = (x1 + x2) / 2, (y1 + y2) / 2, (x2 - x1) * 1.5, (y2 - y1) * 1.5
        g, s = gaussian_label(torch.tensor([[cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2]], device=self.dev),
                              self.size, self.stride)
        self.glm_dyn.append((f['tck'], g, s))
        if len(self.glm_dyn) > MAX_MEMORY - 1:
            self.glm_dyn.pop(0)
        # Static snapshot only when the filter has failed on >60% of the history.
        if self.glm_hits / max(1, self.seen) < 0.4:
            feats_t, gs, ss = self.init_glm
        else:
            feats_t = torch.cat([self.init_glm[0], self.glm_dyn[-1][0]])
            gs = torch.cat([self.init_glm[1], self.glm_dyn[-1][1]]); ss = torch.cat([self.init_glm[2], self.glm_dyn[-1][2]])
        self.c = m.glm.optimise(self.c, feats_t, gs, ss, iters=self.args.update_iters)

    def response_track(self, reader):
        """Median filter (5) -> peaks -> 0.8 x max peak -> last interval above threshold."""
        from evaluation.structures import BBox, ResponseTrack
        n = self.r['query_frame']
        s = np.array([self.scores.get(i, 0.0) for i in range(n)])
        if n == 0 or s.max() <= 0:
            return ResponseTrack([], score=0.0)
        pad = np.pad(s, 2, mode='edge')
        sm = np.median(np.stack([pad[i:i + n] for i in range(5)]), 0)
        peaks = [i for i in range(n) if sm[i] > 0 and sm[i] >= sm[max(0, i - 1)] and sm[i] >= sm[min(n - 1, i + 1)]]
        if not peaks:
            return ResponseTrack([], score=0.0)
        thr = 0.8 * max(sm[p] for p in peaks)
        end = max(p for p in peaks if sm[p] >= thr)
        start = end
        while start > 0 and sm[start - 1] >= thr: start -= 1
        while end < n - 1 and sm[end + 1] >= thr: end += 1
        bboxes, last = [], None
        for fno in range(start, end + 1):
            b = self.boxes.get(fno, last)
            if b is None:
                b = next((self.boxes[k] for k in range(fno, end + 1) if k in self.boxes), None)
            if b is None:
                continue
            last = b
            x1, y1, x2, y2 = reader.to_original(b)
            bboxes.append(BBox(fno, x1, y1, x2, y2))
        return ResponseTrack(bboxes, score=float(sm[start:end + 1].max()))


def vq_metrics(records, predictions):
    """Official VQ2D metrics via the sibling VQLoC repository's evaluation code."""
    sys.path.insert(0, str(ROOT / 'VQLoC'))
    from evaluation.structures import BBox, ResponseTrack
    from metrics.metrics import compute_visual_query_metrics
    preds, gts, vcs = [], [], []
    for r in records:
        if r['key'] not in predictions or not r['response_track']:
            continue
        preds.append([predictions[r['key']]])
        gts.append(ResponseTrack([BBox(f, *b) for f, b in r['response_track']]))
        f, b = r['visual_crop']; vcs.append(BBox(f, *b))
    if not preds:
        return {}
    m = compute_visual_query_metrics(preds, gts, vcs)['all']
    pick = {'tAP25': 'Temporal AP                    @ IoU=0.25     ',
            'tAP': 'Temporal AP                    @ IoU=0.25:0.95',
            'stAP25': 'SpatioTemporal AP              @ IoU=0.25     ',
            'stAP': 'SpatioTemporal AP              @ IoU=0.25:0.95',
            'Succ': 'Success (max scr)              @ IoU=0.05     ',
            'Rec': 'Tracking % recovery (max scr)  @ IoU=0.50     '}
    out = {k: float(m[v]) for k, v in pick.items() if v in m}
    out['queries'] = len(preds)
    return out


def track_to_json(t):
    return dict(score=t.score, bboxes=[[b.fno, b.x1, b.y1, b.x2, b.y2] for b in t.bboxes])   # floats kept


def track_from_json(d):
    from evaluation.structures import BBox, ResponseTrack
    return ResponseTrack([BBox(*b) for b in d['bboxes']], score=d['score'])


@torch.no_grad()
def run_inference(model, records, sam, dev, args, log, batch=16, parts_dir=None):
    """parts_dir: if given, each clip's predictions are saved there as
    <clip_uid>.json when the clip finishes, and clips already saved are loaded
    instead of recomputed (resumable evaluation)."""
    sys.path.insert(0, str(ROOT / 'VQLoC'))
    model.eval()
    by_clip = {}
    for r in records:
        by_clip.setdefault(r['clip_uid'], []).append(r)
    preds = {}
    for ci, (clip_uid, recs) in enumerate(by_clip.items()):
        part = parts_dir / f'{clip_uid}.json' if parts_dir is not None else None
        if part is not None and part.exists():
            saved = json.loads(part.read_text(encoding='utf-8'))
            preds.update({k: track_from_json(v) for k, v in saved.items()})
            continue
        reader = D.ClipReader(clip_uid, recs[0]['orig_size'])
        q_imgs = dict(reader.frames({r['visual_crop'][0] for r in recs}))
        states = []
        for r in recs:
            f0, box = r['visual_crop']
            if f0 not in q_imgs:
                log(f'  {r["key"]}: visual-crop frame not decodable; skipped'); continue
            ibox = reader.to_input(box)
            mask = sam(q_imgs[f0], ibox) if sam is not None else D.box_to_mask(ibox)
            aug = augment_query(q_imgs[f0], mask, ibox)
            imgs = torch.from_numpy(np.stack([a[0] for a in aug])).to(dev)
            masks = torch.from_numpy(np.stack([a[1] for a in aug])).float()[:, None].to(dev)
            with torch.autocast(dev.type, dtype=torch.bfloat16, enabled=dev.type == 'cuda'):
                feats = model.features(imgs)
            feats = {k: v.float() for k, v in feats.items()}
            states.append(QueryState(model, r, feats, masks, [a[2] for a in aug], dev, args))
        stop = max((s.r['query_frame'] for s in states), default=0)
        buf = []
        def flush():
            imgs = torch.from_numpy(np.stack([b[1] for b in buf])).to(dev)
            with torch.autocast(dev.type, dtype=torch.bfloat16, enabled=dev.type == 'cuda'):
                feats = model.features(imgs)
            feats = {k: v.float() for k, v in feats.items()}
            for j, (fno, _) in enumerate(buf):
                fj = {k: v[j:j + 1] for k, v in feats.items()}
                for s in states:
                    if fno < s.r['query_frame']:
                        s.step(fno, fj)
            buf.clear()
        for fno, img in reader.all_frames(stop):
            buf.append((fno, img))
            if len(buf) == batch: flush()
        if buf: flush()
        done = {s.r['key']: s.response_track(reader) for s in states}
        preds.update(done)
        reader.close()
        if part is not None:
            write_json(part, {k: track_to_json(v) for k, v in done.items()})
        log(f'  [{ci + 1}/{len(by_clip)}] clip {clip_uid}: {len(states)} queries, {stop} frames searched')
    return preds
