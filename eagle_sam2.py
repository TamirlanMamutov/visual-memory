"""Refine EAGLE's response-track boxes with SAM 2 video tracking (idea from REN / RELOCATE).

EAGLE finds *when* the object was last seen about as well as the paper (tAP25), but its per-frame
boxes are weaker (stAP25, Recovery). This script keeps EAGLE's predicted time window and lets SAM 2
redraw the boxes: SAM 2 is prompted with EAGLE's box on the track's most confident frame (the
anchor) and propagates the object forward and backward. Nothing is retrained; it runs on the
predictions an evaluation already saved in <run>/eval_parts/.

For each query SAM 2 tracks over EAGLE's window widened by --margin frames on each side, and the
raw SAM 2 boxes are saved per clip (resumable). Three variants are then built and scored next to
EAGLE's own tracks, on the same queries:
  replace - EAGLE's frames, SAM 2's box where SAM 2 sees the object (EAGLE's box elsewhere)
  trim    - replace, then drop end frames where SAM 2 lost the object
  extend  - replace, then grow each end into the margin while SAM 2 still sees the object

  python eagle_sam2.py output/eagle-<run> --clips 0:20
  python eagle_sam2.py output/eagle-<run> --clips 0:351 --model facebook/sam2.1-hiera-small --margin 10

Writes <run>/sam2/<tag>/<clip>.json (raw SAM 2 boxes) and <run>/sam2/<tag>-summary.json.
"""
import argparse
import contextlib
import io
import json
import sys
import time
from pathlib import Path

import numpy as np

from common import device, write_json
from experiment_paths import ROOT

sys.path.insert(0, str(ROOT / 'VQLoC'))
from eagle import data as D                                         # noqa: E402
from eagle_snapshot import score_raw                                # noqa: E402

LONG_SIDE = 1024          # frames are downscaled to this long side before SAM 2 (its own input size)


class FullResReader(D.ClipReader):
    """ClipReader that yields RGB frames at up to LONG_SIDE pixels (not EAGLE's 448 letterbox)."""

    def frames(self, wanted):
        import cv2
        wanted = sorted(set(int(f) for f in wanted))
        if not wanted:
            return
        w, h = self.frame_size
        s = min(1.0, LONG_SIDE / max(w, h))
        size = (round(w * s), round(h * s))
        self.sam_scale = s
        i = 0
        self.container.seek(int(max(0, wanted[0] / D.ANN_FPS - 1.0) / self.tb), stream=self.stream, backward=True)
        for frame in self.container.decode(self.stream):
            if frame.pts is None:
                continue
            t = frame.pts * self.tb
            while i < len(wanted) and wanted[i] / D.ANN_FPS <= t + 0.5 / 30:
                img = frame.to_ndarray(format='rgb24')
                yield wanted[i], (cv2.resize(img, size, interpolation=cv2.INTER_AREA) if s < 1 else img)
                i += 1
            if i >= len(wanted):
                return


class Sam2Tracker:
    def __init__(self, dev, model):
        import torch
        from transformers import Sam2VideoModel, Sam2VideoProcessor
        self.torch, self.dev = torch, dev
        self.dtype = torch.bfloat16 if dev.type == 'cuda' else torch.float32
        self.processor = Sam2VideoProcessor.from_pretrained(model)
        self.model = Sam2VideoModel.from_pretrained(model).to(dev, dtype=self.dtype).eval()

    def track(self, frames, anchor, box):
        """frames: list of RGB arrays; anchor: index of the prompted frame; box: [x1, y1, x2, y2] in
        frame pixels. Returns one box (frame pixels) or None (object not visible) per frame."""
        torch = self.torch
        with torch.no_grad():
            sess = self.processor.init_video_session(video=frames, inference_device=self.dev, dtype=self.dtype)
            self.processor.add_inputs_to_inference_session(inference_session=sess, frame_idx=anchor, obj_ids=1,
                                                           input_boxes=[[[float(v) for v in box]]])
            size = [[sess.video_height, sess.video_width]]
            out = [None] * len(frames)

            def keep(o):
                m = self.processor.post_process_masks([o.pred_masks], original_sizes=size, binarize=True)[0]
                m = m[0, 0].cpu().numpy() if m.ndim == 4 else m[0].cpu().numpy()
                ys, xs = np.nonzero(m)
                out[o.frame_idx] = None if len(xs) == 0 else [float(xs.min()), float(ys.min()), float(xs.max() + 1), float(ys.max() + 1)]

            keep(self.model(inference_session=sess, frame_idx=anchor))
            for rev in (False, True):
                for o in self.model.propagate_in_video_iterator(sess, start_frame_idx=anchor, reverse=rev):
                    keep(o)
        return out


def refine_clip(tracker, clip_uid, preds, recs, margin, max_frames):
    """SAM 2 boxes (annotation pixels) for every query of one clip: {key: {anchor, boxes: {fno: box|None}}}."""
    rec = {r['key']: r for r in recs}
    reader = FullResReader(clip_uid, recs[0]['orig_size'])
    to_sam = (reader.frame_size[0] / recs[0]['orig_size'][0])          # annotation px -> decoded px
    out = {}
    try:
        for key, p in preds.items():
            track = p['bboxes']
            if not track:
                out[key] = dict(anchor=None, boxes={}); continue
            scores = p['trace']['scores']
            fnos = [b[0] for b in track]
            anchor_f = max(fnos, key=lambda f: scores[f] if f < len(scores) else 0.0)
            lo = max(0, fnos[0] - margin); hi = min(rec[key]['query_frame'] - 1, fnos[-1] + margin)
            if hi - lo + 1 > max_frames:                               # very long track: keep the window round the anchor
                lo = max(lo, anchor_f - max_frames // 2); hi = min(hi, lo + max_frames - 1)
            got = dict(reader.frames(range(lo, hi + 1)))
            window = [f for f in range(lo, hi + 1) if f in got]
            if anchor_f not in got:
                out[key] = dict(anchor=anchor_f, boxes={}, note='anchor frame not decodable'); continue
            k = to_sam * reader.sam_scale
            abox = next(b[1:] for b in track if b[0] == anchor_f)
            boxes = tracker.track([got[f] for f in window], window.index(anchor_f), [v * k for v in abox])
            out[key] = dict(anchor=anchor_f, boxes={str(f): (None if b is None else [round(v / k, 2) for v in b])
                                                   for f, b in zip(window, boxes)})
    finally:
        reader.close()
    return out


def variants(pred, sam):
    """Build replace / trim / extend tracks (bboxes lists) from EAGLE's track and SAM 2's boxes."""
    track, boxes = pred['bboxes'], {int(f): b for f, b in sam.get('boxes', {}).items()}
    if not track or not boxes:
        return {v: track for v in ('replace', 'trim', 'extend')}
    rep = [[b[0]] + (boxes[b[0]] if boxes.get(b[0]) else b[1:]) for b in track]
    seen = [b[0] for b in track if boxes.get(b[0])]
    trim = [b for b in rep if seen and seen[0] <= b[0] <= seen[-1]] or rep
    ext = list(rep)
    f = track[0][0] - 1
    while boxes.get(f):
        ext.insert(0, [f] + boxes[f]); f -= 1
    f = track[-1][0] + 1
    while boxes.get(f):
        ext.append([f] + boxes[f]); f += 1
    return dict(replace=rep, trim=trim, extend=ext)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('run', help='evaluated run folder (eval_parts with per-frame traces)')
    ap.add_argument('--clips', default='0:20', help='val clips A:B in annotation order (default 0:20)')
    ap.add_argument('--model', default='facebook/sam2.1-hiera-small', help='SAM 2 checkpoint on Hugging Face')
    ap.add_argument('--margin', type=int, default=10, help='frames added on each side of EAGLE\'s window for SAM 2')
    ap.add_argument('--max-frames', type=int, default=300, help='cap on frames tracked per query')
    ap.add_argument('--tag', help='output subfolder name (default: model name + margin)')
    a = ap.parse_args()

    run = Path(a.run).resolve(); parts = run / 'eval_parts'
    meta = json.loads((parts / 'meta.json').read_text(encoding='utf-8'))
    recs = [r for r in D.load_vq(meta.get('split', 'val')) if r['response_track']]
    order = list(dict.fromkeys(r['clip_uid'] for r in recs))
    s, e = (int(x) if x else None for x in a.clips.split(':'))
    clips = [c for c in order[s:e] if (parts / f'{c}.json').exists()]
    tag = a.tag or f'{a.model.split("/")[-1]}-m{a.margin}'
    dest = run / 'sam2' / tag; dest.mkdir(parents=True, exist_ok=True)
    by_clip = {}
    for r in recs: by_clip.setdefault(r['clip_uid'], []).append(r)

    tracker, t0, done = None, time.time(), 0
    for i, c in enumerate(clips):
        part = dest / f'{c}.json'
        if part.exists():
            continue                                                   # resumable: clip already refined
        tracker = tracker or Sam2Tracker(device(), a.model)
        preds = json.loads((parts / f'{c}.json').read_text(encoding='utf-8'))
        write_json(part, refine_clip(tracker, c, preds, by_clip[c], a.margin, a.max_frames))
        done += 1
        print(f'[{i + 1}/{len(clips)}] {c}: {len(preds)} queries ({time.time() - t0:.0f}s, '
              f'{(time.time() - t0) / done:.1f}s/clip)', flush=True)

    base, var = {}, {'replace': {}, 'trim': {}, 'extend': {}}
    for c in clips:
        preds = json.loads((parts / f'{c}.json').read_text(encoding='utf-8'))
        sam = json.loads((dest / f'{c}.json').read_text(encoding='utf-8'))
        for k, p in preds.items():
            base[k] = dict(score=p['score'], bboxes=p['bboxes'])
            for v, bb in variants(p, sam.get(k, {})).items():
                var[v][k] = dict(score=p['score'], bboxes=bb)

    def quiet(raw):
        with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
            return score_raw(raw, recs)
    res = dict(eagle=quiet(base), **{v: quiet(r) for v, r in var.items()})
    summary = dict(written=time.strftime('%Y-%m-%d %H:%M'), run=run.name, clips=a.clips, n_clips=len(clips),
                   queries=len(base), model=a.model, margin=a.margin, results=res)
    write_json(run / 'sam2' / f'{tag}-summary.json', summary)
    keys = ['stAP25', 'tAP25', 'stAP', 'tAP', 'Succ', 'Rec', 'positive_frame_mean_iou', 'presence_f1']
    print(f'\n{len(clips)} clips, {len(base)} queries, {a.model}, margin {a.margin}')
    print(f'{"":<10}' + ''.join(f'{k[:10]:>11}' for k in keys))
    for name, m in res.items():
        print(f'{name:<10}' + ''.join(f'{m[k]:>11.4f}' for k in keys))
    print(f'\nSaved {run / "sam2" / (tag + "-summary.json")}')


if __name__ == '__main__':
    main()
