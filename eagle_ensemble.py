"""Ensemble several evaluated EAGLE runs and apply the visual-crop constraint, offline.

Two inference-time changes, no retraining. Both work on the per-frame traces that evaluation saved
in <run>/eval_parts/:

  1. Trace ensembling. Each run's per-frame confidences are averaged (optionally weighted), and each
     frame keeps the box of the run that is most confident on that frame. The paper's
     post-processing rule (eagle/infer.py track_from_trace) then turns the averaged trace into a
     response track. Models trained differently (frozen vs fine-tuned backbone, different data
     mixes) fire on different false peaks, so averaging mostly removes wrong time windows.

  2. Visual-crop constraint. The visual crop is itself a sighting of the object, and the response
     track is the object's *last* occurrence before the query frame. So whenever the crop's frame
     comes before the query frame, the response track must come after it. Frames up to and
     including crop frame + --crop-margin get confidence 0. The crop's frame number is part of
     the VQ2D query input, so this uses no ground truth.

The result is written as a pseudo run folder that eagle_snapshot.py, eagle_postproc.py and
eagle_sam2.py accept like any evaluated run:

  python eagle_ensemble.py output/eagle-20261111-221053 output/eagle-20261114-002928 output/eagle-20261114-204835 --out output/ens3-mask0
  python eagle_sam2.py output/ens3-mask0 --clips 0: --margin 30 --tag small-m30

Writes <out>/eval_parts/<clip>.json (tracks + averaged traces), <out>/eval_parts/meta.json and
<out>/metrics.json (official VQ2D metrics of the ensemble before SAM 2).
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np

from common import write_json
from experiment_paths import ROOT

sys.path.insert(0, str(ROOT / 'VQLoC'))
from eagle import data as D                                         # noqa: E402
from eagle.infer import POSTPROC_DEFAULTS, track_from_trace, track_to_json  # noqa: E402
from eagle_snapshot import score_raw                                # noqa: E402


def fuse(traces, weights):
    """Weighted mean of per-frame scores; per frame, the box of the most confident run."""
    ss = [np.asarray(t['scores'], dtype=float) for t in traces]
    s = sum(w * x for w, x in zip(weights, ss)) / sum(weights)
    best = {}
    for t, x in zip(traces, ss):
        for b in t['boxes']:
            f = b[0]
            if f not in best or x[f] > best[f][0]:
                best[f] = (x[f], b)
    return dict(scores=[round(float(v), 6) for v in s], boxes=[best[f][1] for f in sorted(best)])


def crop_constraint(trace, record, margin):
    vc = record['visual_crop'][0]
    if margin is None or vc >= record['query_frame']:
        return trace
    s = list(trace['scores']); cut = min(len(s), vc + margin + 1)
    s[:cut] = [0.0] * cut
    return dict(scores=s, boxes=trace['boxes'])


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('runs', nargs='+', help='evaluated run folders (eval_parts with per-frame traces)')
    ap.add_argument('--out', required=True, help='pseudo run folder to write')
    ap.add_argument('--weights', type=lambda s: [float(x) for x in s.split(',')], help='one weight per run (default equal)')
    ap.add_argument('--crop-margin', default='0', help="suppress frames <= crop frame + this ('none' = off; default 0)")
    a = ap.parse_args()

    runs = [Path(r).resolve() for r in a.runs]
    weights = a.weights or [1.0] * len(runs)
    if len(weights) != len(runs):
        raise SystemExit('--weights needs one value per run')
    margin = None if a.crop_margin == 'none' else int(a.crop_margin)
    metas = [json.loads((r / 'eval_parts' / 'meta.json').read_text(encoding='utf-8')) for r in runs]
    split = metas[0].get('split', 'val')
    recs = [r for r in D.load_vq(split) if r['response_track']]
    by_clip = {}
    for r in recs:
        by_clip.setdefault(r['clip_uid'], []).append(r)

    out = Path(a.out).resolve(); parts = out / 'eval_parts'; parts.mkdir(parents=True, exist_ok=True)
    raw, skipped = {}, 0
    for clip, rs in by_clip.items():
        files = [r / 'eval_parts' / f'{clip}.json' for r in runs]
        if not all(f.exists() for f in files):
            skipped += 1; continue
        preds = [json.loads(f.read_text(encoding='utf-8')) for f in files]
        done = {}
        for rec in rs:
            k = rec['key']
            if not all(k in p for p in preds):
                continue
            if any('trace' not in p[k] for p in preds):
                raise SystemExit(f'{clip}: a run has no per-frame trace; re-evaluate it with the current code.')
            fused = fuse([p[k]['trace'] for p in preds], weights)
            track = track_to_json(track_from_trace(crop_constraint(fused, rec, margin), **POSTPROC_DEFAULTS))
            done[k] = dict(**track, trace=fused)
            raw[k] = dict(score=track['score'], bboxes=track['bboxes'])
        write_json(parts / f'{clip}.json', done)
    write_json(parts / 'meta.json', dict(split=split, ensemble=[r.name for r in runs], weights=weights,
                                         crop_margin=margin, queries=len(raw)))
    m = score_raw(raw, recs)
    write_json(out / 'metrics.json', m)
    print(f'{len(raw)} queries ({skipped} clips missing in some run) -> {out}')
    print(' '.join(f'{k} {m[k]:.4f}' for k in ('stAP25', 'tAP25', 'stAP', 'tAP', 'Succ', 'Rec')))


if __name__ == '__main__':
    main()
