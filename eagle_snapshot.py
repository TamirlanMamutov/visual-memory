"""Metrics snapshot of an EAGLE evaluation, now or as it stood at an earlier time.

Each clip's predictions in <run>/eval_parts/ carry the time they were written, so the snapshot at
time T is reproduced exactly by scoring only the clips finished by T. Reports the official VQ2D
metrics plus frame-level positive-frame mean IoU and presence F1 (computed over ground-truth
response-track frames; not official VQ2D metrics), for all finished clips and, once they are done,
for the first --first-n val clips in annotation order.

  python eagle_snapshot.py output/eagle-20261110-084401
  python eagle_snapshot.py output/eagle-20261110-084401 --at "2026-11-10 18:56"
  python eagle_snapshot.py output/eagle-20261110-084401 --at 18:56          # today

Writes <run>/snapshots/metrics-<YYYYMMDD-HHMM>.json (time of the snapshot) and prints it.
"""
import argparse
import contextlib
import io
import json
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np

from common import write_json
from experiment_paths import ROOT

sys.path.insert(0, str(ROOT / 'VQLoC'))
from eagle import data as D                       # noqa: E402
from eagle.infer import track_from_json, vq_metrics  # noqa: E402


def parse_time(s):
    for fmt in ('%Y-%m-%d %H:%M', '%Y-%m-%d %H:%M:%S', '%H:%M'):
        try:
            t = datetime.strptime(s, fmt)
            if fmt == '%H:%M':
                t = datetime.now().replace(hour=t.hour, minute=t.minute, second=0, microsecond=0)
            return t
        except ValueError:
            pass
    raise SystemExit(f'--at: use "YYYY-MM-DD HH:MM" or "HH:MM", not {s!r}')


def box_iou(a, b):
    ix = max(0, min(a[2], b[2]) - max(a[0], b[0])); iy = max(0, min(a[3], b[3]) - max(a[1], b[1]))
    u = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - ix * iy
    return ix * iy / u if u > 0 else 0.0


def score(parts, clips, recs):
    raw = {}
    for c in clips:
        raw.update(json.loads((parts / f'{c}.json').read_text(encoding='utf-8')))
    rs = [r for r in recs if r['key'] in raw]
    if not rs:
        return None
    with contextlib.redirect_stdout(io.StringIO()):
        m = vq_metrics(rs, {k: track_from_json(v) for k, v in raw.items()})
    ious, tp, fp, fn = [], 0, 0, 0
    for r in rs:
        gt = {f: b for f, b in r['response_track']}
        pr = {b[0]: b[1:] for b in raw[r['key']]['bboxes']}
        ious += [box_iou(gt[f], pr[f]) if f in pr else 0.0 for f in gt]
        tp += len(gt.keys() & pr.keys()); fp += len(pr.keys() - gt.keys()); fn += len(gt.keys() - pr.keys())
    p, rc = tp / max(1, tp + fp), tp / max(1, tp + fn)
    m.update(clips=len(clips), empty_predictions=sum(1 for k in raw if not raw[k]['bboxes']),
             positive_frame_mean_iou=float(np.mean(ious)), presence_precision=p, presence_recall=rc,
             presence_f1=2 * p * rc / max(1e-12, p + rc))
    return m


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('run', help='run folder, e.g. output/eagle-20261110-084401')
    ap.add_argument('--at', help='snapshot as of this time: "YYYY-MM-DD HH:MM" or "HH:MM" (today); default now')
    ap.add_argument('--first-n', type=int, default=351, help='also score the first N val clips (0 = skip)')
    a = ap.parse_args()
    run = Path(a.run).resolve(); parts = run / 'eval_parts'
    if not parts.is_dir():
        raise SystemExit(f'{parts} not found')
    meta = json.loads((parts / 'meta.json').read_text(encoding='utf-8'))
    at = parse_time(a.at) if a.at else datetime.now()
    cutoff = at.timestamp()

    recs = [r for r in D.load_vq(meta.get('split', 'val')) if r['response_track']]
    order = list(dict.fromkeys(r['clip_uid'] for r in recs))
    done = [c for c in order if (parts / f'{c}.json').exists() and (parts / f'{c}.json').stat().st_mtime <= cutoff]
    if not done:
        raise SystemExit(f'No clips had finished by {at:%Y-%m-%d %H:%M}.')
    last = max((parts / f'{c}.json').stat().st_mtime for c in done)
    first = order[:a.first_n] if a.first_n else []

    out = dict(snapshot_at=f'{at:%Y-%m-%d %H:%M}', last_clip_finished=time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(last)),
               split=meta.get('split'), checkpoint=meta.get('checkpoint'),
               total_clips=len(order), total_queries=len(recs),
               note='Frame-level metrics use ground-truth response-track frames; they are not official VQ2D metrics.',
               all_finished_clips=score(parts, done, recs))
    if first:
        out[f'first_{a.first_n}_clips'] = score(parts, first, recs) if set(first) <= set(done) else \
            f'not available: only {len(set(first) & set(done))} of these {a.first_n} clips had finished'
    dest = run / 'snapshots' / f'metrics-{at:%Y%m%d-%H%M}.json'
    dest.parent.mkdir(exist_ok=True)
    write_json(dest, out)
    print(json.dumps(out, indent=2))
    print(f'\nSaved {dest}')


if __name__ == '__main__':
    main()
