"""Tune EAGLE's response-track post-processing offline, from the per-frame scores saved during evaluation.

Evaluation (run_eagle.py --stage evaluate) stores each query's per-frame confidences and boxes in
<run>/eval_parts/<clip>.json ('trace'). This script rebuilds the response tracks under a grid of
post-processing settings without re-running the model, picks the best setting on a TUNING range of
clips, and reports it, next to the paper's default rule, on a separate REPORT range. Clip ranges
index the val clips in annotation order (the order evaluation uses), so the default report range
0:351 is the subset used in the milestone's comparison and the default tuning range is all other clips.

  python eagle_postproc.py output/eagle-<run>
  python eagle_postproc.py output/eagle-<run> --tune 351:551 --report 0:351 --window 5,9,15 --gap 0,10

Settings (see eagle/infer.py track_from_trace): window = median filter length; peak = threshold as a
fraction of the max peak; extend = keep extending while above this fraction (hysteresis, <= peak);
gap = bridge dips of up to this many frames; pick = last or best peak.
To apply a chosen setting to a new evaluation: run_eagle.py --stage evaluate --pp-window ... etc.

Writes <run>/postproc/sweep-<YYYYMMDD-HHMM>.json.
"""
import argparse
import contextlib
import io
import itertools
import json
import sys
import time
from pathlib import Path

from common import write_json
from experiment_paths import ROOT

sys.path.insert(0, str(ROOT / 'VQLoC'))
from eagle import data as D                                         # noqa: E402
from eagle.infer import POSTPROC_DEFAULTS, track_from_trace, track_to_json  # noqa: E402
from eagle_snapshot import score_raw                                # noqa: E402


def clip_range(s, order):
    a, b = (int(x) if x else None for x in s.split(':'))
    return order[a:b]


def floats(s): return [float(x) for x in s.split(',')]
def ints(s): return [int(x) for x in s.split(',')]


def rebuild(traces, params):
    return {k: track_to_json(track_from_trace(t, **params)) for k, t in traces.items()}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('run', help='run folder whose eval_parts hold per-frame traces')
    ap.add_argument('--tune', default='351:', help='clip range to choose the setting on (default 351:)')
    ap.add_argument('--report', default='0:351', help='clip range to report on (default 0:351)')
    ap.add_argument('--metric', default='stAP25', help='metric to maximise on the tuning range')
    ap.add_argument('--window', type=ints, default=[5, 9, 15])
    ap.add_argument('--peak', type=floats, default=[0.6, 0.7, 0.8])
    ap.add_argument('--extend', type=floats, default=[1.0, 0.75, 0.5],
                    help='extension threshold relative to --peak (1.0 = no hysteresis)')
    ap.add_argument('--gap', type=ints, default=[0, 5, 15])
    ap.add_argument('--pick', type=lambda s: s.split(','), default=['last', 'best'])
    a = ap.parse_args()

    run = Path(a.run).resolve(); parts = run / 'eval_parts'
    meta = json.loads((parts / 'meta.json').read_text(encoding='utf-8'))
    recs = [r for r in D.load_vq(meta.get('split', 'val')) if r['response_track']]
    order = list(dict.fromkeys(r['clip_uid'] for r in recs))

    def load(clips):
        traces, stored, missing = {}, {}, 0
        for c in clips:
            p = parts / f'{c}.json'
            if not p.exists():
                missing += 1; continue
            for k, v in json.loads(p.read_text(encoding='utf-8')).items():
                if 'trace' not in v:
                    raise SystemExit(f'{p.name} has no per-frame trace: it was evaluated before traces were '
                                     'saved. Re-run the evaluation into a new run folder.')
                traces[k], stored[k] = v['trace'], v
        return traces, stored, missing

    tune, stored_t, miss_t = load(clip_range(a.tune, order))
    report, stored_r, miss_r = load(clip_range(a.report, order))
    if not tune or not report:
        raise SystemExit('No evaluated clips with traces in the tuning or report range.')
    if set(tune) & set(report):
        print('WARNING: tuning and report ranges overlap; the reported improvement is optimistic.')
    print(f'tuning: {len(tune)} queries ({miss_t} clips not evaluated) | report: {len(report)} queries ({miss_r} not evaluated)')

    # Sanity check: replaying the setting used online must reproduce the stored tracks.
    online = {**POSTPROC_DEFAULTS, **meta.get('postproc', {})}
    replay = rebuild(report, online)
    bad = sum(1 for k in report if replay[k]['bboxes'] != stored_r[k]['bboxes'])
    print(f'replay check: {bad} of {len(report)} stored tracks differ from the offline rebuild')

    def quiet_score(raw, recs_):
        with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
            return score_raw(raw, recs_)

    grid = []
    for w, pk, ex, g, pi in itertools.product(a.window, a.peak, a.extend, a.gap, a.pick):
        grid.append(dict(window=w, peak=pk, extend=None if ex == 1.0 else round(pk * ex, 4), gap=g, pick=pi))
    if POSTPROC_DEFAULTS not in grid:
        grid.insert(0, dict(POSTPROC_DEFAULTS))
    results, t0 = [], time.time()
    for i, params in enumerate(grid):
        m = quiet_score(rebuild(tune, params), recs)
        results.append(dict(params=params, tune=m))
        print(f'[{i + 1}/{len(grid)}] {params} -> {a.metric} {m[a.metric]:.4f} '
              f'Rec {m["Rec"]:.2f} Succ {m["Succ"]:.2f} ({time.time() - t0:.0f}s)', flush=True)
    results.sort(key=lambda r: -r['tune'][a.metric])
    best = results[0]['params']
    out = dict(written=time.strftime('%Y-%m-%d %H:%M'), run=run.name, metric=a.metric,
               tune_range=a.tune, tune_queries=len(tune), report_range=a.report, report_queries=len(report),
               best_params=best,
               report_default=quiet_score(rebuild(report, POSTPROC_DEFAULTS), recs),
               report_best=quiet_score(rebuild(report, best), recs),
               grid=results)
    dest = run / 'postproc' / f'sweep-{time.strftime("%Y%m%d-%H%M")}.json'
    dest.parent.mkdir(exist_ok=True)
    write_json(dest, out)
    keys = ['tAP25', 'stAP25', 'tAP', 'stAP', 'Succ', 'Rec', 'positive_frame_mean_iou', 'presence_f1']
    print(f'\nbest on tuning range: {best}')
    print(f'{"report range " + a.report:<24}' + ''.join(f'{k[:10]:>11}' for k in keys))
    for name in ('default', 'best'):
        m = out[f'report_{name}']
        print(f'{name:<24}' + ''.join(f'{m[k]:>11.4f}' for k in keys))
    print(f'\nSaved {dest}')


if __name__ == '__main__':
    main()
