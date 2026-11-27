"""Rescore finished EAGLE evaluations with the official VQ2D setting (every query).

Runs evaluated before commit ae35b68 have metrics.json scored with VQLoC's default
large-only filter (only queries whose ground-truth box reaches 192x192 px). This
rescores the saved predictions.json with the official all-queries pairing and writes
metrics_official.json next to it; metrics.json is left as it was. The launcher's
results table shows metrics_official.json when it exists.

  python eagle_rescore.py                      every output/eagle-* run with predictions.json
  python eagle_rescore.py output/eagle-<id> ...
"""
import argparse
import contextlib
import io
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from common import ROOT, write_json  # noqa: E402


def rescore(run, records_by_split):
    sys.path.insert(0, str(ROOT / 'VQLoC'))
    from evaluation.structures import BBox, ResponseTrack
    import eagle.data as D
    from eagle.infer import vq_metrics, OFFICIAL_PAIRING
    old = json.loads((run / 'metrics.json').read_text(encoding='utf-8')) if (run / 'metrics.json').exists() else {}
    split = old.get('split') or json.loads((run / 'config.json').read_text(encoding='utf-8')).get('eval_split', 'val')
    raw = json.loads((run / 'predictions.json').read_text(encoding='utf-8'))
    preds = {k: ResponseTrack([BBox(b['fno'], b['x1'], b['y1'], b['x2'], b['y2']) for b in v if 'fno' in b],
                              score=next((b['score'] for b in v if 'score' in b), 0.0)) for k, v in raw.items()}
    source = 'predictions.json'
    # eval_parts/ keeps the unrounded boxes (predictions.json rounds to whole pixels); use it when it covers every query
    parts = {}
    for p in (run / 'eval_parts').glob('*.json'):
        if p.name != 'meta.json':
            parts.update(json.loads(p.read_text(encoding='utf-8')))
    if parts.keys() >= preds.keys():
        preds = {k: ResponseTrack([BBox(*b) for b in parts[k]['bboxes']], score=parts[k]['score']) for k in preds}
        source = 'eval_parts'
    if split not in records_by_split:
        records_by_split[split] = D.load_vq(split, local_only=False)
    records = [r for r in records_by_split[split] if r['key'] in preds and r['response_track']]
    with contextlib.redirect_stdout(io.StringIO()):     # VQLoC prints a count and "empty track" notes
        m = vq_metrics(records, preds, pairing=OFFICIAL_PAIRING)
    out = dict(final=m, split=split, pairing='all', checkpoint=old.get('checkpoint'), source=source,
               note=f'Rescored from {source} by eagle_rescore.py with the official all-queries setting; '
                    'metrics.json holds the original score.')
    write_json(run / 'metrics_official.json', out)
    return m, source


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('runs', nargs='*', type=Path)
    a = p.parse_args()
    runs = a.runs or sorted(r for r in (HERE / 'output').glob('eagle-*') if (r / 'predictions.json').exists())
    cache = {}
    for run in runs:
        if not (run / 'predictions.json').exists():
            print(f'{run.name}: no predictions.json, skipped'); continue
        m, source = rescore(run, cache)
        print(f"{run.name}: tAP25 {m['tAP25']:.3f}  stAP25 {m['stAP25']:.3f}  tAP {m['tAP']:.3f}  stAP {m['stAP']:.3f}  "
              f"Succ {m['Succ']:.2f}  Rec {m['Rec']:.2f}  ({m['queries']} queries, from {source})", flush=True)


if __name__ == '__main__':
    main()
