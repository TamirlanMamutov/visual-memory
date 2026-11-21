"""Score several VQ2D prediction sets on exactly the same queries, with the official evaluator.

Use it to compare our pipeline with a teammate's (for example REN + box refinement) on their
subset: each prediction set is scored only on the queries that every set (and --queries, if given)
covers, so the numbers are head to head. A paired bootstrap over queries gives a confidence
interval for each set's stAP25 / tAP25 difference from the first set.

  python vq2d_compare.py --pred ours=run6 --pred ren=path\\to\\ren.json --pred ren_refine=path\\to\\refined.json
  python vq2d_compare.py --pred ours=run6 --pred ren=ren.json --queries subset_keys.txt

--pred NAME=SOURCE, where SOURCE is one of
  run6                      our final method: output/ens3-mask0 + SAM 2 extend (sam2/small-m30)
  <run folder>              an evaluated run (its eval_parts/), e.g. output/eagle-20261111-221053
  <file>.json               either our flat format  {query key: {"score": s, "bboxes": [[frame, x1, y1, x2, y2], ...]}}
                            or the VQ2D challenge format
                            {"results": {"videos": [{"clips": [{"clip_uid": ..., "predictions": [   # one per annotation
                                {"query_sets": {"1": {"bboxes": [{"fno", "x1", "y1", "x2", "y2"}, ...], "score": s}}}]}]}]}}
Query keys are <clip_uid>_<annotation index>_<query set id>, as in eagle/data.py load_vq.
Boxes are in original video pixels (the annotation's original_width x original_height).
--queries: a .txt (one key per line) or .json list of keys, or a clip range A:B in val annotation order.

Writes nothing; prints the table (and --json PATH saves it).
"""
import argparse
import json
import random
import sys
from pathlib import Path

from common import write_json
from experiment_paths import ROOT

sys.path.insert(0, str(ROOT / 'VQLoC'))
from eagle import data as D                                         # noqa: E402
from eagle_snapshot import score_raw                                # noqa: E402

HERE = Path(__file__).resolve().parent
KEYS = ['stAP25', 'tAP25', 'stAP', 'tAP', 'Succ', 'Rec']


def load_parts(run):
    raw = {}
    for f in (Path(run) / 'eval_parts').glob('*.json'):
        if f.name != 'meta.json':
            raw.update({k: dict(score=v['score'], bboxes=v['bboxes']) for k, v in json.loads(f.read_text(encoding='utf-8')).items()})
    return raw


def load_run6():
    from eagle_sam2 import variants
    run = HERE / 'output' / 'ens3-mask0'; sam = run / 'sam2' / 'small-m30'
    raw = {}
    for f in (run / 'eval_parts').glob('*.json'):
        if f.name == 'meta.json':
            continue
        preds = json.loads(f.read_text(encoding='utf-8'))
        s = json.loads((sam / f.name).read_text(encoding='utf-8')) if (sam / f.name).exists() else {}
        for k, p in preds.items():
            raw[k] = dict(score=p['score'], bboxes=variants(p, s.get(k, {}))['extend'])
    return raw


def load_json(path):
    d = json.loads(Path(path).read_text(encoding='utf-8'))
    if 'results' not in d:                                           # flat {key: {score, bboxes}}
        return {k: dict(score=float(v['score']), bboxes=[list(b) for b in v['bboxes']]) for k, v in d.items()}
    raw = {}
    for v in d['results']['videos']:
        for c in v['clips']:
            for ai, ann in enumerate(c['predictions']):
                for qid, q in ann['query_sets'].items():
                    bb = [[b['fno'], b['x1'], b['y1'], b['x2'], b['y2']] for b in q['bboxes']]
                    raw[f"{c['clip_uid']}_{ai}_{qid}"] = dict(score=float(q['score']), bboxes=bb)
    return raw


def load_source(src):
    if src == 'run6':
        return load_run6()
    p = Path(src)
    return load_parts(p) if p.is_dir() else load_json(p)


def subset(spec, order, recs):
    if spec is None:
        return None
    if ':' in spec and not Path(spec).exists():
        a, b = (int(x) if x else None for x in spec.split(':'))
        clips = set(order[a:b]); return {r['key'] for r in recs if r['clip_uid'] in clips}
    text = Path(spec).read_text(encoding='utf-8')
    return set(json.loads(text)) if spec.endswith('.json') else {l.strip() for l in text.splitlines() if l.strip()}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--pred', action='append', required=True, metavar='NAME=SOURCE')
    ap.add_argument('--queries', help='restrict to these query keys (.txt/.json) or clip range A:B')
    ap.add_argument('--boot', type=int, default=500, help='paired bootstrap resamples (0 = off)')
    ap.add_argument('--json', help='also save the results here')
    a = ap.parse_args()

    recs = [r for r in D.load_vq('val') if r['response_track']]
    order = list(dict.fromkeys(r['clip_uid'] for r in recs))
    known = {r['key'] for r in recs}
    sets = {}
    for spec in a.pred:
        name, src = spec.split('=', 1)
        sets[name] = load_source(src)
        unknown = len(set(sets[name]) - known)
        print(f'{name}: {len(sets[name])} queries from {src}' + (f' ({unknown} keys not in val, ignored)' if unknown else ''))
    common = known.intersection(*map(set, sets.values()))
    want = subset(a.queries, order, recs)
    if want is not None:
        miss = {n: len(want - set(s)) for n, s in sets.items() if want - set(s)}
        if miss:
            print(f'WARNING: queries in --queries missing from a set (scored only where all have them): {miss}')
        common &= want
    keys = sorted(common)
    if not keys:
        raise SystemExit('No query is covered by every prediction set.')
    print(f'\nScoring {len(keys)} common queries from {len({k.rsplit("_", 2)[0] for k in keys})} clips\n')

    def score(name, ks):
        import contextlib, io
        with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
            return score_raw({k: sets[name][k] for k in ks}, recs)

    res = {n: score(n, keys) for n in sets}
    print(f'{"":<14}' + ''.join(f'{k:>9}' for k in KEYS))
    for n, m in res.items():
        print(f'{n:<14}' + ''.join(f'{m[k]:>9.4f}' if k not in ('Succ', 'Rec') else f'{m[k]:>9.2f}' for k in KEYS))

    out = dict(queries=len(keys), metrics=res)
    names = list(sets)
    if a.boot and len(names) > 1:
        rng = random.Random(230); base = names[0]; boot = {n: {'stAP25': [], 'tAP25': []} for n in names[1:]}
        for _ in range(a.boot):
            ks = [keys[rng.randrange(len(keys))] for _ in keys]
            mb = score_dupes(sets, base, ks, recs)
            for n in names[1:]:
                mn = score_dupes(sets, n, ks, recs)
                for k in boot[n]:
                    boot[n][k].append(mn[k] - mb[k])
        print(f'\nPaired bootstrap ({a.boot} resamples) of the difference from {base}, 95% interval:')
        out['bootstrap'] = {}
        for n, d in boot.items():
            line = []
            for k, v in d.items():
                v = sorted(v); lo, hi = v[int(0.025 * len(v))], v[int(0.975 * len(v)) - 1]
                line.append(f'{k} {res[n][k] - res[base][k]:+.4f} [{lo:+.4f}, {hi:+.4f}]')
                out['bootstrap'].setdefault(n, {})[k] = dict(diff=res[n][k] - res[base][k], lo=lo, hi=hi)
            print(f'  {n:<12} ' + '   '.join(line))
    if a.json:
        write_json(Path(a.json), out)


def score_dupes(sets, name, ks, recs):
    """Score a bootstrap sample that may repeat queries: give each copy its own key."""
    import contextlib, io
    rec = {r['key']: r for r in recs}
    raw, rs = {}, []
    for i, k in enumerate(ks):
        nk = f'{k}#{i}'; raw[nk] = sets[name][k]; rs.append(dict(rec[k], key=nk))
    with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
        return score_raw(raw, rs)


if __name__ == '__main__':
    main()
