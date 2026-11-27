"""EAGLE evaluation demo: browse a finished VQ2D evaluation query by query.

For one query it draws the visual crop, the per-frame score timeline with the
ground-truth and predicted response tracks, and frames with both boxes. It only
reads saved predictions (eval_parts/ or predictions.json) and decodes a few
frames, so it needs no GPU or model.

  python eagle_demo.py --run output/eagle-<id> --index            per-query table (JSON)
  python eagle_demo.py --run output/eagle-<id> --key <key> --out q.png
  python eagle_demo.py --serve        JSON requests on stdin (used by the launcher)

Per-query numbers follow the official VQ2D definitions in ../VQLoC/metrics:
tIoU = IoU of the predicted and ground-truth frame spans; stIoU = tube IoU
(shared box area / both track volumes minus it); success = stIoU >= 0.05, so the
success rate over all queries equals the official Success @ 0.05. AP metrics are
dataset-level and stay in the run's metrics.json.
"""
import argparse
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
DEMO_OUT = HERE / 'output' / 'eval-demo'

GT_COLOR, PRED_COLOR, CROP_COLOR = (40, 200, 90), (235, 64, 52), (250, 200, 30)


# ------------------------------------------------------------- data ---------

def run_split(run):
    for name in ('metrics.json', 'config.json'):
        p = run / name
        if p.exists():
            d = json.loads(p.read_text(encoding='utf-8'))
            if d.get('split') or d.get('eval_split'):
                return d.get('split') or d.get('eval_split')
    return 'val'


def load_predictions(run):
    """{key: dict(score, bboxes=[[fno, x1, y1, x2, y2], ...], trace or None)} in annotation pixels."""
    out = {}
    parts = run / 'eval_parts'
    if parts.is_dir():
        for p in parts.glob('*.json'):
            if p.name != 'meta.json':
                out.update(json.loads(p.read_text(encoding='utf-8')))
    if not out and (run / 'predictions.json').exists():
        for k, v in json.loads((run / 'predictions.json').read_text(encoding='utf-8')).items():
            boxes = [[b['fno'], b['x1'], b['y1'], b['x2'], b['y2']] for b in v if 'fno' in b]
            score = next((b['score'] for b in v if 'score' in b), 0.0)
            out[k] = dict(score=score, bboxes=boxes)
    return out


_records = {}


def records(split):
    if split not in _records:
        import eagle.data as D
        _records[split] = {r['key']: r for r in D.load_vq(split, local_only=False)}
    return _records[split]


def box_inter(a, b):
    return max(0.0, min(a[2], b[2]) - max(a[0], b[0])) * max(0.0, min(a[3], b[3]) - max(a[1], b[1]))


def box_iou(a, b):
    inter = box_inter(a, b)
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / union if union > 0 else 0.0


def query_stats(rec, pred):
    gt = {f: b for f, b in rec['response_track']}
    pr = {int(b[0]): b[1:] for b in pred.get('bboxes', [])}
    row = dict(key=rec['key'], clip=rec['clip_uid'], object=rec['object_title'], query_frame=rec['query_frame'],
               gt_span=[min(gt), max(gt)] if gt else None, pred_span=[min(pr), max(pr)] if pr else None,
               score=round(float(pred.get('score', 0.0)), 3), tiou=0.0, stiou=0.0, success=False)
    if gt and pr:
        (g0, g1), (p0, p1) = row['gt_span'], row['pred_span']
        inter = max(0, min(g1, p1) - max(g0, p0) + 1)
        row['tiou'] = round(inter / ((g1 - g0 + 1) + (p1 - p0 + 1) - inter), 3)
        # tube IoU as in VQLoC metrics/utils.py: shared box area / (both track volumes - shared area)
        area = lambda b: (b[2] - b[0]) * (b[3] - b[1])
        shared = sum(box_inter(gt[f], pr[f]) for f in gt.keys() & pr.keys())
        vol = sum(map(area, gt.values())) + sum(map(area, pr.values())) - shared
        st = shared / vol if vol > 0 else 0.0
        row['stiou'] = round(st, 3)
        row['success'] = st >= 0.05                          # official Success @ 0.05
    return row


def build_index(run):
    run = Path(run)
    preds = load_predictions(run)
    if not preds:
        raise SystemExit(f'{run} has no saved predictions (eval_parts/ or predictions.json).')
    recs = records(run_split(run))
    rows = [query_stats(recs[k], v) for k, v in preds.items() if k in recs and recs[k]['response_track']]
    rows.sort(key=lambda r: r['key'])
    n = len(rows) or 1
    summary = dict(run=run.name, split=run_split(run), queries=len(rows),
                   success=round(100 * sum(r['success'] for r in rows) / n, 2),
                   mean_tiou=round(sum(r['tiou'] for r in rows) / n, 3),
                   mean_stiou=round(sum(r['stiou'] for r in rows) / n, 3))
    return dict(summary=summary, rows=rows)


# ------------------------------------------------------------ drawing -------

def _font(size, bold=False):
    from PIL import ImageFont
    for name in (('segoeuib.ttf' if bold else 'segoeui.ttf'), ('DejaVuSans-Bold.ttf' if bold else 'DejaVuSans.ttf'), 'arial.ttf'):
        try:
            return ImageFont.truetype(name, size)
        except OSError:
            pass
    try:
        return ImageFont.load_default(size)
    except TypeError:
        return ImageFont.load_default()


def _rect(draw, box, color, width=3):
    draw.rectangle([box[0], box[1], box[2], box[3]], outline=color, width=width)


def render_query(run, key, out_png, n_frames=6):
    """Draw one query: visual crop, score timeline, frames with GT and predicted boxes."""
    import numpy as np
    from PIL import Image, ImageDraw
    import eagle.data as D
    run = Path(run)
    rec = records(run_split(run))[key]
    preds = load_predictions(run)
    pred = preds.get(key, dict(score=0.0, bboxes=[]))
    stats = query_stats(rec, pred)
    gt = {f: b for f, b in rec['response_track']}
    pr = {int(b[0]): b[1:] for b in pred.get('bboxes', [])}

    # frames to show: spread across the union of both tracks (always both ends)
    track = sorted(gt.keys() | pr.keys())
    if len(track) > n_frames:
        idx = np.linspace(0, len(track) - 1, n_frames).round().astype(int)
        track = [track[i] for i in dict.fromkeys(idx)]
    vc_f, vc_box = rec['visual_crop']
    reader = D.ClipReader(rec['clip_uid'], rec['orig_size'])
    try:
        imgs = dict(reader.frames(set(track) | {vc_f}))
        w, h = reader.frame_size
        content = (round(w * reader.scale), round(h * reader.scale))   # letterbox without its padding
        to_in = reader.to_input
    finally:
        reader.close()

    W, pad = 1380, 16
    thumb_w = (W - pad * 4) // 3
    big, small, body = _font(22, True), _font(15), _font(17)
    canvas = Image.new('RGB', (W, 2000), (246, 247, 249))
    d = ImageDraw.Draw(canvas)

    # header: query crop + facts
    y = pad
    if vc_f in imgs:
        frame = Image.fromarray(imgs[vc_f]).crop((0, 0, *content))
        b = to_in(vc_box)
        crop = frame.crop([int(max(0, b[0])), int(max(0, b[1])), int(min(content[0], b[2])) or 1, int(min(content[1], b[3])) or 1])
        s = 200 / max(crop.size); crop = crop.resize((max(1, round(crop.width * s)), max(1, round(crop.height * s))))
        canvas.paste(crop, (pad, y)); _rect(d, [pad - 2, y - 2, pad + crop.width + 1, y + crop.height + 1], CROP_COLOR, 3)
    x0 = pad + 220
    verdict = 'found' if stats['success'] else ('missed' if pr else 'no prediction')
    d.text((x0, y), f'Query: "{rec["object_title"] or "object"}"  ·  {verdict}', font=big, fill=(20, 32, 46))
    lines = [f'{key}   ·   clip {rec["clip_uid"]}   ·   split {run_split(run)}   ·   run {run.name}',
             f'Visual crop at frame {vc_f}; query frame {rec["query_frame"]} (search covers frames 0–{rec["query_frame"] - 1}, 5 fps).',
             f'Ground truth frames {stats["gt_span"][0]}–{stats["gt_span"][1]}  ·  prediction '
             + (f'frames {stats["pred_span"][0]}–{stats["pred_span"][1]}, score {stats["score"]:.3f}' if pr else 'none'),
             f'tIoU {stats["tiou"]:.3f}   ·   stIoU {stats["stiou"]:.3f}   ·   success (stIoU ≥ 0.05): {"yes" if stats["success"] else "no"}']
    for i, t in enumerate(lines):
        d.text((x0, y + 36 + i * 26), t, font=body, fill=(60, 76, 92))
    ly = y + 36 + len(lines) * 26 + 6
    for i, (c, t) in enumerate([(GT_COLOR, 'ground truth'), (PRED_COLOR, 'EAGLE prediction'), (CROP_COLOR, 'visual crop')]):
        lx = x0 + i * 190
        d.rectangle([lx, ly + 4, lx + 22, ly + 18], outline=c, width=3); d.text((lx + 30, ly), t, font=small, fill=(60, 76, 92))
    y = max(y + 210, ly + 30) + pad

    # timeline: per-frame score with both tracks shaded
    th, tx0, tx1 = 150, pad + 40, W - pad
    n = max(1, rec['query_frame'])
    fx = lambda f: tx0 + (tx1 - tx0) * f / n
    d.rectangle([tx0, y, tx1, y + th], fill=(255, 255, 255), outline=(200, 206, 214))
    for span, color in ((stats['gt_span'], GT_COLOR), (stats['pred_span'], PRED_COLOR)):
        if span:   # faint column behind the curve, so the track position reads against the scores
            tint = tuple(int(c * 0.2 + 255 * 0.8) for c in color)
            d.rectangle([fx(span[0]) - 1, y + 1, max(fx(span[0]) + 3, fx(span[1] + 1)), y + th - 1], fill=tint)
    trace = (pred.get('trace') or {}).get('scores')
    yv = lambda s: y + th - 6 - (th - 12) * float(s)
    if trace:   # frames scored 0 had no detection; draw only detections, joined when consecutive
        seg = []
        for i, s in enumerate(trace + [0.0]):
            if s > 0:
                seg.append((fx(i), yv(s)))
            elif seg:
                d.line(seg, fill=(31, 92, 168), width=2) if len(seg) > 1 else d.point(seg, fill=(31, 92, 168))
                seg = []
        d.text((tx0 + 6, y + 4), 'confidence on frames with a detection', font=small, fill=(31, 92, 168))
    else:
        d.text((tx0 + 6, y + 4), 'no per-frame scores saved for this run (predictions.json only)', font=small, fill=(110, 120, 130))
    for v in (0.0, 0.5, 1.0):
        d.text((pad, yv(v) - 9), f'{v:.1f}', font=small, fill=(110, 120, 130))
    # solid track bars under the plot (at least 5 px wide, so short tracks stay visible)
    for row, (span, color, name) in enumerate(((stats['gt_span'], GT_COLOR, 'GT'), (stats['pred_span'], PRED_COLOR, 'pred'))):
        by = y + th + 12 + row * 13
        d.text((pad, by - 5), name, font=_font(12), fill=color)
        if span:
            d.rectangle([fx(span[0]), by, max(fx(span[0]) + 5, fx(span[1] + 1)), by + 8], fill=color)
    y += th + 40
    step = next(s for s in (25, 50, 100, 250, 500, 1000, 2500, 5000) if n / s <= 10)
    for f in range(0, n, step):
        d.line([fx(f), y - 4, fx(f), y + 1], fill=(150, 156, 164)); d.text((fx(f) - 10, y + 2), str(f), font=small, fill=(110, 120, 130))
    d.line([tx1, y - 4, tx1, y + 1], fill=(150, 156, 164))
    cap = f'frame (5 fps)  ·  query frame {rec["query_frame"]} at right edge'
    d.text((tx1 - d.textlength(cap, font=small), y + 22), cap, font=small, fill=(110, 120, 130))
    y += 50

    # frames with both boxes
    sc = thumb_w / content[0]
    tht = round(content[1] * sc)
    for i, f in enumerate(track):
        cx, cy = pad + (i % 3) * (thumb_w + pad), y + (i // 3) * (tht + 44)
        if f in imgs:
            im = Image.fromarray(imgs[f]).crop((0, 0, *content)).resize((thumb_w, tht))
            canvas.paste(im, (cx, cy))
            for boxes, color in ((gt, GT_COLOR), (pr, PRED_COLOR)):
                if f in boxes:
                    b = [v * sc for v in to_in(boxes[f])]
                    _rect(d, [cx + b[0], cy + b[1], cx + b[2], cy + b[3]], color, 3)
        else:
            d.rectangle([cx, cy, cx + thumb_w, cy + tht], fill=(225, 228, 232)); d.text((cx + 10, cy + 10), 'frame not decodable', font=small, fill=(110, 120, 130))
        tag = []
        if f in gt: tag.append('GT')
        if f in pr: tag.append('pred')
        iou = f'  ·  IoU {box_iou(gt[f], pr[f]):.2f}' if f in gt and f in pr else ''
        d.text((cx, cy + tht + 6), f'frame {f}  ·  {" + ".join(tag)}{iou}', font=small, fill=(60, 76, 92))
    y += ((len(track) + 2) // 3) * (tht + 44) + pad
    out_png = Path(out_png); out_png.parent.mkdir(parents=True, exist_ok=True)
    canvas.crop((0, 0, W, y)).save(out_png)
    return dict(png=str(out_png), stats=stats)


# ------------------------------------------------------------- driver -------

def handle(req):
    if req['op'] == 'index':
        out = DEMO_OUT / f'{Path(req["run"]).name}-index.json'
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(build_index(req['run'])), encoding='utf-8')
        return dict(index=str(out))
    if req['op'] == 'render':
        png = DEMO_OUT / f'{Path(req["run"]).name}-{req["key"]}.png'
        return render_query(req['run'], req['key'], png)
    raise ValueError(f'unknown op {req["op"]}')


def serve():
    """One JSON request per stdin line; one reply per line on stdout, prefixed 'DEMO '."""
    print('DEMO ' + json.dumps(dict(ready=True)), flush=True)
    for line in sys.stdin:
        if not line.strip():
            continue
        req = {}
        try:
            req = json.loads(line)
            reply = handle(req)
        except BaseException as e:      # SystemExit included: report, keep serving
            reply = dict(error=str(e) or type(e).__name__)
        reply['id'] = req.get('id')
        print('DEMO ' + json.dumps(reply), flush=True)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--run', help='EAGLE run folder, e.g. output/eagle-20261114-204835')
    p.add_argument('--index', action='store_true', help='print the per-query table as JSON')
    p.add_argument('--key', help='query key to draw')
    p.add_argument('--out', default='eval-demo.png', help='PNG to write with --key')
    p.add_argument('--serve', action='store_true', help='answer JSON requests on stdin (launcher backend)')
    a = p.parse_args()
    if a.serve:
        serve()
    elif a.run and a.index:
        print(json.dumps(build_index(a.run)['summary'], indent=2))
    elif a.run and a.key:
        print(json.dumps(render_query(a.run, a.key, a.out)['stats'], indent=2))
    else:
        p.error('use --serve, or --run with --index or --key')


if __name__ == '__main__':
    main()
