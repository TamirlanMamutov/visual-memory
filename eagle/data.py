"""Data for the EAGLE re-implementation: Ego4D VQ2D (boxes -> SAM pseudo-masks)
and EPIC-KITCHENS VISOR (ground-truth masks)."""
import io
import json
import pickle
import random
import zipfile
from pathlib import Path

import numpy as np

from common import CACHE, ROOT, atomic_write

EGO4D = ROOT / 'Ego4D' / 'v2'
CLIPS = EGO4D / 'clips'
VISOR = ROOT / 'EPIC-KITCHENS-VISOR'
EAGLE_CACHE = CACHE / 'eagle'
SIZE = 448                       # paper: frames scaled to 448x448 (long side + zero padding)
ANN_FPS = 5                      # VQ2D annotation frame numbers are at 5 fps


# ------------------------------------------------------------ geometry ------

def letterbox(img, size=SIZE):
    """Resize the long side to `size` and zero-pad (bottom/right). Returns
    (padded uint8 image, scale)."""
    import cv2
    h, w = img.shape[:2]
    s = size / max(h, w)
    img = cv2.resize(img, (round(w * s), round(h * s)), interpolation=cv2.INTER_AREA)
    out = np.zeros((size, size, 3), np.uint8)
    out[:img.shape[0], :img.shape[1]] = img
    return out, s


def box_to_mask(box, size=SIZE):
    m = np.zeros((size, size), np.uint8)
    x1, y1, x2, y2 = [int(round(v)) for v in box]
    m[max(0, y1):max(0, y2), max(0, x1):max(0, x2)] = 1
    return m


def mask_to_box(mask):
    ys, xs = np.nonzero(mask)
    if len(xs) == 0:
        return None
    return [float(xs.min()), float(ys.min()), float(xs.max() + 1), float(ys.max() + 1)]


# ---------------------------------------------------------------- VQ2D ------

def local_clip_uids():
    return {p.stem for p in CLIPS.glob('*.mp4') if p.stat().st_size > 0}


def load_vq(split, local_only=True):
    """Flatten vq_{split}.json into one record per valid query set."""
    data = json.loads((EGO4D / 'annotations' / f'vq_{split}.json').read_text())
    have = local_clip_uids() if local_only else None
    out = []
    for video in data['videos']:
        for clip in video['clips']:
            if not clip.get('clip_uid') or (have is not None and clip['clip_uid'] not in have):
                continue
            for ai, ann in enumerate(clip['annotations']):
                for qid, q in ann['query_sets'].items():
                    if not q.get('is_valid', True) or 'visual_crop' not in q:
                        continue
                    vc = q['visual_crop']
                    rt = sorted(q.get('response_track', []), key=lambda r: r['frame_number'])
                    out.append(dict(
                        key=f"{clip['clip_uid']}_{ai}_{qid}", clip_uid=clip['clip_uid'], annotation=ai, query_set=qid,
                        query_frame=int(q['query_frame']), object_title=q.get('object_title', ''),
                        orig_size=(vc['original_width'], vc['original_height']),
                        visual_crop=(int(vc['frame_number']), _xyxy(vc)),
                        response_track=[(int(r['frame_number']), _xyxy(r)) for r in rt]))
    return out


def _xyxy(b):
    return [b['x'], b['y'], b['x'] + b['width'], b['y'] + b['height']]


class ClipReader:
    """Reads 5-fps annotation frames from a 30-fps Ego4D clip (frame f -> time f/5 s).
    Images are letterboxed to 448; boxes in original-resolution pixels are
    mapped with `to_input`."""

    def __init__(self, clip_uid, orig_size):
        import av
        self.path = CLIPS / f'{clip_uid}.mp4'
        self.container = av.open(str(self.path))
        self.stream = self.container.streams.video[0]
        self.stream.thread_type = 'AUTO'
        self.tb = float(self.stream.time_base)
        w, h = self.stream.codec_context.width, self.stream.codec_context.height
        self.frame_size = (w, h)
        self.scale = SIZE / max(w, h)                     # decoded pixels -> input pixels
        self.box_scale = self.scale * w / orig_size[0]    # annotation pixels -> input pixels

    def to_input(self, box):
        return [v * self.box_scale for v in box]

    def to_original(self, box):
        return [v / self.box_scale for v in box]

    def frames(self, wanted):
        """Yield (frame_number, letterboxed RGB) for the sorted 5-fps frame numbers."""
        wanted = sorted(set(int(f) for f in wanted))
        if not wanted:
            return
        i = 0
        def seek(f):
            self.container.seek(int(max(0, f / ANN_FPS - 1.0) / self.tb), stream=self.stream, backward=True)
        seek(wanted[0])
        while i < len(wanted):
            reseek = False
            for frame in self.container.decode(self.stream):
                if frame.pts is None:
                    continue
                t = frame.pts * self.tb
                while i < len(wanted) and wanted[i] / ANN_FPS <= t + 0.5 / 30:
                    yield wanted[i], letterbox(frame.to_ndarray(format='rgb24'))[0]
                    i += 1
                if i >= len(wanted):
                    return
                if wanted[i] / ANN_FPS - t > 10:
                    seek(wanted[i]); reseek = True; break
            if not reseek:
                return                                     # end of stream

    def all_frames(self, stop):
        """Sequentially yield every 5-fps frame number < stop (fast path for inference)."""
        self.container.seek(0)
        f = 0
        for frame in self.container.decode(self.stream):
            if f >= stop:
                return
            if frame.pts is None:
                continue
            t = frame.pts * self.tb
            while f < stop and f / ANN_FPS <= t + 0.5 / 30:
                yield f, letterbox(frame.to_ndarray(format='rgb24'))[0]
                f += 1

    def close(self):
        self.container.close()


# ---------------------------------------------- training cache (VQ2D) ------

def vq_cache_path(key):
    return EAGLE_CACHE / 'vq2d' / f'{key}.pkl'


def build_vq_training_cache(records, sam, log=print, negatives=6, max_track=24, seed=0):
    """For each query store: the visual-crop frame, up to `max_track` response
    track frames and a few object-free frames, each as JPEG + SAM pseudo-mask
    (box + centre-point prompt). Masks for negatives are empty."""
    import cv2
    rng = random.Random(seed)
    (EAGLE_CACHE / 'vq2d').mkdir(parents=True, exist_ok=True)
    todo = [r for r in records if not vq_cache_path(r['key']).exists()]
    log(f'VQ2D training cache: {len(records) - len(todo)} cached, {len(todo)} to build in {EAGLE_CACHE / "vq2d"}')
    by_clip = {}
    for r in todo:
        by_clip.setdefault(r['clip_uid'], []).append(r)
    for ci, (clip_uid, recs) in enumerate(by_clip.items()):
        reader = ClipReader(clip_uid, recs[0]['orig_size'])
        plan = {}
        for r in recs:
            track = r['response_track']
            if len(track) > max_track:
                track = [track[i] for i in sorted(rng.sample(range(len(track)), max_track))]
            in_track = {f for f, _ in r['response_track']}
            pool = [f for f in range(0, r['query_frame']) if f not in in_track]
            neg = sorted(rng.sample(pool, min(negatives, len(pool))))
            plan[r['key']] = dict(query=r['visual_crop'], track=track, neg=neg)
        wanted = {p['query'][0] for p in plan.values()}
        for p in plan.values():
            wanted |= {f for f, _ in p['track']} | set(p['neg'])
        images = dict(reader.frames(wanted))
        for r in recs:
            p = plan[r['key']]
            items = []
            for role, entries in [('query', [p['query']]), ('track', p['track']), ('neg', [(f, None) for f in p['neg']])]:
                for f, box in entries:
                    if f not in images:
                        continue
                    img = images[f]
                    if box is None:
                        mask = np.zeros(img.shape[:2], np.uint8); ibox = None
                    else:
                        ibox = reader.to_input(box)
                        mask = sam(img, ibox)
                    items.append(dict(role=role, frame=f, box=ibox,
                                      jpg=cv2.imencode('.jpg', img[:, :, ::-1], [cv2.IMWRITE_JPEG_QUALITY, 92])[1].tobytes(),
                                      mask=cv2.imencode('.png', mask)[1].tobytes()))
            # atomic: a file under the final name is always complete, so resuming can skip it
            atomic_write(vq_cache_path(r['key']), lambda fh: pickle.dump(dict(record=r, items=items), fh))
        reader.close()
        log(f'  [{ci + 1}/{len(by_clip)}] clip {clip_uid}: {len(recs)} queries')


# ------------------------------------------------------------ EgoTracks ------
# EgoTracks extends VQ2D queries with a long-term track: the object's box in every frame where it is
# visible anywhere in the clip, not just in the last occurrence. Same clips, 5-fps frame numbers and
# visual crops as VQ2D, so the VQ2D reader and cache format are reused.

EGOTRACKS = EGO4D / 'egotracks'


def egotracks_available():
    return (EGOTRACKS / 'egotracks_train.json').is_file()


def load_egotracks(split='train', local_only=True):
    """One record per query set with a long-term track. 'track' holds every box of the long-term
    track as (frame, xyxy, interpolated); 'clip_frames' is the clip length in 5-fps frames."""
    data = json.loads((EGOTRACKS / f'egotracks_{split}.json').read_text())
    have = local_clip_uids() if local_only else None
    out = []
    for video in data['videos']:
        for clip in video['clips']:
            uid = clip.get('clip_uid')
            if not uid or (have is not None and uid not in have) or clip.get('exported_clip_uid', uid) != uid:
                continue
            n = int(round((clip['clip_end_sec'] - clip['clip_start_sec']) * ANN_FPS))
            for ai, ann in enumerate(clip['annotations']):
                for qid, q in ann['query_sets'].items():
                    lt = q.get('lt_track') or []
                    if not q.get('is_valid', True) or 'visual_crop' not in q or not lt:
                        continue
                    vc = q['visual_crop']
                    out.append(dict(
                        key=f'{uid}_{ai}_{qid}', clip_uid=uid, object_title=q.get('object_title', ''),
                        orig_size=(vc['original_width'], vc['original_height']), clip_frames=n,
                        visual_crop=(int(vc['frame_number']), _xyxy(vc)),
                        track=sorted((int(b['frame_number']), _xyxy(b), bool(b.get('interpolated'))) for b in lt)))
    return out


def egotracks_cache_path(key):
    return EAGLE_CACHE / 'egotracks' / f'{key}.pkl'


def plan_egotracks(r, rng, max_track=24, negatives=6, margin=10):
    """Training frames for one long-term track: `max_track` boxes spread over the whole track (one
    per equal-size stretch, preferring hand-annotated over interpolated boxes, so different
    occurrences and viewpoints are covered) and object-free frames at least `margin` frames
    (2 s) from any box of the track."""
    track = r['track']
    picks = []
    for s in np.array_split(np.arange(len(track)), min(max_track, len(track))):
        cand = [i for i in s if not track[i][2]] or list(s)
        picks.append(track[rng.choice(cand)])
    near = np.zeros(max(r['clip_frames'], max(f for f, _, _ in track) + 1) + margin + 1, bool)
    for f, _, _ in track:
        near[max(0, f - margin):f + margin + 1] = True
    pool = [f for f in range(r['clip_frames']) if not near[f]]
    neg = sorted(rng.sample(pool, min(negatives, len(pool))))
    return dict(query=r['visual_crop'], track=[(f, b) for f, b, _ in picks], neg=neg)


def build_egotracks_training_cache(records, sam, log=print, seed=0):
    """Same cache format as build_vq_training_cache (query, track and negative frames as JPEG + SAM
    pseudo-mask), in <cache>/eagle/egotracks/."""
    _build_cache(records, sam, log, egotracks_cache_path, 'EgoTracks', lambda r, rng: plan_egotracks(r, rng), seed)


def _build_cache(records, sam, log, path_of, name, plan_of, seed):
    import cv2
    rng = random.Random(seed)
    todo = [r for r in records if not path_of(r['key']).exists()]
    if todo:
        path_of(todo[0]['key']).parent.mkdir(parents=True, exist_ok=True)
    log(f'{name} training cache: {len(records) - len(todo)} cached, {len(todo)} to build'
        + (f' in {path_of(todo[0]["key"]).parent}' if todo else ''))
    by_clip = {}
    for r in todo:
        by_clip.setdefault(r['clip_uid'], []).append(r)
    for ci, (clip_uid, recs) in enumerate(by_clip.items()):
        reader = ClipReader(clip_uid, recs[0]['orig_size'])
        plan = {r['key']: plan_of(r, rng) for r in recs}
        wanted = set()
        for p in plan.values():
            wanted |= {p['query'][0]} | {f for f, _ in p['track']} | set(p['neg'])
        images = dict(reader.frames(wanted))
        for r in recs:
            p = plan[r['key']]
            items = []
            for role, entries in [('query', [p['query']]), ('track', p['track']), ('neg', [(f, None) for f in p['neg']])]:
                for f, box in entries:
                    if f not in images:
                        continue
                    img = images[f]
                    if box is None:
                        mask = np.zeros(img.shape[:2], np.uint8); ibox = None
                    else:
                        ibox = reader.to_input(box)
                        mask = sam(img, ibox)
                    items.append(dict(role=role, frame=f, box=ibox,
                                      jpg=cv2.imencode('.jpg', img[:, :, ::-1], [cv2.IMWRITE_JPEG_QUALITY, 92])[1].tobytes(),
                                      mask=cv2.imencode('.png', mask)[1].tobytes()))
            rec = {k: v for k, v in r.items() if k != 'track'}       # the full track is large; keep the summary
            rec['track_len'] = len(r.get('track', ()))
            # atomic: a file under the final name is always complete, so resuming can skip it
            atomic_write(path_of(r['key']), lambda fh: pickle.dump(dict(record=rec, items=items), fh))
        reader.close()
        log(f'  [{ci + 1}/{len(by_clip)}] clip {clip_uid}: {len(recs)} queries')


def decode_item(item):
    import cv2
    img = cv2.imdecode(np.frombuffer(item['jpg'], np.uint8), cv2.IMREAD_COLOR)[:, :, ::-1].copy()
    mask = cv2.imdecode(np.frombuffer(item['mask'], np.uint8), cv2.IMREAD_GRAYSCALE)
    return img, (mask > 0).astype(np.uint8)


class SamMasker:
    """SAM pseudo-masks from a box + centre-point prompt (paper Sec. 4.1/Appendix)."""

    def __init__(self, device, model='facebook/sam-vit-base'):
        import torch
        from transformers import SamModel, SamProcessor
        self.torch = torch
        self.processor = SamProcessor.from_pretrained(model)
        self.model = SamModel.from_pretrained(model).to(device).eval()
        self.device = device

    def __call__(self, img, box):
        torch = self.torch
        cx, cy = (box[0] + box[2]) / 2, (box[1] + box[3]) / 2
        inputs = self.processor(img, input_boxes=[[box]], input_points=[[[[cx, cy]]]], return_tensors='pt').to(self.device)
        with torch.no_grad():
            out = self.model(**inputs, multimask_output=True)
        masks = self.processor.image_processor.post_process_masks(
            out.pred_masks.cpu(), inputs['original_sizes'].cpu(), inputs['reshaped_input_sizes'].cpu())[0][0]
        best = int(out.iou_scores[0, 0].argmax())
        m = masks[best].numpy().astype(np.uint8)
        # Keep the mask inside a slightly enlarged box: SAM sometimes grabs the background.
        x1, y1, x2, y2 = box; pad = 0.1 * max(x2 - x1, y2 - y1)
        keep = np.zeros_like(m); keep[max(0, int(y1 - pad)):int(y2 + pad) + 1, max(0, int(x1 - pad)):int(x2 + pad) + 1] = 1
        m = m * keep
        return m if m.sum() > 0 else box_to_mask(box, img.shape[0])


# --------------------------------------------------------------- VISOR ------

def visor_available():
    return (VISOR / 'GroundTruth-SparseAnnotations').is_dir()


def load_visor_sequences(split='train', min_len=3):
    """Group VISOR sparse annotations into per-object sequences:
    [(video_id, object_name, [(image_name, polygons), ...]), ...]."""
    root = VISOR / 'GroundTruth-SparseAnnotations' / 'annotations' / split
    seqs = []
    for jf in sorted(root.glob('*.json')):
        data = json.loads(jf.read_text())
        per_obj = {}
        for entry in data.get('video_annotations', []):
            name = entry['image']['name']
            for a in entry.get('annotations', []):
                if a.get('name', '').startswith(('left hand', 'right hand')):
                    continue                                  # queries are objects, not hands
                per_obj.setdefault(a['name'], []).append((name, a['segments']))
        for obj, frames in per_obj.items():
            if len(frames) >= min_len:
                seqs.append((jf.stem, obj, sorted(frames)))
    return seqs


class VisorFrames:
    """Reads VISOR rgb_frames/<split>/<Pxx>/<video>.zip lazily."""

    def __init__(self, split='train'):
        self.root = VISOR / 'GroundTruth-SparseAnnotations' / 'rgb_frames' / split
        self.zips = {}

    def image(self, video_id, name):
        import cv2
        z = self.zips.get(video_id)
        if z is None:
            z = self.zips[video_id] = zipfile.ZipFile(self.root / video_id.split('_')[0] / f'{video_id}.zip')
        buf = np.frombuffer(z.read(name), np.uint8)
        return cv2.imdecode(buf, cv2.IMREAD_COLOR)[:, :, ::-1].copy()


def polygons_to_mask(polys, shape):
    import cv2
    m = np.zeros(shape[:2], np.uint8)
    for poly in polys:
        pts = np.asarray(poly, np.float32).reshape(-1, 2)
        if len(pts) >= 3:
            cv2.fillPoly(m, [np.round(pts).astype(np.int32)], 1)
    return m
