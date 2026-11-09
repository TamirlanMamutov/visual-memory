"""EPIC-KITCHENS-100 Multi-Instance Retrieval data: annotations, relevance,
frame extraction from the raw MP4s and cached CLIP frame features."""
import pickle
from pathlib import Path

import numpy as np
import pandas as pd

from common import CACHE, ROOT

ANN = ROOT / 'epic-kitchens-100-annotations' / 'retrieval_annotations'
VIDEOS = ROOT / 'EPIC-KITCHENS'
FRAME_CACHE = CACHE / 'tempret' / 'frames'
FEATURE_CACHE = CACHE / 'tempret' / 'features'
STORED_FRAMES = 16          # frames decoded per clip (segment centres)
SHORT_SIDE = 224
NUM_NOUNS = 300


def video_path(video_id):
    return VIDEOS / video_id.split('_')[0] / 'videos' / f'{video_id}.MP4'


def local_video_ids():
    return sorted(p.stem for p in VIDEOS.glob('P*/videos/*.MP4') if p.stat().st_size > 0)


def load_split(split, local_only=True):
    """split: 'train' or 'test' (the MIR 'test' file is the public validation set)."""
    df = pd.read_pickle(ANN / f'EPIC_100_retrieval_{split}.pkl')
    if local_only:
        df = df[df.video_id.isin(set(local_video_ids()))]
    return df


def load_test_sentences(test_df_full):
    """The 3,842 unique MIR evaluation captions with their verb/noun classes."""
    sent = pd.read_pickle(ANN / 'EPIC_100_retrieval_test_sentence.pkl')
    src = test_df_full.loc[sent.index]
    return pd.DataFrame(dict(narration=sent.narration.values, verb_class=src.verb_class.values,
                             noun_classes=src.all_noun_classes.values), index=sent.index)


def multi_hot(noun_lists):
    m = np.zeros((len(noun_lists), NUM_NOUNS), np.float32)
    for i, nouns in enumerate(noun_lists):
        m[i, list(nouns)] = 1
    return m


def relevance(v_verbs, v_nouns, s_verbs, s_nouns):
    """Official EK-100 MIR semantic relevance: 0.5 * verb IoU + 0.5 * noun-set IoU.
    Inputs are numpy arrays (verbs: [N], nouns: multi-hot [N, 300])."""
    verb = (np.asarray(v_verbs)[:, None] == np.asarray(s_verbs)[None]).astype(np.float32)
    inter = v_nouns @ s_nouns.T
    union = v_nouns.sum(1)[:, None] + s_nouns.sum(1)[None] - inter
    return 0.5 * verb + 0.5 * inter / np.maximum(union, 1)


def torch_relevance(verbs, nouns):
    """Batch relevance between clips i and j (both modalities share labels)."""
    import torch
    verb = (verbs[:, None] == verbs[None]).float()
    inter = nouns @ nouns.T
    union = nouns.sum(1)[:, None] + nouns.sum(1)[None] - inter
    return 0.5 * verb + 0.5 * inter / union.clamp(min=1)


def segment_frames(start, stop, n=STORED_FRAMES):
    """n frame numbers (EPIC 1-indexed) at the centres of n equal segments."""
    edges = np.linspace(start, stop, n + 1)
    return np.clip(np.round((edges[:-1] + edges[1:]) / 2).astype(int), start, stop)


# ---------------------------------------------------------------- frames ----

def extract_frames(df, log=print, overwrite=False):
    """Decode STORED_FRAMES frames per clip, resized to a 224 short side, and
    store them as JPEG bytes in <cache>/tempret/frames/<narration_id>.pkl."""
    import av
    import cv2
    FRAME_CACHE.mkdir(parents=True, exist_ok=True)
    if len(df) == 0:
        return
    # np.array: an empty plain list would select columns instead of rows
    todo = df if overwrite else df[np.array([not (FRAME_CACHE / f'{n}.pkl').exists() for n in df.index], dtype=bool)]
    log(f'Frame extraction: {len(df) - len(todo)} clips cached, {len(todo)} to decode '
        f'from {todo.video_id.nunique()} videos into {FRAME_CACHE}')
    for v_i, (video_id, clips) in enumerate(todo.groupby('video_id')):
        wanted = {}                      # frame number -> list of (clip, slot)
        for nid, row in clips.iterrows():
            for slot, f in enumerate(segment_frames(row.start_frame, row.stop_frame)):
                wanted.setdefault(int(f), []).append((nid, slot))
        results = {nid: [None] * STORED_FRAMES for nid in clips.index}
        targets = sorted(wanted)
        with av.open(str(video_path(video_id))) as container:
            stream = container.streams.video[0]
            stream.thread_type = 'AUTO'
            fps = float(stream.average_rate)
            tb = float(stream.time_base)
            ti = 0
            def seek_to(frame_no):
                # EPIC frame numbers are 1-indexed; seek a little early to a keyframe.
                t = max(0.0, (frame_no - 1) / fps - 2.0)
                container.seek(int(t / tb), stream=stream, backward=True, any_frame=False)
            seek_to(targets[0])
            last = None
            while ti < len(targets):
                restart = False
                for frame in container.decode(stream):
                    if frame.pts is None:
                        continue
                    no = int(round(frame.pts * tb * fps)) + 1
                    last = frame
                    while ti < len(targets) and targets[ti] < no:
                        # Missed (e.g. dropped frame): use this nearest later frame.
                        img = _to_jpeg(frame, cv2)
                        for nid, slot in wanted[targets[ti]]:
                            results[nid][slot] = img
                        ti += 1
                    if ti < len(targets) and targets[ti] == no:
                        img = _to_jpeg(frame, cv2)
                        for nid, slot in wanted[targets[ti]]:
                            results[nid][slot] = img
                        ti += 1
                    if ti >= len(targets):
                        break
                    if targets[ti] - no > 20 * fps:   # long gap: seek instead of decoding
                        seek_to(targets[ti]); restart = True
                        break
                if not restart:
                    break
            if ti < len(targets) and last is not None:   # targets past the end of the stream
                img = _to_jpeg(last, cv2)
                for f in targets[ti:]:
                    for nid, slot in wanted[f]:
                        results[nid][slot] = img
        for nid, frames in results.items():
            if any(f is None for f in frames):
                log(f'  WARNING: {nid} missing frames; skipped'); continue
            with (FRAME_CACHE / f'{nid}.pkl').open('wb') as f:
                pickle.dump(frames, f)
        log(f'  [{v_i + 1}/{todo.video_id.nunique()}] {video_id}: {len(clips)} clips')


def _to_jpeg(frame, cv2):
    img = frame.to_ndarray(format='bgr24')
    h, w = img.shape[:2]
    s = SHORT_SIDE / min(h, w)
    img = cv2.resize(img, (round(w * s), round(h * s)), interpolation=cv2.INTER_AREA)
    return cv2.imencode('.jpg', img, [cv2.IMWRITE_JPEG_QUALITY, 90])[1].tobytes()


def load_frames(nid):
    """List of STORED_FRAMES RGB uint8 arrays (224 short side)."""
    import cv2
    with (FRAME_CACHE / f'{nid}.pkl').open('rb') as f:
        frames = pickle.load(f)
    return [cv2.cvtColor(cv2.imdecode(np.frombuffer(b, np.uint8), cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB) for b in frames]


def cached(df):
    return df[[(FRAME_CACHE / f'{n}.pkl').exists() for n in df.index]]


# ------------------------------------------------------- feature cache ------

def feature_file(tag):
    return FEATURE_CACHE / f'{tag}.pt'


def build_feature_cache(df, clip_model, preprocess_tensor, tag, device, log=print, batch_clips=32):
    """Encode every stored frame with the frozen CLIP image tower (CLS token
    after the visual projection). Saves {nid: fp16 [STORED_FRAMES, D]}."""
    import torch
    path = feature_file(tag)
    feats = torch.load(path) if path.exists() else {}
    todo = [n for n in df.index if n not in feats]
    log(f'Feature cache {path.name}: {len(feats)} clips cached, {len(todo)} to encode')
    clip_model.eval()
    for i in range(0, len(todo), batch_clips):
        ids = todo[i:i + batch_clips]
        imgs = torch.stack([preprocess_tensor(f, train=False) for n in ids for f in load_frames(n)]).to(device)
        with torch.no_grad(), torch.autocast(device.type, dtype=torch.float16, enabled=device.type == 'cuda'):
            out = clip_model.encode_image(imgs).float()
        out = out.view(len(ids), STORED_FRAMES, -1).half().cpu()
        feats.update({n: out[j] for j, n in enumerate(ids)})
        if (i // batch_clips) % 50 == 0:
            log(f'  encoded {min(i + batch_clips, len(todo))}/{len(todo)} clips')
    FEATURE_CACHE.mkdir(parents=True, exist_ok=True)
    if todo:
        torch.save(feats, path)
    return feats
