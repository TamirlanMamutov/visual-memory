"""EAGLE manual test: a local web page for trying the model on a query you make yourself.

Pick an Ego4D VQ2D query, any local Ego4D clip or a video of your own, scrub to a
frame, draw a box around an object, choose the last frame to search, and run
EAGLE. The page plays the video and overlays the predicted response track (and
the ground truth when the query comes from VQ2D).

  python eagle_manual.py             serve on http://127.0.0.1:8765 (or the next free port)
  python eagle_manual.py --open      also open the page in the browser

Runs in the GPU environment. The server only listens on this computer. The search
is the same online inference as the evaluation (eagle/infer.py run_inference) on a
single query, so a VQ2D query run unchanged reproduces that run's saved prediction.
Frame numbers are VQ2D annotation frames at 5 fps (frame f = time f / 5 s).

Your own videos are converted to H.264 at 30 fps (rotation applied, long side at most
1440 px) and kept in output/manual-test/videos; saved queries go to
output/manual-test/queries.
"""
import argparse
import json
import re
import shutil
import socket
import sys
import tempfile
import threading
import time
import traceback
import webbrowser
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, unquote, urlparse

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from common import CACHE, ROOT, write_json  # noqa: E402
import eagle.data as D  # noqa: E402

MT = HERE / 'output' / 'manual-test'
VIDEOS, QUERIES, TMP = MT / 'videos', MT / 'queries', MT / 'tmp'
PAGE = HERE / 'eagle_manual.html'
D.EXTRA_CLIP_DIRS.append(VIDEOS)
FPS = D.ANN_FPS
NAME_RE = re.compile(r'^[A-Za-z0-9_-]{1,80}$')


# ------------------------------------------------------------- catalogue ----

def resolve_checkpoint(path):
    """Checkpoint paths in run files may be absolute paths from another machine; re-root them
    on this machine's cs230-cache."""
    p = Path(path)
    if p.exists():
        return p
    parts = [x.lower() for x in p.parts]
    if CACHE.name.lower() in parts:
        q = CACHE.joinpath(*p.parts[parts.index(CACHE.name.lower()) + 1:])
        if q.exists():
            return q
    q = ROOT / path
    return q if q.exists() else None


def checkpoints():
    """Checkpoints of finished EAGLE evaluations, best official Success first."""
    out = []
    for run in sorted((HERE / 'output').glob('eagle-*')):
        for name in ('metrics_official.json', 'metrics.json'):
            f = run / name
            if f.exists():
                m = json.loads(f.read_text(encoding='utf-8'))
                ck = resolve_checkpoint(m['checkpoint']) if m.get('checkpoint') else None
                fin = m.get('final', {})
                if ck and fin.get('queries', 0) >= 100:
                    official = m.get('pairing') == 'all'
                    out.append(dict(path=str(ck), run=run.name, succ=fin.get('Succ'), stap25=fin.get('stAP25'),
                                    label=f"{run.name} · Succ {fin.get('Succ', 0):.1f}" + ('' if official else ' (old scorer)')))
                break
    out.sort(key=lambda c: -(c['succ'] or 0))
    return out


_val = None


def val_queries():
    global _val
    if _val is None:
        _val = {r['key']: r for r in D.load_vq('val') if r['response_track']}
    return _val


_info = {}


def video_info(clip):
    if clip not in _info:
        import av
        path = D.clip_path(clip)
        if not path.exists():
            raise ValueError(f'No video for {clip}')
        with av.open(str(path)) as c:
            s = c.streams.video[0]
            dur = float(s.duration * s.time_base) if s.duration else c.duration / 1e6
            _info[clip] = dict(clip=clip, width=s.codec_context.width, height=s.codec_context.height,
                               duration=round(dur, 3), frames=int(dur * FPS), own=path.parent == VIDEOS)
    return _info[clip]


def sources():
    q = val_queries()
    queries = sorted(([k, r['clip_uid'], r['object_title'], r['query_frame']] for k, r in q.items()),
                     key=lambda x: (x[2].lower(), x[0]))
    with_q = {r['clip_uid'] for r in q.values()}
    clips = sorted(D.local_clip_uids(), key=lambda c: (c not in with_q, c))
    own = sorted((p.stem for p in VIDEOS.glob('*.mp4')), reverse=True) if VIDEOS.exists() else []
    return dict(queries=queries, clips=clips, own=own)


def scaled(box, sx, sy):
    return [box[0] * sx, box[1] * sy, box[2] * sx, box[3] * sy]


def query_detail(key):
    """A VQ2D query with every box converted to the video's decoded pixels (what the page draws in)."""
    r = val_queries()[key]
    info = video_info(r['clip_uid'])
    sx, sy = info['width'] / r['orig_size'][0], info['height'] / r['orig_size'][1]
    return dict(key=key, clip=r['clip_uid'], object=r['object_title'], query_frame=r['query_frame'],
                crop_frame=r['visual_crop'][0], crop_box=scaled(r['visual_crop'][1], sx, sy),
                gt=[[f] + scaled(b, sx, sy) for f, b in r['response_track']], video=info)


# ----------------------------------------------------------------- model ----

class Engine:
    """Holds the model and SAM on the GPU; runs one query at a time."""

    def __init__(self):
        self.lock = threading.Lock()
        self.model = self.args = self.sam = self.dev = None
        self.checkpoint = None
        self.state = 'idle'            # idle | loading | ready | error
        self.error = None

    def load(self, checkpoint):
        import torch
        import run_eagle
        from common import device
        from eagle.model import EAGLE
        if self.checkpoint == checkpoint and self.model is not None:
            return
        self.state = 'loading'
        try:
            self.dev = self.dev or device()
            ck = torch.load(checkpoint, map_location='cpu', weights_only=False)
            argv, sys.argv = sys.argv, ['run_eagle.py']
            try:
                args = run_eagle.parse()                      # evaluation defaults
            finally:
                sys.argv = argv
            for k, v in ck.get('args', {}).items():           # architecture and memory settings of the checkpoint
                if k in ('backbone', 'no_amm', 'no_glm', 'finetune_blocks', 'init_iters', 'update_iters', 'sam'):
                    setattr(args, k, v)
            self.model = None
            if self.dev.type == 'cuda':
                torch.cuda.empty_cache()
            model = EAGLE(args.backbone, use_glm=not args.no_glm, use_amm=not args.no_amm,
                          finetune_blocks=args.finetune_blocks).to(self.dev)
            run_eagle.load_weights(model, ck)                 # copies the CPU tensors into the GPU model
            if self.sam is None:
                self.sam = D.SamMasker(self.dev, args.sam)
            self.model, self.args, self.checkpoint = model, args, checkpoint
            self.state, self.error = 'ready', None
        except BaseException as e:
            self.state, self.error = 'error', f'{type(e).__name__}: {e}'
            raise

    def run(self, clip, crop_frame, crop_box, query_frame, checkpoint, job, decode_until=0):
        from eagle.infer import run_inference
        with self.lock:
            job.update(phase='Loading the model')
            self.load(checkpoint)
            self.args.decode_until = decode_until
            info = video_info(clip)
            rec = dict(key='manual', clip_uid=clip, annotation=0, query_set='1', object_title='',
                       query_frame=int(query_frame), orig_size=(info['width'], info['height']),
                       visual_crop=(int(crop_frame), [float(v) for v in crop_box]), response_track=[])
            orig = D.ClipReader.all_frames

            def counted(reader, stop):
                for f, img in orig(reader, stop):
                    job.update(phase='Searching', progress=[f + 1, stop])
                    yield f, img
            TMP.mkdir(parents=True, exist_ok=True)
            D.ClipReader.all_frames = counted
            try:
                with tempfile.TemporaryDirectory(dir=TMP) as tmp:
                    job.update(phase='Reading the query frame')
                    run_inference(self.model, [rec], self.sam, self.dev, self.args, lambda *a: None, parts_dir=Path(tmp))
                    part = Path(tmp) / f'{clip}.json'
                    if not part.exists():
                        raise ValueError(f'Frame {crop_frame} of {clip} could not be decoded.')
                    return json.loads(part.read_text(encoding='utf-8'))['manual']
            finally:
                D.ClipReader.all_frames = orig


ENGINE = Engine()


# ------------------------------------------------------------ background ----

class Job:
    """The one long task (search or video conversion) running at a time; the page polls it."""

    def __init__(self):
        self.lock = threading.Lock()
        self.d = dict(id=0, kind=None, state='idle')

    def start(self, kind, fn):
        with self.lock:
            if self.d['state'] == 'running':
                return None
            self.d = dict(id=self.d['id'] + 1, kind=kind, state='running', phase='Starting', progress=None,
                          started=time.time())
            jid = self.d['id']

        def work():
            try:
                result = fn(self)
                self.update(state='done', result=result)
            except BaseException as e:
                traceback.print_exc()
                self.update(state='error', error=str(e) or type(e).__name__)
            self.update(seconds=round(time.time() - self.d['started'], 1))
        threading.Thread(target=work, daemon=True).start()
        return jid

    def update(self, **kw):
        with self.lock:
            self.d.update(kw)

    def snapshot(self):
        with self.lock:
            d = dict(self.d)
        if d['state'] == 'running':
            d['seconds'] = round(time.time() - d['started'], 1)
        return d


JOB = Job()


def search(req, job):
    clip, key = req['clip'], req.get('key')
    # A VQ2D clip's evaluation decoded up to its last query frame in batches of 16 frames, and the
    # feature BatchNorm uses batch statistics; decoding as far gives the same features, so an
    # unchanged VQ2D query reproduces the saved evaluation prediction.
    until = max((r['query_frame'] for r in val_queries().values() if r['clip_uid'] == clip), default=0)
    pred = ENGINE.run(clip, req['crop_frame'], req['crop_box'], req['query_frame'], req['checkpoint'], job,
                      decode_until=until if req.get('match_eval', True) else 0)
    out = dict(pred=pred, query=req)
    if key in val_queries():
        from eagle_demo import query_stats
        d = query_detail(key)
        if d['query_frame'] == int(req['query_frame']):    # the ground truth only answers its own query frame
            rec = dict(key=key, clip_uid=clip, object_title=d['object'], query_frame=d['query_frame'],
                       response_track=[(g[0], g[1:]) for g in d['gt']])
            out['stats'] = query_stats(rec, pred)
    return out


def convert(src, name, job):
    """Your own video -> H.264, 30 fps constant, rotation applied, long side <= 1440, no audio."""
    import av
    import cv2
    import numpy as np
    VIDEOS.mkdir(parents=True, exist_ok=True)
    dst = VIDEOS / f'{name}.mp4'
    part = dst.with_suffix('.part.mp4')
    try:
        with av.open(str(src)) as inp:
            vs = inp.streams.video[0]
            vs.thread_type = 'AUTO'
            dur = float(vs.duration * vs.time_base) if vs.duration else (inp.duration or 0) / 1e6
            with av.open(str(part), 'w', format='mp4', options={'movflags': 'faststart'}) as out:
                os_ = None
                prev, i, t0 = None, 0, None

                def emit(img):
                    nonlocal os_
                    if os_ is None:
                        h, w = img.shape[:2]
                        os_ = out.add_stream('libx264', rate=30)
                        os_.width, os_.height, os_.pix_fmt = w, h, 'yuv420p'
                        os_.options = {'crf': '20', 'preset': 'veryfast'}
                    for pkt in os_.encode(av.VideoFrame.from_ndarray(img, format='rgb24')):
                        out.mux(pkt)
                for frame in inp.decode(vs):
                    if frame.pts is None:
                        continue
                    t = float(frame.pts * vs.time_base)
                    t0 = t if t0 is None else t0
                    t -= t0
                    img = frame.to_ndarray(format='rgb24')
                    k = round(getattr(frame, 'rotation', 0) / 90) % 4      # counter-clockwise quarter turns
                    if k:
                        img = np.ascontiguousarray(np.rot90(img, k))
                    h, w = img.shape[:2]
                    s = min(1.0, 1440 / max(h, w))
                    W, H = int(w * s) // 2 * 2, int(h * s) // 2 * 2
                    if (W, H) != (w, h):
                        img = cv2.resize(img, (W, H), interpolation=cv2.INTER_AREA)
                    while prev is not None and i / 30 < t:
                        emit(prev); i += 1
                    prev = img
                    if dur:
                        job.update(phase='Converting', progress=[round(min(t, dur), 1), round(dur, 1)])
                while prev is not None and (i == 0 or i / 30 < (dur or 0)):
                    emit(prev); i += 1
                if os_ is None:
                    raise ValueError('No video frames found in the file.')
                for pkt in os_.encode():
                    out.mux(pkt)
        part.replace(dst)
    finally:
        part.unlink(missing_ok=True)
        Path(src).unlink(missing_ok=True)
    return dict(clip=name, video=video_info(name))


def safe_name(filename):
    stem = re.sub(r'[^A-Za-z0-9_-]+', '-', Path(filename).stem).strip('-')[:40] or 'video'
    return f'{stem}-{time.strftime("%Y%m%d-%H%M%S")}'


# ------------------------------------------------------------------ http ----

class Handler(BaseHTTPRequestHandler):
    protocol_version = 'HTTP/1.1'

    def log_message(self, *a):
        pass

    def send(self, code, body, ctype='application/json'):
        if not isinstance(body, bytes):
            body = json.dumps(body).encode('utf-8')
        self.send_response(code)
        self.send_header('Content-Type', ctype)
        self.send_header('Content-Length', str(len(body)))
        self.send_header('Cache-Control', 'no-store')
        self.end_headers()
        self.wfile.write(body)

    def body(self):
        n = int(self.headers.get('Content-Length') or 0)
        return json.loads(self.rfile.read(n) or b'{}')

    def do_GET(self):
        u = urlparse(self.path)
        q = {k: v[0] for k, v in parse_qs(u.query).items()}
        try:
            if u.path in ('/', '/index.html'):
                return self.send(200, PAGE.read_bytes(), 'text/html; charset=utf-8')
            if u.path == '/api/status':
                return self.send(200, dict(model=ENGINE.state, error=ENGINE.error, checkpoint=ENGINE.checkpoint,
                                           job=JOB.snapshot()))
            if u.path == '/api/checkpoints':
                return self.send(200, checkpoints())
            if u.path == '/api/sources':
                return self.send(200, sources())
            if u.path == '/api/query':
                return self.send(200, query_detail(q['key']))
            if u.path == '/api/video_info':
                if not NAME_RE.match(q.get('clip', '')):
                    return self.send(400, dict(error='bad clip name'))
                return self.send(200, video_info(q['clip']))
            if u.path.startswith('/video/'):
                return self.video(unquote(u.path[len('/video/'):]).removesuffix('.mp4'))
            self.send(404, dict(error='not found'))
        except (BrokenPipeError, ConnectionResetError, ConnectionAbortedError):
            pass
        except Exception as e:
            self.send(500, dict(error=f'{type(e).__name__}: {e}'))

    def video(self, clip):
        if not NAME_RE.match(clip):
            return self.send(400, dict(error='bad clip name'))
        path = D.clip_path(clip)
        if not path.exists():
            return self.send(404, dict(error='no such video'))
        size = path.stat().st_size
        start, end = 0, size - 1
        m = re.match(r'bytes=(\d*)-(\d*)', self.headers.get('Range', ''))
        if m:
            if m.group(1):
                start = int(m.group(1)); end = int(m.group(2)) if m.group(2) else size - 1
            else:
                start = size - int(m.group(2))
            end = min(end, size - 1)
        self.send_response(206 if m else 200)
        self.send_header('Content-Type', 'video/mp4')
        self.send_header('Accept-Ranges', 'bytes')
        self.send_header('Content-Length', str(end - start + 1))
        if m:
            self.send_header('Content-Range', f'bytes {start}-{end}/{size}')
        self.end_headers()
        with path.open('rb') as f:
            f.seek(start)
            left = end - start + 1
            while left > 0:
                chunk = f.read(min(1 << 20, left))
                if not chunk:
                    break
                self.wfile.write(chunk)
                left -= len(chunk)

    def do_POST(self):
        u = urlparse(self.path)
        try:
            if u.path == '/api/run':
                req = self.body()
                if not NAME_RE.match(req.get('clip', '')):
                    return self.send(400, dict(error='bad clip name'))
                if int(req['query_frame']) < 1:
                    return self.send(400, dict(error='The last frame to search must be after frame 0.'))
                jid = JOB.start('search', lambda job: search(req, job))
                return self.send(202 if jid else 409, dict(job=jid) if jid else dict(error='Another task is running.'))
            if u.path == '/api/save':
                req = self.body()
                QUERIES.mkdir(parents=True, exist_ok=True)
                name = f"{time.strftime('%Y%m%d-%H%M%S')}-{req.get('query', {}).get('clip', 'query')[:40]}.json"
                write_json(QUERIES / name, dict(req, saved=time.strftime('%Y-%m-%d %H:%M:%S')))
                return self.send(200, dict(saved=f'output/manual-test/queries/{name}'))
            self.send(404, dict(error='not found'))
        except Exception as e:
            self.send(500, dict(error=f'{type(e).__name__}: {e}'))

    def do_PUT(self):
        u = urlparse(self.path)
        if u.path != '/api/upload':
            return self.send(404, dict(error='not found'))
        q = {k: v[0] for k, v in parse_qs(u.query).items()}
        if JOB.snapshot()['state'] == 'running':
            return self.send(409, dict(error='Another task is running.'))
        name = safe_name(q.get('name', 'video'))
        TMP.mkdir(parents=True, exist_ok=True)
        raw = TMP / f'{name}.upload'
        left = int(self.headers.get('Content-Length') or 0)
        with raw.open('wb') as f:
            while left > 0:
                chunk = self.rfile.read(min(1 << 20, left))
                if not chunk:
                    break
                f.write(chunk); left -= len(chunk)
        if left:
            raw.unlink(missing_ok=True)
            return self.send(400, dict(error='Upload incomplete.'))
        jid = JOB.start('convert', lambda job: convert(raw, name, job))
        if not jid:
            raw.unlink(missing_ok=True)
        self.send(202 if jid else 409, dict(job=jid) if jid else dict(error='Another task is running.'))


class Server(ThreadingHTTPServer):
    """Refuses a port another program already listens on (the default SO_REUSEADDR would share it on Windows)."""
    allow_reuse_address = False
    daemon_threads = True

    def server_bind(self):
        if hasattr(socket, 'SO_EXCLUSIVEADDRUSE'):
            self.socket.setsockopt(socket.SOL_SOCKET, socket.SO_EXCLUSIVEADDRUSE, 1)
        super().server_bind()


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--port', type=int, default=8765, help='first port to try (the next free one is used)')
    p.add_argument('--open', action='store_true', help='open the page in the browser')
    p.add_argument('--no-preload', action='store_true', help='load the model on the first search instead of now')
    a = p.parse_args()
    if TMP.exists():
        shutil.rmtree(TMP, ignore_errors=True)        # leftovers of an interrupted upload or search
    for port in range(a.port, a.port + 20):
        try:
            server = Server(('127.0.0.1', port), Handler)
            break
        except OSError:
            continue
    else:
        raise SystemExit(f'No free port in {a.port}-{a.port + 19}.')
    url = f'http://127.0.0.1:{server.server_port}/'
    print(f'MANUAL {url}', flush=True)
    if not a.no_preload:
        cks = checkpoints()
        if cks:
            def preload():
                with ENGINE.lock:
                    try:
                        ENGINE.load(cks[0]['path'])
                    except BaseException:
                        traceback.print_exc()
            threading.Thread(target=preload, daemon=True).start()
    if a.open:
        webbrowser.open(url)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass


if __name__ == '__main__':
    main()
