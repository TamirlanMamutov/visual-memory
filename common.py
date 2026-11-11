"""Shared helpers for the TempRet and EAGLE re-implementations."""
import json
import os
import random
import time
from pathlib import Path

from experiment_paths import HERE, ROOT, project_dir

# Large generated data (decoded frames, feature caches, checkpoints): <project>/cs230-cache.
CACHE = Path(os.environ.get('CS230_CACHE', project_dir('cs230-cache'))).expanduser().resolve()
OUTPUT = HERE / 'output'


def portable(path):
    """A path as it should be recorded in run files: from the cs230-cache folder onwards
    (e.g. 'cs230-cache/eagle/checkpoints/<run>/last.pt') when it lies in a cache, else relative
    to the project folder when inside it, so records stay valid after the folders move."""
    p = Path(path)
    parts = [x.lower() for x in p.parts]
    if CACHE.name.lower() in parts:
        i = len(parts) - 1 - parts[::-1].index(CACHE.name.lower())
        return Path(*p.parts[i:]).as_posix()
    try:
        return p.resolve().relative_to(ROOT).as_posix()
    except ValueError:
        return p.as_posix()


def make_run_dir(prefix, resume=None):
    out = Path(resume).resolve() if resume else OUTPUT / f'{prefix}-{time.strftime("%Y%m%d-%H%M%S")}'
    out.mkdir(parents=True, exist_ok=True)
    print(f'Results: {out}', flush=True)
    return out


def write_json(path, obj):
    atomic_write(path, lambda f: f.write(json.dumps(obj, indent=2, default=str).encode('utf-8')))


def atomic_write(path, write_fn):
    """Write via <path>.tmp + rename, so an interruption never leaves a
    half-written file under the final name (resume logic trusts existence)."""
    path = Path(path)
    tmp = path.with_name(path.name + '.tmp')
    with tmp.open('wb') as f:
        write_fn(f)
        f.flush(); os.fsync(f.fileno())
    os.replace(tmp, path)


def torch_save_atomic(obj, path):
    import torch
    atomic_write(path, lambda f: torch.save(obj, f))


def seed_everything(seed):
    import numpy as np
    import torch
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)


def device():
    import torch
    if not torch.cuda.is_available():
        print('WARNING: CUDA is not available; running on CPU will be very slow.', flush=True)
        return torch.device('cpu')
    return torch.device('cuda')


class Logger:
    """Print a line and append it to <run>/log.txt (the GUI streams stdout)."""

    def __init__(self, run_dir):
        self.path = Path(run_dir) / 'log.txt'

    def __call__(self, *parts):
        line = ' '.join(str(p) for p in parts)
        print(line, flush=True)
        with self.path.open('a', encoding='utf-8') as f:
            f.write(line + '\n')
