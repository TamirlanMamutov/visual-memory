"""Shared helpers for the TempRet and EAGLE re-implementations."""
import json
import os
import random
import time
from pathlib import Path

from experiment_paths import HERE, ROOT

# Large generated data (decoded frames, feature caches, checkpoints) is kept
# outside the synced Google Drive folder by default: <drive root>/cs230-cache.
CACHE = Path(os.environ.get('CS230_CACHE', Path(ROOT.anchor) / 'cs230-cache')).expanduser().resolve()
OUTPUT = HERE / 'output'


def make_run_dir(prefix, resume=None):
    out = Path(resume).resolve() if resume else OUTPUT / f'{prefix}-{time.strftime("%Y%m%d-%H%M%S")}'
    out.mkdir(parents=True, exist_ok=True)
    print(f'Results: {out}', flush=True)
    return out


def write_json(path, obj):
    Path(path).write_text(json.dumps(obj, indent=2, default=str), encoding='utf-8')


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
