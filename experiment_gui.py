"""Local desktop launcher for the CS230 experiments: the JPoSE EPIC baseline,
TempRet (EK-100 retrieval), EAGLE (Ego4D VQ2D) and the Ego-Exo4D review tools."""
import argparse
import json
import os
from pathlib import Path
import queue
import re
import shlex
import subprocess
import sys
import threading
import time
import tkinter as tk
from tkinter import filedialog, messagebox, ttk
import webbrowser
from experiment_paths import HERE, ROOT

LOCAL_PYTHON = Path('.venv') / ('Scripts/python.exe' if os.name == 'nt' else 'bin/python')
PYTHON = './' + LOCAL_PYTHON.as_posix() if (HERE / LOCAL_PYTHON).is_file() else 'python'
# GPU training environment (CUDA PyTorch, open_clip, transformers, PyAV). Kept
# outside Google Drive: <drive root>/cs230-venv/env, or set CS230_PYTHON.
GPU_ENV = Path(ROOT.anchor) / 'cs230-venv' / 'env' / ('Scripts/python.exe' if os.name == 'nt' else 'bin/python')
TRAIN_PYTHON = os.environ.get('CS230_PYTHON') or (str(GPU_ENV) if GPU_ENV.is_file() else PYTHON)
CACHE = Path(os.environ.get('CS230_CACHE', Path(ROOT.anchor) / 'cs230-cache'))
PROGRESS = re.compile(r'(?:step|iter|\[)\s*(\d+)/(\d+)')

def format_command(argv):
    """Resolve portable arguments into an absolute, copyable shell command."""
    argv = [str(arg) for arg in argv]
    if argv[0] == 'python':
        interpreter = Path(sys.executable)
        if os.name == 'nt' and interpreter.name.lower() == 'pythonw.exe':
            interpreter = interpreter.with_name('python.exe')
        argv[0] = os.path.abspath(interpreter)
    else:
        argv[0] = os.path.abspath(HERE / argv[0])
    for i in range(1, len(argv)):
        if argv[i].startswith(('./', '../', '.\\', '..\\')):
            argv[i] = os.path.abspath(HERE / argv[i])
    if os.name == 'nt':
        # PowerShell needs the call operator for a quoted executable. Single
        # quotes also keep dollar signs and backticks literal in every argument.
        return '& ' + ' '.join("'" + str(arg).replace("'", "''") + "'" for arg in argv)
    return shlex.join([str(arg) for arg in argv])


def _int(value, name, lo, hi):
    try:
        n = int(value)
    except ValueError:
        raise ValueError(f'{name} must be a whole number.')
    if not lo <= n <= hi:
        raise ValueError(f'{name} must be between {lo:,} and {hi:,}.')
    return str(n)


def _float(value, name):
    try:
        x = float(value)
    except ValueError:
        raise ValueError(f'{name} must be a number.')
    if x <= 0:
        raise ValueError(f'{name} must be positive.')
    return str(x)


def tempret_args(o):
    a = ['--mode', o['mode'], '--backbone', o['backbone'],
         '--frames', _int(o['frames'], 'Frames per clip', 1, 16),
         '--batch-size', _int(o['batch'], 'Batch size', 2, 4096), '--lr', _float(o['lr'], 'Learning rate'),
         '--max-train-clips', _int(o['max_train'], 'Max train clips', 0, 10**7),
         '--max-test-clips', _int(o['max_test'], 'Max test clips', 0, 10**7)]
    if o['no_temporal']: a.append('--no-temporal')
    if o['no_rerank']: a.append('--no-rerank')
    return a


def eagle_args(o):
    a = ['--iters', _int(o['iters'], 'Iterations', 1, 10**7), '--batch-size', _int(o['batch'], 'Batch size', 1, 256),
         '--lr', _float(o['lr'], 'Learning rate'), '--seq-len', _int(o['seq'], 'Sequence length', 2, 32),
         '--rho', _float(o['rho'], 'rho'),
         '--max-train-queries', _int(o['max_train'], 'Max train queries', 0, 10**7),
         '--max-eval-queries', _int(o['max_eval'], 'Max eval queries', 0, 10**7)]
    for flag in ('no_amm', 'no_glm', 'no_visor'):
        if o[flag]: a.append('--' + flag.replace('_', '-'))
    return a


def commands(epochs='1', tempret=None, eagle=None, checkpoint=None):
    n = int(epochs)
    if not 1 <= n <= 10000:
        raise ValueError('Choose between 1 and 10,000 epochs.')
    cmds = {
        'EPIC pretrained evaluation': [PYTHON, '-u', './run_epic_baseline.py', '--epochs', '0'],
        'EPIC training and validation': [PYTHON, '-u', './run_epic_baseline.py', '--epochs', str(n)],
        'EPIC retrieval error analysis': [PYTHON, '-u', './inspect_epic_retrieval.py'],
        'Ego-Exo4D dataset check': [PYTHON, '-u', './check_egoexo_dataset.py', '--sample-json', '10'],
        'Ego-Exo4D prepare query review': [PYTHON, '-u', './prepare_egoexo_review.py'],
    }
    if tempret is not None:
        base = [TRAIN_PYTHON, '-u', './run_tempret.py'] + tempret_args(tempret)
        ep = _int(tempret['epochs'], 'Epochs', 1, 10000)
        cmds['TempRet decode frames + cache features'] = base + ['--epochs', ep, '--extract-only']
        cmds['TempRet train + evaluate'] = base + ['--epochs', ep]
        cmds['TempRet zero-shot CLIP evaluation'] = base + ['--epochs', '0']
        if checkpoint:
            cmds['TempRet evaluate checkpoint'] = base + ['--epochs', '0', '--checkpoint', checkpoint]
            cmds['TempRet continue training'] = base + ['--epochs', ep, '--checkpoint', checkpoint]
            cmds['TempRet train reranker only'] = base + ['--epochs', '0', '--rerank-only', '--checkpoint', checkpoint]
    if eagle is not None:
        base = [TRAIN_PYTHON, '-u', './run_eagle.py'] + eagle_args(eagle)
        cmds['EAGLE prepare SAM pseudo-masks'] = base + ['--stage', 'prepare']
        cmds['EAGLE train'] = base + ['--stage', 'train']
        cmds['EAGLE prepare + train + evaluate'] = base + ['--stage', 'all']
        if checkpoint:
            cmds['EAGLE evaluate checkpoint'] = base + ['--stage', 'evaluate', '--checkpoint', checkpoint]
            cmds['EAGLE continue training'] = base + ['--stage', 'train', '--checkpoint', checkpoint]
    return cmds


def _load(p):
    try:
        return json.loads(p.read_text(encoding='utf-8'))
    except (OSError, ValueError):
        return None


def _mir(m):
    return f"mAP {m['mAP']['AVG']:.2f} · nDCG {m['nDCG']['AVG']:.2f}  (V2T {m['mAP']['V2T']:.1f}/{m['nDCG']['V2T']:.1f}, T2V {m['mAP']['T2V']:.1f}/{m['nDCG']['T2V']:.1f})"


def saved_metrics():
    """(run, experiment, metrics text, sort key) rows for every saved result."""
    rows = []
    for run in (HERE / 'output').glob('epic-baseline-*'):
        for stage, label in [('pretrained', 'JPoSE pretrained'), ('trained-validation', 'JPoSE trained')]:
            result = _load(run / (stage + '.json'))
            try:
                m = result['metrics']
                epochs = (_load(run / 'run.json') or {}).get('epochs', '?')
                rows.append((run.name, label if stage == 'pretrained' else f'{label} ({epochs} ep)',
                             f"nDCG {m['nDCG']['AVG']:.3f} · mAP {m['mAP']['AVG']:.3f}"))
            except (TypeError, KeyError):
                continue
    for run in (HERE / 'output').glob('tempret-*'):
        cfg = _load(run / 'config.json') or {}
        desc = f"{cfg.get('backbone', '?')} {cfg.get('mode', '')}"
        scope = 'full' if cfg.get('full_benchmark') else f"subset {cfg.get('test_clips', '?')} clips"
        res = _load(run / 'metrics.json')
        if res and 'final' in res:
            ep = cfg.get('epochs', 0)
            name = (f"TempRet {desc}, {ep} ep" if ep else 'TempRet reranker stage' if cfg.get('rerank_only')
                    else 'TempRet checkpoint eval' if cfg.get('checkpoint') else 'Zero-shot CLIP')
            for kind in ('rerank', 'dual'):
                if kind in res['final']:
                    alpha = f" a={res['final']['rerank_alpha']}" if kind == 'rerank' and 'rerank_alpha' in res['final'] else ''
                    rows.append((run.name, f"{name} [{'reranked' + alpha if kind == 'rerank' else 'dual'}; {scope}]", _mir(res['final'][kind])))
        else:
            hist = [json.loads(l) for l in (run / 'history.jsonl').read_text().splitlines()] if (run / 'history.jsonl').exists() else []
            if hist:
                h = hist[-1]
                state = 'stopped/partial' if not (run / 'COMPLETE').exists() else 'done'
                rows.append((run.name, f"TempRet {desc} [epoch {h['epoch']} of {cfg.get('epochs', '?')}, {state}]",
                             f"mAP {h['mAP_AVG']:.2f} · nDCG {h['nDCG_AVG']:.2f} (dual, {scope})"))
    for run in (HERE / 'output').glob('eagle-*'):
        res = _load(run / 'metrics.json')
        if res and res.get('final'):
            m = res['final']
            rows.append((run.name, f"EAGLE [{res.get('split', 'val')}, {m.get('queries', '?')} queries]",
                         f"tAP25 {m.get('tAP25', 0):.3f} · stAP25 {m.get('stAP25', 0):.3f} · Rec {m.get('Rec', 0):.2f} · Succ {m.get('Succ', 0):.2f}"))
        elif (run / 'history.jsonl').exists():
            lines = (run / 'history.jsonl').read_text().splitlines()
            if lines:
                h = json.loads(lines[-1])
                rows.append((run.name, f"EAGLE training [iter {h['iter']}]",
                             f"L_seg {h['seg']:.4f} · L_tck {h['tck']:.5f} · frames IoU>0.5 {h['frame_iou50']:.3f}"))
    rows.sort(key=lambda r: (re.search(r'\d{8}-\d{6}', r[0]) or re.search('', '')).group(0), reverse=True)  # newest first
    reference = HERE / 'results' / 'baseline.json'
    if reference.exists():
        for item in json.loads(reference.read_text(encoding='utf-8'))['models']:
            rows.append(('reference-20261107', item['model'] + ' [reference]',
                         f"nDCG {item['nDCG']:.3f} · mAP {item['mAP']:.3f}"))
    rows.append(('paper', 'TempRet paper, EK-100 MIR test [reference]', 'mAP 67.97 · nDCG 82.92  (V2T 70.91/84.42, T2V 65.03/81.41)'))
    rows.append(('paper', 'EAGLE paper, VQ2D val [reference]', 'tAP25 0.470 · stAP25 0.420 · Rec 52.09 · Succ 61.29'))
    return rows


def data_summary():
    """Short description of locally available data for each new experiment."""
    epic = sorted(p.stem for p in (ROOT / 'EPIC-KITCHENS').glob('P*/videos/*.MP4'))
    clips = ROOT / 'Ego4D' / 'v2' / 'clips'
    n_clips = len(list(clips.glob('*.mp4'))) if clips.is_dir() else 0
    partial = len(list(clips.glob('*.part'))) if clips.is_dir() else 0
    visor = (ROOT / 'EPIC-KITCHENS-VISOR' / 'GroundTruth-SparseAnnotations').is_dir()
    parts = sorted({v.split('_')[0] for v in epic})
    return (f"EPIC videos: {len(epic)} ({', '.join(parts) or 'none'}) of 700 · "
            f"Ego4D VQ2D clips: {n_clips:,} of 5,823{f' (+{partial} downloading)' if partial else ''} · "
            f"VISOR: {'found' if visor else 'not downloaded'}")


class Launcher:
    def __init__(self, root):
        self.root = root
        self.root.title('CS230 · EPIC, Ego4D & Ego-Exo4D experiments')
        self.root.geometry('1280x940')
        self.root.minsize(1060, 760)
        self.messages = queue.Queue()
        self.process = None
        self.busy = False
        self.stopping = False
        self.run_dir = None
        self.stage = ''
        self.offsets = {}
        self.action_buttons = []
        self.epochs = tk.StringVar(value='1')
        self.status = tk.StringVar(value='Ready · select an experiment')
        self.command = tk.StringVar(value='Copied commands use absolute paths detected on this machine.')
        self.dataset_status = tk.StringVar(value='Run the dataset check to inspect local Ego-Exo4D coverage.')
        self.data_status = tk.StringVar(value=data_summary())
        self.progress_text = tk.StringVar(value='')
        self.tr = {k: tk.StringVar(value=v) for k, v in dict(
            mode='frozen', backbone='ViT-L-14', epochs='10', batch='64', lr='1.8e-5', frames='8',
            max_train='0', max_test='0').items()}
        self.tr.update(no_temporal=tk.BooleanVar(value=False), no_rerank=tk.BooleanVar(value=False))
        self.eg = {k: tk.StringVar(value=v) for k, v in dict(
            iters='25000', batch='4', lr='0.0025', seq='4', rho='1.0', max_train='0', max_eval='0').items()}
        self.eg.update(no_amm=tk.BooleanVar(value=False), no_glm=tk.BooleanVar(value=False),
                       no_visor=tk.BooleanVar(value=False))
        style = ttk.Style()
        style.theme_use('clam')
        style.configure('.', font=('Segoe UI', 10))
        style.configure('Title.TLabel', font=('Segoe UI', 21, 'bold'))
        style.configure('Sub.TLabel', foreground='#536578')
        style.configure('TButton', padding=(10, 6))
        style.configure('Treeview', rowheight=25)
        outer = ttk.Frame(root, padding=18)
        outer.pack(fill='both', expand=True)
        ttk.Label(outer, text='CS230 experiment launcher', style='Title.TLabel').pack(anchor='w')
        ttk.Label(outer, text=f'Python (baseline tools): {PYTHON}    ·    Python (GPU training): {TRAIN_PYTHON}', style='Sub.TLabel').pack(anchor='w', pady=(2, 0))
        ttk.Label(outer, textvariable=self.data_status, style='Sub.TLabel').pack(anchor='w', pady=(0, 8))
        tabs = ttk.Notebook(outer)
        tabs.pack(fill='x')
        self.build_tempret(tabs)
        self.build_eagle(tabs)
        self.build_jpose(tabs)
        self.build_egoexo(tabs)
        bar = ttk.Frame(outer); bar.pack(fill='x', pady=(10, 4))
        ttk.Label(bar, textvariable=self.status).pack(side='left')
        self.stop_button = ttk.Button(bar, text='Stop current run', command=self.stop, state='disabled')
        self.stop_button.pack(side='right')
        ttk.Button(bar, text='Open results folder', command=lambda: self.open_path(HERE / 'output')).pack(side='right', padx=6)
        ttk.Button(bar, text='Open checkpoints folder', command=lambda: self.open_path(CACHE)).pack(side='right')
        prog = ttk.Frame(outer); prog.pack(fill='x', pady=(0, 6))
        self.progress = ttk.Progressbar(prog, mode='determinate', maximum=1.0)
        self.progress.pack(side='left', fill='x', expand=True)
        ttk.Label(prog, textvariable=self.progress_text, width=48).pack(side='left', padx=(8, 0))
        command_box = ttk.LabelFrame(outer, text='Command (PowerShell)' if os.name == 'nt' else 'Command (shell)', padding=6)
        command_box.pack(fill='x', pady=(0, 6))
        line = ttk.Frame(command_box); line.pack(fill='x')
        ttk.Entry(line, textvariable=self.command, state='readonly').pack(side='left', fill='x', expand=True)
        ttk.Button(line, text='Copy command', command=self.copy_command).pack(side='left', padx=(6, 0))
        results = ttk.LabelFrame(outer, text='Saved results · double-click a row to open its folder', padding=6)
        results.pack(fill='x')
        self.table = ttk.Treeview(results, columns=('run', 'model', 'metrics'), show='headings', height=7)
        for key, label, width in [('run', 'Run', 230), ('model', 'Experiment', 420), ('metrics', 'Metrics', 560)]:
            self.table.heading(key, text=label); self.table.column(key, width=width, stretch=key != 'run')
        self.table.pack(fill='x')
        self.table.bind('<Double-1>', self.open_selected_run)
        foot = ttk.Frame(results); foot.pack(fill='x')
        ttk.Label(foot, textvariable=self.dataset_status, style='Sub.TLabel', wraplength=1000).pack(side='left', pady=(4, 0))
        ttk.Button(foot, text='Refresh', command=self.refresh).pack(side='right', pady=(4, 0))
        logbox = ttk.LabelFrame(outer, text='Live output', padding=6)
        logbox.pack(fill='both', expand=True, pady=(8, 0))
        self.log = tk.Text(logbox, height=8, bg='#132133', fg='#d6e6f5', insertbackground='white', font=('Consolas', 10), wrap='word', state='disabled')
        scroll = ttk.Scrollbar(logbox, command=self.log.yview)
        self.log.configure(yscrollcommand=scroll.set)
        scroll.pack(side='right', fill='y'); self.log.pack(fill='both', expand=True)
        self.refresh()
        root.protocol('WM_DELETE_WINDOW', self.close)
        root.after(400, self.poll)

    # ------------------------------------------------------------ tabs ----
    def form(self, parent, fields, vars_):
        grid = ttk.Frame(parent); grid.pack(fill='x', pady=(0, 6))
        for i, (key, label, widget, extra) in enumerate(fields):
            r, c = divmod(i, 4)
            ttk.Label(grid, text=label).grid(row=r, column=2 * c, sticky='w', padx=(0 if c == 0 else 14, 4), pady=2)
            if widget == 'combo':
                w = ttk.Combobox(grid, textvariable=vars_[key], values=extra, width=12, state='readonly')
            else:
                w = ttk.Entry(grid, textvariable=vars_[key], width=10)
            w.grid(row=r, column=2 * c + 1, sticky='w', pady=2)
        return grid

    def checks(self, parent, items, vars_):
        line = ttk.Frame(parent); line.pack(fill='x', pady=(0, 6))
        for key, label in items:
            ttk.Checkbutton(line, text=label, variable=vars_[key]).pack(side='left', padx=(0, 16))

    def buttons(self, parent, items):
        line = ttk.Frame(parent); line.pack(fill='x')
        for text, action in items:
            b = ttk.Button(line, text=text, command=action if callable(action) else (lambda a=action: self.start(a)))
            b.pack(side='left', padx=(0, 6)); self.action_buttons.append(b)

    def build_tempret(self, tabs):
        tab = ttk.Frame(tabs, padding=12); tabs.add(tab, text='TempRet · EK-100 retrieval')
        ttk.Label(tab, text='CLIP dual encoder + temporal transformer, SMS loss, ITM cross-encoder reranking (arXiv 2605.24470).', style='Sub.TLabel').pack(anchor='w', pady=(0, 6))
        self.form(tab, [
            ('mode', 'Mode', 'combo', ['frozen', 'full']), ('backbone', 'CLIP backbone', 'combo', ['ViT-L-14', 'ViT-B-16', 'ViT-B-32']),
            ('epochs', 'Epochs', 'entry', None), ('batch', 'Batch size', 'entry', None),
            ('lr', 'Learning rate', 'entry', None), ('frames', 'Frames / clip', 'entry', None),
            ('max_train', 'Max train clips (0 = all)', 'entry', None), ('max_test', 'Max test clips (0 = all)', 'entry', None)], self.tr)
        self.checks(tab, [('no_temporal', 'Ablation: no temporal transformer'), ('no_rerank', 'Ablation: no reranking')], self.tr)
        self.buttons(tab, [('Decode frames + cache features', 'TempRet decode frames + cache features'),
                           ('Train + evaluate', 'TempRet train + evaluate'),
                           ('Zero-shot CLIP eval', 'TempRet zero-shot CLIP evaluation'),
                           ('Evaluate checkpoint…', lambda: self.with_checkpoint('TempRet evaluate checkpoint', 'tempret')),
                           ('Continue from checkpoint…', lambda: self.with_checkpoint('TempRet continue training', 'tempret')),
                           ('Train reranker on checkpoint…', lambda: self.with_checkpoint('TempRet train reranker only', 'tempret'))])
        ttk.Label(tab, text='frozen = cached CLIP image features (fits a 12 GB GPU); full = end-to-end like the paper (needs much more memory; use ViT-B-16 and a small batch).\n'
                  'Evaluation uses only locally available videos and is labelled “subset” unless all 9,668 test clips are present.', style='Sub.TLabel').pack(anchor='w', pady=(8, 0))

    def build_eagle(self, tabs):
        tab = ttk.Frame(tabs, padding=12); tabs.add(tab, text='EAGLE · Ego4D VQ2D')
        ttk.Label(tab, text='DINOv2 backbone, appearance memory (AMM) + geometry memory (GLM), SAM pseudo-masks (arXiv 2511.08007, 2D branch).', style='Sub.TLabel').pack(anchor='w', pady=(0, 6))
        self.form(tab, [
            ('iters', 'Iterations', 'entry', None), ('batch', 'Batch (sequences)', 'entry', None),
            ('lr', 'Peak learning rate', 'entry', None), ('seq', 'Frames / sequence', 'entry', None),
            ('rho', 'ρ (L_tck weight)', 'entry', None), ('max_train', 'Max train queries (0 = all)', 'entry', None),
            ('max_eval', 'Max eval queries (0 = all)', 'entry', None)], self.eg)
        self.checks(tab, [('no_amm', 'Ablation: no AMM'), ('no_glm', 'Ablation: no GLM'), ('no_visor', 'Skip VISOR')], self.eg)
        self.buttons(tab, [('Prepare SAM pseudo-masks', 'EAGLE prepare SAM pseudo-masks'), ('Train', 'EAGLE train'),
                           ('Prepare + train + evaluate', 'EAGLE prepare + train + evaluate'),
                           ('Evaluate checkpoint…', lambda: self.with_checkpoint('EAGLE evaluate checkpoint', 'eagle')),
                           ('Continue from checkpoint…', lambda: self.with_checkpoint('EAGLE continue training', 'eagle'))])
        ttk.Label(tab, text='Uses VQ2D clips in ../Ego4D/v2/clips and VISOR in ../EPIC-KITCHENS-VISOR when present. Prepare is resumable: rerun it after more clips download.\n'
                  'Evaluation searches every frame before the query frame (5 fps) and scores with the official VQ2D metrics from ../VQLoC.', style='Sub.TLabel').pack(anchor='w', pady=(8, 0))

    def build_jpose(self, tabs):
        tab = ttk.Frame(tabs, padding=12); tabs.add(tab, text='EPIC · JPoSE baseline')
        line = ttk.Frame(tab); line.pack(fill='x', pady=(0, 6))
        ttk.Label(line, text='Training epochs').pack(side='left')
        ttk.Spinbox(line, from_=1, to=10000, textvariable=self.epochs, width=6).pack(side='left', padx=8)
        self.buttons(tab, [('Evaluate pretrained model', 'EPIC pretrained evaluation'),
                           ('Train from scratch + validate', 'EPIC training and validation'),
                           ('Export retrieval errors', 'EPIC retrieval error analysis')])
        ttk.Label(tab, text='Joint-Part-of-Speech-Embeddings on the supplied features. Matches the repository README evaluation (nDCG 0.690, mAP 0.734).', style='Sub.TLabel').pack(anchor='w', pady=(8, 0))

    def build_egoexo(self, tabs):
        tab = ttk.Frame(tabs, padding=12); tabs.add(tab, text='Ego-Exo4D · data review')
        self.buttons(tab, [('Check videos & pose annotations', 'Ego-Exo4D dataset check'),
                           ('Prepare 10-video query review', 'Ego-Exo4D prepare query review'),
                           ('Open query review / label objects', self.open_review)])
        ttk.Label(tab, text='No object-localization scores here; query boxes, response tracks and model inference are still required.', style='Sub.TLabel').pack(anchor='w', pady=(8, 0))

    # --------------------------------------------------------- actions ----
    def options(self, vars_):
        return {k: v.get() for k, v in vars_.items()}

    def all_commands(self, checkpoint=None):
        return commands(self.epochs.get(), self.options(self.tr), self.options(self.eg), checkpoint)

    def with_checkpoint(self, action, kind):
        initial = CACHE / kind / 'checkpoints'
        path = filedialog.askopenfilename(title='Choose a checkpoint', initialdir=str(initial if initial.is_dir() else CACHE),
                                          filetypes=[('PyTorch checkpoint', '*.pt *.pth'), ('All files', '*.*')])
        if path:
            self.start(action, checkpoint=path)

    def append(self, text):
        self.log.configure(state='normal')
        self.log.insert('end', text.replace('\r', '\n'))
        if int(self.log.index('end-1c').split('.')[0]) > 1500:
            self.log.delete('1.0', '500.0')
        self.log.see('end'); self.log.configure(state='disabled')

    def refresh(self):
        self.table.delete(*self.table.get_children())
        for row in saved_metrics()[:60]:
            self.table.insert('', 'end', values=row)
        self.data_status.set(data_summary())
        for p in sorted((HERE / 'output').glob('egoexo-check-*/dataset-check.json'), reverse=True):
            try:
                d = json.loads(p.read_text())
                self.dataset_status.set(f"Ego-Exo4D check: {d['status']} · {d['takes']} takes · {d['mp4_streams']:,} streams · {d['video_bytes']/1e9:.2f} GB · {len(d['takes_missing_pose_annotations'])} takes missing pose annotations")
                break
            except (OSError, ValueError, KeyError):
                continue

    def start(self, action, checkpoint=None):
        if self.busy:
            return
        try:
            cmd = self.all_commands(checkpoint)[action]
        except ValueError as e:
            messagebox.showerror('Invalid setting', str(e)); return
        self.command.set(format_command(cmd))
        self.busy, self.stopping = True, False
        self.run_dir, self.stage, self.offsets = None, '', {}
        self.progress['value'] = 0; self.progress_text.set('')
        for button in self.action_buttons: button.configure(state='disabled')
        self.stop_button.configure(state='normal')
        self.status.set(f'Running · {action}')
        self.append(f'\n── {action} ──\n{self.command.get()}\n')
        threading.Thread(target=self.worker, args=(cmd,), daemon=True).start()

    def worker(self, cmd):
        try:
            logs = HERE / 'output' / 'launcher-logs'; logs.mkdir(parents=True, exist_ok=True)
            logfile = logs / (time.strftime('%Y%m%d-%H%M%S') + '.log')
            env = os.environ.copy(); env['PATH'] = str(Path(sys.executable).parent) + os.pathsep + env.get('PATH', ''); env['PYTHONDONTWRITEBYTECODE'] = '1'; env['PYTHONIOENCODING'] = 'utf-8'
            if cmd[0] != TRAIN_PYTHON or TRAIN_PYTHON == PYTHON:
                env['OMP_NUM_THREADS'] = '4'; env['MKL_NUM_THREADS'] = '4'
            with logfile.open('w', encoding='utf-8') as f:
                self.process = subprocess.Popen(cmd, cwd=HERE, env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                                text=True, encoding='utf-8', errors='replace',
                                                creationflags=getattr(subprocess, 'CREATE_NO_WINDOW', 0))
                if self.stopping: self.process.terminate()
                for line in self.process.stdout:
                    f.write(line); f.flush(); self.messages.put(('line', line))
                code = self.process.wait()
            self.messages.put(('done', code))
        except Exception as e:
            self.messages.put(('line', f'ERROR: {e}\n')); self.messages.put(('done', -1))

    def poll(self):
        refresh_due = False
        try:
            while True:
                kind, value = self.messages.get_nowait()
                if kind == 'line':
                    self.append(value)
                    if value.startswith('Results: '):
                        p = Path(value.strip()[9:]).resolve()
                        if p.is_relative_to((HERE / 'output').resolve()): self.run_dir = p
                    if value.startswith('Starting '):
                        self.stage = value.strip()[9:]; self.progress['value'] = 0
                        self.progress_text.set(self.stage)
                        refresh_due = True
                    m = PROGRESS.search(value)
                    if m and int(m.group(2)) > 0:
                        done, total = int(m.group(1)), int(m.group(2))
                        self.progress['value'] = done / total
                        self.progress_text.set(f'{self.stage} · {done:,}/{total:,}')
                else:
                    self.read_stage_log()
                    self.busy = False
                    self.status.set('Stopped · partial results retained' if self.stopping else ('Completed successfully' if value == 0 else f'Failed · exit code {value}; see output below'))
                    self.stop_button.configure(state='disabled')
                    for button in self.action_buttons: button.configure(state='normal')
                    refresh_due = True
        except queue.Empty:
            pass
        if refresh_due: self.refresh()
        if self.busy: self.read_stage_log()
        self.root.after(500, self.poll)

    def read_stage_log(self):
        if not self.run_dir or not self.stage: return
        p = self.run_dir / (self.stage + '.log')
        try:
            with p.open('rb') as f:
                f.seek(self.offsets.get(str(p), 0)); data = f.read(); self.offsets[str(p)] = f.tell()
            if data: self.append(data.decode('utf-8', errors='replace'))
        except OSError:
            pass

    def stop(self):
        if self.busy:
            self.stopping = True
            if self.process and self.process.poll() is None: self.process.terminate()
            self.status.set('Stopping current run…')

    def open_path(self, path):
        if not path.exists():
            messagebox.showinfo('Not available', f'Not found yet:\n{path}'); return
        if os.name == 'nt': os.startfile(str(path))
        else: webbrowser.open(path.as_uri())

    def open_review(self):
        self.open_path(HERE / 'output' / 'egoexo-query-review' / 'review.html')

    def open_selected_run(self, event=None):
        selected = self.table.selection()
        if selected:
            name = self.table.item(selected[0], 'values')[0]
            if name == 'paper':
                return
            self.open_path(HERE / 'results' if name.startswith('reference-') else HERE / 'output' / name)

    def copy_command(self):
        self.root.clipboard_clear(); self.root.clipboard_append(self.command.get()); self.root.update()

    def close(self):
        if self.busy and not messagebox.askyesno('Close launcher', 'Stop the active run and close the launcher?'):
            return
        self.stop(); self.root.destroy()

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--ui-test', action='store_true', help='Construct the interface, verify results load, then close')
    parser.add_argument('--integration-test', action='store_true', help='Run EPIC evaluation and Ego-Exo4D check through GUI callbacks, then close')
    args = parser.parse_args()
    root = tk.Tk()
    app = Launcher(root)
    if args.ui_test:
        assert app.table.get_children(), 'No saved metrics loaded'
        cmds = app.all_commands(checkpoint='x.pt')
        assert len(cmds) == 16, len(cmds)
        print('\n'.join(f'{k}: {format_command(v)}' for k, v in cmds.items() if k.startswith(('TempRet', 'EAGLE'))))
        root.after(500, root.destroy)
    if args.integration_test:
        test_actions = ['EPIC pretrained evaluation', 'Ego-Exo4D dataset check']
        test_results = []
        started = time.monotonic()
        test_output = HERE / 'output' / 'launcher-integration-test.json'
        def next_test():
            if time.monotonic() - started > 180:
                app.stop()
                test_output.write_text(json.dumps({'passed': False, 'error': 'timeout'}, indent=2))
                root.destroy(); return
            if app.busy:
                root.after(500, next_test); return
            if test_results and test_results[-1]['status'] == 'running':
                test_results[-1]['status'] = app.status.get()
                if app.status.get() != 'Completed successfully':
                    test_actions.clear()
            if test_actions:
                action = test_actions.pop(0)
                test_results.append(dict(action=action, status='running'))
                app.start(action)
                root.after(500, next_test)
            else:
                result = dict(passed=len(test_results) == 2 and all(r['status'] == 'Completed successfully' for r in test_results),
                              tests=test_results, saved_result_rows=len(app.table.get_children()),
                              dataset_summary=app.dataset_status.get())
                test_output.write_text(json.dumps(result, indent=2))
                print(json.dumps(result, indent=2), flush=True)
                root.destroy()
        root.after(200, next_test)
    root.mainloop()

if __name__ == '__main__':
    main()
