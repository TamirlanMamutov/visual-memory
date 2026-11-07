"""Local desktop launcher for the existing CS230 experiment scripts."""
import argparse
import json
import os
from pathlib import Path
import queue
import subprocess
import sys
import threading
import time
import tkinter as tk
from tkinter import messagebox, ttk
import webbrowser
from experiment_paths import HERE, ROOT

PYTHON = str(Path(sys.executable).with_name('python.exe')) if os.name == 'nt' else sys.executable

def commands(epochs='1'):
    n = int(epochs)
    if not 1 <= n <= 10000:
        raise ValueError('Choose between 1 and 10,000 epochs.')
    return {
        'EPIC pretrained evaluation': [PYTHON, '-u', str(HERE / 'run_epic_baseline.py'), '--epochs', '0'],
        'EPIC training and validation': [PYTHON, '-u', str(HERE / 'run_epic_baseline.py'), '--epochs', str(n)],
        'EPIC retrieval error analysis': [PYTHON, '-u', str(HERE / 'inspect_epic_retrieval.py')],
        'Ego-Exo4D dataset check': [PYTHON, '-u', str(HERE / 'check_egoexo_dataset.py'), '--sample-json', '10'],
        'Ego-Exo4D prepare query review': [PYTHON, '-u', str(HERE / 'prepare_egoexo_review.py')],
    }

def saved_metrics():
    rows = []
    for run in sorted((HERE / 'output').glob('epic-baseline-*'), reverse=True):
        for stage, label in [('pretrained', 'Pretrained'), ('trained-validation', 'Trained')]:
            p = run / (stage + '.json')
            if p.exists():
                try:
                    result = json.loads(p.read_text(encoding='utf-8'))
                    m = result['metrics']
                    epochs = json.loads((run / 'run.json').read_text()).get('epochs', '?')
                    rows.append((run.name, label if stage == 'pretrained' else f'{label} ({epochs} epochs)',
                                 f"{m['nDCG']['AVG']:.3f}", f"{m['mAP']['AVG']:.3f}"))
                except (OSError, ValueError, KeyError):
                    continue
    reference = HERE / 'results' / 'baseline.json'
    if reference.exists():
        for item in json.loads(reference.read_text(encoding='utf-8'))['models']:
            rows.append(('reference-20261107', item['model'] + ' [reference]',
                         f"{item['nDCG']:.3f}", f"{item['mAP']:.3f}"))
    return rows

class Launcher:
    def __init__(self, root):
        self.root = root
        self.root.title('CS230 · EPIC & Ego-Exo4D experiments')
        self.root.geometry('1220x860')
        self.root.minsize(1000, 720)
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
        self.command = tk.StringVar(value='Commands use the Python interpreter running this app.')
        self.dataset_status = tk.StringVar(value='Run the dataset check to inspect local Ego-Exo4D coverage.')
        style = ttk.Style()
        style.theme_use('clam')
        style.configure('.', font=('Segoe UI', 10))
        style.configure('Title.TLabel', font=('Segoe UI', 23, 'bold'))
        style.configure('Sub.TLabel', foreground='#536578')
        style.configure('TButton', padding=(10, 8))
        style.configure('Treeview', rowheight=27)
        outer = ttk.Frame(root, padding=22)
        outer.pack(fill='both', expand=True)
        ttk.Label(outer, text='CS230 experiment launcher', style='Title.TLabel').pack(anchor='w')
        ttk.Label(outer, text='Run existing data locally · inspect results · keep upstream repositories unchanged', style='Sub.TLabel').pack(anchor='w', pady=(3, 14))
        ttk.Label(outer, text=f'Project: {ROOT}', style='Sub.TLabel').pack(anchor='w')
        ttk.Label(outer, text=f'Python: {PYTHON}', style='Sub.TLabel').pack(anchor='w', pady=(0, 12))
        controls = ttk.Frame(outer)
        controls.pack(fill='x')
        epic = ttk.LabelFrame(controls, text='EPIC-KITCHENS · retrieval baseline', padding=12)
        epic.pack(side='left', fill='both', expand=True, padx=(0, 8))
        ego = ttk.LabelFrame(controls, text='Ego-Exo4D · dataset & query review', padding=12)
        ego.pack(side='left', fill='both', expand=True)
        self.button(epic, 'Evaluate pretrained model', 'EPIC pretrained evaluation')
        line = ttk.Frame(epic); line.pack(fill='x', pady=4)
        ttk.Label(line, text='Training epochs').pack(side='left')
        ttk.Spinbox(line, from_=1, to=10000, textvariable=self.epochs, width=6).pack(side='left', padx=8)
        self.button(epic, 'Train from scratch + validate', 'EPIC training and validation')
        self.button(epic, 'Export retrieval errors', 'EPIC retrieval error analysis')
        ttk.Label(epic, text='Matches the repository README evaluation.\nOne epoch checks the pipeline; it is not the full paper.', style='Sub.TLabel').pack(anchor='w', pady=(8, 0))
        self.button(ego, 'Check videos & pose annotations', 'Ego-Exo4D dataset check')
        self.button(ego, 'Prepare 10-video query review', 'Ego-Exo4D prepare query review')
        ttk.Button(ego, text='Open query review / label objects', command=self.open_review).pack(fill='x', pady=3)
        ttk.Label(ego, text='No object-localization scores yet. Query boxes,\nresponse tracks and model inference are still required.', style='Sub.TLabel').pack(anchor='w', pady=(8, 0))
        bar = ttk.Frame(outer); bar.pack(fill='x', pady=(12, 6))
        ttk.Label(bar, textvariable=self.status).pack(side='left')
        self.stop_button = ttk.Button(bar, text='Stop current run', command=self.stop, state='disabled')
        self.stop_button.pack(side='right')
        ttk.Button(bar, text='Open results folder', command=lambda: self.open_path(HERE / 'output')).pack(side='right', padx=6)
        command_box = ttk.LabelFrame(outer, text='Command', padding=8)
        command_box.pack(fill='x', pady=(0, 8))
        ttk.Entry(command_box, textvariable=self.command, state='readonly').pack(fill='x')
        ttk.Button(command_box, text='Copy command', command=self.copy_command).pack(anchor='e', pady=(4, 0))
        results = ttk.LabelFrame(outer, text='Saved EPIC validation results · averages', padding=8)
        results.pack(fill='x')
        ttk.Button(results, text='Refresh results', command=self.refresh).pack(anchor='e', pady=(0, 4))
        self.table = ttk.Treeview(results, columns=('run', 'model', 'ndcg', 'map'), show='headings', height=4)
        for key, label, width in [('run', 'Run', 300), ('model', 'Model', 200), ('ndcg', 'nDCG', 90), ('map', 'mAP', 90)]:
            self.table.heading(key, text=label); self.table.column(key, width=width)
        self.table.pack(fill='x')
        self.table.bind('<Double-1>', self.open_selected_run)
        ttk.Label(results, textvariable=self.dataset_status, style='Sub.TLabel', wraplength=1100).pack(anchor='w', pady=(5, 0))
        logbox = ttk.LabelFrame(outer, text='Live output', padding=8)
        logbox.pack(fill='both', expand=True, pady=(10, 0))
        self.log = tk.Text(logbox, height=9, bg='#132133', fg='#d6e6f5', insertbackground='white', font=('Consolas', 10), wrap='word', state='disabled')
        scroll = ttk.Scrollbar(logbox, command=self.log.yview)
        self.log.configure(yscrollcommand=scroll.set)
        scroll.pack(side='right', fill='y'); self.log.pack(fill='both', expand=True)
        self.refresh()
        root.protocol('WM_DELETE_WINDOW', self.close)
        root.after(400, self.poll)

    def button(self, parent, text, action):
        button = ttk.Button(parent, text=text, command=lambda: self.start(action))
        button.pack(fill='x', pady=3)
        self.action_buttons.append(button)

    def append(self, text):
        self.log.configure(state='normal')
        self.log.insert('end', text.replace('\r', '\n'))
        if int(self.log.index('end-1c').split('.')[0]) > 600:
            self.log.delete('1.0', '200.0')
        self.log.see('end'); self.log.configure(state='disabled')

    def refresh(self):
        self.table.delete(*self.table.get_children())
        for row in saved_metrics()[:30]:
            self.table.insert('', 'end', values=row)
        for p in sorted((HERE / 'output').glob('egoexo-check-*/dataset-check.json'), reverse=True):
            try:
                d = json.loads(p.read_text())
                self.dataset_status.set(f"Ego-Exo4D check: {d['status']} · {d['takes']} takes · {d['mp4_streams']:,} streams · {d['video_bytes']/1e9:.2f} GB · {len(d['takes_missing_pose_annotations'])} takes missing pose annotations")
                break
            except (OSError, ValueError, KeyError):
                continue

    def start(self, action):
        if self.busy:
            return
        try:
            cmd = commands(self.epochs.get())[action]
        except ValueError as e:
            messagebox.showerror('Epoch count', str(e)); return
        self.command.set(subprocess.list2cmdline(cmd))
        self.busy, self.stopping = True, False
        self.run_dir, self.stage, self.offsets = None, '', {}
        for button in self.action_buttons: button.configure(state='disabled')
        self.stop_button.configure(state='normal')
        self.status.set(f'Running · {action}')
        self.append(f'\n── {action} ──\n{self.command.get()}\n')
        threading.Thread(target=self.worker, args=(cmd,), daemon=True).start()

    def worker(self, cmd):
        try:
            logs = HERE / 'output' / 'launcher-logs'; logs.mkdir(parents=True, exist_ok=True)
            logfile = logs / (time.strftime('%Y%m%d-%H%M%S') + '.log')
            env = os.environ.copy(); env['PYTHONDONTWRITEBYTECODE'] = '1'; env['PYTHONIOENCODING'] = 'utf-8'
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
        try:
            while True:
                kind, value = self.messages.get_nowait()
                if kind == 'line':
                    self.append(value)
                    if value.startswith('Results: '):
                        p = Path(value.strip()[9:]).resolve()
                        if p.is_relative_to((HERE / 'output').resolve()): self.run_dir = p
                    if value.startswith('Starting '): self.stage = value.strip()[9:]
                else:
                    self.read_stage_log()
                    self.busy = False
                    self.status.set('Stopped · partial results retained' if self.stopping else ('Completed successfully' if value == 0 else f'Failed · exit code {value}; see output below'))
                    self.stop_button.configure(state='disabled')
                    for button in self.action_buttons: button.configure(state='normal')
                    self.refresh()
        except queue.Empty:
            pass
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
        assert app.table.get_children(), 'No saved EPIC metrics loaded'
        assert len(commands()) == 5
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
