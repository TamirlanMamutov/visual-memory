"""Run upstream EPIC evaluation and a seeded training check without editing it."""
import argparse
import contextlib
import json
import os
from pathlib import Path
import random
import re
import runpy
import sys
import time

from experiment_paths import HERE, ROOT

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', type=int, default=1)
    parser.add_argument('--finish-run', type=Path, help='Evaluate a checkpoint already saved by an interrupted run')
    args = parser.parse_args()
    out = args.finish_run.resolve() if args.finish_run else HERE / 'output' / ('epic-baseline-' + time.strftime('%Y%m%d-%H%M%S'))
    out.mkdir(parents=True, exist_ok=bool(args.finish_run))
    os.environ['PYTHONDONTWRITEBYTECODE'] = '1'
    sys.dont_write_bytecode = True
    sys.path.insert(0, str(ROOT / 'Joint-Part-of-Speech-Embeddings' / 'src'))
    os.chdir(ROOT / 'JPoSE_data')
    import numpy as np
    import torch
    # Upstream samples integer indices using np.int, removed in modern NumPy.
    # This process-local alias leaves the upstream source untouched.
    if 'int' not in np.__dict__:
        np.int = int
    torch.set_num_threads(4)
    random.seed(230)
    np.random.seed(230)
    torch.manual_seed(230)
    import pandas as pd
    import datasets.jpose_dataset as dataset_module
    upstream_create = dataset_module.create_epic_jpose_dataset
    def create_with_validation_ids(*positional, **kwargs):
        dataset = upstream_create(*positional, **kwargs)
        # The final diagnostic export expects IDs that the validation constructor
        # omits. Restore them from the matching dataframe, without changing data.
        if dataset.action.test_dataset:
            split = 'test' if kwargs.get('is_test', False) else 'validation'
            frame = pd.read_pickle(f'./data/dataframes/EPIC_100_retrieval_{split}.pkl')
            assert len(frame) == dataset.action.x_len
            dataset.action._x_idxs = np.arange(dataset.action.x_len)
            dataset.action.x_idx_to_uid = dict(enumerate(frame.index))
            dataset.action._y_idxs = np.arange(dataset.action.y_len)
        return dataset
    dataset_module.create_epic_jpose_dataset = create_with_validation_ids
    metadata = dict(seed=230, epochs=args.epochs, torch=torch.__version__,
                    numpy=np.__version__, cuda=torch.cuda.is_available(),
                    threads=4, data=str(Path.cwd()), upstream_source=sys.path[0],
                    compatibility='Process-local np.int=int alias; validation ID fields restored from matching dataframe for final diagnostic export',
                    scope='Retrieval baseline evaluation/training; not a full paper reproduction')
    if not args.finish_run:
        (out / 'run.json').write_text(json.dumps(metadata, indent=2))
    print(f'Results: {out}', flush=True)

    def run(name, module, argv):
        print(f'Starting {name}', flush=True)
        start = time.time()
        with (out / (name + '.log')).open('w', encoding='utf-8') as log:
            with contextlib.redirect_stdout(log), contextlib.redirect_stderr(log):
                sys.argv = [module] + argv
                runpy.run_module(module, run_name='__main__')
        text = (out / (name + '.log')).read_text(encoding='utf-8')
        metrics = {}
        for metric, vt, tv, avg in re.findall(r'(nDCG|mAP): VT:([\d.]+) TV:([\d.]+) AVG:([\d.]+)', text):
            metrics[metric] = dict(VT=float(vt), TV=float(tv), AVG=float(avg))
        result = dict(seconds=round(time.time()-start, 2), metrics=metrics)
        (out / (name + '.json')).write_text(json.dumps(result, indent=2))
        print(f'Finished {name}: {result}', flush=True)

    if not args.finish_run:
        run('pretrained', 'train.test_jpose_triplet',
            ['./data/models/JPoSE_BEST/model/EPIC_100_retrieval_JPoSE_BEST.pth'])
    if args.epochs and not args.finish_run:
        run('training', 'train.train_jpose_triplet',
            ['--num-epochs', str(args.epochs), '--out-dir', (out / 'training').as_posix()])
    if args.epochs or args.finish_run:
        checkpoint = max((out / 'training').glob('*/model/*.pth'), key=lambda p: int(p.stem.rsplit('_', 1)[1]))
        run('trained-validation', 'train.test_jpose_triplet', [checkpoint.as_posix()])
    if args.finish_run:
        from train.train_jpose_triplet import create_modality_dicts, initialise_jpose_nDCG_values, test_epoch
        from models.jpose import JPOSE
        from torch.utils.tensorboard import SummaryWriter
        import utils
        model_args = utils.output.load_args(str(checkpoint.parents[1] / 'args.txt'))
        dataset = initialise_jpose_nDCG_values(create_with_validation_ids(is_train=False, num_triplets=0))
        modalities, combination = create_modality_dicts(model_args, dataset.x_size, dataset.y_size)
        model = JPOSE(modalities, comb_func=combination)
        model.load_state_dict(torch.load(checkpoint))
        writer = SummaryWriter(log_dir=str(checkpoint.parents[1] / 'results'))
        with (out / 'recovered-export.log').open('w', encoding='utf-8') as log:
            with contextlib.redirect_stdout(log), contextlib.redirect_stderr(log), torch.no_grad():
                test_epoch(model, dataset, ['verb','noun','action'], writer, model_args.num_epochs-1,
                           str(checkpoint.parents[1]), final_run=True)
        writer.close()
        (out / 'recovery.json').write_text(json.dumps(dict(
            reason='Training completed; upstream final diagnostic export raised missing _x_idxs before process shutdown.',
            recovery='Evaluated the saved checkpoint and completed final diagnostic export with restored validation IDs; no repeated training.',
            checkpoint=str(checkpoint), optional_per_example_export='completed via runner compatibility fix'), indent=2))
    (out / 'COMPLETE').write_text('All requested stages completed.\n')

if __name__ == '__main__':
    main()
