"""Export text-to-video retrieval errors from the existing pretrained checkpoint."""
import csv
import os
from pathlib import Path
import sys

from experiment_paths import HERE, ROOT
OUT = HERE / 'output' / 'epic-retrieval-errors'
sys.dont_write_bytecode = True
sys.path.insert(0, str(ROOT / 'Joint-Part-of-Speech-Embeddings' / 'src'))
os.chdir(ROOT / 'JPoSE_data')
import numpy as np
import pandas as pd
import torch
import utils
from models.jpose import JPOSE
from datasets.jpose_dataset import create_epic_jpose_dataset
from train.train_jpose_triplet import create_modality_dicts, initialise_jpose_nDCG_values
from train.test_jpose_triplet import get_features
from evaluation import nDCG

torch.set_num_threads(2)
args = utils.output.load_args('./data/models/JPoSE_BEST/args.txt')
dataset = initialise_jpose_nDCG_values(create_epic_jpose_dataset(is_train=False, num_triplets=0))
mods, comb = create_modality_dicts(args, dataset.x_size, dataset.y_size)
model = JPOSE(mods, comb_func=comb)
model.load_state_dict(torch.load('./data/models/JPoSE_BEST/model/EPIC_100_retrieval_JPoSE_BEST.pth'))
with torch.no_grad():
    vis, txt = get_features(model, dataset, ['verb','noun','action'])
sim = txt @ vis.T
rel = dataset.action.relevancy_matrix.T
scores = nDCG.calculate_nDCG(sim, rel, dataset.action.k_values['t'], IDCG=dataset.action.IDCG['t'], reduction=None)
videos = pd.read_pickle('./data/dataframes/EPIC_100_retrieval_validation.pkl')
sentences = pd.read_pickle('./data/dataframes/EPIC_100_retrieval_validation_sentence.pkl')
assert sim.shape == (len(sentences), len(videos))
OUT.mkdir(parents=True, exist_ok=True)
with (OUT / 'text-to-video-top5.csv').open('w', encoding='utf-8-sig', newline='') as f:
    writer = csv.DictWriter(f, fieldnames=['query_index','query','query_nDCG','rank','predicted_narration_id','predicted_narration','video_id','start_timestamp','stop_timestamp','similarity','relevance','local_video'])
    writer.writeheader()
    for qi in np.argsort(scores):
        top = np.argsort(-sim[qi])[:5]
        for rank, vi in enumerate(top, 1):
            item = videos.iloc[vi]
            local = ROOT / 'EPIC-KITCHENS' / item['participant_id'] / 'videos' / (item['video_id']+'.MP4')
            writer.writerow(dict(query_index=int(qi), query=sentences.iloc[qi]['sentence'], query_nDCG=float(scores[qi]), rank=rank,
                                 predicted_narration_id=videos.index[vi], predicted_narration=item['narration'], video_id=item['video_id'],
                                 start_timestamp=item['start_timestamp'], stop_timestamp=item['stop_timestamp'], similarity=float(sim[qi,vi]),
                                 relevance=float(rel[qi,vi]), local_video=local.relative_to(ROOT).as_posix() if local.exists() else ''))
print(f'Exported {len(sentences)} queries, five predictions each, lowest nDCG first: {OUT}')
