"""Apply a frozen verifier to a label-free candidate bundle; never read targets."""
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
import argparse,json,hashlib
import numpy as np
import torch
from experiments.track_verifier import core
from experiments.candidate_tools.candidate_common import validate_query,track_prediction

def rows_from_bundle(bundle):
    rows=[]
    for q in bundle['queries']:
        validate_query(q)
        for c in q['candidates']:
            if 'quality_features' not in c:raise ValueError('Candidate needs quality_features; use build_dataset.py or the provided export')
            r=dict(key=q['key'],source=q['source'],features=c['quality_features'],track=[[f,x,y,x2-x,y2-y] for f,x,y,x2,y2 in c['bboxes']],is_baseline=c['id']==q['baseline_id'],candidate_id=c['id'])
            if c.get('empty_native_fallback'):r['empty_native_fallback']=True
            for src,dst in [('image_score','image_cosine'),('name_score','name_cosine'),('shuffled_name_score','shuffled_name_cosine'),('hand_features','hand_features')]:
                if src in c:r[dst]=c[src]
            rows.append(r)
    return rows
def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ['bundle','model','out']:p.add_argument('--'+name,type=Path,required=True)
    a=p.parse_args();torch.set_num_threads(1)
    bundle=json.loads(a.bundle.read_text());saved=json.loads(a.model.read_text())
    if bundle.get('feature_family')!=saved['feature_family']:raise ValueError('Feature-family mismatch; do not apply REN-trained weights to EAGLE/CLIP proxy features without retraining')
    models,cfg,threshold=core.deserialize(saved);rows=core.augment(rows_from_bundle(bundle),cfg)
    scores=np.mean([core.logits(rows,m,cfg) for m in models],axis=0)
    if cfg['risk_sd']:
        seed_scores=[core.logits(rows,m,cfg) for m in models]
        for ids in core.groups(rows).values():
            base=next(i for i in ids if rows[i]['is_baseline'])
            for i in ids:
                d=np.asarray([s[i]-s[base] for s in seed_scores]);scores[i]=np.mean([s[base] for s in seed_scores])+d.mean()-cfg['risk_sd']*d.std(ddof=1)
    selected=core.choose(rows,scores,threshold,cfg)
    predictions=[track_prediction(q,rows[selected[q['key']]]['candidate_id']) for q in bundle['queries']]
    result=dict(method='frozen_track_quality_verifier',queries=predictions,model_sha256=hashlib.sha256(a.model.read_bytes()).hexdigest(),input_sha256=hashlib.sha256(a.bundle.read_bytes()).hexdigest(),AP_confidence='unchanged baseline confidence',no_targets_read=True)
    a.out.parent.mkdir(parents=True,exist_ok=True);a.out.write_text(json.dumps(result,indent=2)+'\n');print(a.out)
if __name__=='__main__':main()
