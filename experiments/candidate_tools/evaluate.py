"""Independent official-evaluator entry point; does not import a model runner."""
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
import argparse
import contextlib
import hashlib
import importlib
import io
import json
import sys
import numpy as np

def official_api(root):
    root=Path(root).resolve();np.float=float;np.int=int;sys.path.insert(0,str(root))
    if (root/'vq2d/metrics/metrics.py').exists():name='vq2d.metrics.metrics'
    elif (root/'metrics/metrics.py').exists():name='metrics.metrics'
    else:raise ValueError('Evaluator root must contain vq2d/metrics/metrics.py or metrics/metrics.py')
    module=importlib.import_module(name)
    return module.compute_visual_query_metrics,module.BBox,module.ResponseTrack

def score_files(bundle,predictions,truth,root,subset='all'):
    compute,BBox,ResponseTrack=official_api(root)
    queries={q['key']:q for q in bundle['queries']};pred={q['key']:q for q in predictions['queries']};gt={q['key']:q for q in truth['queries']}
    if len(pred)!=len(predictions['queries']) or len(gt)!=len(truth['queries']):raise ValueError('Duplicate query keys')
    if set(pred)!=set(queries) or set(gt)!=set(queries):raise ValueError('Exact query coverage required; no silent omissions')
    keys=sorted(k for k,q in queries.items() if subset=='all' or not q.get('old',False))
    if not keys:raise ValueError('Empty scoring cohort')
    def track(d):return ResponseTrack([BBox(*b) for b in d['bboxes']],score=d.get('score'))
    for k in keys:
        q=queries[k];prediction=pred[k]
        if prediction['score']!=q['score']:raise ValueError('Baseline AP confidence changed')
        frames=[b[0] for b in prediction['bboxes']]
        if frames!=sorted(set(frames)) or any(not 0<=f<q['query_frame'] for f in frames):raise ValueError('Invalid or post-query frame')
    with contextlib.redirect_stdout(io.StringIO()):
        raw=compute([[track(pred[k])] for k in keys],[track(gt[k]) for k in keys],
                    [BBox(-1,0,0,1,1) for _ in keys],vc_rt_pairings={'all':('all','all')})['all']
    mapping={'stAP25':('SpatioTemporal AP','IoU=0.25'),'tAP25':('Temporal AP','IoU=0.25'),'Recovery':('Tracking','IoU=0.50'),'Success':('Success','IoU=0.05')}
    metrics={label:float(next(v for k,v in raw.items() if k.strip().startswith(prefix) and k.strip().endswith(suffix))) for label,(prefix,suffix) in mapping.items()}
    return dict(queries=len(keys),source_videos=len({queries[k]['source'] for k in keys}),metrics=metrics,
                raw={k:float(v) for k,v in raw.items()},scope=bundle.get('cohort','unspecified'),
                hint_provenance=predictions.get('hint_provenance'),evaluator_root=str(Path(root).resolve()))

def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ['bundle','predictions','truth','evaluator-root','out']:p.add_argument('--'+name,type=Path,required=True)
    p.add_argument('--subset',choices=['all','new'],default='all');a=p.parse_args()
    load=lambda f:json.loads(f.read_text())
    result=score_files(load(a.bundle),load(a.predictions),load(a.truth),a.evaluator_root,a.subset)
    result['input_sha256']={str(f):hashlib.sha256(f.read_bytes()).hexdigest() for f in [a.bundle,a.predictions,a.truth,Path(__file__)]}
    a.out.parent.mkdir(parents=True,exist_ok=True);a.out.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({k:result[k] for k in ['queries','source_videos','metrics','hint_provenance']},indent=2));print(a.out)

if __name__=='__main__':main()
