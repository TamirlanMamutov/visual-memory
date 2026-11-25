"""Build a training-only verifier dataset from a new candidate bundle and targets.

REN cached evidence and CLIP/EAGLE proxy evidence are different feature families.
The latter is a new experiment requiring retraining, not exact REN reproduction.
"""
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
import argparse,json,random
import numpy as np
from experiments.candidate_tools.metric_cache import CandidateMetrics
from experiments.track_verifier.core import FEATURE_NAMES

def iou(a,b):
    _,x,y,w,h=a;_,u,v,bw,bh=b;z=max(0,min(x+w,u+bw)-max(x,u))*max(0,min(y+h,v+bh)-max(y,v));return z/max(1,w*h+bw*bh-z)
def proxy(c,q):
    track=[[f,x,y,x2-x,y2-y] for f,x,y,x2,y2 in c['bboxes']]
    if not track:return [0.]*12
    samples=[s['image_score'] for s in c.get('samples',[]) if 'image_score' in s]
    s=np.asarray(samples);area=np.asarray([max(1,b[3]*b[4]) for b in track]);centers=np.asarray([[b[1]+b[3]/2,b[2]+b[4]/2] for b in track]);shift=float(np.linalg.norm(np.diff(centers,axis=0),axis=1).max()/max(1,np.sqrt(np.median(area)))) if len(track)>1 else 0.;span=track[-1][0]-track[0][0]+1
    return [float(c.get('native_score',q['score'])),float(s.mean()) if len(s) else 0.,float(np.quantile(s,.1)) if len(s) else 0.,float(s.std()) if len(s) else 0.,len(s)/5,float(np.log(area).std()),shift,float(np.mean([iou(a,b) for a,b in zip(track,track[1:])])) if len(track)>1 else 0.,span/q['query_frame'],track[-1][0]/q['query_frame'],track[0][0]/q['query_frame'],len(track)/span]
def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ['bundle','truth','evaluator-root','out']:p.add_argument('--'+name,type=Path,required=True)
    p.add_argument('--feature-family',choices=['ren_quality_v1','clip_quality_proxy_v1'],required=True)
    p.add_argument('--confirm-training-split',action='store_true',required=True,help='Confirm all inputs are from training/development, never reserved validation')
    p.add_argument('--excluded-sources',type=Path)
    a=p.parse_args();bundle=json.loads(a.bundle.read_text());truth=json.loads(a.truth.read_text());metric=CandidateMetrics(bundle,truth,a.evaluator_root)
    if bundle.get('annotation_split') in ['val','test']:raise ValueError('Do not train on an exported validation/test cohort')
    if a.feature_family=='clip_quality_proxy_v1' and bundle.get('appearance_features_available') is False:raise ValueError('Export actual appearance features first; baseline-only replay scores are placeholders')
    rows=[];mapping={};cases={};old=[];family=a.feature_family
    excluded=json.loads(a.excluded_sources.read_text()) if a.excluded_sources else []
    for q in bundle['queries']:
        if q['source'] in excluded:raise ValueError('Reserved source in training export')
        if q.get('old'):old.append(q['key'])
        cases[q['key']]=dict(source=q['source'],cutoff=q['query_frame'],object=q.get('object',''))
        for c in q['candidates']:
            index=len(rows);feature=c.get('quality_features')
            if family=='ren_quality_v1' and feature is None:raise ValueError('Exact REN features require exported REN evidence; choose clip_quality_proxy_v1 only for a separate new experiment')
            if family=='clip_quality_proxy_v1':feature=proxy(c,q)
            c['quality_features']=feature;quality=float(metric.quality[index])
            row=dict(key=q['key'],source=q['source'],features=feature,track=[[f,x,y,x2-x,y2-y] for f,x,y,x2,y2 in c['bboxes']],is_baseline=c['id']==q['baseline_id'],track_IoU=quality,label=1 if quality>=.25 else 0 if quality<.05 else -1)
            for src,dst in [('image_score','image_cosine'),('name_score','name_cosine'),('hand_features','hand_features')]:
                if src in c:row[dst]=c[src]
            rows.append(row);mapping[str(index)]=index
    sources=sorted({r['source'] for r in rows});random.Random(2026111704).shuffle(sources)
    bundle['feature_family']=family
    result=dict(version=1,training_only=True,feature_family=family,feature_names=FEATURE_NAMES,rows=rows,cases=cases,old_keys=old,bundle=bundle,truth=truth,row_to_candidate=mapping,folds={s:i%min(5,len(sources)) for i,s in enumerate(sources)},excluded_sources=excluded)
    a.out.parent.mkdir(parents=True,exist_ok=True);a.out.write_text(json.dumps(result,indent=2)+'\n')
    a.out.with_name(a.out.stem+'_input_bundle.json').write_text(json.dumps(bundle,indent=2)+'\n');print(a.out)
if __name__=='__main__':main()
