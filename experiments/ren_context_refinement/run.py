"""Portable CLI for real timing hints and fixed-window box refinement."""
from pathlib import Path
import argparse
import json
from core import timing_rank, refine_boxes, track_prediction, validate_query

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('action',choices=['baseline','hint','refine','hint-template'])
    p.add_argument('--bundle',type=Path,required=True);p.add_argument('--out',type=Path,required=True)
    p.add_argument('--hints',type=Path);p.add_argument('--mode',choices=['image','name','description'],default='image')
    p.add_argument('--weight',type=float,default=.05);p.add_argument('--margin',type=float,default=.025)
    p.add_argument('--prior-json',type=Path);p.add_argument('--reliability',type=float,default=.8)
    a=p.parse_args();bundle=json.loads(a.bundle.read_text());queries=bundle['queries']
    for q in queries:validate_query(q)
    if a.action=='hint-template':
        result=dict(provenance='UNFILLED: specify real_user or explicitly simulated',queries=[dict(key=q['key'],object=q.get('object',''),kind='time',seconds=None,sigma_seconds=None,searchable_seconds=(q['query_frame']-1)/q.get('fps',5)) for q in queries])
    else:
        hints={};prior=None;provenance=None
        if a.action=='hint':
            if a.hints is None:p.error('hint requires --hints')
            h=json.loads(a.hints.read_text());provenance=h.get('provenance','unspecified');hints={h['key']:h for h in h['queries']}
            if set(hints)!={q['key'] for q in queries}:raise ValueError('Hint keys must match the bundle exactly; no silent query omissions')
            if a.prior_json:prior=json.loads(a.prior_json.read_text())
        output=[]
        for q in queries:
            if a.action=='baseline':pred=track_prediction(q,q['baseline_id'])
            elif a.action=='refine':pred=refine_boxes(q,a.mode,a.margin)
            else:
                order=timing_rank(q,hints[q['key']],a.weight,a.mode=='description',None if prior is None else prior[q['key']],a.reliability)
                pred=track_prediction(q,order[0]);pred['ranked_candidate_ids']=order
            output.append(pred)
        result=dict(method=a.action,queries=output,cohort=bundle.get('cohort','unspecified'),hint_provenance=provenance,
                    parameters=dict(mode=a.mode,temporal_weight=a.weight,box_margin=a.margin),AP_confidence='unchanged baseline confidence')
    a.out.parent.mkdir(parents=True,exist_ok=True);a.out.write_text(json.dumps(result,indent=2,allow_nan=False)+'\n');print(a.out)

if __name__=='__main__':main()
