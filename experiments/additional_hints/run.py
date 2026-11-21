"""Run extra sighting-time hints, or create a label-free hint template."""
from pathlib import Path
import argparse
import json
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
from experiments.candidate_tools.candidate_common import validate_query,track_prediction
from experiments.additional_hints.hints import timing_rank

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--bundle',type=Path,required=True);p.add_argument('--out',type=Path,required=True)
    p.add_argument('--hints',type=Path);p.add_argument('--template',action='store_true')
    p.add_argument('--use-description',action='store_true');p.add_argument('--weight',type=float,default=.05)
    p.add_argument('--prior-json',type=Path);p.add_argument('--reliability',type=float,default=.8)
    a=p.parse_args();bundle=json.loads(a.bundle.read_text());queries=bundle['queries']
    for q in queries:validate_query(q)
    if a.template:
        result=dict(provenance='UNFILLED: specify real_user or explicitly simulated',queries=[dict(key=q['key'],object=q.get('object',''),kind='time',seconds=None,sigma_seconds=None,searchable_seconds=(q['query_frame']-1)/q.get('fps',5)) for q in queries])
    else:
        if a.hints is None:p.error('Use --hints or --template')
        data=json.loads(a.hints.read_text());hints={h['key']:h for h in data['queries']}
        if len(hints)!=len(data['queries']) or set(hints)!={q['key'] for q in queries}:raise ValueError('Exact hint-key coverage required')
        prior=json.loads(a.prior_json.read_text()) if a.prior_json else None;predictions=[]
        for q in queries:
            order=timing_rank(q,hints[q['key']],a.weight,a.use_description,None if prior is None else prior[q['key']],a.reliability)
            prediction=track_prediction(q,order[0]);prediction['ranked_candidate_ids']=order;predictions.append(prediction)
        result=dict(method='additional_timing_hint',queries=predictions,cohort=bundle.get('cohort','unspecified'),hint_provenance=data.get('provenance','unspecified'),
                    parameters=dict(temporal_weight=a.weight,use_description=a.use_description),AP_confidence='unchanged baseline confidence')
    a.out.parent.mkdir(parents=True,exist_ok=True);a.out.write_text(json.dumps(result,indent=2,allow_nan=False)+'\n');print(a.out)

if __name__=='__main__':main()
