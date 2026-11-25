"""Official top-one metrics plus separately labeled candidate shortlist coverage."""
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
import argparse,json
from experiments.candidate_tools.evaluate import score_files
from experiments.candidate_tools.metric_cache import CandidateMetrics
def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ['bundle','truth','predictions','evaluator-root','out']:p.add_argument('--'+name,type=Path,required=True)
    a=p.parse_args();read=lambda p:json.loads(p.read_text());bundle=read(a.bundle);truth=read(a.truth);pred=read(a.predictions)
    result=score_files(bundle,pred,truth,a.evaluator_root);metric=CandidateMetrics(bundle,truth,a.evaluator_root)
    quality={};offset=0
    for q in bundle['queries']:
        quality[q['key']]={c['id']:float(metric.quality[offset+j]) for j,c in enumerate(q['candidates'])};offset+=len(q['candidates'])
    hits={1:0,3:0}
    for q in pred['queries']:
        order=q.get('ranked_candidate_ids')
        if order is None:raise ValueError('Prediction has no defined candidate ranking')
        for k in hits:hits[k]+=int(any(quality[q['key']][c]>=.25 for c in order[:k]))
    result['retrieval_top1_hits']=hits[1]
    result['retrieval_top3_hits']=None if pred.get('method','').endswith('_baseline') else hits[3]
    result['candidate_oracle_hits']=sum(any(v>=.25 for v in row.values()) for row in quality.values())
    result['shortlist_metric_note']='Target-track spatio-temporal IoU>=0.25; shortlist coverage is separate from official Success and automatic top-one AP'
    a.out.parent.mkdir(parents=True,exist_ok=True);a.out.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2))
if __name__=='__main__':main()
