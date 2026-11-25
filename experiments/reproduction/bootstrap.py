"""Source-paired uncertainty for fixed candidate predictions; no model selection."""
from pathlib import Path
import argparse,json
import numpy as np
from experiments.candidate_tools.metric_cache import CandidateMetrics

def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ['bundle','truth','baseline','evaluator-root','out']:p.add_argument('--'+name,type=Path,required=True)
    p.add_argument('--predictions',type=Path,nargs='+',required=True)
    p.add_argument('--draws',type=int,default=2000);p.add_argument('--seed',type=int,default=2026112101)
    a=p.parse_args();read=lambda f:json.loads(f.read_text());bundle=read(a.bundle)
    queries={q['key']:q for q in bundle['queries']};keys=sorted(queries);offset=0;indices={}
    for q in bundle['queries']:
        for c in q['candidates']:indices[q['key'],tuple(map(tuple,c['bboxes']))]=offset;offset+=1
    def choices(path):
        predictions=read(path)['queries'];pred={q['key']:q for q in predictions}
        if set(pred)!=set(keys) or len(pred)!=len(predictions):raise ValueError('Exact unique prediction coverage required')
        if any(pred[k]['score']!=queries[k]['score'] for k in keys):raise ValueError('This bootstrap assumes unchanged native AP confidence')
        return np.asarray([indices[k,tuple(map(tuple,pred[k]['bboxes']))] for k in keys])
    baseline=choices(a.baseline);arms={f.stem:choices(f) for f in a.predictions}
    if len(arms)!=len(a.predictions):raise ValueError('Use distinct method filenames')
    cache=CandidateMetrics(bundle,read(a.truth),a.evaluator_root)
    assert keys==cache.keys
    sources=sorted({q['source'] for q in queries.values()});groups=[[i for i,k in enumerate(keys) if queries[k]['source']==s] for s in sources]
    rng=np.random.default_rng(a.seed);paired={arm:{} for arm in arms}
    for ordinal,draw in enumerate(rng.integers(len(sources),size=(a.draws,len(sources)))):
        sampled=np.asarray([i for source in draw for i in groups[source]])
        base=cache.score(baseline,sampled)
        if ordinal<3:cache.verify_official(baseline,sampled)
        for arm,selection in arms.items():
            value=cache.score(selection,sampled)
            if ordinal<3:cache.verify_official(selection,sampled)
            for field in ['stAP25','tAP25','Recovery','Success','mean_track_IoU']:paired[arm].setdefault(field,[]).append(value[field]-base[field])
    report=dict(queries=len(keys),sources=len(sources),draws=a.draws,seed=a.seed,paired_source_bootstrap={arm:{field:dict(CI95=np.quantile(values,[.025,.975]).tolist()) for field,values in fields.items()} for arm,fields in paired.items()},
        note='Fixed models; sources resampled with all their queries, pooled AP recomputed. Does not include method-selection or training-set variability; secondary arms are not multiplicity-adjusted.')
    a.out.parent.mkdir(parents=True,exist_ok=True);a.out.write_text(json.dumps(report,indent=2)+'\n');print(a.out)
if __name__=='__main__':main()
