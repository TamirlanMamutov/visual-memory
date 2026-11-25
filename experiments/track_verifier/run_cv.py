"""Source-grouped verifier reproduction on exported training candidates."""
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
import argparse,copy,hashlib,json,random,time
import numpy as np
import torch
from experiments.track_verifier import core
from experiments.candidate_tools.metric_cache import CandidateMetrics

def read(p):return json.loads(Path(p).read_text())
def save(p,d):
    p=Path(p);p.parent.mkdir(parents=True,exist_ok=True);t=p.with_suffix('.tmp');t.write_text(json.dumps(d,indent=2,allow_nan=False)+'\n');t.replace(p)
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def validate(data):
    if data.get('training_only') is not True:raise ValueError('Training requires explicitly training_only data; never pass the reserved validation export')
    rows=data['rows'];g=core.groups(rows);queries={q['key']:q for q in data['bundle']['queries']}
    if set(g)!=set(queries):raise ValueError('Training rows and bundle query IDs differ')
    for key,ids in g.items():
        if sum(bool(rows[i]['is_baseline']) for i in ids)!=1:raise ValueError('Exactly one baseline required per query')
        if len({rows[i]['source'] for i in ids})!=1:raise ValueError('One query cannot span sources')
        for i in ids:
            r=rows[i]
            if r['label'] not in [-1,0,1] or len(r['features'])!=12:raise ValueError('Invalid label/base feature vector')
            if not np.isfinite(r['features']).all():raise ValueError('Nonfinite features')
            if any(not 0<=b[0]<queries[key]['query_frame'] for b in r['track']):raise ValueError('Post-query training track')
    sources={r['source'] for r in rows}
    if sources & set(data.get('excluded_sources',[])):raise ValueError('Reserved source included in training')
    folds=data.get('folds')
    if folds is None:
        ordered=sorted(sources);random.Random(2026111704).shuffle(ordered);folds={s:i%min(5,len(sources)) for i,s in enumerate(ordered)}
    if set(folds)!=sources or len(sources)<6 or len(set(folds.values()))<2:raise ValueError('Need all source-fold assignments, at least two folds and six training sources')
    return folds
def flat(data,metric,selection):return np.asarray([data['row_to_candidate'][str(selection[k])] for k in metric.keys],dtype=int)
def metrics_for(data,metric,selection):
    values=flat(data,metric,selection);n=len(metric.keys);cohorts={'all':np.arange(n)}
    new=np.asarray([i for i,k in enumerate(metric.keys) if k not in data.get('old_keys',[])],dtype=int)
    if len(new):cohorts['new']=new
    return {name:{k:float(v) for k,v in metric.score(values,ids).items()} for name,ids in cohorts.items()},values
def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ['data','recipes','evaluator-root','out']:p.add_argument('--'+name,type=Path,required=True)
    p.add_argument('--names',nargs='+');p.add_argument('--fit-full-model',action='store_true')
    a=p.parse_args();torch.set_num_threads(1);torch.set_num_interop_threads(1)
    data=read(a.data);folds=validate(data);protocol=read(a.recipes)
    configs=protocol['head_configs'] if isinstance(protocol,dict) else protocol
    if a.names:
        if set(a.names)-{c['name'] for c in configs}:p.error('Unknown recipe name')
        configs=[c for c in configs if c['name'] in a.names]
    metric=CandidateMetrics(data['bundle'],data['truth'],a.evaluator_root)
    confidence={q['key']:q['score'] for q in data['bundle']['queries']};base={k:next(i for i in ids if data['rows'][i]['is_baseline']) for k,ids in core.groups(data['rows']).items()}
    baseline,_=metrics_for(data,metric,base)
    signature=dict(data_sha256=sha(a.data),recipes_sha256=sha(a.recipes),core_sha256=sha(Path(core.__file__)),driver_sha256=sha(__file__))
    for recipe in configs:
        cfg=dict(core.DEFAULT,**recipe);name=cfg['name']
        if not name or name in ['.','..'] or any(c not in 'abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_.-' for c in name):raise ValueError('Unsafe recipe name')
        out=a.out/name;marker=out/'COMPLETE.json'
        if marker.exists():
            if read(marker)['input_signature']!=signature:raise ValueError('Inputs/code changed; use a fresh output directory')
            print('COMPLETE_CACHE_HIT',name,flush=True);continue
        rows=core.augment(data['rows'],cfg);selection={};individuals={s:{} for s in (0,1,2)};audits=[]
        for fold in sorted(set(folds.values())):
            train=[r for r in rows if folds[r['source']]!=fold];ids=[i for i,r in enumerate(rows) if folds[r['source']]==fold];held=[rows[i] for i in ids]
            assert not {r['source'] for r in train}&{r['source'] for r in held}
            save(a.out/'progress.json',dict(phase='source_cv',recipe=name,outer_fold=fold,updated=time.time()))
            threshold,inner=core.select_threshold(train,fold,cfg,confidence)
            scores,seeds,_,detail=core.ensemble(train,held,cfg)
            for key,j in core.choose(held,scores,threshold,cfg).items():selection[key]=ids[j]
            for seed,values in enumerate(seeds):
                for key,j in core.choose(held,values,threshold,cfg).items():individuals[seed][key]=ids[j]
            audits.append(dict(fold=fold,threshold=threshold,inner=inner,training=detail,training_sources=sorted({r['source'] for r in train}),held_sources=sorted({r['source'] for r in held})))
            print('SOURCE_FOLD',name,fold,'threshold',threshold,flush=True)
        save(out/'prediction_freeze.json',dict(choices=selection,seed_choices=individuals,folds=audits,input_signature=signature,predictions_frozen_before_scoring=True))
        point,choices=metrics_for(data,metric,selection)
        metric.verify_official(choices,np.arange(len(metric.keys)))
        seed_metrics=[metrics_for(data,metric,x)[0] for x in individuals.values()]
        np.savez_compressed(out/'choices.npz',choices=choices,keys=np.asarray(metric.keys),baseline=flat(data,metric,base))
        if a.fit_full_model:
            threshold,inner=core.select_threshold(rows,100,cfg,confidence);_,_,models,_=core.ensemble(rows,rows,cfg)
            model=core.serialize(models,cfg,threshold,data.get('feature_family','ren_quality_v1'))
            model.update(training_queries=len(core.groups(rows)),training_sources=len(folds),training_source_ids=sorted(folds),role='Fitted on training only; development-selected method needs prospective evaluation')
            save(out/'model.json',model)
        save(marker,dict(completed=time.time(),configuration=cfg,metrics=point,baseline=baseline,seed_metrics=seed_metrics,input_signature=signature,
            changed_queries=sum(selection[k]!=base[k] for k in base),queries=len(metric.keys),sources=len(folds),
            evaluation_pairing={'all':['all','all']},development_only=True))
        print('RECIPE_COMPLETE',name,json.dumps(point),flush=True)
    save(a.out/'progress.json',dict(phase='complete',updated=time.time(),recipes=[c['name'] for c in configs]))
if __name__=='__main__':main()
