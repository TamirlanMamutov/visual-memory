"""Prediction-only clarification requests and ORACLE-DERIVED budget simulation.

Request selection never reads GT. The simulation reuses explicitly synthetic hint
inputs from simulate.py; it is not a real human-in-the-loop evaluation.
"""
from pathlib import Path
import argparse
import hashlib
import json
import sys
import time
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
from experiments.candidate_tools.candidate_common import validate_query,track_prediction
from experiments.additional_hints.hints import temporal_log_density,native_gated_choice

BUDGETS=[.1,.25,.5]
METHODS=['REN_native','latest_center','all_true_point','all_true_gate','all_shuffled_gate',
         'active10_true_gate','active25_true_gate','active50_true_gate',
         'active25_shuffled_gate','random25_all_true_gate','random25_eligible_true_gate',
         'eligible_all_true_gate','active25_true_point']

def read(p):return json.loads(Path(p).read_text())
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def save(p,d):
    p=Path(p);p.parent.mkdir(parents=True,exist_ok=True);t=p.with_suffix('.tmp');t.write_text(json.dumps(d,indent=2,allow_nan=False)+'\n');t.replace(p)

def temporal_overlap(a,b):
    a={x[0] for x in a};b={x[0] for x in b}
    return len(a&b)/max(1,len(a|b))

def request_policy(bundle,budget):
    details=[]
    for q in bundle['queries']:
        validate_query(q);c=q['candidates'];base=next(x for x in c if x['id']==q['baseline_id'])
        order=sorted(range(len(c)),key=lambda i:(-c[i]['image_score'],c[i]['id']!=q['baseline_id'],i))
        best=c[order[0]]['image_score'];plausible=[x for x in c if x['bboxes'] and x['image_score']>=best-.05]
        ambiguous=any(temporal_overlap(x['bboxes'],y['bboxes'])<.2 for i,x in enumerate(plausible) for y in plausible[i+1:])
        disagreement=bool(base['bboxes'] and c[order[0]]['bboxes'] and temporal_overlap(base['bboxes'],c[order[0]]['bboxes'])<.2)
        margin=float(c[order[0]]['image_score']-c[order[1]]['image_score']) if len(order)>1 else 1.
        details.append(dict(key=q['key'],source=q['source'],object=q.get('object',''),ambiguous=ambiguous,
                            native_image_temporal_disagreement=disagreement,image_top2_margin=margin))
    eligible=sorted([d for d in details if d['ambiguous']],key=lambda d:(not d['native_image_temporal_disagreement'],d['image_top2_margin'],d['key']))
    count=min(int(np.floor(budget*len(details))),len(eligible))
    return dict(budget_fraction=budget,queries=len(details),eligible=len(eligible),requested=count,
                requested_keys=[d['key'] for d in eligible[:count]],eligible_keys=[d['key'] for d in eligible],details=details,
                rule='Image scores within .05 of best; an eligible pair has temporal frame-set IoU<.2. Prioritize native/image temporal disagreement, then smaller top2 margin. No GT or hint used.')

def gated_choices(bundle,simulation):
    keys=list(simulation['keys']);lookup={q['key']:q for q in bundle['queries']};flat=[];ids={};native=[]
    for q in bundle['queries']:
        ids[q['key']]=list(range(len(flat),len(flat)+len(q['candidates'])));flat+=q['candidates']
    for k in keys:native.append(next(i for i in ids[k] if flat[i]['id']==lookup[k]['baseline_id']))
    native=np.asarray(native);source_methods=list(simulation['methods']);conditions=list(simulation['conditions'])
    point=np.stack([simulation['first'][:,:,source_methods.index(m),:] for m in ['image_hint','image_shuffled_hint','image_random_hint']],axis=2)
    # The simulation's candidate order must be the bundle's flattened order.
    for n,k in enumerate(keys):
        if not np.isin(point[:,:,:,n],ids[k]).all():raise ValueError('Simulation/bundle candidate-index mismatch')
    gate=np.broadcast_to(native,point.shape).copy();hints=simulation['hints'];definitions=simulation['definitions']
    changes=0
    for ci,name in enumerate(conditions):
        condition=definitions[ci]
        for draw in range(point.shape[1]):
            for control in range(3):
                for n,k in enumerate(keys):
                    chosen=int(point[ci,draw,control,n]);base=int(native[n])
                    if chosen==base:continue
                    q=lookup[k];duration=(q['query_frame']-1)/q.get('fps',5)
                    hint=dict(kind='third',value=['early','middle','late'][int(hints[ci,draw,control,n])]) if name=='coarse_thirds' else dict(kind='time',seconds=float(hints[ci,draw,control,n]),sigma_seconds=condition['sigma'])
                    alt_time=temporal_log_density(flat[chosen],q.get('fps',5),hint,duration);base_time=temporal_log_density(flat[base],q.get('fps',5),hint,duration)
                    alt_image=flat[chosen]['image_score'];base_image=flat[base]['image_score']
                    allow=alt_image>=base_image-.025 and alt_time>base_time+np.log(3.) and alt_image+.05*alt_time>base_image+.05*base_time+.025
                    if allow:gate[ci,draw,control,n]=chosen;changes+=1
    return native,point,gate,changes

def main():
    ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('--bundle',type=Path,required=True);ap.add_argument('--out',type=Path,required=True)
    ap.add_argument('--simulation-dir',type=Path);ap.add_argument('--truth',type=Path);ap.add_argument('--evaluator-root',type=Path)
    ap.add_argument('--requests-only',action='store_true');ap.add_argument('--budget',type=float,default=.25)
    ap.add_argument('--real-hints',type=Path,help='Human clue file for exactly the prediction-only requested keys; no GT required')
    a=ap.parse_args()
    if not 0<=a.budget<=1:ap.error('--budget must be in [0,1]')
    bundle=read(a.bundle)
    if a.requests_only:
        result=request_policy(bundle,a.budget);result['hint_template']=[dict(key=k,kind='time',seconds=None,sigma_seconds=None) for k in result['requested_keys']]
        result['ground_truth_read']=False;save(a.out,result);print(a.out);return
    if a.real_hints:
        policy=request_policy(bundle,a.budget);data=read(a.real_hints);hints={h['key']:h for h in data['queries']}
        if len(hints)!=len(data['queries']) or set(hints)!=set(policy['requested_keys']):
            raise ValueError('Human clue keys must match the requested set, including explicit declined entries')
        predictions=[];decisions=[]
        for q in bundle['queries']:
            if q['key'] in hints:
                selected,detail=native_gated_choice(q,hints[q['key']])
            else:selected,detail=q['baseline_id'],dict(reason='not_requested',changed=False)
            predictions.append(track_prediction(q,selected));decisions.append(dict(key=q['key'],**detail))
        result=dict(method='human_hint_clarification_native_gate',queries=predictions,
                    hint_provenance=data.get('provenance','unspecified'),requested=policy['requested'],
                    supplied=sum(not h.get('declined',False) for h in hints.values()),decisions=decisions,
                    ground_truth_read=False,cohort=bundle.get('cohort','unspecified'))
        save(a.out,result);print(a.out);return
    if not all([a.simulation_dir,a.truth,a.evaluator_root]):ap.error('Simulation analysis requires --simulation-dir --truth --evaluator-root')
    a.out.mkdir(parents=True,exist_ok=True)
    paths=[Path(__file__),a.bundle,a.truth,a.simulation_dir/'rankings.npz',a.simulation_dir/'SIMULATED_INPUTS.npz',a.simulation_dir/'protocol.json']
    protocol=dict(created=time.time(),classification='ORACLE-DERIVED clarification-budget simulation, not real user input',
                  primary='new-tagged queries,primary_5s,active25_true_gate versus random25_all_true_gate and active25_shuffled_gate',
                  eligibility='Prediction-only: >=2plausible image candidates within.05cosine of best, with temporal IoU<.2. Request priority: native/image temporal disagreement, then small image margin.',
                  budgets=BUDGETS,budget_rounding='floor(fraction*N), capped by eligible count; random controls use the SAME actual count',
                  gate='Only replace the point-fusion winner if its image score >=native-.025, temporal likelihood >3*native, and fused score >native+.025. No parameter sweep.',
                  controls='Native baseline,latestcenter,all hints,ungated requests,randomall,randomeligible,andshuffledhints. Same synthetic draws and source groups as input simulation.',
                  randomness='Seed2026112021forrequestmasks; masks reused across all conditions. No GT used for eligibility, ordering, request masks or baseline gate.',
                  uncertainty='2000source-bootstrap Top1differences after averaging synthetic draws perquery. Results remain repeated development.',input_sha256={str(p):sha(p) for p in paths})
    if (a.out/'protocol.json').exists():
        for p,h in read(a.out/'protocol.json')['input_sha256'].items():
            if sha(p)!=h:raise ValueError('Changed frozen inputs; choose a new output folder')
    else:save(a.out/'protocol.json',protocol)
    z=dict(np.load(a.simulation_dir/'rankings.npz'));supplied=dict(np.load(a.simulation_dir/'SIMULATED_INPUTS.npz'))
    if list(z['keys'])!=list(supplied['keys']):raise ValueError('Hint/key mismatch')
    z['hints']=supplied['hints'];z['definitions']=read(a.simulation_dir/'protocol.json')['conditions']
    native,point,gate,changes=gated_choices(bundle,z);keys=list(z['keys']);lookup={q['key']:q for q in bundle['queries']}
    cohorts=dict(all_queries=np.arange(len(keys)),new_queries=np.asarray([i for i,k in enumerate(keys) if not lookup[k].get('old',False)]))
    choices={};costs={};policies={};masks={};n_draws=point.shape[1]
    for cohort,idx in cohorts.items():
        if not len(idx):continue
        view=dict(queries=[lookup[keys[i]] for i in idx]);policy={str(f):request_policy(view,f) for f in BUDGETS};policies[cohort]=policy
        selected=np.broadcast_to(native,(point.shape[0],n_draws,len(METHODS),len(keys))).copy()
        m=lambda name:METHODS.index(name)
        selected[:,:,m('latest_center'),:]=z['first'][:,:,list(z['methods']).index('latest_center'),:]
        selected[:,:,m('all_true_point'),:]=point[:,:,0,:];selected[:,:,m('all_true_gate'),:]=gate[:,:,0,:];selected[:,:,m('all_shuffled_gate'),:]=gate[:,:,1,:]
        costs[cohort]={'REN_native':0,'latest_center':0,'all_true_point':len(idx),'all_true_gate':len(idx),'all_shuffled_gate':len(idx)}
        keyindex={k:i for i,k in enumerate(keys)}
        for fraction in BUDGETS:
            p=policy[str(fraction)];which=np.asarray([keyindex[k] for k in p['requested_keys']],dtype=int);name=f'active{int(fraction*100)}_true_gate'
            selected[:,:,m(name),which]=gate[:,:,0,which];costs[cohort][name]=len(which)
        p=policy['0.25'];active=np.asarray([keyindex[k] for k in p['requested_keys']],dtype=int);eligible=np.asarray([keyindex[k] for k in p['eligible_keys']],dtype=int)
        selected[:,:,m('active25_shuffled_gate'),active]=gate[:,:,1,active];selected[:,:,m('active25_true_point'),active]=point[:,:,0,active]
        selected[:,:,m('eligible_all_true_gate'),eligible]=gate[:,:,0,eligible]
        for name in ['active25_shuffled_gate','active25_true_point']:costs[cohort][name]=len(active)
        costs[cohort]['eligible_all_true_gate']=len(eligible)
        rng=np.random.default_rng(2026112021);random_all=np.zeros((n_draws,len(keys)),bool);random_eligible=random_all.copy()
        for draw in range(n_draws):
            for pool,mask in [(idx,random_all),(eligible,random_eligible)]:
                if len(active):mask[draw,rng.choice(pool,len(active),replace=False)]=True
        masks[cohort]=dict(random_all=random_all,random_eligible=random_eligible)
        for name,mask in [('random25_all_true_gate',random_all),('random25_eligible_true_gate',random_eligible)]:
            selected[:,:,m(name),:]=np.where(mask[None,:,:],gate[:,:,0,:],native[None,None,:]);costs[cohort][name]=len(active)
        choices[cohort]=selected
        # Every query without a requested hint must retain native prediction.
        for fraction in BUDGETS:
            name=f'active{int(fraction*100)}_true_gate';requested=set(policy[str(fraction)]['requested_keys'])
            for n,k in enumerate(keys):
                if k not in requested:assert np.all(selected[:,:,m(name),n]==native[n])
    save(a.out/'request_policy.json',dict(policies=policies,ground_truth_used=False))
    output=a.out/'frozen_choices.npz'
    with output.open('wb') as f:np.savez_compressed(f,**choices,keys=np.asarray(keys),methods=np.asarray(METHODS))
    save(a.out/'prediction_freeze.json',dict(frozen_before_metrics=time.time(),sha256=sha(output),gate_changes_across_conditions_controls_draws=changes,request_counts=costs))
    from experiments.candidate_tools.metric_cache import CandidateMetrics
    cache=CandidateMetrics(bundle,read(a.truth),a.evaluator_root);assert cache.keys==keys
    summary={};intervals={};checks=[]
    for cohort,selected in choices.items():
        idx=cohorts[cohort];metrics=cache.score(selected,idx);summary[cohort]={};intervals[cohort]={}
        for ci,condition in enumerate(z['conditions']):
            summary[cohort][str(condition)]={name:{metric:dict(mean=float(v[ci,:,mi].mean()),hint_draw_SD=float(v[ci,:,mi].std(ddof=1))) for metric,v in metrics.items()} for mi,name in enumerate(METHODS)}
        for ci,draw,mi in [(0,0,m('REN_native')),(1,17,m('active25_true_gate')),(4,n_draws-1,m('random25_all_true_gate'))]:
            cache.verify_official(selected[ci,draw,mi],idx);checks.append(dict(cohort=cohort,condition=str(z['conditions'][ci]),draw=draw,method=METHODS[mi],matches=True))
        sources=sorted({lookup[keys[i]]['source'] for i in idx});counts=np.asarray([sum(lookup[keys[i]]['source']==s for i in idx) for s in sources]);rng=np.random.default_rng(2026112022);boot=rng.integers(len(sources),size=(2000,len(sources)))
        pairs=[('active25_true_gate','REN_native'),('active25_true_gate','random25_all_true_gate'),('active25_true_gate','random25_eligible_true_gate'),('active25_true_gate','active25_shuffled_gate'),('all_true_gate','all_true_point')]
        for ci,condition in enumerate(z['conditions']):
            intervals[cohort][str(condition)]={}
            for left,right in pairs:
                a_hit=(cache.quality[selected[ci,:,m(left),:]]>=.25).mean(0);b_hit=(cache.quality[selected[ci,:,m(right),:]]>=.25).mean(0);delta=a_hit-b_hit
                sums=np.asarray([sum(delta[i] for i in idx if lookup[keys[i]]['source']==s) for s in sources]);values=sums[boot].sum(1)/counts[boot].sum(1)
                intervals[cohort][str(condition)][left+' minus '+right]=dict(metric='Top1_hit_rate_delta',estimate=float(sums.sum()/counts.sum()),CI95=[float(x) for x in np.quantile(values,[.025,.975])])
    save(a.out/'metrics.json',dict(classification='ORACLE-DERIVED SIMULATION',summary=summary,source_intervals=intervals,hint_costs=costs,official_checks=checks))
    lines=['# Ambiguity-aware hint requests: ORACLE-DERIVED simulation','',
           'Same fixed candidates and synthetic hints as the robustness study. No real-user context or new video evaluation. Request decisions use prediction information only. Full official metric calls verify six scenario/cohort combinations.',
           '', '## Primary5second-noise condition, new-tagged queries','', '| Policy | Hints requested | stAP25 mean | tAP25 mean | Recovery mean | Top1 mean |','|---|---:|---:|---:|---:|---:|']
    for name,values in summary['new_queries']['primary_5s'].items():lines.append(f'| {name} | {costs["new_queries"][name]} | {values["stAP25"]["mean"]:.4f} | {values["tAP25"]["mean"]:.4f} | {values["Recovery"]["mean"]:.2f}% | {100*values["Top1"]["mean"]:.2f}% |')
    lines+=['','## Stress conditions: new-tagged stAP25','', '| Condition | REN | All point hints | All gated hints | Active25 gated | Same-budget random all | Same-budget random eligible | Active25 shuffled |','|---|---:|---:|---:|---:|---:|---:|---:|']
    for condition,values in summary['new_queries'].items():lines.append('| '+condition+' | '+' | '.join(f'{values[name]["stAP25"]["mean"]:.4f}' for name in ['REN_native','all_true_point','all_true_gate','active25_true_gate','random25_all_true_gate','random25_eligible_true_gate','active25_shuffled_gate'])+' |')
    lines+=['','## Primary source-bootstrap Top1 differences','']
    for pair,v in intervals['new_queries']['primary_5s'].items():lines.append(f'- {pair}: {v["estimate"]:+.3f}, 95% CI [{v["CI95"][0]:+.3f}, {v["CI95"][1]:+.3f}].')
    lines+=['','Actual request counts are capped by prediction-only eligibility. Random policies use exactly the same count, not an easier query budget. Candidate pool limitations and the short-history development cohort still apply. Intervals describe fixed source-cluster development results;200hint draws are not200sources.']
    (a.out/'RESULTS.md').write_text('\n'.join(lines)+'\n');save(a.out/'COMPLETE.json',dict(completed=time.time(),classification='ORACLE-DERIVED SIMULATION',queries=len(keys)))
    print(a.out/'RESULTS.md')

if __name__=='__main__':main()
