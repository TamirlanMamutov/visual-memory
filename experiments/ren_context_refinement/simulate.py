"""ORACLE-DERIVED context robustness simulation; no real user-memory evidence."""
from pathlib import Path
import argparse
import contextlib
import hashlib
import io
import json
import sys
import time
import numpy as np

# Set by CLI; simulation inputs and outputs always live outside this code folder.
W=B=D=VQ=None
SEEDS=[2026112101,2026112102,2026112103,2026112104,2026112105]
PER_SEED=40
N=PER_SEED*len(SEEDS)
WEIGHT=.05
CONDITIONS=[
    dict(name='precise_2s',sigma=2.,bias=0.,corruption=0.),
    dict(name='primary_5s',sigma=5.,bias=0.,corruption=0.),
    dict(name='broad_10s',sigma=10.,bias=0.,corruption=0.),
    dict(name='wrong20pct_5s',sigma=5.,bias=0.,corruption=.2),
    dict(name='earlier5s_2s',sigma=2.,bias=-5.,corruption=0.),
    dict(name='coarse_thirds',sigma=0.,bias=0.,corruption=0.),
]
METHODS=['REN_native','CLIP_image','latest_center','image_OOF_time_prior','image_last_third',
         'hint_only','image_hint','image_robust_hint','image_description_hint',
         'image_shuffled_hint','image_random_hint']

def read(p):return json.loads(Path(p).read_text())
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def save(p,d):
    p=Path(p);p.parent.mkdir(parents=True,exist_ok=True);t=p.with_suffix('.tmp');t.write_text(json.dumps(d,indent=2,allow_nan=False)+'\n');t.replace(p)
def state(phase,**kw):
    save(W/'status.json',dict(phase=phase,updated=time.time(),**kw));print('CONTEXT_ROBUSTNESS',phase,kw,flush=True)
def guard():
    import shutil
    if (W/'PAUSED.json').exists() or shutil.disk_usage(W).free<8*1024**3:
        state('paused_or_disk_safety_stop');raise SystemExit(30)

def prepare(data,keys):
    paths=[Path(__file__),B/'dataset.json',D/'text_features.npz',D/'queries.json',B/'protocol.json',B.parent/'DESCRIPTION_STATUS.json']
    paths+=sorted((B/'embeddings').glob('*.npz'))+[Path(data['cases'][k]['baseline']) for k in keys]
    protocol=dict(created=time.time(),classification='ORACLE-DERIVED SIMULATION; NOT REAL USER PERFORMANCE OR BENCHMARK IMPROVEMENT',
        motivation='Previous single-permutation control outperformed noisy true hints; test many shuffled assignments, recency confounding and hint robustness.',
        cohort='User-supplied candidate bundle; source backbone training overlap depends on its provenance.',
        primary='primary_5s image_hint compared with image_shuffled_hint and image_OOF_time_prior on new-tagged queries; all conditions/methods reported.',
        conditions=CONDITIONS,seeds=SEEDS,draws_per_seed=PER_SEED,total_hint_draws=N,
        noise='Common source-correlated Gaussian errors, same draw across compared methods/conditions; clipped to searchable history.',
        wrong_hints='20% source/draw Bernoulli corruption substitutes a different-source normalized target time before adding noise. No inference method sees the corruption flag.',
        shuffle='Independently sampled source-disjoint query permutation for every draw; normalized source midpoints mapped to each destination history; same noise and uncertainty.',
        prior='Five source-grouped folds inherited from v3. Estimate density of relative target midpoints from other-fold sources only; equal source weights; KDE bandwidth10% of current searchable history. No tested source contributes its labels.',
        fusion='image cosine+.05*log mean temporal density across candidate frames. Description uses .5image+.5description. Robust arm uses fixed .8hint+.2OOFprior density; heuristic mixture, not calibrated confidence.',
        coarse='GT-derived early/middle/late third; uniform within supplied third with3%uniform-history smoothing. Shuffled/random category controls and constant last-third prior.',
        methods=METHODS,confidence='Original REN AP confidence unchanged; boxes and frame IDs never modified.',
        metrics='Official pooled stAP25/tAP25/Recovery/Success plus Top1 and Top2; fast per-candidate metric statistics checked against full official evaluator in12scenario/cohort combinations.',
        uncertainty='Mean and Monte Carlo draw SD; five seed-block means; paired source-bootstrap Top1 intervals after averaging draws per query.200draws are not200independent videos.',
        interpretation='Privileged synthetic sighting cues are not remembered actions. Broad/mistaken cues and majority timing priors may explain gains. No final-validation or paper-level claim.',
        description_scores_available=read(B.parent/'DESCRIPTION_STATUS.json')['available'],
        input_sha256={str(p):sha(p) for p in paths})
    if (W/'protocol.json').exists():
        p=read(W/'protocol.json');assert p['conditions']==CONDITIONS and p['methods']==METHODS
        for path,h in p['input_sha256'].items():assert sha(path)==h,('Frozen input changed',path)
    else:save(W/'protocol.json',protocol)
    assert len(keys)>0 and len(data['rows'])>=len(keys)

def generate_inputs(data,keys):
    path=W/'SIMULATED_INPUTS.npz'
    if path.exists():assert sha(path)==read(W/'input_generation.json')['sha256'];return
    state('generating_simulated_cues')
    cases=data['cases'];sources=[cases[k]['item']['video_uid'] for k in keys];unique=sorted(set(sources));source_ids=np.asarray([unique.index(s) for s in sources])
    centers=[];duration=[]
    for k in keys:
        case=cases[k];s=case['item']['selection'];p=read(case['baseline']);gt=p['results']['videos'][0]['clips'][0]['predictions'][0]['query_sets'][s['query_set_id']+'-gt']['bboxes']
        centers.append((gt[0]['fno']+gt[-1]['fno'])/10.);duration.append((s['query_frame']-1)/5.)
    centers=np.asarray(centers);duration=np.asarray(duration);relative=centers/duration
    noises=[];permutations=[];uniform=[];bad=[]
    for seed in SEEDS:
        rng=np.random.default_rng(seed)
        noises.extend(rng.standard_normal((PER_SEED,len(unique)))[:,source_ids])
        bad.extend((rng.uniform(size=(PER_SEED,len(unique)))<.2)[:,source_ids])
        uniform.extend(rng.uniform(size=(PER_SEED,len(keys))))
        for draw in range(PER_SEED):
            for attempt in range(10000):
                p=rng.permutation(len(keys))
                if all(sources[i]!=sources[j] for i,j in enumerate(p)):
                    permutations.append(p);break
            else:raise AssertionError('Failed to make a source-disjoint permutation')
    noises=np.asarray(noises);permutations=np.asarray(permutations);uniform=np.asarray(uniform);bad=np.asarray(bad)
    wrong=relative[permutations]*duration;random=uniform*duration
    hints=[];clipping={}
    for condition in CONDITIONS:
        name=condition['name']
        if name=='coarse_thirds':
            hints.append(np.stack([np.minimum(2,np.floor(relative*3))[None,:].repeat(N,0),np.minimum(2,np.floor(relative[permutations]*3)),np.minimum(2,np.floor(uniform*3))],axis=1));continue
        actual=np.where(bad,wrong,centers) if condition['corruption'] else np.broadcast_to(centers,(N,len(keys)))
        raw=np.stack([actual,wrong,random],axis=1)+condition['bias']+condition['sigma']*noises[:,None,:]
        values=np.clip(raw,0,duration);hints.append(values);clipping[name]=float(np.mean(raw[:,0]!=values[:,0]))
    with path.with_suffix('.tmp').open('wb') as f:np.savez_compressed(f,hints=np.asarray(hints),keys=np.asarray(keys),relative_midpoints=relative,history_seconds=duration,permutations=permutations)
    path.with_suffix('.tmp').replace(path)
    save(W/'input_generation.json',dict(created=time.time(),sha256=sha(path),classification='Privileged GT-derived simulation inputs',source_videos=len(unique),queries=len(keys),draws=N,
        source_disjoint_permutations_verified=True,unique_permutations=len({tuple(p) for p in permutations}),
        clipping_fraction=clipping,coarse_target_bin_counts=np.bincount(np.minimum(2,np.floor(relative*3)).astype(int),minlength=3).tolist(),
        corrupted_source_draw_fraction=float(bad.mean()),search_history_seconds=dict(min=float(duration.min()),median=float(np.median(duration)),max=float(duration.max()))))

def logmean_pdf(times,centers,tau):
    if not len(times):return np.full(len(centers),-1e6)
    a=-.5*((times[None,:]-centers[:,None])/tau)**2-np.log(tau*np.sqrt(2*np.pi))
    maximum=a.max(1);return maximum+np.log(np.exp(a-maximum[:,None]).mean(1))

def rank(data,keys):
    path=W/'rankings.npz'
    if path.exists():assert sha(path)==read(W/'prediction_freeze.json')['sha256'];return
    state('ranking_with_controls')
    z=dict(np.load(W/'SIMULATED_INPUTS.npz'));assert list(z['keys'])==keys
    data_text=dict(np.load(D/'text_features.npz'));description=dict(zip(data_text['keys'],data_text['vectors']))
    rows=data['rows'];cases=data['cases'];folds=read(B/'protocol.json')['outer_assignment'];sources=[cases[k]['item']['video_uid'] for k in keys]
    first=np.empty((len(CONDITIONS),N,len(METHODS),len(keys)),dtype=np.int32);second=first.copy();prior_audit=[]
    for n,k in enumerate(keys):
        guard();ids=np.asarray([i for i,r in enumerate(rows) if r['key']==k]);native=next(int(i) for i in ids if rows[i]['is_baseline'])
        image_z=dict(np.load(B/'embeddings'/(k+'.npz')));assert list(image_z['row_ids'])==ids.tolist()
        image=image_z['candidates']@image_z['query'];desc=image_z['candidates']@description[k];fused=(image+desc)/2
        duration=z['history_seconds'][n];times=[np.asarray([t[0]/5. for t in rows[i]['track']]) for i in ids]
        train=[i for i,s in enumerate(sources) if folds[s]!=folds[sources[n]]];assert all(sources[i]!=sources[n] for i in train)
        counts={s:sum(sources[i]==s for i in train) for s in {sources[i] for i in train}}
        weights=np.asarray([1/counts[sources[i]] for i in train]);weights/=weights.sum()
        prior=[]
        for t in times:
            density=logmean_pdf(t,z['relative_midpoints'][train]*duration,max(.2,.1*duration))
            top=float(density.max());prior.append(top+np.log(np.sum(weights*np.exp(density-top))))
        prior=np.asarray(prior)
        prior_audit.append(dict(key=k,held_source=sources[n],held_fold=folds[sources[n]],training_sources=sorted(set(sources[i] for i in train)),source_weight_sum=float(weights.sum())))
        last_third=np.asarray([np.log(np.mean(np.where(t>=2*duration/3,2.94/duration,.03/duration))) if len(t) else -1e6 for t in times])
        centers=np.asarray([t.mean() if len(t) else -1 for t in times]);latest=int(ids[np.argmax(centers)])
        for ci,condition in enumerate(CONDITIONS):
            evidence=[]
            for control in range(3):
                supplied=z['hints'][ci,:,control,n]
                if condition['name']=='coarse_thirds':
                    columns=[]
                    for t in times:
                        bins=np.minimum(2,np.floor(t/duration*3))
                        likelihood=np.mean(np.where(bins[None,:]==supplied[:,None],2.94/duration,.03/duration),axis=1) if len(t) else np.full(N,1e-100)
                        columns.append(np.log(likelihood))
                    evidence.append(np.stack(columns,axis=1))
                else:evidence.append(np.stack([logmean_pdf(t,supplied,max(.2,condition['sigma'])) for t in times],axis=1))
            robust=np.logaddexp(np.log(.8)+evidence[0],np.log(.2)+prior[None,:])
            scores=[None,image,None,image+WEIGHT*prior,image+WEIGHT*last_third,evidence[0],
                    image+WEIGHT*evidence[0],image+WEIGHT*robust,fused+WEIGHT*evidence[0],
                    image+WEIGHT*evidence[1],image+WEIGHT*evidence[2]]
            for mi,score in enumerate(scores):
                if mi in [0,2]:
                    choice=native if mi==0 else latest;first[ci,:,mi,n]=second[ci,:,mi,n]=choice;continue
                a=np.broadcast_to(score,(N,len(ids)));tie=np.broadcast_to(ids!=native,a.shape);index=np.broadcast_to(np.arange(len(ids)),a.shape)
                order=np.lexsort((index,tie,-a),axis=-1)
                first[ci,:,mi,n]=ids[order[:,0]];second[ci,:,mi,n]=ids[order[:,min(1,len(ids)-1)]]
    with path.with_suffix('.tmp').open('wb') as f:np.savez_compressed(f,first=first,second=second,keys=np.asarray(keys),methods=np.asarray(METHODS),conditions=np.asarray([c['name'] for c in CONDITIONS]))
    path.with_suffix('.tmp').replace(path)
    save(W/'prediction_freeze.json',dict(frozen_before_metrics=time.time(),sha256=sha(path),hint_sha256=sha(W/'SIMULATED_INPUTS.npz'),OOF_prior=prior_audit,ranking_uses_no_GT_except_explicit_simulated_cue_and_other_fold_prior=True))

def evaluate(data,keys):
    state('evaluating_frozen_rankings')
    from evaluate import official_api
    import importlib
    compute_visual_query_metrics,BBox,ResponseTrack=official_api(VQ)
    prefix='vq2d.metrics' if (VQ/'vq2d/metrics').exists() else 'metrics'
    utils=importlib.import_module(prefix+'.utils')
    spatial_iou=utils.spatial_iou
    spatio_temporal_iou_response_track=utils.spatio_temporal_iou_response_track
    segment_iou=utils.segment_iou
    cases=data['cases'];rows=data['rows'];keyindex={k:i for i,k in enumerate(keys)};truth=[];confidence=[];length=[]
    for k in keys:
        c=cases[k];qid=c['item']['selection']['query_set_id'];p=read(c['baseline']);sets=p['results']['videos'][0]['clips'][0]['predictions'][0]['query_sets']
        g=ResponseTrack([BBox(b['fno'],b['x1'],b['y1'],b['x2'],b['y2']) for b in sets[qid+'-gt']['bboxes']]);truth.append(g);length.append(len(g.bboxes));confidence.append(sets[qid]['score'])
    tracks=[];quality=[];temporal=[];recovered=[]
    for row in rows:
        n=keyindex[row['key']];t=ResponseTrack([BBox(f,x,y,x+w,y+h) for f,x,y,w,h in row['track']],score=confidence[n]);g=truth[n];tracks.append(t)
        value=float(spatio_temporal_iou_response_track(g,t));assert abs(value-row['track_IoU'])<1e-10
        quality.append(value);temporal.append(float(segment_iou(np.asarray(g.temporal_extent),np.asarray([t.temporal_extent]))[0]))
        by={b.fno:b for b in t.bboxes};recovered.append(sum(b.fno in by and spatial_iou(b,by[b.fno])>=.5 for b in g.bboxes))
    quality=np.asarray(quality);temporal=np.asarray(temporal);recovered=np.asarray(recovered);confidence=np.asarray(confidence);length=np.asarray(length)
    z=dict(np.load(W/'rankings.npz'));assert list(z['keys'])==keys;first=z['first'];second=z['second']
    assert sha(W/'rankings.npz')==read(W/'prediction_freeze.json')['sha256']
    cohorts={'all_queries':np.arange(len(keys)), 'new_queries':np.asarray([i for i,k in enumerate(keys) if not cases[k]['old']])}
    def fast(chosen,idx,other):
        a=chosen[...,idx];q=quality[a];order=confidence[idx].argsort()[::-1]
        def ap(v):
            hit=(v>=.25).astype(float)[...,order];tp=hit.cumsum(-1);prec=tp/np.arange(1,len(idx)+1);envelope=np.maximum.accumulate(prec[...,::-1],axis=-1)[...,::-1]
            return (envelope*hit).sum(-1)/len(idx)
        return dict(stAP25=ap(q),tAP25=ap(temporal[a]),Recovery=100*recovered[a].sum(-1)/length[idx].sum(),Success=100*(q>=.05).mean(-1),Top1=(q>=.25).mean(-1),Top2=((q>=.25)|(quality[other[...,idx]]>=.25)).mean(-1))
    metrics={label:fast(first,idx,second) for label,idx in cohorts.items()}
    checks=[]
    for ci,draw,mi in [(0,0,0),(0,1,1),(1,17,6),(2,199,9),(3,80,7),(5,120,8)]:
        for label,idx in cohorts.items():
            chosen=first[ci,draw,mi]
            with contextlib.redirect_stdout(io.StringIO()):official=compute_visual_query_metrics([[tracks[chosen[i]]] for i in idx],[truth[i] for i in idx],[BBox(-1,0,0,1,1) for _ in idx],vc_rt_pairings={'all':('all','all')})['all']
            for metric,prefix,suffix in [('stAP25','SpatioTemporal AP','IoU=0.25'),('tAP25','Temporal AP','IoU=0.25'),('Recovery','Tracking','IoU=0.50'),('Success','Success','IoU=0.05')]:
                value=float(next(v for k,v in official.items() if k.strip().startswith(prefix) and k.strip().endswith(suffix)))
                assert abs(value-metrics[label][metric][ci,draw,mi])<1e-10,(label,metric,value,metrics[label][metric][ci,draw,mi])
            checks.append(dict(cohort=label,condition=CONDITIONS[ci]['name'],draw=draw,method=METHODS[mi],matches=True))
    save(W/'official_checks.json',dict(scenarios=checks,candidate_IoUs_verified=len(rows),evaluator_sha256={str(p):sha(p) for p in (VQ/'vq2d/metrics' if (VQ/'vq2d/metrics').exists() else VQ/'metrics').glob('*.py')}))
    summary={};intervals={}
    pairs=[('image_hint','CLIP_image'),('image_hint','image_shuffled_hint'),('image_hint','image_OOF_time_prior'),('image_hint','latest_center'),('image_robust_hint','image_hint')]
    for label,idx in cohorts.items():
        summary[label]={};intervals[label]={}
        sources=sorted({cases[keys[i]]['item']['video_uid'] for i in idx});counts=np.asarray([sum(cases[keys[i]]['item']['video_uid']==s for i in idx) for s in sources]);rng=np.random.default_rng(2026112011);boot=rng.integers(len(sources),size=(2000,len(sources)))
        for ci,condition in enumerate(CONDITIONS):
            cname=condition['name'];summary[label][cname]={};intervals[label][cname]={}
            for mi,method in enumerate(METHODS):
                summary[label][cname][method]={metric:dict(mean=float(value[ci,:,mi].mean()),hint_draw_SD=float(value[ci,:,mi].std(ddof=1)),seed_block_means=[float(value[ci,s*PER_SEED:(s+1)*PER_SEED,mi].mean()) for s in range(len(SEEDS))]) for metric,value in metrics[label].items()}
            for a,b in pairs:
                ahit=(quality[first[ci,:,METHODS.index(a),:]]>=.25).mean(0);bhit=(quality[first[ci,:,METHODS.index(b),:]]>=.25).mean(0);delta=ahit-bhit
                sums=np.asarray([sum(delta[i] for i in idx if cases[keys[i]]['item']['video_uid']==s) for s in sources]);values=sums[boot].sum(1)/counts[boot].sum(1)
                intervals[label][cname][a+' minus '+b]=dict(metric='Top1_hit_rate_delta',estimate=float(sums.sum()/counts.sum()),CI95=[float(x) for x in np.quantile(values,[.025,.975])],unit='source video',source_count=len(sources))
    save(W/'metrics.json',dict(classification='ORACLE-DERIVED SIMULATION',summary=summary,paired_source_intervals=intervals))
    lines=['# ORACLE-DERIVED timing robustness simulation','',
        '**Not real user-memory input, remembered-action understanding, or benchmark improvement.** User-supplied fixed candidate bank. Five independent simulation seed blocks,40draws each; source-level uncertainty uses actual source counts, not200independent samples.',
        '', '## Primary5second-noise condition, new-tagged queries','',
        '| Method | stAP25 mean | tAP25 mean | Recovery mean | Top1 mean |','|---|---:|---:|---:|---:|']
    for method,m in summary['new_queries']['primary_5s'].items():lines.append(f'| {method} | {m["stAP25"]["mean"]:.4f} | {m["tAP25"]["mean"]:.4f} | {m["Recovery"]["mean"]:.2f}% | {100*m["Top1"]["mean"]:.2f}% |')
    lines+=['','## All conditions, new-tagged queriesstAP25','', '| Condition | REN | Latest center | OOF prior | True hint | Robust mixture | Shuffled | Random |','|---|---:|---:|---:|---:|---:|---:|']
    for condition in CONDITIONS:
        name=condition['name'];d=summary['new_queries'][name];values=[d[m]['stAP25']['mean'] for m in ['REN_native','latest_center','image_OOF_time_prior','image_hint','image_robust_hint','image_shuffled_hint','image_random_hint']];lines.append('| '+name+' | '+' | '.join(f'{v:.4f}' for v in values)+' |')
    lines+=['','## Paired source-bootstrap Top1 differences: true versus shuffled hints','']
    for name,d in intervals['new_queries'].items():
        v=d['image_hint minus image_shuffled_hint'];lines.append(f'- {name}: {v["estimate"]:+.3f}, 95% CI [{v["CI95"][0]:+.3f}, {v["CI95"][1]:+.3f}].')
    lines+=['','OOF time prior uses labels only from other source folds. Latest-center and last-third rules use no user hint. Shuffled controls use200distinct source-disjoint permutations with matched noise. The robust mixture has fixed weights and was not tuned. All metrics retain original REN confidence; target boxes enter only cue generation/prior fitting and evaluation.',
        '', 'Earlier-sighting bias is a stress test, not an annotated interaction. Early/middle/late labels are generated from GT relative midpoints; they are not NLP or user memory evidence. Natural descriptions remain assistant-authored. A real-user study and independently defined context targets are still required. This script never starts or stops another model run.']
    if not read(B.parent/'DESCRIPTION_STATUS.json')['available']:
        lines+=['', 'DESCRIPTION SCORES NOT PROVIDED: the description arm duplicates image scores and must not be reported as text fusion. The EAGLE adapter supplies object-name scores but no independently authored description scores.']
    (W/'RESULTS.md').write_text('\n'.join(lines)+'\n')
    save(W/'COMPLETE.json',dict(completed=time.time(),queries=len(keys),simulation_draws=N,classification='ORACLE-DERIVED SIMULATION',does_not_control_other_runs=True));state('complete',report=str(W/'RESULTS.md'))

def main():
    global W,B,D,VQ
    from portable_io import prepare_cache
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--bundle',type=Path,required=True)
    ap.add_argument('--truth',type=Path,required=True)
    ap.add_argument('--evaluator-root',type=Path,required=True)
    ap.add_argument('--out',type=Path,required=True)
    ap.add_argument('--evaluate-only',action='store_true')
    ap.add_argument('--check',action='store_true')
    args=ap.parse_args()
    W=args.out.resolve();W.mkdir(parents=True,exist_ok=True);VQ=args.evaluator_root.resolve()
    B,D=prepare_cache(args.bundle,args.truth,VQ,W/'oracle_internal_cache')
    data=read(B/'dataset.json');keys=sorted(data['cases']);prepare(data,keys);guard()
    if args.check:print('CHECK OK: fixed context-robustness simulation protocol.');return
    if not args.evaluate_only:generate_inputs(data,keys);rank(data,keys)
    evaluate(data,keys)

if __name__=='__main__':
    try:main()
    except SystemExit:raise
    except Exception as exc:
        if W is not None:state('error',error=repr(exc))
        raise
