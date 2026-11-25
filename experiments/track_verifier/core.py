"""Portable matched track-quality heads; no local-machine imports or paths."""
import copy,json,math,random
from pathlib import Path
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

FEATURE_NAMES=['REN_score','query_similarity_mean','query_similarity_p10','query_similarity_std',
 'valid_sample_fraction','log_area_std','relative_center_shift','neighbor_box_IoU',
 'duration_fraction','end_fraction','start_fraction','frame_density']
DEFAULT=dict(lr=.003,weight_decay=.01,epochs=200,width=8,dropout=0.,normalization='std',clip_z=None,
 rank_weight=0.,risk_sd=0.,minimum_overlap=0.,source_balance=True,selector='AP',features='stats',matched_init=True)

def groups(rows):
    return {k:[i for i,r in enumerate(rows) if r['key']==k] for k in sorted({r['key'] for r in rows})}

def augment(raw,cfg):
    rows=copy.deepcopy(raw);mode=cfg.get('features','stats');g=groups(rows)
    rotated={i:i for i in range(len(rows))}
    if mode=='hand_shuffled_control':
        for ids in g.values():
            for a,b in zip(ids,ids[1:]+ids[:1]):rotated[a]=b
    for key,ids in g.items():
        base=next(rows[i] for i in ids if rows[i]['is_baseline'])
        for i in ids:
            r=rows[i];v=list(r['features'])
            if len(v)!=12:raise ValueError('Expected the 12 base track-quality features')
            if mode.startswith('hand_'):
                extra=list(raw[rotated[i]]['hand_features'])
                if len(extra)!=12:raise ValueError('Expected 12 hand proxy features')
                if mode=='hand_presence':extra=extra[:1]
                elif mode=='hand_proximity':extra=extra[:4]
                elif mode=='hand_proximity_motion':extra=extra[:8]
                elif mode=='hand_zero_control':extra=[0.]*12
                elif mode not in ['hand_all','hand_shuffled_control']:raise ValueError(mode)
                v+=extra
            elif mode!='stats':
                v.append(r['image_cosine'])
                if mode in ['name','relative']:v.append(r['name_cosine'])
                elif mode=='shuffled_name':v.append(r['shuffled_name_cosine'])
                if mode=='relative':
                    a={b[0] for b in r['track']};b={b[0] for b in base['track']}
                    v += [len(a&b)/max(1,len(a|b)),r['features'][9]-base['features'][9],
                          r['features'][1]-base['features'][1],r['image_cosine']-base['image_cosine']]
            r['features']=v
    return rows

class Head(nn.Module):
    def __init__(self,dimensions,cfg):
        super().__init__();matched=cfg.get('matched_init',True)
        layers=[nn.Linear(12 if matched else dimensions,cfg['width']),nn.GELU()]
        if cfg['dropout']:layers.append(nn.Dropout(cfg['dropout']))
        layers.append(nn.Linear(cfg['width'],1));self.net=nn.Sequential(*layers)
        nn.init.zeros_(self.net[-1].weight);nn.init.zeros_(self.net[-1].bias)
        if matched and dimensions>12:
            self.net[0].weight=nn.Parameter(torch.cat([self.net[0].weight.detach(),torch.zeros(cfg['width'],dimensions-12)],dim=1))
            self.net[0].in_features=dimensions
    def forward(self,x):return self.net(x).squeeze(-1)

def transform(x,mean,scale,cfg):
    x=(x-mean)/scale
    return x.clamp(-cfg['clip_z'],cfg['clip_z']) if cfg['clip_z'] else x

def fit(rows,seed,cfg):
    torch.manual_seed(seed);np.random.seed(seed);random.seed(seed)
    positive=sum(r['label']==1 for r in rows);negative=sum(r['label']==0 for r in rows)
    if min(positive,negative)<2:return None,dict(seed=seed,reason='insufficient_classes')
    x=torch.tensor(np.asarray([r['features'] for r in rows]),dtype=torch.float32)
    if cfg['normalization']=='robust':
        mean=torch.quantile(x,.5,dim=0);scale=((torch.quantile(x,.75,dim=0)-torch.quantile(x,.25,dim=0))/1.349).clamp_min(.05)
    else:mean=x.mean(0);scale=x.std(0,unbiased=False).clamp_min(.05)
    x=transform(x,mean,scale,cfg);mask=torch.tensor([r['label']>=0 for r in rows]);y=torch.tensor([max(0,r['label']) for r in rows],dtype=torch.float32)
    g=groups(rows);source_queries={s:len({r['key'] for r in rows if r['source']==s}) for s in {r['source'] for r in rows}}
    weights=torch.tensor([1/(len(g[r['key']])*(source_queries[r['source']] if cfg['source_balance'] else 1)) for r in rows]);weights/=weights[mask].mean()
    pairs=[];pair_weights=[]
    if cfg['rank_weight']:
        for ids in g.values():
            current=[(p,n) for p in ids for n in ids if rows[p]['label']==1 and rows[n]['label']==0]
            pairs+=current;pair_weights += [1/len(current)]*len(current) if current else []
    net=Head(x.shape[1],cfg);opt=torch.optim.AdamW(net.parameters(),lr=cfg['lr'],weight_decay=cfg['weight_decay'])
    for _ in range(cfg['epochs']):
        opt.zero_grad(set_to_none=True);scores=net(x)
        loss=(F.binary_cross_entropy_with_logits(scores[mask],y[mask],pos_weight=torch.tensor(negative/positive),reduction='none')*weights[mask]).mean()
        if pairs:
            values=torch.stack([F.softplus(scores[n]-scores[p]) for p,n in pairs]);w=torch.tensor(pair_weights)
            loss=loss+cfg['rank_weight']*(values*w).sum()/w.sum()
        loss.backward();torch.nn.utils.clip_grad_norm_(net.parameters(),5);opt.step()
    net.eval()
    return (net,mean,scale),dict(seed=seed,training_sources=len(source_queries),positive_tracks=positive,negative_tracks=negative,loss=float(loss.detach()))

def logits(rows,model,cfg):
    if model is None:return np.zeros(len(rows))
    net,mean,scale=model
    with torch.inference_mode():return net(transform(torch.tensor(np.asarray([r['features'] for r in rows]),dtype=torch.float32),mean,scale,cfg)).numpy()

def choose(rows,scores,threshold,cfg):
    result={}
    for key,ids in groups(rows).items():
        base=next(i for i in ids if rows[i]['is_baseline']);alts=[i for i in ids if i!=base]
        if rows[base].get('empty_native_fallback'):
            result[key]=base;continue
        best=max(alts,key=lambda i:(float(scores[i]),rows[i]['features'][9],-i)) if alts else base
        preference=1/(1+np.exp(-float(np.clip(scores[best]-scores[base],-50,50))))
        pick=best if threshold is not None and best!=base and scores[best]>0 and preference>threshold else base
        if cfg['minimum_overlap']:
            a={b[0] for b in rows[pick]['track']};b={x[0] for x in rows[base]['track']}
            if len(a&b)/max(1,len(a|b))<cfg['minimum_overlap']:pick=base
        result[key]=pick
    return result

def ensemble(train,test,cfg,seeds=(0,1,2)):
    scores=[];models=[];details=[]
    for seed in seeds:
        model,detail=fit(train,seed,cfg);models.append(model);details.append(detail);scores.append(logits(test,model,cfg))
    mean=np.mean(scores,axis=0)
    if cfg['risk_sd']:
        adjusted=mean.copy()
        for ids in groups(test).values():
            base=next(i for i in ids if test[i]['is_baseline'])
            for i in ids:
                delta=np.asarray([s[i]-s[base] for s in scores]);adjusted[i]=mean[base]+delta.mean()-cfg['risk_sd']*delta.std(ddof=1)
        mean=adjusted
    return mean,scores,models,details

def comparison(rows,selection):
    base={k:next(i for i in ids if rows[i]['is_baseline']) for k,ids in groups(rows).items()}
    d=[rows[selection[k]]['track_IoU']-rows[base[k]]['track_IoU'] for k in base]
    return dict(wins=sum(v>1e-8 for v in d),losses=sum(v< -1e-8 for v in d),changes=sum(selection[k]!=base[k] for k in base),
                mean_track_IoU_gain=float(np.mean(d)),mean_track_IoU=float(np.mean([rows[i]['track_IoU'] for i in selection.values()])))

def ap(rows,selection,confidence):
    keys=sorted(selection);scores=np.asarray([confidence[k] for k in keys]);hit=np.asarray([rows[selection[k]]['track_IoU']>=.25 for k in keys],dtype=float)
    hit=hit[scores.argsort()[::-1]];precision=hit.cumsum()/np.arange(1,len(hit)+1)
    return float((np.maximum.accumulate(precision[::-1])[::-1]*hit).sum()/len(hit))

def select_threshold(rows,outer_fold,cfg,confidence,seeds=(0,1,2)):
    sources=sorted({r['source'] for r in rows});random.Random(2026111801+outer_fold).shuffle(sources)
    assignment={s:i%3 for i,s in enumerate(sources)};scores=np.full(len(rows),np.nan);audits=[]
    for fold in range(3):
        train=[r for r in rows if assignment[r['source']]!=fold];ids=[i for i,r in enumerate(rows) if assignment[r['source']]==fold];held=[rows[i] for i in ids]
        if not held:continue
        assert not {r['source'] for r in train}&{r['source'] for r in held}
        value,_,_,detail=ensemble(train,held,cfg,seeds);scores[ids]=value
        audits.append(dict(fold=fold,training_sources=sorted({r['source'] for r in train}),held_sources=sorted({r['source'] for r in held}),training=detail))
    assert np.isfinite(scores).all()
    base={k:next(i for i in ids if rows[i]['is_baseline']) for k,ids in groups(rows).items()};native=ap(rows,base,confidence);options=[]
    for threshold in [.5,.65,.8,.9]:
        selection=choose(rows,scores,threshold,cfg);stats=comparison(rows,selection);point=ap(rows,selection,confidence)
        passed=(point>native+1e-10 and stats['mean_track_IoU_gain']>=0) if cfg['selector']=='AP' else (stats['wins']>=1 and stats['losses']<=stats['wins'] and stats['mean_track_IoU_gain']>=.005)
        options.append(dict(threshold=threshold,AP=point,AP_gain=point-native,passes=passed,**stats))
    valid=[x for x in options if x['passes']]
    order=(lambda x:(x['AP_gain'],x['mean_track_IoU_gain'],-x['changes'],x['threshold'])) if cfg['selector']=='AP' else (lambda x:(round(x['mean_track_IoU_gain'],9),-x['changes'],x['threshold']))
    selected=max(valid,key=order)['threshold'] if valid else None
    return selected,dict(selected=selected,options=options,source_assignment=assignment,folds=audits)

def serialize(models,cfg,threshold,feature_family):
    def values(model):
        if model is None:return None
        net,mean,scale=model
        return dict(state_dict={k:v.detach().cpu().tolist() for k,v in net.state_dict().items()},mean=mean.tolist(),scale=scale.tolist())
    return dict(version=1,configuration=cfg,threshold=threshold,feature_family=feature_family,seeds=[values(m) for m in models])

def deserialize(bundle):
    cfg=dict(DEFAULT,**bundle['configuration']);models=[]
    for seed in bundle['seeds']:
        if seed is None:models.append(None);continue
        net=Head(len(seed['mean']),cfg);net.load_state_dict({k:torch.tensor(v,dtype=torch.float32) for k,v in seed['state_dict'].items()});net.eval()
        models.append((net,torch.tensor(seed['mean']),torch.tensor(seed['scale'])))
    return models,cfg,bundle['threshold']
