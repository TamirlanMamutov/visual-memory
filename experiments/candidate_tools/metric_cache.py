"""Fast pooled metric summaries for fixed candidate banks, checked with official code."""
import contextlib
import importlib
import io
from pathlib import Path
import numpy as np
from experiments.candidate_tools.evaluate import official_api

class CandidateMetrics:
    def __init__(self,bundle,truth,evaluator_root):
        self.compute,self.BBox,self.ResponseTrack=official_api(evaluator_root)
        root=Path(evaluator_root)
        prefix='vq2d.metrics' if (root/'vq2d/metrics').exists() else 'metrics'
        utils=importlib.import_module(prefix+'.utils')
        self.queries={q['key']:q for q in bundle['queries']};self.keys=sorted(self.queries)
        gt={q['key']:q for q in truth['queries']}
        if set(gt)!=set(self.keys):raise ValueError('Exact query/GT coverage required')
        self.truth=[];self.confidence=[];self.gt_length=[]
        for k in self.keys:
            g=self.ResponseTrack([self.BBox(*b) for b in gt[k]['bboxes']])
            if not g.bboxes:raise ValueError('Empty target track')
            self.truth.append(g);self.confidence.append(self.queries[k]['score']);self.gt_length.append(len(g.bboxes))
        index={k:i for i,k in enumerate(self.keys)}
        self.candidates=[];self.candidate_keys=[];self.quality=[];self.temporal=[];self.recovered=[]
        for q in bundle['queries']:
            n=index[q['key']];g=self.truth[n]
            for c in q['candidates']:
                t=self.ResponseTrack([self.BBox(*b) for b in c['bboxes']],score=q['score'])
                self.candidates.append(t);self.candidate_keys.append(q['key'])
                self.quality.append(float(utils.spatio_temporal_iou_response_track(g,t)))
                self.temporal.append(float(utils.segment_iou(np.asarray(g.temporal_extent),np.asarray([t.temporal_extent]))[0]))
                by={b.fno:b for b in t.bboxes}
                self.recovered.append(sum(b.fno in by and utils.spatial_iou(b,by[b.fno])>=.5 for b in g.bboxes))
        for field in ['confidence','gt_length','quality','temporal','recovered']:setattr(self,field,np.asarray(getattr(self,field)))

    def score(self,choices,idx):
        choices=np.asarray(choices);idx=np.asarray(idx);a=choices[...,idx];q=self.quality[a]
        order=self.confidence[idx].argsort()[::-1]
        def ap(values):
            hit=(values>=.25).astype(float)[...,order];tp=hit.cumsum(-1);precision=tp/np.arange(1,len(idx)+1)
            envelope=np.maximum.accumulate(precision[...,::-1],axis=-1)[...,::-1]
            return (envelope*hit).sum(-1)/len(idx)
        return dict(stAP25=ap(q),tAP25=ap(self.temporal[a]),Recovery=100*self.recovered[a].sum(-1)/self.gt_length[idx].sum(),Success=100*(q>=.05).mean(-1),Top1=(q>=.25).mean(-1),mean_track_IoU=q.mean(-1))

    def verify_official(self,choices,idx):
        choices=np.asarray(choices);idx=np.asarray(idx)
        with contextlib.redirect_stdout(io.StringIO()):
            raw=self.compute([[self.candidates[choices[i]]] for i in idx],[self.truth[i] for i in idx],
                             [self.BBox(-1,0,0,1,1) for _ in idx],vc_rt_pairings={'all':('all','all')})['all']
        fast=self.score(choices,idx)
        for label,prefix,suffix in [('stAP25','SpatioTemporal AP','IoU=0.25'),('tAP25','Temporal AP','IoU=0.25'),('Recovery','Tracking','IoU=0.50'),('Success','Success','IoU=0.05')]:
            value=float(next(v for k,v in raw.items() if k.strip().startswith(prefix) and k.strip().endswith(suffix)))
            if abs(value-fast[label])>1e-10:raise AssertionError((label,value,fast[label]))
        return True
