"""Guards against reserved-source leakage and cross-model feature substitution."""
import copy,unittest
import numpy as np
import torch
from experiments.track_verifier import core
from experiments.track_verifier.run_cv import validate

class TrainingContracts(unittest.TestCase):
    def dataset(self):
        rows=[];queries=[]
        for i in range(6):
            key='q'+str(i);source='s'+str(i)
            rows.append(dict(key=key,source=source,is_baseline=True,label=1,features=[0.]*12,track=[[0,0,0,1,1]],track_IoU=1.))
            queries.append(dict(key=key,source=source,query_frame=10))
        return dict(training_only=True,rows=rows,bundle=dict(queries=queries),folds={'s'+str(i):i%3 for i in range(6)})
    def test_reserved_source_rejected(self):
        d=self.dataset();d['excluded_sources']=['s2']
        with self.assertRaisesRegex(ValueError,'Reserved'):validate(d)
    def test_validation_export_rejected(self):
        d=self.dataset();d['training_only']=False
        with self.assertRaisesRegex(ValueError,'training_only'):validate(d)
    def test_source_assignment_required(self):
        d=self.dataset();d['folds'].pop('s0')
        with self.assertRaisesRegex(ValueError,'source-fold'):validate(d)
    def test_single_fold_rejected(self):
        d=self.dataset();d['folds']={s:0 for s in d['folds']}
        with self.assertRaisesRegex(ValueError,'two folds'):validate(d)
    def test_future_track_rejected(self):
        d=self.dataset();d['rows'][0]['track'][0][0]=10
        with self.assertRaisesRegex(ValueError,'Post-query'):validate(d)
    def test_matched_initialization_zero_padding(self):
        cfg=dict(core.DEFAULT);torch.manual_seed(0);a=core.Head(12,cfg)
        torch.manual_seed(0);b=core.Head(24,cfg)
        self.assertTrue(torch.equal(a.net[0].weight,b.net[0].weight[:,:12]))
        self.assertTrue(torch.equal(torch.zeros(8,12),b.net[0].weight[:,12:]))
    def test_empty_native_fallback_preserved(self):
        rows=self.dataset()['rows'][:1];rows[0]['empty_native_fallback']=True
        alternative=copy.deepcopy(rows[0]);alternative.update(is_baseline=False,empty_native_fallback=False)
        self.assertEqual(core.choose(rows+[alternative],np.asarray([-10.,10.]),.5,core.DEFAULT),{'q0':0})
    def test_frozen_head_roundtrip(self):
        cfg=dict(core.DEFAULT);torch.manual_seed(2);net=core.Head(12,cfg)
        saved=core.serialize([(net,torch.zeros(12),torch.ones(12))],cfg,.65,'ren_quality_v1')
        models,loaded,threshold=core.deserialize(saved)
        rows=self.dataset()['rows'];np.testing.assert_array_equal(core.logits(rows,models[0],loaded),core.logits(rows,(net,torch.zeros(12),torch.ones(12)),cfg))
        self.assertEqual(threshold,.65)

if __name__=='__main__':unittest.main()
