import unittest
from experiments.multimodal_queries.run import rank

class RankingContracts(unittest.TestCase):
    def query(self):
        return dict(key='q',source='s',query_frame=10,score=.4,baseline_id='a',candidates=[dict(id='a',image_score=.5,name_score=.1,bboxes=[[0,0,0,1,1]]),dict(id='b',image_score=.5,name_score=.9,bboxes=[[1,0,0,1,1]])])
    def test_tie_retains_native(self):self.assertEqual(rank(self.query(),'image')[0],0)
    def test_text_uses_text_evidence(self):self.assertEqual(rank(self.query(),'name')[0],1)
    def test_empty_candidate_cannot_win(self):
        q=self.query();q['candidates'][1].update(bboxes=[],appearance_valid=False,image_score=1.)
        self.assertEqual(rank(q,'image')[0],0)
    def test_missing_description_rejected(self):
        with self.assertRaises((KeyError,ValueError)):rank(self.query(),'description')
    def test_cached_scores_replayed_without_refusion(self):
        q=self.query()
        for c,v in zip(q['candidates'],[1.,0.]):c['modality_scores']={'image_text_name':v}
        self.assertEqual(rank(q,'image_name')[0],0)

if __name__=='__main__':unittest.main()
