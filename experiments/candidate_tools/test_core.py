"""Behavioral contracts: timing units, immutable windows and unchanged confidence."""
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
import copy
import unittest
from experiments.box_refinement.refinement import refine_boxes
from experiments.additional_hints.hints import timing_rank,native_gated_choice
from experiments.additional_hints.clarification import request_policy
from experiments.candidate_tools.candidate_common import validate_query
from experiments.candidate_tools.adapt_eagle import choose_records, export_pool

def candidate(identifier,start,score,offset=0):
    return dict(id=identifier,bboxes=[[f,offset,0,offset+10,10] for f in range(start,start+5)],
                image_score=score,name_score=score,description_score=score,
                samples=[dict(frame=f,image_score=score,name_score=score,description_score=score) for f in [start,start+2,start+4]])

def query():
    return dict(key='synthetic_query',source='synthetic_source',query_frame=50,fps=5,score=.81,
                baseline_id='base',candidates=[candidate('base',0,.6),candidate('other',20,.7,2)])

class CoreTests(unittest.TestCase):
    def test_declined_hint_keeps_native_without_a_timestamp(self):
        q=query();selected,detail=native_gated_choice(q,dict(declined=True))
        self.assertEqual(selected,'base');self.assertFalse(detail['changed'])

    def test_conservative_hint_gate_accepts_compatible_appearance(self):
        q=query();q['candidates'][1]['image_score']=.59
        selected,detail=native_gated_choice(q,dict(kind='time',seconds=4.4,sigma_seconds=.2))
        self.assertEqual(selected,'other');self.assertTrue(detail['changed'])

    def test_clarification_policy_needs_no_labels(self):
        a=query();a['candidates'][1]['image_score']=.62
        b=copy.deepcopy(a);b.update(key='second',source='second_source')
        b['candidates'][1]['bboxes']=copy.deepcopy(b['candidates'][0]['bboxes'])
        b['candidates'][1]['samples']=copy.deepcopy(b['candidates'][0]['samples'])
        policy=request_policy(dict(queries=[a,b]),.5)
        self.assertEqual(policy['requested_keys'],['synthetic_query']);self.assertEqual(policy['eligible'],1)

    def test_timing_clue_uses_seconds_not_frame_numbers(self):
        q=query();before=copy.deepcopy(q)
        self.assertEqual(timing_rank(q,dict(kind='time',seconds=.4,sigma_seconds=.1))[0],'base')
        self.assertEqual(timing_rank(q,dict(kind='time',seconds=4.4,sigma_seconds=.1))[0],'other')
        self.assertEqual(q,before)

    def test_window_refinement_rejects_wrong_episode(self):
        q=query();q['candidates'][1]['image_score']=.99
        result=refine_boxes(q)
        self.assertEqual(result['bboxes'],q['candidates'][0]['bboxes'])
        self.assertEqual(result['score'],q['score'])

    def test_spatial_change_keeps_all_frames_and_ap_score(self):
        q=query();q['candidates'].append(candidate('overlap',0,.9,3));original=copy.deepcopy(q)
        result=refine_boxes(q)
        self.assertEqual(result['selected_candidate'],'overlap')
        self.assertEqual([b[0] for b in result['bboxes']],list(range(5)))
        self.assertEqual(result['score'],.81);self.assertEqual(q,original)

    def test_future_frames_are_rejected(self):
        q=query();q['candidates'][1]['bboxes'][-1][0]=50
        with self.assertRaises(ValueError):validate_query(q)

    def test_empty_baseline_stays_empty_in_window_refinement(self):
        q=query();q['candidates'][0]['bboxes']=[];q['candidates'][0]['samples']=[]
        self.assertEqual(refine_boxes(q)['bboxes'],[])

    def test_eagle_pool_deduplicates_identical_tracks(self):
        boxes=[[1,0,0,10,10],[2,0,0,10,10]]
        parts=[{'q':dict(bboxes=boxes,score=.8)},{'q':dict(bboxes=copy.deepcopy(boxes),score=.9)}]
        before=copy.deepcopy(parts)
        baseline,pool=export_pool(dict(key='q'),parts)
        self.assertEqual(len(pool),1);self.assertEqual(pool[0]['id'],'baseline');self.assertEqual(parts,before)

    def test_eagle_selection_spreads_sources_before_extra_queries(self):
        records=[dict(key='a1',source='a'),dict(key='a2',source='a'),dict(key='b1',source='b'),dict(key='c1',source='c')]
        selected=choose_records(records,3,230)
        self.assertEqual(len({r['source'] for r in selected}),3)
        self.assertEqual(selected,choose_records(records,3,230))

if __name__=='__main__':unittest.main()
