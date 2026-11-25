"""Independently replay saved frozen heads, box refinement and modality rankings."""
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
import argparse,json,subprocess,time
import numpy as np
from experiments.candidate_tools.evaluate import score_files
from experiments.candidate_tools.candidate_common import track_prediction
from experiments.box_refinement.refinement import refine_boxes
from experiments.multimodal_queries.run import rank
from experiments.reproduction.verify_handoff import verify

def read(p):return json.loads(p.read_text())
def save(p,d):p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(d,indent=2)+'\n')
def compare(actual,expected):
    for field in ['stAP25','tAP25','Recovery','Success']:
        if abs(actual[field]-expected[field])>1e-10:raise AssertionError((field,actual[field],expected[field]))
def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--handoff',type=Path,required=True);p.add_argument('--evaluator-root',type=Path,required=True);p.add_argument('--out',type=Path,required=True)
    a=p.parse_args();a.out.mkdir(parents=True,exist_ok=True);report=dict(created=time.time(),verified_files=verify(a.handoff),checks=[])
    box=a.handoff/'box_hints51';bundle=read(box/'bundle.json');truth=read(box/'ground_truth.json')
    baseline=dict(queries=[track_prediction(q,q['baseline_id']) for q in bundle['queries']])
    refined=dict(queries=[refine_boxes(q) for q in bundle['queries']])
    expected=read(Path(__file__).resolve().parents[1]/'box_refinement/results/ren_box_refinement.json')['box_refinement']
    for name,pred in [('REN_baseline',baseline),('window_image',refined)]:
        save(a.out/'box51'/(name+'.json'),pred)
        for subset,cohort in [('all','all51'),('new','new31')]:
            scored=score_files(bundle,pred,truth,a.evaluator_root,subset);compare(scored['metrics'],expected[cohort][name]);report['checks'].append(dict(experiment='box_refinement',method=name,cohort=cohort,metrics=scored['metrics']))
    for directory in sorted(a.handoff.glob('validation[0-9][0-9]')):
        bundle=read(directory/'bundle.json');truth=read(directory/'ground_truth.json');expected=read(directory/'expected_metrics.json')
        native=score_files(bundle,read(directory/'REN_baseline.json'),truth,a.evaluator_root)
        compare(native['metrics'],expected['methods']['REN_baseline'])
        report['validation_baseline_metrics_verified']=native['metrics']
        for arm in ['expanded_stats','expanded_stats_image','expanded_image_name']:
            out=a.out/directory.name/(arm+'.json')
            subprocess.check_call([sys.executable,'-m','experiments.track_verifier.apply','--bundle',str(directory/'bundle.json'),'--model',str(a.handoff/'models'/(arm+'.json')),'--out',str(out)])
            actual=read(out);frozen=read(directory/(arm+'.json'))
            got={q['key']:q for q in actual['queries']};want={q['key']:q for q in frozen['queries']}
            assert set(got)==set(want)
            assert all(got[k]['score']==want[k]['score'] and got[k]['bboxes']==want[k]['bboxes'] for k in got),'Frozen checkpoint predictions changed'
            score=score_files(bundle,actual,truth,a.evaluator_root);compare(score['metrics'],expected['methods'][arm])
            report['checks'].append(dict(experiment=directory.name,method=arm,all_saved_tracks_exact=True,metrics=score['metrics']))
    mm=a.handoff/'multimodal20';bundle=read(mm/'bundle.json');truth=read(mm/'ground_truth.json');orders=read(mm/'expected_rankings.json')
    names={'image':'CLIP_image','name':'text_name','description':'text_description','image_name':'image_text_name','image_description':'image_text_description','speech_transcript':'synthetic_speech_transcript','image_speech':'image_synthetic_speech'}
    for mode,oldname in names.items():
        predictions=[]
        for q in bundle['queries']:
            order=rank(q,mode);assert order==orders[q['key']]['orders'][oldname]
            predictions.append(track_prediction(q,q['candidates'][order[0]]['id']))
        pred=dict(queries=predictions);save(a.out/'multimodal20'/(mode+'.json'),pred)
        score=score_files(bundle,pred,truth,a.evaluator_root)
        report['checks'].append(dict(experiment='multimodal20',method=mode,all_cached_rankings_exact=True,metrics=score['metrics']))
    for folder,script,clues,saved,options in [
        ('human_pilot7','clarification','human_clues_original.json','hinted.json',['--budget','.25','--real-hints']),
        ('human_clues8','uncertain','clues.json','uncertain.json',['--hints'])]:
        directory=a.handoff/folder
        if not directory.exists():continue
        output=a.out/folder/'predictions.json'
        subprocess.check_call([sys.executable,'-m','experiments.additional_hints.'+script,'--bundle',str(directory/'bundle.json'),*options,str(directory/clues),'--out',str(output)])
        actual={q['key']:q for q in read(output)['queries']};frozen={q['key']:q for q in read(directory/saved)['queries']}
        assert set(actual)==set(frozen)
        assert all(actual[k]['score']==frozen[k]['score'] and actual[k]['bboxes']==frozen[k]['bboxes'] for k in actual)
        scored=score_files(read(directory/'bundle.json'),read(output),read(directory/'ground_truth.json'),a.evaluator_root)
        report['checks'].append(dict(experiment=folder,all_saved_tracks_exact=True,metrics=scored['metrics'],provenance=read(directory/clues).get('provenance')))
    report.update(passed=True,finished=time.time(),pairing={'all':['all','all']},no_video_or_large_model_loaded=True)
    save(a.out/'REPLAY_CHECK.json',report);print('REPLAY_CHECK_PASS',len(report['checks']),'comparisons')
if __name__=='__main__':main()
