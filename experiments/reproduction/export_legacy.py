"""Owner-side export of the existing REN studies into a private portable handoff.

No local paths are defaults. The generated data/targets/crops belong OUTSIDE Git.
Teammates use the resulting archive; this exporter is not required on their PC.
"""
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
import argparse,copy,hashlib,json,subprocess,time,zipfile
import numpy as np

def read(p):return json.loads(Path(p).read_text())
def save(p,d):
    p=Path(p);p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(d,indent=2,allow_nan=False)+'\n')
def digest(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda:f.read(1024**2),b''):h.update(b)
    return h.hexdigest()
def xyxy(track):return [[f,x,y,x+w,y+h] for f,x,y,w,h in track]
def export_development(w,r,out):
    data=read(w/'ren_development_v10/dataset.json');data['training_only']=True;data['feature_family']='ren_quality_v1'
    protocol=read(w/'ren_hand_interaction_v1/protocol.json');data['excluded_sources']=protocol['excluded_reserved_sources']
    hand={}
    for p in (w/'ren_hand_interaction_v1/features').glob('*.json'):
        for row in read(p)['rows']:hand[row['global_row']]=row['features']
    assert len(hand)==len(data['rows'])
    for i,row in enumerate(data['rows']):row['hand_features']=hand[i]
    for q in data['bundle']['queries']:
        for c in q['candidates']:
            row=data['rows'][int(c['id'])];c['quality_features']=row['features'];c['hand_features']=row['hand_features'];c['shuffled_name_score']=row['shuffled_name_cosine']
        info=data['cases'][q['key']];q.update(frame_width=info['width'],frame_height=info['height'])
    data['bundle']['feature_family']='ren_quality_v1';data['bundle']['coordinate_space']='processed REN pixels; targets share these coordinates'
    save(out/'development60/dataset.json',data);save(out/'development60/bundle.json',data['bundle']);save(out/'development60/ground_truth.json',data['truth'])
    save(out/'development60/hand_recipes.json',dict(head_configs=protocol['head_configs']))
    for version in ['v10','v11']:
        p=read(w/f'ren_development_{version}/protocol.json')
        configs=[]
        for c in p['head_configs']:configs.append(dict(c,matched_init=version=='v11',source_balance=c.get('source_balance',False),selector=c.get('selector','mean_IoU')))
        save(out/f'development60/{version}_recipes.json',dict(head_configs=configs))
    expected={}
    for group,root in [('hand',w/'ren_hand_interaction_v1'),('v10',w/'ren_development_v10'),('v11',w/'ren_development_v11')]:
        expected[group]={}
        for marker in (root/'heads').glob('*/COMPLETE.json'):
            d=read(marker);choices=dict(np.load(marker.parent/'choices.npz'))
            expected[group][marker.parent.name]=dict(metrics=d['metrics'],keys=choices['keys'].tolist(),choices=choices['choices'].tolist())
    save(out/'development60/expected.json',expected)
    plan=read(w/'ren_hand_interaction_v1/frame_plan.json');videos={}
    for q in data['bundle']['queries']:
        info=plan['cases'][q['key']];geometry=read(w/'ren_hand_interaction_v1/features'/(q['key']+'.json'))
        name=('full_scale/'+info['source']+'.mp4') if info['full_source_video'] else ('clips/'+info['clip_uid']+'.mp4')
        q.update(video_path=name,clip_uid=info['clip_uid'],video_start_sec=info['video_start_sec'],full_source_video=info['full_source_video'],geometry=dict(bounds=geometry['bounds'],output_size=[info['width'],info['height']],pixel_exact_owner_check=True))
        refs=[]
        for j,ref in enumerate(info['references']):
            dest=out/'development60/geometry_refs'/q['key']/f'{j}.png';dest.parent.mkdir(parents=True,exist_ok=True);dest.write_bytes(Path(ref['path']).read_bytes())
            refs.append(dict(frame=ref['frame'],box=ref['box'],path=str(dest.relative_to(out/'development60')),sha256=ref['sha256']))
        q['geometry_references']=refs
        videos[name]=dict(relative_path=name,video_uid=info['source'],clip_uid=info['clip_uid'],full_source_video=info['full_source_video'],video_start_sec=info['video_start_sec'],video_end_sec=info['video_end_sec'],owner_file_bytes=Path(info['video_path']).stat().st_size,split='train')
    save(out/'development60/hand_input_bundle.json',data['bundle']);save(out/'development60/videos_needed.json',dict(files=list(videos.values()),raw_video_included=False,license='Each recipient needs their own Ego4D access'))

def export_models(r,out):
    import torch
    from experiments.track_verifier.core import DEFAULT
    torch.set_num_threads(1)
    for arm in ['expanded_stats','expanded_stats_image','expanded_image_name']:
        frozen=read(r/'final_study/evaluation_v1/models'/arm/'frozen_model.json');seeds=[];hashes=[]
        for p,h in frozen['checkpoints'].items():
            assert digest(p)==h;d=torch.load(p,map_location='cpu',weights_only=False)
            seeds.append(dict(state_dict={k:v.detach().cpu().tolist() for k,v in d['state_dict'].items()},mean=d['mean'].tolist(),scale=d['scale'].tolist()));hashes.append(h)
        cfg=dict(DEFAULT,features={'expanded_stats':'stats','expanded_stats_image':'image','expanded_image_name':'name'}[arm],lr=.001,source_balance=False,selector='mean_IoU')
        model=dict(version=1,arm=arm,configuration=cfg,threshold=frozen['threshold'],feature_family='ren_quality_v1',seeds=seeds,original_checkpoint_sha256=hashes,training_queries=60,training_sources=40,role='Frozen before reserved validation; no retraining in apply.py')
        save(out/'models'/(arm+'.json'),model)

def export_validation(r,out):
    e=r/'final_study/evaluation_v1'
    files=sorted(e.glob('final_batch_*_metrics.json'),key=lambda p:p.stat().st_mtime)
    if not files:return
    metrics=read(files[-1]);count=metrics['queries'];planned=read(e/'manifest.json')['items'][:count]
    queries=[];targets=[];predictions={};arms=['REN_baseline','expanded_image_name','expanded_stats','expanded_stats_image']
    for item in planned:
        k=f"{item['clip']['clip_uid']}_a{item['selection']['annotation_index']}_q{item['selection']['query_set_id']}";d=read(e/'decisions'/k/'decision.json')
        emb=dict(np.load(e/'clip_embeddings'/(k+'.npz')));image=emb['candidates']@emb['query'];name=emb['candidates']@emb['text_name']
        candidates=[]
        for i,row in enumerate(d['rows']):
            assert len(row['features'])==12
            c=dict(id=str(i),bboxes=xyxy(row['track']),quality_features=row['features'],image_score=float(image[i]),name_score=float(name[i]))
            if row.get('empty_native_fallback'):c['empty_native_fallback']=True
            candidates.append(c)
        base=next(i for i,row in enumerate(d['rows']) if row['is_baseline'])
        queries.append(dict(key=k,source=item['video_uid'],query_frame=item['selection']['query_frame'],fps=5,score=d['original_AP_confidence'],baseline_id=str(base),candidates=candidates))
        for arm in arms:
            payload=read(e/'predictions'/k/arm/'prediction.json');qid=item['selection']['query_set_id'];sets=payload['results']['videos'][0]['clips'][0]['predictions'][0]['query_sets']
            convert=lambda bs:[[b['fno'],b['x1'],b['y1'],b['x2'],b['y2']] for b in bs]
            predictions.setdefault(arm,[]).append(dict(key=k,bboxes=convert(sets[qid]['bboxes']),score=sets[qid]['score']))
            if arm=='REN_baseline':targets.append(dict(key=k,bboxes=convert(sets[qid+'-gt']['bboxes'])))
    dest=out/f'validation{count:02d}';save(dest/'bundle.json',dict(version=1,feature_family='ren_quality_v1',cohort=f'Frozen partial official-validation cohort: {count}/50 queries; not the full benchmark',queries=queries))
    save(dest/'ground_truth.json',dict(evaluation_only=True,queries=targets))
    for arm,pred in predictions.items():save(dest/(arm+'.json'),dict(method=arm,queries=pred))
    save(dest/'expected_metrics.json',dict(queries=count,sources=metrics['source_videos'],methods=metrics['methods'],evaluation_pairing={'all':['all','all']},final_analysis_complete=(e/'COMPLETE.json').exists()))
    analysis=e/'readouts'/f'batch_{count:02d}_analysis.json'
    if analysis.exists():save(dest/'expected_analysis.json',read(analysis))

def export_multimodal20(r,out):
    m=r/'multimodal_ren_pilot_v1';rank=read(m/'evaluation/with_synthetic_speech/prediction_freeze.json')['rankings'];raw=read(r/'pairwise_verifier_v1/training/plus_two/dataset.json')['rows'];queries=[];targets=[]
    for k,d in rank.items():
        ids=d['row_ids'];base=d['orders']['REN_baseline'][0]
        path=m/'evaluation/with_synthetic_speech/predictions'/k/'REN_baseline/prediction.json';payload=read(path);qid=k.rsplit('_q',1)[1];video=payload['results']['videos'][0];clip=video['clips'][0];sets=clip['predictions'][0]['query_sets']
        source=raw[ids[0]]['source'];qframe=next(i['cutoff'] for i in read(m/'crop_manifest.json')['items'] if i['key']==k)
        candidates=[]
        for j,i in enumerate(ids):
            c=dict(id=str(j),bboxes=xyxy(raw[i]['track']),image_score=float(d['scores']['CLIP_image'][j]),name_score=float(d['scores']['text_name'][j]),description_score=float(d['scores']['text_description'][j]))
            c['modality_scores']={mode:float(values[j]) for mode,values in d['scores'].items()};candidates.append(c)
        queries.append(dict(key=k,source=source,query_frame=qframe,fps=5,score=sets[qid]['score'],baseline_id=str(base),candidates=candidates))
        targets.append(dict(key=k,bboxes=[[b['fno'],b['x1'],b['y1'],b['x2'],b['y2']] for b in sets[qid+'-gt']['bboxes']]))
    save(out/'multimodal20/bundle.json',dict(version=1,cohort='20 official-training queries; synthetic speech and assistant-authored appearance descriptions; image-conditioned candidate reranking',queries=queries))
    save(out/'multimodal20/ground_truth.json',dict(evaluation_only=True,queries=targets));save(out/'multimodal20/expected_rankings.json',rank)
    # Original20-query quality-head study: preserve its original row order and
    # source folds, which differ from the later60-query development assignment.
    mapping={};flat=0;training_queries=copy.deepcopy(queries)
    for q in training_queries:
        ids=rank[q['key']]['row_ids']
        for c,index in zip(q['candidates'],ids):c['quality_features']=raw[index]['features'];mapping[str(index)]=flat;flat+=1
    config=dict(name='quality_only_original20',features='stats',lr=.001,weight_decay=.01,epochs=200,width=8,dropout=0.,normalization='std',clip_z=None,rank_weight=0.,risk_sd=0.,minimum_overlap=0.,source_balance=False,selector='mean_IoU',matched_init=True)
    d=dict(version=1,training_only=True,feature_family='ren_quality_v1',rows=raw,cases={q['key']:dict(source=q['source'],cutoff=q['query_frame']) for q in training_queries},
        old_keys=[],bundle=dict(version=1,feature_family='ren_quality_v1',annotation_split='train',queries=training_queries),truth=dict(queries=targets),row_to_candidate=mapping,folds=read(r/'quality_verifier_v1/protocol.json')['outer_assignment'])
    save(out/'development20/dataset.json',d);save(out/'development20/recipes.json',dict(head_configs=[config]))
    original=read(r/'quality_verifier_v1/training/quality_only/source_nested_CV_metrics.json')
    save(out/'development20/expected_metrics.json',original['methods']['quality_only'])

def export_human_clues(w,out):
    first=w/'ren_active_hint_v8/human_evaluation/20261120_210811_563688'
    if first.exists():
        save(out/'human_pilot7/bundle.json',read(w/'ren_active_hint_v8/human_bundle_new31.json'))
        save(out/'human_pilot7/ground_truth.json',read(w/'ren_active_hint_v8/human_evaluation_targets_new31.json'))
        for name in ['human_clues_original.json','native.json','hinted.json','comparison.json','input_provenance.json']:
            save(out/'human_pilot7'/name,read(first/name))
    source=w/'ren_uncertain_hint_v9/human_evaluation/20261120_212135_279573'
    if not source.exists():return
    clues=read(source/'original_clues.json');keys={q['key'] for q in clues['queries']}
    bundle=read(out/'box_hints51/bundle.json');truth=read(out/'box_hints51/ground_truth.json')
    assert keys<={q['key'] for q in bundle['queries']}
    bundle['queries']=[q for q in bundle['queries'] if q['key'] in keys];truth['queries']=[q for q in truth['queries'] if q['key'] in keys]
    save(out/'human_clues8/bundle.json',bundle);save(out/'human_clues8/ground_truth.json',truth);save(out/'human_clues8/clues.json',clues)
    for name in ['native.json','uncertain.json','legacy_gate.json','comparison.json']:
        save(out/'human_clues8'/name,read(source/name))
    save(out/'human_clues8/PROVENANCE.json',dict(classification=clues.get('provenance'),
        scope='Eight all-declined synthetic interface records; NO human-clue performance evidence',
        warning='The new confidence-annotated human study remains pending. The separate human_pilot7 folder contains the actual first review.'))

def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ['workspace','ren-root','out']:p.add_argument('--'+name,type=Path,required=True)
    a=p.parse_args();root=Path(__file__).resolve().parents[2]
    if root in a.out.resolve().parents:raise ValueError('Private handoff must be outside the Git checkout')
    if a.out.exists() and any(a.out.iterdir()):raise ValueError('Use a new empty handoff directory')
    a.out.mkdir(parents=True,exist_ok=True)
    export_development(a.workspace,a.ren_root,a.out);export_models(a.ren_root,a.out);export_validation(a.ren_root,a.out);export_multimodal20(a.ren_root,a.out)
    subprocess.check_call([sys.executable,str(root/'experiments/candidate_tools/export_local.py'),'--runs-root',str(a.workspace),'--out',str(a.out/'box_hints51')])
    export_human_clues(a.workspace,a.out)
    manifest=dict(created=time.time(),license_notice='PRIVATE: derived Ego4D annotations/crops/predictions. Share only with independently licensed teammates through authorized private storage. Do not commit this archive.',files={str(p.relative_to(a.out)):digest(p) for p in sorted(a.out.rglob('*')) if p.is_file()},raw_video_included=False,large_pretrained_weights_included=False,credentials_included=False)
    save(a.out/'MANIFEST.json',manifest)
    archive=a.out.with_suffix('.zip')
    with zipfile.ZipFile(archive,'w',zipfile.ZIP_DEFLATED,compresslevel=1) as z:
        for f in sorted(a.out.rglob('*')):
            if f.is_file():z.write(f,str(f.relative_to(a.out)))
    print('PRIVATE_HANDOFF',archive,'bytes',archive.stat().st_size,'sha256',digest(archive),flush=True)
if __name__=='__main__':main()
