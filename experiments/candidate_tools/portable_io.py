"""Bridge a label-free candidate bundle to the internal simulation cache.

The separate truth file is read ONLY for the explicitly oracle-derived simulation.
Real hint inference and box refinement never import this module.
"""
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
import hashlib
import json
import random
import numpy as np
from experiments.candidate_tools.candidate_common import validate_query
from experiments.candidate_tools.evaluate import official_api

def digest(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def prepare_cache(bundle_path,truth_path,evaluator_root,cache):
    cache=Path(cache);marker=cache/'IMPORT.json'
    signature=dict(bundle=digest(bundle_path),truth=digest(truth_path),bridge=digest(__file__))
    if marker.exists():
        d=json.loads(marker.read_text())
        if d['inputs']!=signature:raise ValueError('Simulation inputs changed; use a new output directory')
        return cache/'data',cache/'text'
    bundle=json.loads(Path(bundle_path).read_text());truth=json.loads(Path(truth_path).read_text());gt={g['key']:g for g in truth['queries']}
    queries=bundle['queries'];keys=[q['key'] for q in queries]
    if len(keys)!=len(set(keys)) or set(keys)!=set(gt):raise ValueError('Bundle/GT keys must match exactly')
    if any('/' in k or '\\' in k or k in ['.','..'] for k in keys):raise ValueError('Query IDs cannot contain path separators')
    if not any(not q.get('old',False) for q in queries):raise ValueError('Need at least one query not tagged old for the primary cohort')
    for q in queries:
        validate_query(q)
        if q.get('fps',5)!=5:raise ValueError('The simulation currently requires 5-FPS annotation frame numbers')
    _,BBox,ResponseTrack=official_api(evaluator_root)
    # Import the matching official geometry helper, not a local substitute.
    import importlib
    prefix='vq2d.metrics' if (Path(evaluator_root)/'vq2d/metrics').exists() else 'metrics'
    iou=importlib.import_module(prefix+'.utils').spatio_temporal_iou_response_track
    sources=sorted({q['source'] for q in queries})
    if len(sources)<2:raise ValueError('At least two source videos are needed for a source-disjoint prior')
    if all('source_fold' in q for q in queries):
        for s in sources:
            if len({q['source_fold'] for q in queries if q['source']==s})!=1:raise ValueError('Queries of one source cannot cross folds')
        folds={q['source']:q['source_fold'] for q in queries}
    else:
        shuffled=list(sources);random.Random(230).shuffle(shuffled);folds={s:i%min(5,len(sources)) for i,s in enumerate(shuffled)}
    if len(set(folds.values()))<2:raise ValueError('Need at least two different source folds')
    b=cache/'data';d=cache/'text';(b/'embeddings').mkdir(parents=True,exist_ok=True);d.mkdir(parents=True,exist_ok=True)
    rows=[];cases={};titles=[]
    for q in queries:
        key=q['key'];truth_boxes=gt[key]['bboxes'];target=ResponseTrack([BBox(*x) for x in truth_boxes])
        baseline=next(c for c in q['candidates'] if c['id']==q['baseline_id'])
        to_dict=lambda boxes:[dict(fno=f,x1=x1,y1=y1,x2=x2,y2=y2) for f,x1,y1,x2,y2 in boxes]
        payload=dict(results=dict(videos=[dict(clips=[dict(predictions=[dict(query_sets={'0':dict(score=q['score'],bboxes=to_dict(baseline['bboxes'])),'0-gt':dict(score=None,bboxes=to_dict(truth_boxes))})])])]))
        path=b/'baseline'/f'{len(cases):06d}.json';path.parent.mkdir(exist_ok=True);path.write_text(json.dumps(payload)+'\n')
        cases[key]=dict(old=q.get('old',False),baseline=str(path.resolve()),item=dict(video_uid=q['source'],selection=dict(query_set_id='0',query_frame=q['query_frame'],object_title=q.get('object',''))))
        vectors=[];ids=[]
        for c in q['candidates']:
            index=len(rows);ids.append(index);track=ResponseTrack([BBox(*x) for x in c['bboxes']],score=q['score'])
            rows.append(dict(key=key,source=q['source'],is_baseline=c['id']==q['baseline_id'],object=q.get('object',''),
                             track=[[f,x1,y1,x2-x1,y2-y1] for f,x1,y1,x2,y2 in c['bboxes']],track_IoU=float(iou(target,track))))
            vectors.append([c['image_score'],c.get('description_score',c['image_score'])])
        # This cache stores scalar scores in two coordinates. No vision features
        # are trained or approximated; the dot products just select cached scores.
        np.savez_compressed(b/'embeddings'/(key+'.npz'),candidates=np.asarray(vectors),query=np.asarray([1.,0.]),row_ids=np.asarray(ids))
        titles.append(dict(key=key,description=q.get('description','')))
    (b/'dataset.json').write_text(json.dumps(dict(rows=rows,cases=cases))+'\n')
    (b/'protocol.json').write_text(json.dumps(dict(outer_assignment=folds))+'\n')
    (d/'queries.json').write_text(json.dumps(dict(items=titles))+'\n')
    np.savez_compressed(d/'text_features.npz',keys=np.asarray(keys),vectors=np.tile([0.,1.],(len(keys),1)))
    marker.write_text(json.dumps(dict(inputs=signature,classification='ORACLE-DERIVED SIMULATION CACHE',queries=len(queries)),indent=2)+'\n')
    (cache/'DESCRIPTION_STATUS.json').write_text(json.dumps(dict(available=all('description_score' in c for q in queries for c in q['candidates']),note='If absent, the description arm duplicates image scores; it is not evidence of text fusion.'))+'\n')
    return b,d
