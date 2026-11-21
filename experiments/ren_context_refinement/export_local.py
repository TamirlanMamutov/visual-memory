"""Export completed local REN candidates; data stays outside Git.

Usage: python export_local.py --runs-root PATH_TO_LOCAL_EXPERIMENTS --out OUTPUT
Requires the completed cached_v3 and descriptions_v4 artifacts. Writes a model-
input bundle without labels and a SEPARATE ground-truth file for local scoring.
"""
from pathlib import Path
import argparse
import json
import numpy as np

def read(p):return json.loads(p.read_text())

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--runs-root',type=Path,required=True);p.add_argument('--out',type=Path,required=True);a=p.parse_args()
    b=a.runs_root/'ren_multimodal_cached_v3';d=a.runs_root/'ren_multimodal_descriptions_v4'
    data=read(b/'dataset.json');rows=data['rows'];texts=dict(np.load(d/'text_features.npz'));text=dict(zip(texts['keys'],texts['vectors']))
    samples=dict(np.load(d/'window_followup/sample_embeddings.npz'));folds=read(b/'protocol.json')['outer_assignment']
    descriptions={q['key']:q['description'] for q in read(d/'queries.json')['items']}
    queries=[];truth=[]
    for key,case in data['cases'].items():
        ids=[i for i,r in enumerate(rows) if r['key']==key];z=dict(np.load(b/'embeddings'/(key+'.npz')));assert list(z['row_ids'])==ids
        image=z['candidates']@z['query'];name=z['candidates']@z['text_name'];desc=z['candidates']@text[key]
        item=case['item'];qid=item['selection']['query_set_id'];payload=read(Path(case['baseline']));sets=payload['results']['videos'][0]['clips'][0]['predictions'][0]['query_sets'];candidates=[]
        for j,index in enumerate(ids):
            row=rows[index];which=np.where(samples['rows']==index)[0]
            per_sample=[dict(frame=int(samples['frames'][s]),image_score=float(samples['vectors'][s]@z['query']),name_score=float(samples['vectors'][s]@z['text_name']),description_score=float(samples['vectors'][s]@text[key])) for s in which]
            candidates.append(dict(id=str(index),bboxes=[[f,x,y,x+w,y+h] for f,x,y,w,h in row['track']],image_score=float(image[j]),name_score=float(name[j]),description_score=float(desc[j]),samples=per_sample))
        query=dict(key=key,source=item['video_uid'],object=item['selection']['object_title'],description=descriptions[key],
                   query_frame=item['selection']['query_frame'],fps=5,score=sets[qid]['score'],old=case['old'],source_fold=folds[item['video_uid']],
                   baseline_id=str(next(i for i in ids if rows[i]['is_baseline'])),candidates=candidates)
        queries.append(query)
        truth.append(dict(key=key,source=query['source'],query_frame=query['query_frame'],bboxes=[[b['fno'],b['x1'],b['y1'],b['x2'],b['y2']] for b in sets[qid+'-gt']['bboxes']]))
    a.out.mkdir(parents=True,exist_ok=True)
    for name,value in [('bundle.json',dict(version=1,coordinate_space='processed REN frames, shared by all candidates and exported targets',cohort='official-training development; released REN may have seen these examples',queries=queries)),('ground_truth.json',dict(version=1,evaluation_only=True,queries=truth))]:
        path=a.out/name
        if path.exists() and read(path)!=value:raise ValueError('Refusing to replace different existing export: '+str(path))
        path.write_text(json.dumps(value,indent=2,allow_nan=False)+'\n')
    print(f'Exported {len(queries)} queries/{len(rows)} candidates to {a.out}. No images, videos, weights or credentials copied.')

if __name__=='__main__':main()
