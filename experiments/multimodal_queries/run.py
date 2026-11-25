"""Replay cached query-modality rankings, or rank a newly exported candidate bank."""
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
import argparse,json
import numpy as np
from experiments.candidate_tools.candidate_common import validate_query,track_prediction
MODES=['baseline','image','name','description','image_name','image_description','speech_transcript','image_speech']
def standardize(x):return (x-x.mean())/max(float(x.std()),.01)
def rank(q,mode):
    validate_query(q);base=next(i for i,c in enumerate(q['candidates']) if c['id']==q['baseline_id'])
    if mode=='baseline':return [base]+[i for i in range(len(q['candidates'])) if i!=base]
    cached={'image':'CLIP_image','name':'text_name','description':'text_description','image_name':'image_text_name',
        'image_description':'image_text_description','speech_transcript':'synthetic_speech_transcript','image_speech':'image_synthetic_speech'}[mode]
    if all(cached in c.get('modality_scores',{}) for c in q['candidates']):scores=np.asarray([c['modality_scores'][cached] for c in q['candidates']])
    else:
        def values(field):return np.asarray([c[field] for c in q['candidates']],dtype=float)
        image=values('image_score')
        if mode=='image':scores=image
        elif mode=='name':scores=values('name_score')
        elif mode=='description':scores=values('description_score')
        elif mode=='speech_transcript':scores=values('speech_score')
        else:
            field={'image_name':'name_score','image_description':'description_score','image_speech':'speech_score'}[mode]
            scores=.5*(standardize(image)+standardize(values(field)))
        valid=np.asarray([c.get('appearance_valid',bool(c['bboxes'])) for c in q['candidates']])
        if not valid.any():return [base]+[i for i in range(len(scores)) if i!=base]
        scores=np.where(valid,scores,-1e6)
    return sorted(range(len(scores)),key=lambda i:(-float(scores[i]),i!=base,i))
def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--bundle',type=Path,required=True);p.add_argument('--mode',choices=MODES,required=True);p.add_argument('--out',type=Path,required=True);a=p.parse_args()
    b=json.loads(a.bundle.read_text());pred=[]
    for q in b['queries']:
        order=rank(q,a.mode);row=track_prediction(q,q['candidates'][order[0]]['id']);row['ranked_candidate_ids']=[q['candidates'][i]['id'] for i in order];pred.append(row)
    d=dict(method='cached_modality_'+a.mode,queries=pred,AP_confidence='unchanged baseline confidence',cohort=b.get('cohort'),no_targets_read=True,limitations='Fixed image-generated candidate pool. Transcript ranking is not end-to-end acoustic understanding.')
    a.out.parent.mkdir(parents=True,exist_ok=True);a.out.write_text(json.dumps(d,indent=2)+'\n');print(a.out)
if __name__=='__main__':main()
