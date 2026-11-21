"""Export EAGLE eval_parts/SAM2 alternatives into the common candidate bundle.

Only the optional feature-export step loads CLIP or decodes existing local video.
No EAGLE training/inference or SAM propagation is invoked. Uses the teammate's
existing FullResReader and variants helpers without modifying their files.
"""
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
import argparse
import json
import random
import sys
import numpy as np

def annotations(path):
    records=[]
    for video in json.loads(Path(path).read_text())['videos']:
        for clip in video['clips']:
            uid=clip.get('clip_uid')
            if not uid:continue
            for ai,ann in enumerate(clip['annotations']):
                for qid,q in ann['query_sets'].items():
                    if not q.get('is_valid',True) or not q.get('response_track') or 'visual_crop' not in q:continue
                    records.append(dict(key=f'{uid}_{ai}_{qid}',source=video['video_uid'],clip=uid,query=q))
    return records

def choose_records(records,maximum,seed):
    # Round-robin source videos before extra queries; no metric-based selection.
    grouped={}
    for r in sorted(records,key=lambda r:r['key']):grouped.setdefault(r['source'],[]).append(r)
    sources=sorted(grouped);random.Random(seed).shuffle(sources);result=[]
    for index in range(max(len(v) for v in grouped.values())):
        for source in sources:
            if index<len(grouped[source]):result.append(grouped[source][index])
    return result[:maximum] if maximum else result

def export_pool(record,parts,sam=None,variants=None):
    key=record['key'];base=parts[0][key];pool=[];seen=set()
    entries=[('baseline',base['bboxes'])]+[(f'run{i}',p[key]['bboxes']) for i,p in enumerate(parts[1:],1)]
    if sam is not None:
        entries += [('sam2_'+name,boxes) for name,boxes in variants(base,sam.get(key,{})).items()]
    for name,boxes in entries:
        identity=tuple(tuple(b) for b in boxes)
        if identity in seen:continue
        seen.add(identity);pool.append(dict(id=name,bboxes=boxes))
    return base,pool

def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--runs',type=Path,nargs='+',required=True,help='Baseline run FIRST, then constituent/alternative run folders')
    ap.add_argument('--annotations',type=Path,required=True)
    ap.add_argument('--clips',type=Path,required=True)
    ap.add_argument('--out',type=Path,required=True)
    ap.add_argument('--sam2-dir',type=Path,help='Optional raw SAM2 per-clip output directory from eagle_sam2.py')
    ap.add_argument('--clip-model',help='Existing local CLIP model directory or explicit HF ID; ID may download weights')
    ap.add_argument('--local-files-only',action='store_true')
    ap.add_argument('--device',default='cpu')
    ap.add_argument('--max-queries',type=int,default=20)
    ap.add_argument('--seed',type=int,default=230)
    ap.add_argument('--inventory-only',action='store_true',help='Check coverage without decoding or loading models')
    a=ap.parse_args();records=annotations(a.annotations)
    available=[r for r in records if all((run/'eval_parts'/(r['clip']+'.json')).exists() for run in a.runs)]
    records=choose_records(available,a.max_queries,a.seed)
    if not records:raise ValueError('No annotated queries have saved files in every supplied run')
    print(f'{len(records)} queries from {len({r["source"] for r in records})} sources selected before scoring; {len(available)} eligible queries total.')
    if a.inventory_only:return
    if not a.clip_model:ap.error('Feature export requires --clip-model; use --inventory-only for a no-model check')
    root=Path(__file__).resolve().parents[2];sys.path.insert(0,str(root))
    from eagle import data as eagle_data
    from eagle_sam2 import FullResReader, variants
    eagle_data.CLIPS=a.clips.resolve()
    import torch
    from transformers import CLIPModel,CLIPProcessor
    torch.set_num_threads(1)
    model=CLIPModel.from_pretrained(a.clip_model,local_files_only=a.local_files_only).to(a.device).eval()
    processor=CLIPProcessor.from_pretrained(a.clip_model,local_files_only=a.local_files_only,use_fast=False)
    def encode(images):
        with torch.inference_mode():
            x=processor(images=images,input_data_format='channels_last',return_tensors='pt').to(a.device)
            return torch.nn.functional.normalize(model.get_image_features(**x),dim=-1).cpu().numpy()
    def text_features(text):
        with torch.inference_mode():
            x=processor(text=[text],return_tensors='pt',truncation=True).to(a.device)
            return torch.nn.functional.normalize(model.get_text_features(**x),dim=-1).cpu().numpy()[0]
    from PIL import Image
    queries=[];truth=[];byclip={}
    for r in records:byclip.setdefault(r['clip'],[]).append(r)
    for uid,selected in byclip.items():
        parts=[json.loads((run/'eval_parts'/(uid+'.json')).read_text()) for run in a.runs]
        sam=json.loads((a.sam2_dir/(uid+'.json')).read_text()) if a.sam2_dir else None
        for r in selected:
            if not all(r['key'] in p for p in parts):raise ValueError('Selected query missing from a run: '+r['key'])
            base,pool=export_pool(r,parts,sam,variants);q=r['query'];vc=q['visual_crop'];wanted={int(vc['frame_number'])}
            for c in pool:
                indices=sorted(set(np.linspace(0,len(c['bboxes'])-1,min(3,len(c['bboxes'])),dtype=int))) if c['bboxes'] else []
                c['sample_boxes']=[c['bboxes'][i] for i in indices];wanted.update(b[0] for b in c['sample_boxes'])
            reader=FullResReader(uid,(vc['original_width'],vc['original_height']))
            try:
                frames=dict(reader.frames(wanted))
                if set(frames)!=wanted:raise ValueError('Requested frame is undecodable; no silent query skipping: '+r['key'])
                scale=reader.frame_size[0]/vc['original_width']*reader.sam_scale
                def crop(frame,box):
                    image=Image.fromarray(frames[frame]);w,h=image.size;x1,y1,x2,y2=[int(round(v*scale)) for v in box]
                    x1=max(0,min(w-1,x1));y1=max(0,min(h-1,y1));x2=max(x1+1,min(w,x2));y2=max(y1+1,min(h,y2))
                    return image.crop((x1,y1,x2,y2)).convert('RGB')
                query=encode([crop(int(vc['frame_number']),[vc['x'],vc['y'],vc['x']+vc['width'],vc['y']+vc['height']])])[0]
                title=q.get('object_title','object');name=text_features('a photo of '+title)
                for c in pool:
                    boxes=c.pop('sample_boxes');vectors=encode([crop(b[0],b[1:]) for b in boxes]) if boxes else np.empty((0,len(query)))
                    mean=vectors.mean(0) if len(vectors) else np.zeros_like(query)
                    c.update(image_score=float(mean@query),name_score=float(mean@name),
                             samples=[dict(frame=int(b[0]),image_score=float(v@query),name_score=float(v@name)) for b,v in zip(boxes,vectors)])
                queries.append(dict(key=r['key'],source=r['source'],object=title,query_frame=int(q['query_frame']),fps=5,score=float(base['score']),baseline_id='baseline',candidates=pool,old=False))
                truth.append(dict(key=r['key'],source=r['source'],query_frame=int(q['query_frame']),bboxes=[[int(b['frame_number']),b['x'],b['y'],b['x']+b['width'],b['y']+b['height']] for b in q['response_track']]))
            finally:reader.close()
            print('EXPORTED',len(queries),'/',len(records),flush=True)
    a.out.mkdir(parents=True,exist_ok=True)
    bundle=dict(version=1,cohort='EAGLE subset; record annotation split and selection in any reported result',coordinate_space='original annotation image pixels',appearance_model=a.clip_model,appearance_revision=getattr(model.config,'_commit_hash',None),queries=queries)
    for filename,data in [('bundle.json',bundle),('ground_truth.json',dict(evaluation_only=True,queries=truth))]:
        path=a.out/filename
        if path.exists() and json.loads(path.read_text())!=data:raise ValueError('Refusing to replace different export: '+str(path))
        path.write_text(json.dumps(data,indent=2,allow_nan=False)+'\n')
    print(a.out)

if __name__=='__main__':main()
