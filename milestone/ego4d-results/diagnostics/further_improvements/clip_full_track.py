"""Rerank original DINOv2 box proposals with CLIP inside fixed response tracks."""
import os
import argparse
import json
import math
from pathlib import Path
import subprocess
import sys

import numpy as np
from PIL import Image
from transformers import CLIPModel, CLIPProcessor
import torch
import torch.nn.functional as F

parser = argparse.ArgumentParser()
parser.add_argument('root', type=Path)
parser.add_argument('output', type=Path)
parser.add_argument('--smoke', action='store_true')
args = parser.parse_args()
root = args.root
data = Path(os.environ['VQ2D_BATCH_DIR'])
assets = root / 'dinov2_assets'
sys.path[:0] = [str(root / 'visual-memory'), str(assets / 'dinov2')]
import run_dinov2_pilot as pilot
from dinov2.hub.backbones import dinov2_vits14

W,H,P = pilot.WIDTH,pilot.HEIGHT,pilot.PATCH
clip_id = "openai/clip-vit-base-patch32"
clip_model = CLIPModel.from_pretrained(clip_id, local_files_only=True).eval()
clip_processor = CLIPProcessor.from_pretrained(clip_id, local_files_only=True)

def clip_embed(images):
    inputs = clip_processor(images=images, return_tensors="pt")
    return F.normalize(clip_model.get_image_features(**inputs), dim=-1)
torch.set_num_threads(4)
model=dinov2_vits14(pretrained=False).eval()
model.load_state_dict(torch.load(assets/'dinov2_vits14_pretrain.pth',map_location='cpu',weights_only=True))
baseline=json.loads((data/'baseline.json').read_text())['results']['videos']
gt=json.loads((Path(os.environ['VQ2D_BATCH_DIR'])/'annotations.json').read_text())
if args.smoke:
    gt['videos']=gt['videos'][:1]
    gt['videos'][0]['clips']=gt['videos'][0]['clips'][:1]
    clip=gt['videos'][0]['clips'][0]
    for annotation in clip['annotations']:
        valid={k:q for k,q in annotation['query_sets'].items() if q['is_valid']}
        if valid:
            clip['annotations']=[dict(annotation,query_sets=dict([next(iter(valid.items()))]))]
            break

def boxes_for_shape(h,w):
    return [(x*P,y*P,(x+w)*P,(y+h)*P) for y in range(H//P-h+1) for x in range(W//P-w+1)]

boxes=np.array([b for h,w in pilot.SIZES for b in boxes_for_shape(h,w)])

def query_embedding(path,q,fps):
    crop=q['visual_crop']
    w,h=crop['original_width'],crop['original_height']
    raw=subprocess.check_output([str(assets/'ffmpeg'),'-v','error','-ss',str(crop['frame_number']/fps),'-i',str(path),'-frames:v','1','-vf',f'scale={w}:{h}','-f','rawvideo','-pix_fmt','rgb24','-threads','1','-'])
    image=Image.frombytes('RGB',(w,h),raw)
    l,t=crop['x'],crop['y']
    image=image.crop((math.floor(l),math.floor(t),math.ceil(l+crop['width']),math.ceil(t+crop['height'])))
    patch = image.resize((112,112),Image.Resampling.BICUBIC)
    dino = F.normalize(model.forward_features(pilot.normalize([np.asarray(patch)]))['x_norm_patchtokens'].mean(1),dim=1)[0]
    return dino, clip_embed([image])[0]

def score_clip(path,queries,fps):
    pairs=[query_embedding(path,q,fps) for q in queries]
    embeddings=torch.stack([p[0] for p in pairs])
    clip_queries=torch.stack([p[1] for p in pairs])
    ranges = {(q['forced_window']['start'], q['forced_window']['end']) for q in queries}
    frames_needed = sorted({f for start, end in ranges for f in range(start, end)})
    lookup = {frame: index for index, frame in enumerate(frames_needed)}
    selection = '+'.join(f'between(n\\,{start}\\,{end-1})' for start, end in sorted(ranges))
    frame_bytes=W*H*3
    cmd=[str(assets/'ffmpeg'),'-v','error','-i',str(path),'-an','-vf',f'fps={fps}:start_time=0:round=up,select={selection},scale={W}:{H}','-fps_mode','passthrough','-frames:v',str(len(frames_needed)),'-f','rawvideo','-pix_fmt','rgb24','-threads','1','-']
    scores=[]; indices=[]; images=[]
    with subprocess.Popen(cmd,stdout=subprocess.PIPE) as p:
        while raw:=p.stdout.read(frame_bytes*8):
            frames=np.frombuffer(raw,dtype=np.uint8).reshape(-1,H,W,3)
            z=model.forward_features(pilot.normalize(frames))['x_norm_patchtokens']
            grid=z.reshape(len(frames),H//P,W//P,-1).permute(0,3,1,2)
            sims=[]
            for h,w in pilot.SIZES:
                pool=F.normalize(F.avg_pool2d(grid,(h,w),stride=1),dim=1)
                sims.append(torch.einsum('bdhw,qd->bhwq',pool,embeddings).flatten(1,2))
            score,idx=torch.cat(sims,dim=1).topk(5,dim=1)
            images.extend(Image.fromarray(frame) for frame in frames)
            scores.append(score.numpy())
            indices.append(idx.numpy())
        if p.wait(): raise RuntimeError(f'Decode failed: {path.name}')
    s=np.concatenate(scores)
    ix=np.concatenate(indices)
    if len(s) != len(frames_needed):
        raise ValueError(f'{path.name}: unexpected number of decoded frames')
    predictions=[]
    for j,q in enumerate(queries):
        a,b=q['forced_window']['start'],q['forced_window']['end']
        crop=q['visual_crop']
        ow,oh=crop['original_width'],crop['original_height']
        track=[]
        for start in range(a,b,8):
            frames = list(range(start,min(start+8,b)))
            frame_indices = [lookup[f] for f in frames]
            candidates = boxes[ix[frame_indices,:,j]]
            crops = [images[i].crop(tuple(box)) for i, group in zip(frame_indices,candidates)
                     for box in group]
            clip_scores = (clip_embed(crops) @ clip_queries[j]).numpy().reshape(len(frames),5)
            dino_scores = s[frame_indices,:,j]
            normalize = lambda x: (x-x.mean(1,keepdims=True))/np.maximum(x.std(1,keepdims=True),.01)
            best = (normalize(clip_scores)+normalize(dino_scores)).argmax(1)
            for f,box in zip(frames,candidates[np.arange(len(frames)),best]):
                x1,y1,x2,y2=np.rint(box*np.array([ow/W,oh/H,ow/W,oh/H])).astype(int).tolist()
                track.append(dict(fno=f,x1=x1,y1=y1,x2=x2,y2=y2))
        predictions.append(dict(score=q['forced_window']['score'], bboxes=track))
    return predictions

videos=[]
with torch.inference_mode():
    for v,pv in zip(gt['videos'],baseline,strict=True):
        clips=[]
        for c,pc in zip(v['clips'],pv['clips'],strict=True):
            queries=[]; patches=[]
            for a,pa in zip(c['annotations'],pc['predictions'],strict=True):
                for k,q in a['query_sets'].items():
                    if not q['is_valid']:continue
                    old=pa['query_sets'][k]
                    boxes_old=old['bboxes']
                    if boxes_old:
                        assert [b['fno'] for b in boxes_old]==list(range(boxes_old[0]['fno'],boxes_old[-1]['fno']+1))
                        queries.append(dict(q,forced_window=dict(start=boxes_old[0]['fno'],end=boxes_old[-1]['fno']+1,score=old['score'])))
                        patches.append(old)
            if queries:
                refined=score_clip(data/'v2/clips'/f"{c['clip_uid']}.mp4",queries,c['clip_fps'])
                for old,new in zip(patches,refined,strict=True):old.update(new)
            clips.append(pc)
            print(c['clip_uid'],len(queries),flush=True)
        videos.append(dict(video_uid=v['video_uid'],clips=clips))
args.output.mkdir(parents=True,exist_ok=True)
(args.output/'predictions.json').write_text(json.dumps(dict(version=gt['version'],challenge='ego4d_vq2d_challenge',results=dict(videos=videos))))
if args.smoke:
    (args.output/'gt.json').write_text(json.dumps(gt))
