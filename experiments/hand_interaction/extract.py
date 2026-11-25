"""Add hand-proxy features to new or exported candidate bundles, label-free.

Requires local licensed videos, explicit coordinate dimensions/transform, and
optional saved geometry-reference crops. Does not regenerate object proposals.
"""
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
import argparse,functools,hashlib,io,json,os,re,shutil,subprocess
import numpy as np
from PIL import Image
from experiments.candidate_tools.candidate_common import validate_query
from experiments.hand_interaction.proxies import detect,proxy_features

def digest(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def save(p,d):
    p.parent.mkdir(parents=True,exist_ok=True);tmp=p.with_suffix('.tmp');tmp.write_text(json.dumps(d,indent=2)+'\n');tmp.replace(p)
def samples(track,cutoff):
    frames=sorted({int(b[0]) for b in track})
    if not frames:return []
    anchors={frames[round(j*(len(frames)-1)/5)] for j in range(6)};available=set(frames)
    result=sorted({f+d for f in anchors for d in [-1,0,1] if f+d in available})
    if any(f<0 or f>=cutoff for f in result):raise ValueError('Post-query hand sampling forbidden')
    return result
def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ['bundle','video-root','model','out']:p.add_argument('--'+name,type=Path,required=True)
    p.add_argument('--ffmpeg',required=True,help='Path to an installed FFmpeg binary; no executable is downloaded automatically')
    p.add_argument('--max-queries',type=int)
    a=p.parse_args()
    import mediapipe as mp
    from mediapipe.tasks import python
    from mediapipe.tasks.python import vision
    options=vision.HandLandmarkerOptions(base_options=python.BaseOptions(model_asset_path=str(a.model),delegate=python.BaseOptions.Delegate.CPU),running_mode=vision.RunningMode.IMAGE,num_hands=2,min_hand_detection_confidence=.3,min_hand_presence_confidence=.3)
    bundle=json.loads(a.bundle.read_text());a.out.mkdir(parents=True,exist_ok=True)
    signature=dict(bundle=digest(a.bundle),model=digest(a.model),extractor=digest(__file__),proxies=digest(Path(__file__).with_name('proxies.py')))
    @functools.lru_cache(maxsize=8)
    def raw(path,timecode):
        b=subprocess.check_output([a.ffmpeg,'-loglevel','error','-ss',timecode,'-threads','1','-i',path,'-r','1','-q:v','2','-pix_fmt','rgb24','-frames:v','1','-threads','1','-f','image2pipe','-vcodec','png','-'],timeout=90)
        return Image.open(io.BytesIO(b)).convert('RGB')
    with vision.HandLandmarker.create_from_options(options) as detector:
        for ordinal,q in enumerate(bundle['queries'],1):
            if a.max_queries is not None and ordinal>a.max_queries:break
            validate_query(q);key=q['key']
            if not re.fullmatch(r'[A-Za-z0-9_.-]+',key) or key in ['.','..']:raise ValueError('Unsafe query key')
            marker=a.out/'features'/(key+'.json')
            if marker.exists():
                d=json.loads(marker.read_text())
                if d['signature']!=signature:raise ValueError('Inputs/code changed; use a fresh output directory')
                for c,row in zip(q['candidates'],d['rows']):
                    assert c['id']==row['candidate_id'];c['hand_features']=row['features']
                continue
            if shutil.disk_usage(a.out).free<2*1024**3:raise RuntimeError('Hand extraction stopped below2GiB disk headroom')
            path=Path(q['video_path']);path=path if path.is_absolute() else a.video_root/path
            if not path.is_file():raise FileNotFoundError('Missing licensed video: '+str(path))
            width=q.get('frame_width');height=q.get('frame_height')
            if not width or not height:raise ValueError('Bundle must specify frame_width/frame_height in its box coordinate space')
            geometry=q.get('geometry',{});offset=float(q.get('video_start_sec',0)) if q.get('full_source_video',False) else 0.
            def frame(f):
                if not 0<=f<q['query_frame']:raise ValueError('Future frame forbidden')
                image=raw(str(path),f'{offset+f/q.get("fps",5):.3f}')
                if geometry.get('bounds'):image=image.crop(geometry['bounds'])
                if image.size!=(width,height):image=image.resize((width,height),Image.Resampling.BICUBIC)
                return image
            references=q.get('geometry_references',[])
            for ref in references:
                refpath=(a.bundle.parent/ref['path']).resolve()
                if a.bundle.parent.resolve() not in refpath.parents:raise ValueError('Reference path must remain inside the handoff dataset folder')
                if digest(refpath)!=ref['sha256']:raise ValueError('Geometry reference changed')
                got=np.asarray(frame(ref['frame']).crop(ref['box']));want=np.asarray(Image.open(refpath).convert('RGB'))
                if got.shape!=want.shape or not np.array_equal(got,want):raise ValueError('Frame geometry or raw video pixels differ from original: '+key)
            tracks=[[[f,x,y,x2-x,y2-y] for f,x,y,x2,y2 in c['bboxes']] for c in q['candidates']]
            sample_ids=[samples(track,q['query_frame']) for track in tracks]
            images={};hands={}
            for f in sorted({f for values in sample_ids for f in values}):
                image=frame(f);images[f]=np.asarray(image);hands[f]=detect(detector,image)
            rows=[]
            for c,track,ids in zip(q['candidates'],tracks,sample_ids):
                vector,coverage=proxy_features(track,ids,hands,images);c['hand_features']=vector
                rows.append(dict(candidate_id=c['id'],features=vector,coverage=coverage))
            save(marker,dict(key=key,rows=rows,signature=signature,geometry_reference_verified=bool(references),all_frames_prequery=True,contact_is_not_labeled=True))
            print('HAND_QUERY',ordinal,key,'hand_frames',sum(bool(h['hands']) for h in hands.values()),'/',len(hands),flush=True);raw.cache_clear()
    bundle['hand_feature_provenance']=dict(model_sha256=digest(a.model),delegate='CPU',meaning='2D proximity/camera-corrected co-motion/release proxies, not confirmed contact')
    if a.max_queries is not None:bundle['queries']=bundle['queries'][:a.max_queries]
    save(a.out/'bundle_with_hands.json',bundle);print(a.out/'bundle_with_hands.json')
if __name__=='__main__':main()
