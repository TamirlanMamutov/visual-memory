"""Unchanged numeric hand proxies from the completed 60-query development study.
Source SHA256: fda7e6bd910c949537c7c822919d930cc768cd383080209c200f70eca1f50f14
These are interaction proxies, not labeled physical contact.
"""
import math
import numpy as np
import cv2
cv2.setNumThreads(1)

def box_distance(a,b):
    # Rectangles are xyxy. A zero gap is overlap, not proof of physical contact.
    dx=max(a[0]-b[2],b[0]-a[2],0.);dy=max(a[1]-b[3],b[1]-a[3],0.)
    return math.hypot(dx,dy)

def overlap_fraction(a,b):
    intersection=max(0.,min(a[2],b[2])-max(a[0],b[0]))*max(0.,min(a[3],b[3])-max(a[1],b[1]))
    return intersection/max(1.,min((a[2]-a[0])*(a[3]-a[1]),(b[2]-b[0])*(b[3]-b[1])))

def camera_affine(rgb0,rgb1,exclusions):
    g0=cv2.cvtColor(rgb0,cv2.COLOR_RGB2GRAY);g1=cv2.cvtColor(rgb1,cv2.COLOR_RGB2GRAY)
    mask=np.full(g0.shape,255,dtype=np.uint8)
    h,w=mask.shape
    for box in exclusions:
        x1,y1,x2,y2=[int(round(v)) for v in box]
        mask[max(0,y1-5):min(h,y2+5),max(0,x1-5):min(w,x2+5)]=0
    p=cv2.goodFeaturesToTrack(g0,maxCorners=120,qualityLevel=.01,minDistance=7,mask=mask)
    if p is None or len(p)<8:return None
    p1,status,_=cv2.calcOpticalFlowPyrLK(g0,g1,p,None,winSize=(21,21),maxLevel=2)
    if p1 is None or status is None:return None
    valid=status.ravel().astype(bool)
    a=p[valid].reshape(-1,2);b=p1[valid].reshape(-1,2)
    valid=np.isfinite(a).all(1)&np.isfinite(b).all(1);a=a[valid];b=b[valid]
    if len(a)<8:return None
    matrix,inliers=cv2.estimateAffinePartial2D(a,b,method=cv2.RANSAC,ransacReprojThreshold=3,maxIters=500,confidence=.99)
    if matrix is None or inliers is None or inliers.mean()<.5:return None
    if not np.isfinite(matrix).all() or abs(np.linalg.det(matrix[:,:2]))<.5 or abs(np.linalg.det(matrix[:,:2]))>2:return None
    return matrix

def proxy_features(track,frames,detections,images):
    boxes={int(f):[float(x),float(y),float(x+bw),float(y+bh)] for f,x,y,bw,bh in track}
    selected={};near=[];prox=[];over=[]
    for f in frames:
        obj=boxes[f];scale=max(1.,math.hypot(obj[2]-obj[0],obj[3]-obj[1]))
        hands=detections[f]['hands']
        hand=min(hands,key=lambda h:box_distance(obj,h['box'])) if hands else None
        if hand is not None:
            gap=box_distance(obj,hand['box'])/scale
            selected[f]=hand;near.append(float(gap<.5));prox.append(math.exp(-gap));over.append(overlap_fraction(obj,hand['box']))
        else:near.append(0.);prox.append(0.);over.append(0.)
    pairs=[(f,f+1) for f in frames if f+1 in frames and f+1 in boxes and f+1 in detections and f+1 in images]
    motion=[];release=[];stationary=[];observed=0;fits=0
    for f,g in pairs:
        if f not in selected or g not in selected:continue
        h0=selected[f]
        same=[h for h in detections[g]['hands'] if h['side']==h0['side']]
        if not same:continue
        h1=min(same,key=lambda h:np.linalg.norm(np.asarray(h['center'])-h0['center']))
        o0=boxes[f];o1=boxes[g];scale=max(1.,math.hypot(o0[2]-o0[0],o0[3]-o0[1]))
        if np.linalg.norm(np.asarray(h1['center'])-h0['center'])>3*scale:continue
        observed+=1
        affine=camera_affine(images[f],images[g],[o0]+[h['box'] for h in detections[f]['hands']])
        if affine is None:continue
        fits+=1
        c0=np.asarray([(o0[0]+o0[2])/2,(o0[1]+o0[3])/2]);c1=np.asarray([(o1[0]+o1[2])/2,(o1[1]+o1[3])/2])
        object_move=(c1-affine@np.r_[c0,1.])/scale
        hand_move=(np.asarray(h1['center'])-affine@np.r_[h0['center'],1.])/scale
        on=float(np.linalg.norm(object_move));hn=float(np.linalg.norm(hand_move))
        similarity=0.
        if on>.02 and hn>.02:
            cosine=max(0.,float(object_move@hand_move/(on*hn)))
            similarity=cosine*math.exp(-float(np.linalg.norm(object_move-hand_move)))*min(1.,on/.05)
        motion.append(similarity)
        d0=box_distance(o0,h0['box'])/scale;d1=box_distance(o1,h1['box'])/scale
        # Missing hands never count as release. Release requires two detections
        # and a valid background-motion fit, plus a stationary object proxy.
        stable=max(0.,1-on/.04)
        release.append(math.exp(-d0)*min(1.,max(0.,d1-d0))*stable)
        stationary.append(float(d0<.5 and on<.04))
    n=max(1,len(frames));p=max(1,len(pairs))
    vector=[len(selected)/n,float(np.mean(near)) if near else 0.,max(over,default=0.),float(np.mean(prox)) if prox else 0.,
        observed/p,fits/p,float(np.mean(motion)) if motion else 0.,max(motion,default=0.),fits/p,
        float(np.mean(release)) if release else 0.,max(release,default=0.),float(np.mean(stationary)) if stationary else 0.]
    assert len(vector)==12 and np.isfinite(vector).all()
    return vector,dict(sampled_frames=len(frames),hand_detected_frames=len(selected),adjacent_pairs=len(pairs),observed_hand_pairs=observed,camera_valid_pairs=fits)

def detect(detector,image):
    import mediapipe as mp
    rgb=np.ascontiguousarray(np.asarray(image))
    result=detector.detect(mp.Image(image_format=mp.ImageFormat.SRGB,data=rgb))
    width,height=image.size;hands=[]
    for points,categories in zip(result.hand_landmarks,result.handedness):
        xy=np.asarray([[p.x*width,p.y*height] for p in points]);lo=xy.min(0);hi=xy.max(0);margin=(hi-lo)*.1
        lo=np.maximum(0,lo-margin);hi=np.minimum([width,height],hi+margin)
        center=xy[[0,5,9,13,17]].mean(0)
        category=categories[0]
        hands.append(dict(box=[*map(float,lo),*map(float,hi)],center=center.tolist(),landmarks=xy.tolist(),
            side=category.category_name,handedness_score=float(category.score)))
    return dict(hands=hands,model_is_contact_classifier=False)
