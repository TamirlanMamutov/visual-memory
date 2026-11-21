"""Box-only refinement inside the baseline time window; no hint/GT input."""
import numpy as np
from experiments.candidate_tools.candidate_common import validate_query

def refine_boxes(query, mode='image', margin=.025):
    """Choose eligible overlapping boxes; preserve every baseline frame/score.

    Samples must be query-conditioned CLIP similarities from cached crop images.
    Eligibility requires >=50% frame overlap and >=min(2, anchor length)
    distinct crop-sample frames inside the baseline window. Text modes also
    require image similarity no lower than baseline minus the fixed margin.
    """
    validate_query(query)
    if mode not in ['image', 'name', 'description']:
        raise ValueError('Unknown box-refinement mode')
    candidates = query['candidates']; anchor = next(c for c in candidates if c['id']==query['baseline_id'])
    frames = {b[0] for b in anchor['bboxes']}; evidence = {}
    for index,candidate in enumerate(candidates):
        overlap = len({b[0] for b in candidate['bboxes']}&frames)/max(1,len(frames))
        samples = {s['frame']:s for s in candidate.get('samples',[]) if s['frame'] in frames}
        if overlap < .5 or not samples or len(samples) < min(2,len(frames)):
            continue
        field = {'image':'image_score','name':'name_score','description':'description_score'}[mode]
        evidence[candidate['id']] = dict(
            image=float(np.mean([s['image_score'] for s in samples.values()])),
            selected_score=float(np.mean([s[field] for s in samples.values()])), index=index)
    chosen = anchor
    if anchor['id'] in evidence:
        ref = evidence[anchor['id']]
        eligible = [c for c in candidates if c['id'] in evidence
                    and evidence[c['id']]['selected_score'] > ref['selected_score']+margin
                    and (mode=='image' or evidence[c['id']]['image'] >= ref['image']-margin)]
        if eligible:
            chosen = max(eligible,key=lambda c:(evidence[c['id']]['selected_score'],
                                                evidence[c['id']]['image'],-evidence[c['id']]['index']))
    by_frame = {b[0]:b for b in chosen['bboxes']}
    boxes = [list(by_frame.get(b[0],b)) for b in anchor['bboxes']]
    assert [b[0] for b in boxes] == [b[0] for b in anchor['bboxes']]
    return dict(key=query['key'],score=query['score'],bboxes=boxes,
                selected_candidate=chosen['id'],frame_ids_preserved=True)
