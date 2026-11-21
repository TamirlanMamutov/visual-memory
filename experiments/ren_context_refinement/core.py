"""Candidate-level timing hints and window-preserving box refinement.

All boxes use [frame, x1, y1, x2, y2] in one consistent image coordinate space.
This module never reads ground-truth tracks, videos, or model checkpoints.
"""
import math
import numpy as np


def validate_query(query):
    candidates = query['candidates']
    ids = [c['id'] for c in candidates]
    if not candidates or len(ids) != len(set(ids)):
        raise ValueError('Candidate IDs must be nonempty and unique')
    if query['baseline_id'] not in ids:
        raise ValueError('Missing baseline candidate')
    if not math.isfinite(query['score']):
        raise ValueError('Invalid AP confidence')
    for candidate in candidates:
        if not math.isfinite(candidate['image_score']):
            raise ValueError('Non-finite appearance score')
        frames = [b[0] for b in candidate['bboxes']]
        if frames != sorted(set(frames)):
            raise ValueError('Duplicate/unsorted candidate frames')
        if frames and frames != list(range(frames[0], frames[-1]+1)):
            raise ValueError('Candidate track must be contiguous')
        for f,x1,y1,x2,y2 in candidate['bboxes']:
            if not 0 <= f < query['query_frame'] or x2 < x1 or y2 < y1:
                raise ValueError('Invalid box or prediction reaches query/future frame')
            if not all(math.isfinite(v) for v in [f,x1,y1,x2,y2]):
                raise ValueError('Non-finite box')
        for sample in candidate.get('samples',[]):
            if sample['frame'] not in frames or not math.isfinite(sample['image_score']):
                raise ValueError('Invalid crop sample or score')


def temporal_log_density(candidate, fps, hint, duration):
    times = np.asarray([b[0]/fps for b in candidate['bboxes']], dtype=float)
    if not len(times):
        return -1e6
    if hint['kind'] == 'third':
        label = {'early':0, 'middle':1, 'late':2}[hint['value']]
        bins = np.minimum(2, np.floor(times/duration*3))
        return float(np.log(np.mean(np.where(bins == label, 2.94/duration, .03/duration))))
    if hint['kind'] != 'time':
        raise ValueError('Hint kind must be time or third')
    center = float(hint['seconds']); sigma = float(hint['sigma_seconds'])
    if sigma < 0 or not 0 <= center <= duration:
        raise ValueError('Hint seconds must lie in searchable clip time; sigma must be nonnegative')
    tau = max(1/fps, sigma)
    values = -.5*((times-center)/tau)**2 - np.log(tau*np.sqrt(2*np.pi))
    maximum = float(values.max())
    return maximum + float(np.log(np.exp(values-maximum).mean()))


def timing_rank(query, hint, weight=.05, use_description=False, prior=None, reliability=.8):
    """Return candidate IDs. Hints use CLIP appearance + fixed temporal density.

    Optional prior is a candidate-ID -> log-density mapping fitted externally on
    training sources. Providing it enables the fixed robust mixture. This is a
    heuristic, not calibrated user-confidence modeling.
    """
    validate_query(query)
    fps = float(query.get('fps', 5)); duration = (query['query_frame']-1)/fps
    if duration <= 0:
        raise ValueError('Need nonempty searchable history')
    scores = []
    for index,candidate in enumerate(query['candidates']):
        appearance = float(candidate['image_score'])
        if use_description:
            appearance = .5*(appearance+float(candidate['description_score']))
        time_score = temporal_log_density(candidate, fps, hint, duration)
        if prior is not None:
            if not 0 < reliability < 1:
                raise ValueError('Mixture reliability must be in (0,1)')
            time_score = float(np.logaddexp(np.log(reliability)+time_score,
                                           np.log(1-reliability)+prior[candidate['id']]))
        scores.append((appearance+weight*time_score, candidate['id']==query['baseline_id'], -index, candidate['id']))
    return [item[-1] for item in sorted(scores, reverse=True)]


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


def track_prediction(query, candidate_id):
    candidate = next(c for c in query['candidates'] if c['id']==candidate_id)
    return dict(key=query['key'],score=query['score'],bboxes=candidate['bboxes'],selected_candidate=candidate_id)
