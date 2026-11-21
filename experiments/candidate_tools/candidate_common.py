"""Label-free candidate validation and unchanged-confidence prediction export."""
import math

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

def track_prediction(query, candidate_id):
    candidate = next(c for c in query['candidates'] if c['id']==candidate_id)
    return dict(key=query['key'],score=query['score'],bboxes=candidate['bboxes'],selected_candidate=candidate_id)
