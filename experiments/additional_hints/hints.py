"""Candidate ranking from extra sighting-time hints; no GT is read here."""
import numpy as np
from experiments.candidate_tools.candidate_common import validate_query

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


def native_gated_choice(query,hint):
    """Conservative fixed gate used by the clarification pilot; no labels read."""
    validate_query(query)
    if hint.get('declined',False):
        return query['baseline_id'],dict(reason='user_declined',changed=False)
    best_id=timing_rank(query,hint)[0]
    base=next(c for c in query['candidates'] if c['id']==query['baseline_id'])
    best=next(c for c in query['candidates'] if c['id']==best_id)
    if best_id==query['baseline_id']:
        return best_id,dict(reason='native_is_winner',changed=False)
    duration=(query['query_frame']-1)/query.get('fps',5)
    a=temporal_log_density(best,query.get('fps',5),hint,duration)
    b=temporal_log_density(base,query.get('fps',5),hint,duration)
    allow=(best['image_score']>=base['image_score']-.025 and a>b+np.log(3.)
           and best['image_score']+.05*a>base['image_score']+.05*b+.025)
    return (best_id if allow else query['baseline_id']),dict(reason='gate_passed' if allow else 'gate_retained_native',changed=bool(allow))
