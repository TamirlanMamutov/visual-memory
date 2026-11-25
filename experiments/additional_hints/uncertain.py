"""Uncertain human sighting clues: native fallback plus ranked alternatives.

No labels are read. Confidence weights are fixed heuristics, not calibrated
probabilities. Missing scope/confidence never authorizes a primary override.
"""
from pathlib import Path
import argparse
import json
import sys
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from experiments.additional_hints.hints import temporal_log_density
from experiments.candidate_tools.candidate_common import validate_query, track_prediction

POLICY = 'uncertain_sighting_v1'
RELIABILITY = {'unknown': .2, 'low': .2, 'medium': .5, 'high': .8}
SCOPES = {'unknown', 'remembered_sighting', 'latest_estimate'}
WEIGHT = .05
APPEARANCE_MARGIN = .025


def context_density(candidate, query, hint, scope):
    """Log support for a target episode; not a calibrated posterior."""
    fps = float(query.get('fps', 5))
    duration = query['query_frame']/fps  # exclusive cutoff: includes every pre-query frame
    # Validate the timing fields even when their scope is only a recalled sighting.
    density = temporal_log_density(candidate, fps, hint, duration)
    if scope == 'latest_estimate' or not candidate['bboxes']:
        return density
    # A recalled sighting may precede the actual last one. It supplies only a
    # weak lower time bound, with tolerance for a noisy remembered point.
    if hint['kind'] == 'third':
        lower = {'early': 0, 'middle': 1, 'late': 2}[hint['value']]*duration/3
    else:
        lower = max(0., float(hint['seconds'])-2*float(hint['sigma_seconds']))
    times = np.asarray([b[0]/fps for b in candidate['bboxes']])
    return float(np.log(np.mean(np.where(times >= lower, 1., .03))/duration))


def uncertain_prediction(query, hint=None):
    validate_query(query)
    fps = float(query.get('fps', 5))
    duration = query['query_frame']/fps
    if duration <= 0:
        raise ValueError('Need nonempty searchable history')
    scope = 'unknown' if hint is None else hint.get('scope', 'unknown')
    confidence = 'unknown' if hint is None else hint.get('confidence', 'unknown')
    if scope not in SCOPES or confidence not in RELIABILITY:
        raise ValueError('Invalid clue scope or confidence category')
    supplied = hint is not None and not hint.get('declined', False)
    uniform = -np.log(duration)
    reliability = RELIABILITY[confidence]
    rows = []
    for index, candidate in enumerate(query['candidates']):
        log_support = uniform
        if supplied:
            value = context_density(candidate, query, hint, scope)
            log_support = float(np.logaddexp(np.log(reliability)+value,
                                             np.log(1-reliability)+uniform))
        rows.append(dict(id=candidate['id'], image_score=candidate['image_score'],
                         mixed_log_support=float(log_support),
                         fused_score=float(candidate['image_score']+WEIGHT*(log_support-uniform)),
                         index=index))
    rows.sort(key=lambda row: (-row['fused_score'], row['id'] != query['baseline_id'], row['index']))
    base = next(row for row in rows if row['id'] == query['baseline_id'])
    best = rows[0]
    selected = query['baseline_id']
    reason = 'not_requested' if hint is None else 'user_declined'
    if supplied:
        if scope != 'latest_estimate':
            reason = 'recalled_sighting_keeps_native'
        elif confidence != 'high':
            reason = 'uncertain_latest_keeps_native'
        elif best['id'] == query['baseline_id']:
            reason = 'native_is_winner'
        elif (best['image_score'] >= base['image_score']-APPEARANCE_MARGIN
              and best['mixed_log_support'] > base['mixed_log_support']+np.log(3.)
              and best['fused_score'] > base['fused_score']+APPEARANCE_MARGIN):
            selected = best['id']
            reason = 'explicit_high_confidence_gate_passed'
        else:
            reason = 'gate_retained_native'
    # Keep primary distinct from the full context-ranked alternatives. A shortlist
    # is not an automatic multi-track answer or a replacement for top1 evaluation.
    order = [selected]+[row['id'] for row in rows if row['id'] != selected]
    prediction = track_prediction(query, selected)
    prediction.update(ranked_candidate_ids=order,
                      hint_ranked_candidate_ids=[row['id'] for row in rows],
                      suggested_candidate_ids=order[:2],
                      decision=dict(reason=reason, changed=selected != query['baseline_id'],
                                    scope=scope, confidence=confidence,
                                    needs_confirmation=bool(supplied and order[0] != rows[0]['id'])),
                      ranking_scores=[{k: v for k, v in row.items() if k != 'index'} for row in rows])
    return prediction


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--bundle', type=Path, required=True)
    parser.add_argument('--hints', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    bundle = json.loads(args.bundle.read_text())
    data = json.loads(args.hints.read_text())
    hints = {h['key']: h for h in data['queries']}
    keys = {q['key'] for q in bundle['queries']}
    if len(hints) != len(data['queries']) or not set(hints) <= keys:
        raise ValueError('Duplicate or unknown hint keys')
    result = dict(method=POLICY, queries=[uncertain_prediction(q, hints.get(q['key']))
                                         for q in bundle['queries']],
                  cohort=bundle.get('cohort', 'unspecified'),
                  hint_provenance=data.get('provenance', 'unspecified'),
                  ground_truth_read=False,
                  parameters=dict(reliability=RELIABILITY, weight=WEIGHT,
                                  appearance_margin=APPEARANCE_MARGIN,
                                  note='Fixed heuristic weights, not calibrated confidence probabilities'))
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2, allow_nan=False)+'\n')
    print(args.out)


if __name__ == '__main__':
    main()
