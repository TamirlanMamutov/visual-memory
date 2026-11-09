"""EK-100 MIR metrics, matching the official evaluation code (JPoSE repo,
src/evaluation): mAP over relevance == 1, nDCG over graded relevance with the
first k = |relevant| retrievals."""
import numpy as np


def mean_ap(sim, rel):
    order = np.argsort(-sim, axis=1, kind='stable')
    r = np.take_along_axis(rel, order, 1) == 1
    cum = np.cumsum(r, 1)
    prec = cum / np.arange(1, r.shape[1] + 1)
    n_rel = r.sum(1)
    ap = (prec * r).sum(1) / np.maximum(n_rel, 1)
    return float(ap[n_rel > 0].mean())


def ndcg(sim, rel):
    order = np.argsort(-sim, axis=1, kind='stable')
    k = (np.sort(rel, 1)[:, ::-1] > 0)
    disc = 1 / np.log2(np.arange(rel.shape[1]) + 2)
    dcg = (np.take_along_axis(rel, order, 1) * k * disc).sum(1)
    idcg = (np.sort(rel, 1)[:, ::-1] * k * disc).sum(1)
    valid = idcg > 0
    return float((dcg[valid] / idcg[valid]).mean())


def evaluate(sim_vt, rel_vt, sim_tv=None):
    """sim_vt / rel_vt: [videos, sentences]; sim_tv optionally gives separate
    [sentences, videos] scores (e.g. after per-direction reranking).
    Returns percentages like the leaderboard."""
    sim_vt = np.asarray(sim_vt, np.float32); rel_vt = np.asarray(rel_vt, np.float32)
    sim_tv = sim_vt.T if sim_tv is None else np.asarray(sim_tv, np.float32)
    out = {'mAP': {}, 'nDCG': {}}
    for name, s, r in [('V2T', sim_vt, rel_vt), ('T2V', sim_tv, rel_vt.T)]:
        out['mAP'][name] = 100 * mean_ap(s, r)
        out['nDCG'][name] = 100 * ndcg(s, r)
    for m in out.values():
        m['AVG'] = (m['V2T'] + m['T2V']) / 2
    return out


def fmt(m):
    return (f"mAP V2T {m['mAP']['V2T']:.2f} T2V {m['mAP']['T2V']:.2f} AVG {m['mAP']['AVG']:.2f} | "
            f"nDCG V2T {m['nDCG']['V2T']:.2f} T2V {m['nDCG']['T2V']:.2f} AVG {m['nDCG']['AVG']:.2f}")
