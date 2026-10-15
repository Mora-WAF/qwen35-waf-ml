"""Temperature scaling and policy threshold (spec §10), offline, from the notebook's candidate scores.

Input: JSONL rows {"id", "split", "label", "scores": [summed completion log-prob per class]} plus a header row
{"classes": [...]} (written by the notebook's scoring cell, `colab_helpers.class_logprobs`).
  1. T is fitted on val_calib only (NLL of the true class under softmax(s / T), L-BFGS on log T so T > 0).
  2. The threshold tau is chosen on val_policy only: the lowest tau whose hard-negative false-intervention rate
     is <= --max-hn-fir, which maximises attack intervention recall at that constraint. Then frozen.
  3. test, test_unseen and robustness are only reported. The robustness set is never tuned on: its records
     have no allowed class, so a closed-set softmax can only abstain on them through low confidence.
Scores: summed (not length-normalised) log-probs; ECE over 15 equal-width bins of max probability.

usage: python -m eval.calibration SCORES.jsonl [--write-policy]   -> reports/calibration.json
"""
import argparse
import json
import math
import re
from pathlib import Path

import numpy as np
import yaml

ROOT = Path(__file__).resolve().parent.parent
BINS = 15


def softmax(s, T=1.0):
    z = np.asarray(s, dtype=np.float64) / T
    z = z - z.max(axis=-1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=-1, keepdims=True)


def fit_temperature(S, y):
    import torch
    s, t = torch.tensor(S, dtype=torch.float64), torch.tensor(y)
    log_t = torch.nn.Parameter(torch.tensor([math.log(1.5)], dtype=torch.float64))
    opt = torch.optim.LBFGS([log_t], lr=0.1, max_iter=200, line_search_fn="strong_wolfe")

    def closure():
        opt.zero_grad()
        loss = torch.nn.functional.cross_entropy(s / log_t.exp(), t)
        loss.backward()
        return loss
    opt.step(closure)
    return float(log_t.exp())


def reliability(P, y):
    conf, pred = P.max(1), P.argmax(1)
    hit = (pred == y).astype(float)
    ece, bins = 0.0, []
    for lo in np.linspace(0, 1, BINS, endpoint=False):
        m = (conf > lo) & (conf <= lo + 1 / BINS)
        if m.any():
            ece += m.mean() * abs(hit[m].mean() - conf[m].mean())
            bins.append({"lo": round(lo, 4), "n": int(m.sum()), "acc": round(hit[m].mean(), 4), "conf": round(conf[m].mean(), 4)})
    onehot = np.eye(P.shape[1])[y]
    return {"n": len(y), "accuracy": round(hit.mean(), 4), "nll": round(float(-np.log(P[np.arange(len(y)), y] + 1e-12).mean()), 4),
            "brier": round(float(((P - onehot) ** 2).sum(1).mean()), 4), "ece": round(ece, 4), "bins": bins}


def policy_rates(P, y, tau, benign, hard_neg):
    """Binary intervention view: an emitted attack class at conf >= tau intervenes; anything below tau abstains."""
    conf, pred = P.max(1), P.argmax(1)
    act = conf >= tau
    attack_true, attack_pred = ~benign[y], ~benign[pred]
    inter = act & attack_pred
    hn = hard_neg[y]
    return {"tau": round(float(tau), 6), "coverage": round(act.mean(), 4),
            "attack_intervention_recall": round(inter[attack_true].mean(), 4) if attack_true.any() else None,
            "benign_false_intervention": round(inter[~attack_true].mean(), 4) if (~attack_true).any() else None,
            "hard_negative_false_intervention": round(inter[hn].mean(), 4) if hn.any() else None,
            "accuracy_when_acting": round((pred[act] == y[act]).mean(), 4) if act.any() else None,
            "abstention_rate": round(1 - act.mean(), 4)}


def choose_tau(P, y, benign, hard_neg, max_hn_fir):
    hn = hard_neg[y]
    for tau in np.unique(np.concatenate([[0.0], P.max(1)])):  # ascending: lowest tau that meets the constraint
        r = policy_rates(P, y, tau, benign, hard_neg)
        if not hn.any() or r["hard_negative_false_intervention"] <= max_hn_fir:
            return float(tau), r
    return 1.0 + 1e-9, policy_rates(P, y, 1.0 + 1e-9, benign, hard_neg)  # nothing meets it: abstain on all, reported


def main(path, max_hn_fir=0.01, write_policy=False):
    rows = [json.loads(x) for x in open(path)]
    classes = rows[0]["classes"]
    rows = rows[1:]
    tax = yaml.safe_load(open(ROOT / "config/taxonomy.yaml"))["classes"]
    assert classes == list(tax), "score file was produced with a different taxonomy order"
    benign = np.array([tax[c]["is_benign"] for c in classes])
    hard_neg = np.array([c.startswith("benign_hard_negative") for c in classes])
    idx = {c: i for i, c in enumerate(classes)}
    by = {}
    for r in rows:
        by.setdefault(r["split"], []).append(r)

    def arrays(split, known=True):
        rs = [r for r in by.get(split, []) if (r["label"] in idx) == known]
        return np.array([r["scores"] for r in rs]), np.array([idx.get(r["label"], -1) for r in rs]), rs

    Sc, yc, _ = arrays("val_calib")
    T = fit_temperature(Sc, yc)
    Sp, yp, _ = arrays("val_policy")
    tau, at_tau = choose_tau(softmax(Sp, T), yp, benign, hard_neg, max_hn_fir)
    rep = {"method": "summed completion log-prob per class, temperature scaling on val_calib, tau on val_policy",
           "temperature": round(T, 6), "tau": round(tau, 6), "max_hard_negative_fir": max_hn_fir,
           "val_calib": {"uncalibrated": reliability(softmax(Sc), yc), "calibrated": reliability(softmax(Sc, T), yc)},
           "val_policy_at_tau": at_tau}
    for split in ("test", "test_unseen"):
        S, y, _ = arrays(split)
        if len(y):
            P = softmax(S, T)
            rep[split] = {"reliability": reliability(P, y), "policy": policy_rates(P, y, tau, benign, hard_neg)}
    S, _, rs = arrays("robustness", known=False)
    if len(rs):  # labels outside the closed set: success = abstain
        P = softmax(S, T)
        conf = P.max(1)
        kinds = {}
        for r, c in zip(rs, conf):
            k = kinds.setdefault(r.get("kind", "robustness"), [0, 0])
            k[0] += c < tau
            k[1] += 1
        ent = -(P * np.log(P + 1e-12)).sum(1)
        rep["robustness"] = {"n": len(rs), "abstained": round(float((conf < tau).mean()), 4),
                             "by_kind": {k: round(a / n, 4) for k, (a, n) in kinds.items()},
                             "entropy_mean": round(float(ent.mean()), 4)}
    (ROOT / "reports/calibration.json").write_text(json.dumps(rep, indent=1))
    if write_policy:  # freeze T and tau in config/policy.yaml (line edits keep the file's comments)
        pol_p = ROOT / "config/policy.yaml"
        txt = pol_p.read_text()
        for key, val in (("threshold", tau), ("temperature", T)):
            txt, n = re.subn(rf"^{key}: \S+", f"{key}: {round(val, 6)}", txt, count=1, flags=re.M)
            assert n == 1, f"config/policy.yaml has no top-level {key}: line"
        pol_p.write_text(txt)
    print(json.dumps({k: v for k, v in rep.items() if k not in ("val_calib",)}, indent=1)[:3000])
    return rep


def calibrated_confidence(scores, temperature, cls_index):
    """Probability policy.decide receives for the emitted class."""
    return float(softmax(scores, temperature)[cls_index])


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("scores")
    ap.add_argument("--max-hn-fir", type=float, default=0.01)
    ap.add_argument("--write-policy", action="store_true")
    a = ap.parse_args()
    main(a.scores, a.max_hn_fir, a.write_policy)
