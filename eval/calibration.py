"""Temperature scaling and policy threshold (spec §10), offline, from the notebook's candidate scores.

Input: JSONL rows {"id", "split", "label", "scores": [summed completion log-prob per class]} plus a header row
{"classes": [...]} (written by the notebook's scoring cell, `colab_helpers.class_logprobs`).
  1. T is fitted on val_calib only (NLL of the true class under softmax(s / T), L-BFGS on log T so T > 0).
  2. The threshold tau is chosen on val_policy only: the highest tau whose hard-negative false-intervention rate
     is <= --max-hn-fir and whose attack intervention recall is within --max-recall-loss of the best feasible tau
     (the lowest one). Abstaining is free up to that recall budget, so tau is not stuck at 0 when hard negatives
     are already clean (v7/v8: tau 0 never abstained). Then frozen.
  2b. Flow envelope: 0.5-99.5% quantiles of benign_standard_traffic flow features in the build's train split. A flow
     the model calls benign_standard_traffic outside it abstains (eval.policy): out-of-taxonomy floods and scans
     land in the background class with p ~ 1, which no confidence threshold can catch (v8: 0/25).
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


def choose_tau(P, y, benign, hard_neg, max_hn_fir, max_recall_loss=0.005):
    hn = hard_neg[y]
    ok = [(float(t), r) for t in np.unique(np.concatenate([[0.0], P.max(1)]))
          for r in [policy_rates(P, y, t, benign, hard_neg)]
          if not hn.any() or r["hard_negative_false_intervention"] <= max_hn_fir]
    if not ok:  # nothing meets it: abstain on all, reported
        return 1.0 + 1e-9, policy_rates(P, y, 1.0 + 1e-9, benign, hard_neg)
    best = ok[0][1]["attack_intervention_recall"] or 0.0  # recall only falls as tau rises
    return [x for x in ok if (x[1]["attack_intervention_recall"] or 0.0) >= best - max_recall_loss][-1]


ENVELOPE = ("pkts_per_s", "bytes_per_s", "pkts_total", "bytes_total", "duration_s")


def flow_features(mode):
    """id -> flow features for the build that produced the scores (core splits + robustness set)."""
    out = {}
    for fn in sorted((ROOT / "datasets_ready").glob(f"{mode}_*.jsonl")) + [ROOT / "datasets_ready/robustness_pilot.jsonl"]:
        for line in open(fn):
            r = json.loads(line)
            f = json.loads(r["messages"][1]["content"].split("] ", 1)[1]).get("flow", {}).get("features")
            if f:
                out[r["id"]] = (f, r["metadata"]["split"], r["metadata"]["threat_type"])
    return out


def fit_envelope(feats):
    tr = [f for f, sp, c in feats.values() if sp == "train" and c == "benign_standard_traffic"]
    env = {}
    for k in ENVELOPE:
        v = sorted(f[k] for f in tr if k in f)
        if v:
            env[k] = [v[int(0.005 * len(v))], v[int(0.995 * len(v)) - 1]]
    return env


def main(path, max_hn_fir=0.01, write_policy=False, max_recall_loss=0.005):
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
    tau, at_tau = choose_tau(softmax(Sp, T), yp, benign, hard_neg, max_hn_fir, max_recall_loss)
    feats = flow_features(json.loads(open(path).readline()).get("mode", "fast"))
    env = fit_envelope(feats)
    from eval.policy import outside_envelope
    bst = idx["benign_standard_traffic"]

    def env_abstain(rs, P):  # model says benign_standard_traffic, flow outside the train envelope
        return np.array([P[i].argmax() == bst and r["id"] in feats and outside_envelope(feats[r["id"]][0], env)
                         for i, r in enumerate(rs)], dtype=bool)
    rep = {"method": "summed completion log-prob per class, temperature scaling on val_calib, tau on val_policy",
           "temperature": round(T, 6), "tau": round(tau, 6), "max_hard_negative_fir": max_hn_fir,
           "val_calib": {"uncalibrated": reliability(softmax(Sc), yc), "calibrated": reliability(softmax(Sc, T), yc)},
           "val_policy_at_tau": at_tau, "flow_envelope": env}
    for split in ("test", "test_unseen"):
        S, y, rs = arrays(split)
        if len(y):
            P = softmax(S, T)
            ea = env_abstain(rs, P)
            rep[split] = {"reliability": reliability(P, y), "policy": policy_rates(P, y, tau, benign, hard_neg),
                          "envelope_abstained": {"benign": round(float(ea[benign[y]].mean()), 4),
                                                 "attack": round(float(ea[~benign[y]].mean()), 4)}}
    S, _, rs = arrays("robustness", known=False)
    if len(rs):  # labels outside the closed set: success = abstain
        P = softmax(S, T)
        conf = P.max(1)
        absn = (conf < tau) | env_abstain(rs, P)
        kinds = {}
        for r, a in zip(rs, absn):
            k = kinds.setdefault(r.get("kind", "robustness"), [0, 0])
            k[0] += a
            k[1] += 1
        ent = -(P * np.log(P + 1e-12)).sum(1)
        rep["robustness"] = {"n": len(rs), "abstained": round(float(absn.mean()), 4),
                             "by_kind": {k: round(a / n, 4) for k, (a, n) in kinds.items()},
                             "entropy_mean": round(float(ent.mean()), 4)}
    (ROOT / "reports/calibration.json").write_text(json.dumps(rep, indent=1))
    if write_policy:  # freeze T and tau in config/policy.yaml (line edits keep the file's comments)
        pol_p = ROOT / "config/policy.yaml"
        txt = pol_p.read_text()
        for key, val in (("threshold", round(tau, 6)), ("temperature", round(T, 6)), ("flow_envelope", json.dumps(env))):
            txt, n = re.subn(rf"^{key}: [^#\n]*?(\s*(?:#.*)?)$", lambda m: f"{key}: {val}{m.group(1)}", txt, count=1, flags=re.M)
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
    ap.add_argument("--max-recall-loss", type=float, default=0.005, help="attack recall tau may give up for abstention")
    ap.add_argument("--write-policy", action="store_true")
    a = ap.parse_args()
    main(a.scores, a.max_hn_fir, a.write_policy, a.max_recall_loss)
