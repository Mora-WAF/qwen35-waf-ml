"""Probability-averaging ensemble of classifiers fine-tuned on the same build (19 Oct: Qwen3.5-4B v11 on the A100 +
Gemma 4 E2B-it 1 epoch on the TPU), offline, from the members' candidate score files. No training.

  1. Each member's summed class log-probs -> calibrated probabilities with that member's frozen temperature (its policy).
  2. The ensemble distribution is their mean (eval.policy.ensemble_log_mean / ensemble_probs). The merged score file
     stores log(mean) as `scores`, so eval.calibration fits the ensemble's own temperature on val_calib and its
     threshold on val_policy exactly as for one model, and freezes them into the ensemble policy.
  3. The flow envelope is the members' (same build: --envelope-from the first member's policy), so no local build is
     needed. Records scored by every member only (ids matched per split); the rest are counted, never guessed.
  4. Report: each member under its own frozen policy vs the ensemble under its frozen policy, on test and
     test_unseen, plus paired counts of enforcement errors (attack vs benign) the ensemble adds or removes.

The decision view is chosen on val_policy (19 Oct: top_class and attack_mass tie on recall and false intervention,
top_class abstains less and is more accurate when acting), never on test.

usage: python -m eval.ensemble build --member NAME SCORES[,SCORES...] POLICY [--member ...] [--out-dir reports/ensemble]
                                     [--policy config/policy_ensemble.yaml] [--act-on top_class]
"""
import argparse
import json
from pathlib import Path

import numpy as np
import yaml

from eval import calibration as CAL
from eval.policy import decision_view, ensemble_log_mean

ROOT = Path(__file__).resolve().parent.parent
TEMPLATE = "config/policy.yaml"  # the deployed policy's keys (required_context, reason_codes, tier-0 rules, ...)


def _load(path):
    """(header, {(split, id): row}) from one score file or several, comma-separated (e.g. Gemma's SCORES_fast.jsonl +
    SCORES_robustness.jsonl). Window ids "<id>#<k>" (training.gemma_tpu.data bench) map back to their record when it
    is one window; multi-window records are left out (counted as dropped by merge)."""
    head, out = None, {}
    for f in path.split(","):
        rows = [json.loads(x) for x in open(ROOT / f)]
        head = head or rows[0]
        assert rows[0]["classes"] == head["classes"], f"{f}: other class list"
        for r in rows[1:]:
            out[(r["split"], r["id"])] = r
    wins = {}
    for (split, rid) in list(out):
        base, sep, k = rid.rpartition("#")
        if sep and k.isdigit():
            wins.setdefault((split, base), []).append(out.pop((split, rid)))
    for key, ws in wins.items():
        if len(ws) == 1:
            out[key] = {**ws[0], "id": key[1]}
    return head, out


def merge(members, out):
    """members: [(name, scores path, policy path)] -> merged score file at `out`; returns (header, counts)."""
    loaded = []
    for name, scores, policy in members:
        head, rows = _load(scores)
        pol = yaml.safe_load(open(ROOT / policy))
        loaded.append((name, scores, policy, float(pol["temperature"]), head, rows))
    classes = loaded[0][4]["classes"]
    assert all(m[4]["classes"] == classes for m in loaded), "members score different class lists"
    keys = set(loaded[0][5])
    for m in loaded[1:]:
        keys &= set(m[5])
    head = {"classes": classes, "mode": loaded[0][4].get("mode", "fast"),
            "ensemble": {"combine": "mean_probability",
                         "members": [{"name": n, "scores": s, "policy": p, "temperature": T,
                                      "fingerprint": h.get("fingerprint")} for n, s, p, T, h, _ in loaded]}}
    split_order = {s: i for i, s in enumerate(("val_calib", "val_policy", "val_select", "test", "test_unseen", "robustness"))}
    counts = {"merged": {}, "dropped": {n: {} for n, *_ in loaded}}
    with open(ROOT / out, "w") as fh:
        fh.write(json.dumps(head) + "\n")
        for k in sorted(keys, key=lambda k: (split_order.get(k[0], 99), k[0], k[1])):
            first = loaded[0][5][k]  # labels and kinds from the first member (Gemma writes null for classless records)
            assert all(m[5][k]["label"] in (first["label"], None) for m in loaded[1:]), f"label mismatch for {k}"
            P = [CAL.softmax(m[5][k]["scores"], m[3]) for m in loaded]
            fh.write(json.dumps({"id": k[1], "split": k[0], "kind": first.get("kind"), "label": first["label"],
                                 "scores": [round(float(x), 6) for x in ensemble_log_mean(P)]}) + "\n")
            counts["merged"][k[0]] = counts["merged"].get(k[0], 0) + 1
    for n, *_, rows in loaded:  # records a member scored that another did not (val_select: Qwen skips it)
        for k in set(rows) - keys:
            counts["dropped"][n][k[0]] = counts["dropped"][n].get(k[0], 0) + 1
    return head, counts


def write_policy_template(path, head, act_on):
    """The ensemble policy before calibration: the deployed policy's keys + an `ensemble:` block. eval.calibration
    --write-policy then freezes threshold, temperature, act_on and flow_envelope into it."""
    txt = (ROOT / TEMPLATE).read_text()
    members = "\n".join(f"    - {{name: {m['name']}, policy: {m['policy']}, temperature: {m['temperature']}}}"
                        for m in head["ensemble"]["members"])
    hdr = ("# Ensemble policy (eval.ensemble, 19 Oct): mean of the members' calibrated distributions, re-tempered with\n"
           "# `temperature`; threshold, temperature, act_on and flow_envelope frozen by eval.calibration --write-policy.\n"
           "# Members score every request (each with its own prompt template and frozen temperature).\n")
    block = ("\n# Members in order (eval.policy.ensemble_probs); each member's distribution uses its own frozen temperature.\n"
             f"ensemble:\n  combine: mean_probability\n  members:\n{members}\n")
    txt = txt.replace("act_on: top_class", f"act_on: {act_on}", 1)
    (ROOT / path).write_text(hdr + txt.rstrip("\n") + "\n" + block)


def _arrays(rows, split, classes, T):
    rs = [r for r in rows.values() if r["split"] == split and r["label"] in classes]
    rs.sort(key=lambda r: r["id"])
    return [r["id"] for r in rs], CAL.softmax(np.array([r["scores"] for r in rs]), T), np.array([classes.index(r["label"]) for r in rs])


def report(members, merged, ens_policy, tax):
    classes = json.loads(open(ROOT / merged).readline())["classes"]  # the members' classes (a prefix of today's taxonomy)
    benign = np.array([tax[c]["is_benign"] for c in classes])
    hard = np.array([c.startswith("benign_hard_negative") for c in classes])
    epol = yaml.safe_load(open(ROOT / ens_policy))
    systems = [(n, _load(s)[1], yaml.safe_load(open(ROOT / p))) for n, s, p in members]
    systems.append(("ensemble", _load(merged)[1], epol))
    rep = {}
    for split in ("test", "test_unseen"):
        common = None
        for _, rows, _ in systems:  # the ensemble's records = those every member scored
            ids = {r["id"] for r in rows.values() if r["split"] == split}
            common = ids if common is None else common & ids
        out, wrong = {}, {}
        for name, rows, pol in systems:
            rows = {k: v for k, v in rows.items() if k[0] == split and k[1] in common}
            ids, P, y = _arrays(rows, split, classes, float(pol["temperature"]))
            act_on = pol.get("act_on", "top_class")
            rates = CAL.policy_rates(P, y, float(pol["threshold"]), benign, hard, None, act_on)
            pred = P.argmax(1)
            conf, vpred = decision_view(P, benign, act_on)
            acting = conf >= float(pol["threshold"])
            # enforcement error at the frozen policy: an attack not intervened on, or a benign record blocked
            err = np.where(benign[y], acting & ~benign[vpred], ~(acting & ~benign[vpred]))
            wrong[name] = dict(zip(ids, err))
            out[name] = {"n": len(y), "accuracy": round(float((pred == y).mean()), 4),
                         "nll": round(float(-np.log(np.maximum(P[np.arange(len(y)), y], 1e-12)).mean()), 4),
                         "ece": CAL.reliability(P, y)["ece"],
                         "argmax_attack_to_benign": int((~benign[y] & benign[pred]).sum()),
                         "argmax_benign_to_attack": int((benign[y] & ~benign[pred]).sum()),
                         "policy": {"act_on": act_on, "tau": float(pol["threshold"]), **rates}}
        paired = {}
        for name, *_ in members:  # McNemar-style counts at record level (lineage groups are not in the score files)
            a, b = wrong[name], wrong["ensemble"]
            paired[name] = {"member_wrong_ensemble_right": int(sum(a[i] and not b[i] for i in common)),
                            "member_right_ensemble_wrong": int(sum(b[i] and not a[i] for i in common))}
        out["paired_enforcement_errors_vs_ensemble"] = paired
        rep[split] = out
    return rep


def build(members, out_dir="reports/ensemble", policy="config/policy_ensemble.yaml", act_on="top_class"):
    out = Path(out_dir)
    (ROOT / out).mkdir(parents=True, exist_ok=True)
    merged = str(out / "SCORES_fast.jsonl")
    head, counts = merge(members, merged)
    write_policy_template(policy, head, act_on)
    cal = CAL.main(merged, write_policy=True, out=str(out / "calibration.json"), act_on=act_on, policy=policy,
                   envelope_from=members[0][2])
    tax = yaml.safe_load(open(ROOT / "config/taxonomy.yaml"))["classes"]
    rep = {"members": head["ensemble"]["members"], "records": counts, "policy": policy,
           "temperature": cal["temperature"], "tau": cal["tau"], "act_on": act_on,
           "val_policy_at_tau": cal["val_policy_at_tau"],
           "not_measured": "robustness set (the Gemma member has no robustness scores); flow-envelope abstention on "
                           "the test splits when the build is not on this machine (records_with_flow_features)",
           "records_with_flow_features": cal["records_with_flow_features"],
           **report(members, merged, policy, tax)}
    (ROOT / out / "REPORT.json").write_text(json.dumps(rep, indent=1))
    print(json.dumps({k: rep[k] for k in ("records", "temperature", "tau", "val_policy_at_tau")}, indent=1))
    for split in ("test", "test_unseen"):
        for name, v in rep[split].items():
            if name != "paired_enforcement_errors_vs_ensemble":
                p = v["policy"]
                print(f"{split:11s} {name:20s} acc {v['accuracy']:.4f} nll {v['nll']:.4f} | recall "
                      f"{p['attack_intervention_recall']} benign FI {p['benign_false_intervention']} hard-neg FI "
                      f"{p['hard_negative_false_intervention']} abstain {p['abstention_rate']}")
        print(split, "paired:", rep[split]["paired_enforcement_errors_vs_ensemble"])
    return rep


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["build"])
    ap.add_argument("--member", nargs=3, action="append", metavar=("NAME", "SCORES", "POLICY"), required=True)
    ap.add_argument("--out-dir", default="reports/ensemble")
    ap.add_argument("--policy", default="config/policy_ensemble.yaml")
    ap.add_argument("--act-on", choices=("top_class", "attack_mass"), default="top_class")
    a = ap.parse_args()
    build([tuple(m) for m in a.member], a.out_dir, a.policy, a.act_on)
