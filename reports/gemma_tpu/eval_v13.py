"""Gemma v13 (20 Oct) evaluation extras, after eval.calibration: argmax accuracy per class on test / test_unseen vs Gemma
v12 (different build: compare per class, not record by record), and the real observed flows held out in
val_select by the campaign split (CIC-2017 attacks, CIC-2018 benign + slowloris, DNS-EXF).
usage: .venv/bin/python reports/gemma_tpu/eval_v13.py"""
import collections
import json
import sys

import numpy as np

NEW, OLD = "reports/gemma_tpu/colab_v13/SCORES_fast.jsonl", "reports/gemma_tpu/colab_v12/SCORES_fast.jsonl"


def load(path):
    f = open(path)
    classes = json.loads(f.readline())["classes"]
    return classes, [json.loads(line) for line in f]


def per_class(classes, rows, split):
    acc = collections.defaultdict(lambda: [0, 0])
    for r in rows:
        if r["split"] == split:
            acc[r["label"]][0] += classes[int(np.argmax(r["scores"]))] == r["label"]
            acc[r["label"]][1] += 1
    return acc


def main():
    cn, new = load(NEW)
    co, old = load(OLD)
    out = {}
    for split in ("test", "test_unseen"):
        a, b = per_class(cn, new, split), per_class(co, old, split)
        tot = lambda d: sum(v[0] for v in d.values()) / max(1, sum(v[1] for v in d.values()))
        out[split] = {"acc_v13": round(tot(a), 4), "acc_v12": round(tot(b), 4), "n_v13": sum(v[1] for v in a.values()),
                      "per_class": {c: {"v13": f"{a[c][0]}/{a[c][1]}", "v12": f"{b[c][0]}/{b[c][1]}" if c in b else "-"}
                                    for c in sorted(a) if a[c][1] and (a[c][0] < a[c][1] or (c in b and b[c][0] < b[c][1]))}}
    ids = {}
    for line in open("datasets_ready/fast_val.jsonl"):
        r = json.loads(line)
        if r["metadata"]["provenance"] == "observed":
            ids[r["id"]] = r["id"].split(":")[1]
    cic = collections.defaultdict(lambda: [0, 0, collections.Counter()])
    for r in new:
        if r["id"] in ids and r["split"] == "val_select":
            p = cn[int(np.argmax(r["scores"]))]
            k = f"{ids[r['id']]}/{r['label']}"
            cic[k][0] += p == r["label"]
            cic[k][1] += 1
            cic[k][2][p] += 1
    out["observed_val_select"] = {k: {"right": v[0], "n": v[1], "predicted": dict(v[2].most_common(3))} for k, v in sorted(cic.items())}
    json.dump(out, open("reports/gemma_tpu/eval_v13.json", "w"), indent=1)
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    sys.exit(main())
