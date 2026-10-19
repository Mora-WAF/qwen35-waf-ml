"""Endpoint package metadata, from the run directory and the bundle the run trained on (deploy.sh).

  fingerprint.json   tokenizer/template fingerprint of the run (inference.model_fn checks it)
  selfcheck.jsonl    test records of the bundle with the run's own scores (one per input mode first, then the 3 hardest)
  serving.json       train/serve parity of the sensor's jwt/base64 decoded views: on when re-applying
                     training.prepare.add_decoded_views to the bundle's HTTP training records changes (almost) none of
                     them, i.e. the build already carried the views (v11); off when many gain a view (v10)

The records come from the bundle, never from datasets_ready/: a v10 package built on a machine whose datasets_ready
holds a later build must still check v10 records against the v10 scores.

usage: python -m deploy.package_meta RUN_DIR BUNDLE_ZIP OUT_DIR
"""
import json
import sys
import zipfile

from training.prepare import add_decoded_views, deserialize_telemetry

MAX_GAINING = 0.005  # share of HTTP training records that may gain a view and still count as trained with views


def views_evidence(tels):
    """(decoded_views, HTTP records, records gaining a view when the sensor views are re-applied)."""
    total = gaining = 0
    for tel in tels:
        h = tel.get("http")
        if not h:
            continue
        total += 1
        before = len(h.get("decoded_views") or [])
        gaining += len(add_decoded_views(json.loads(json.dumps(tel)))["http"].get("decoded_views") or []) > before
    return total > 0 and gaining <= MAX_GAINING * total, total, gaining


def pick_selfcheck(test_rows, scores, n_first=5, n_hard=3):
    """One record per input mode first (then fill to n_first), plus the n_hard records with the smallest top-2 score
    gap: on confident records only the top class is compared, which is ~0 anyway."""
    picked, modes = [], set()
    for r in test_rows:
        tel = deserialize_telemetry(r["messages"][1]["content"])
        if r["id"] in scores and (tel["mode"] not in modes or len(modes) >= 3):
            modes.add(tel["mode"])
            picked.append({"id": r["id"], "telemetry": tel, "scores": scores[r["id"]]})
        if len(picked) == n_first:
            break
    gap = sorted(scores, key=lambda i: (lambda s: s[0] - s[1])(sorted(scores[i], reverse=True)))[:n_hard]
    have = {x["id"] for x in picked}
    for r in test_rows:
        if r["id"] in gap and r["id"] not in have:
            picked.append({"id": r["id"], "telemetry": deserialize_telemetry(r["messages"][1]["content"]),
                           "scores": scores[r["id"]]})
    return picked


def main(run, bundle, out):
    rep = json.load(open(f"{run}/train_fast_report.json"))
    json.dump(rep["fingerprint"], open(f"{out}/fingerprint.json", "w"))
    f = open(f"{run}/SCORES_fast.jsonl")
    f.readline()
    scores = {r["id"]: r["scores"] for r in map(json.loads, f) if r["split"] == "test"}
    z = zipfile.ZipFile(bundle)
    test = [json.loads(line) for line in z.open("datasets_ready/fast_test.jsonl")]
    missing = sum(r["id"] not in scores for r in test)
    assert missing <= 0.01 * len(test), f"{missing} of {len(test)} bundle test records have no score in {run}: run and bundle do not belong together"
    picked = pick_selfcheck(test, scores)
    open(f"{out}/selfcheck.jsonl", "w").write("".join(json.dumps(x) + "\n" for x in picked))
    views, total, gaining = views_evidence(deserialize_telemetry(json.loads(line)["messages"][1]["content"])
                                           for line in z.open("datasets_ready/fast_train.jsonl"))
    serving = {"decoded_views": views, "train_http_records": total, "records_gaining_views": gaining,
               "bundle": bundle, "run": run}
    json.dump(serving, open(f"{out}/serving.json", "w"), indent=1)
    print("selfcheck:", [(x["id"], x["telemetry"]["mode"]) for x in picked])
    print("serving:", serving)


if __name__ == "__main__":
    main(*sys.argv[1:4])
