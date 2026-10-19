"""Gemma 4 E2B-it data export for the TPU (JAX/Tunix) run. Runs locally on CPU.

datasets_ready/<prefix>{train,val,test,test_unseen}.jsonl -> reports/gemma_tpu/data/<split>_<L>.npz, one file per
length bucket L (fixed shapes: XLA compiles once per bucket). The user turn is re-serialised with Gemma's control
strings (the build escaped Qwen's), so the model input is the same telemetry the Qwen run saw.

Arrays per bucket (right padding with the pad id):
  tokens [N, L] int32; length [N]; tgt_pos [N, K] (position whose output predicts the target token), tgt_tok [N, K],
  tgt_mask [N, K] (the supervised answer + <turn|>); prompt_len [N] (scoring); label [N] (class index); ids;
  split [N] (the record's own split, e.g. val_calib / val_policy: eval.calibration needs it).
meta.json: classes, the class completion suffixes (scoring), shared prefix length, tokenizer fingerprint, counts.

usage: .venv/bin/python -m training.gemma_tpu.data [--prefix fast_] [--max-tokens 2048] [--src DIR] [--out DIR]
       .venv/bin/python -m training.gemma_tpu.data bench [--names gotestwaf,openappsec,unsw] [--robustness]
         -> reports/gemma_tpu/data_bench/: bench_<L>.npz (+ robustness_<L>.npz), meta.json, bench_manifest.json.
         Prompt-only records (no answer), the endpoint's sensor contract: budget_windows over the raw telemetry,
         counted with the decoded views added (each window one record, id "<record>#<k>"), so the TPU scorer
         (train_tunix.py score bench) sees what the endpoint would. eval.public_bench gemma turns the scores into results.
"""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import yaml

from training import prepare as P

ROOT = Path(__file__).resolve().parents[2]
MODEL_ID = "google/gemma-4-E2B-it"
MODEL_REVISION = "3e22461f65e89153144f8adb70e3b8c2cc9845a7"
BUCKETS = (512, 1024, 2048)
K = 20  # max supervised tokens per record (longest class: 17, tls_ja4_ua_spoofing) ({"threat_type":"<class>"} + <turn|>); checked below
TURN_START, TURN_END = "<|turn>", "<turn|>"


def load_tokenizer():
    from transformers import AutoTokenizer
    return AutoTokenizer.from_pretrained(MODEL_ID, revision=MODEL_REVISION)


def encode(tok, ctrl, rec):
    """(ids, n_prompt, target ids incl. <turn|>) for one record, Gemma template, Gemma control strings escaped."""
    msgs = rec["messages"]
    user = P.serialize_telemetry(P.deserialize_telemetry(msgs[1]["content"]), ctrl)
    m = [msgs[0], {"role": "user", "content": user}, msgs[2]]
    prompt = tok.apply_chat_template(m[:2], tokenize=False, add_generation_prompt=True)
    full = tok.apply_chat_template(m, tokenize=False)
    assert full.startswith(prompt), "template prefix mismatch"
    p_ids = tok(prompt, add_special_tokens=False)["input_ids"]
    f_ids = tok(full, add_special_tokens=False)["input_ids"]
    assert f_ids[:len(p_ids)] == p_ids, "prompt tokens not a prefix"
    end = tok.convert_tokens_to_ids(TURN_END)
    last = max(i for i, t in enumerate(f_ids) if t == end)
    ids = f_ids[:last + 1]  # through <turn|>, as supervised; the trailing newline is not trained
    assert ids.count(tok.convert_tokens_to_ids(TURN_START)) == 3, "user content created extra turns"
    return ids, len(p_ids)


def class_suffixes(tok, classes):
    """Completion ids per class through <turn|> (template-fixed boundary, so a dummy prompt suffices)."""
    msgs = [{"role": "system", "content": "s"}, {"role": "user", "content": "u"}]
    p = tok(tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True), add_special_tokens=False)["input_ids"]
    end, out = tok.convert_tokens_to_ids(TURN_END), []
    for c in classes:
        tgt = json.dumps({"threat_type": c}, separators=(",", ":"))
        f = tok(tok.apply_chat_template(msgs + [{"role": "assistant", "content": tgt}], tokenize=False),
                add_special_tokens=False)["input_ids"]
        assert f[:len(p)] == p
        s = f[len(p):]
        out.append(s[:max(i for i, t in enumerate(s) if t == end) + 1])
    return out


def main(prefix="fast_", max_tokens=P.MAX_TOKENS, src="datasets_ready", out="reports/gemma_tpu/data"):
    tok = load_tokenizer()
    ctrl = P.control_strings(tok)
    classes = list(yaml.safe_load(open(ROOT / "config/taxonomy.yaml"))["classes"])
    cidx = {c: i for i, c in enumerate(classes)}
    suf = class_suffixes(tok, classes)
    assert max(map(len, suf)) <= K, f"class completion longer than K={K}"
    out = ROOT / out
    out.mkdir(parents=True, exist_ok=True)
    meta = {"model": MODEL_ID, "revision": MODEL_REVISION, "classes": classes, "suffixes": suf, "K": K,
            "buckets": list(BUCKETS), "pad_id": tok.pad_token_id,
            "chat_template_sha256": hashlib.sha256(tok.chat_template.encode()).hexdigest(),
            "system_prompt_sha256": P.sha256(P.system_prompt()), "splits": {}}
    for split in ("train", "val", "test", "test_unseen"):
        fp = ROOT / src / f"{prefix}{split}.jsonl"
        if not fp.exists():
            continue
        rows, dropped = {b: [] for b in BUCKETS}, 0
        for line in open(fp):
            rec = json.loads(line)
            ids, n_p = encode(tok, ctrl, rec)
            if len(ids) > max_tokens:
                dropped += 1
                continue
            b = next(x for x in BUCKETS if len(ids) <= x)
            rows[b].append((rec["id"], ids, n_p, cidx[rec["metadata"]["threat_type"]], rec["metadata"]["split"]))
        counts = {}
        for b, rs in rows.items():
            if not rs:
                continue
            n = len(rs)
            tokens = np.full((n, b), tok.pad_token_id, np.int32)
            tgt_pos, tgt_tok, tgt_mask = (np.zeros((n, K), np.int32) for _ in range(3))
            for i, (_, ids, n_p, _, _) in enumerate(rs):
                tokens[i, :len(ids)] = ids
                tgt = list(range(n_p, len(ids)))
                assert len(tgt) <= K
                tgt_pos[i, :len(tgt)] = [t - 1 for t in tgt]  # output at t-1 predicts token t
                tgt_tok[i, :len(tgt)] = [ids[t] for t in tgt]
                tgt_mask[i, :len(tgt)] = 1
            np.savez_compressed(out / f"{split}_{b}.npz", tokens=tokens, length=np.array([len(r[1]) for r in rs], np.int32),
                                tgt_pos=tgt_pos, tgt_tok=tgt_tok, tgt_mask=tgt_mask,
                                prompt_len=np.array([r[2] for r in rs], np.int32),
                                label=np.array([r[3] for r in rs], np.int32), ids=np.array([r[0] for r in rs]),
                                split=np.array([r[4] for r in rs]))  # val_select / val_calib / val_policy / test ...
            counts[b] = n
        meta["splits"][split] = {"buckets": counts, "dropped_over_budget": dropped}
        print(split, counts, "dropped", dropped, flush=True)
    (out / "meta.json").write_text(json.dumps(meta, indent=1))


def prompt_ids(tok, ctrl, tel):
    """Serving prompt (system + user, generation prompt), the same template call as encode() makes for training."""
    m = [{"role": "system", "content": P.system_prompt()}, {"role": "user", "content": P.serialize_telemetry(tel, ctrl)}]
    return tok(tok.apply_chat_template(m, tokenize=False, add_generation_prompt=True), add_special_tokens=False)["input_ids"]


def bench_rows(recs, ids_of, budget, contract=1):
    """Sensor contract per record, as deploy/sagemaker/inference.predict_fn applies it: P.budget_windows over the raw
    telemetry, counted with the decoded views added; each window -> its prompt ids (views added). -> (rows, manifest)
    rows: (window id "<record>#<k>", prompt ids); manifest: record id -> windows (0 = fits no budget), truncated."""
    def enriched(x):
        return P.add_decoded_views(json.loads(json.dumps(x)))
    rows, man = [], {}
    for r in recs:
        ws = P.budget_windows(r["telemetry"], lambda x: len(ids_of(enriched(x))), budget, contract=contract)
        man[r["id"]] = {"windows": len(ws), "bench": r.get("bench"),
                        "truncated": bool(r["telemetry"].get("truncated")) or any(bool(w.get("truncated")) for w in ws)}
        rows += [(f"{r['id']}#{k}", ids_of(enriched(w))) for k, w in enumerate(ws)]
    return rows, man


def write_prompt_buckets(out, split, rows, labels, pad_id):
    """Prompt-only npz buckets in the training export's format (train_tunix.load_split / score): the bucket holds
    the prompt plus the longest class completion (K); targets are empty. labels: window id -> class index or -1."""
    by = {b: [] for b in BUCKETS}
    for rid, ids in rows:
        by[next(b for b in BUCKETS if len(ids) + K <= b)].append((rid, ids))
    counts = {}
    for b, rs in by.items():
        if not rs:
            continue
        n = len(rs)
        tokens = np.full((n, b), pad_id, np.int32)
        for i, (_, ids) in enumerate(rs):
            tokens[i, :len(ids)] = ids
        zeros = np.zeros((n, K), np.int32)
        np.savez_compressed(out / f"{split}_{b}.npz", tokens=tokens, length=np.array([len(x[1]) for x in rs], np.int32),
                            tgt_pos=zeros, tgt_tok=zeros, tgt_mask=zeros,
                            prompt_len=np.array([len(x[1]) for x in rs], np.int32),
                            label=np.array([labels.get(x[0], -1) for x in rs], np.int32), ids=np.array([x[0] for x in rs]),
                            split=np.array([split] * n))
        counts[b] = n
    return counts


def bench(names=("gotestwaf", "openappsec", "unsw"), robustness=False, out="reports/gemma_tpu/data_bench",
          max_tokens=P.MAX_TOKENS, contract=1):  # contract: the sensor contract of the build Gemma trained on (v11: 1)
    tok = load_tokenizer()
    ctrl = P.control_strings(tok)
    classes = list(yaml.safe_load(open(ROOT / "config/taxonomy.yaml"))["classes"])
    suf = class_suffixes(tok, classes)
    ref = json.loads((ROOT / "reports/gemma_tpu/data/meta.json").read_text())  # the trained model's export
    sha = hashlib.sha256(tok.chat_template.encode()).hexdigest()
    assert suf == ref["suffixes"] and sha == ref["chat_template_sha256"], "tokenizer differs from the training export"
    out = ROOT / out
    out.mkdir(parents=True, exist_ok=True)
    ids_of = lambda tel: prompt_ids(tok, ctrl, tel)  # noqa: E731
    budget = max_tokens - K  # prompt + the longest class completion fit the largest bucket
    recs = [json.loads(line) for n in names for line in open(ROOT / f"reports/benchmarks/{n}.jsonl")]
    rows, man = bench_rows(recs, ids_of, budget, contract)
    first = lambda e: classes.index(e[0]) if isinstance(e, list) else classes.index("benign_standard_traffic") if e == "benign" else -1  # noqa: E731
    exp = {r["id"]: first(r["expected"]) for r in recs}
    meta = {**{k: ref[k] for k in ("model", "revision", "classes", "suffixes", "K", "buckets", "pad_id",
                                   "chat_template_sha256", "system_prompt_sha256")}, "splits": {}}
    meta["splits"]["bench"] = {"buckets": write_prompt_buckets(out, "bench", rows, {w: exp[w.rpartition("#")[0]] for w, _ in rows},
                                                               tok.pad_token_id), "records": len(recs), "windows": len(rows),
                               "over_budget": sum(m["windows"] == 0 for m in man.values())}
    if robustness:  # the robustness set (training-format records): ensemble robustness needs Gemma scores of it
        rob = [json.loads(line) for line in open(ROOT / "datasets_ready/robustness_pilot.jsonl")]
        rr = [{"id": r["id"], "telemetry": P.deserialize_telemetry(r["messages"][1]["content"])} for r in rob]
        rows_r, man_r = bench_rows(rr, ids_of, budget, contract)
        lab = {r["id"]: classes.index(r["metadata"]["threat_type"]) if r["metadata"]["threat_type"] in classes else -1 for r in rob}
        meta["splits"]["robustness"] = {"buckets": write_prompt_buckets(out, "robustness", rows_r,
                                                                        {w: lab[w.rpartition("#")[0]] for w, _ in rows_r},
                                                                        tok.pad_token_id), "records": len(rr), "windows": len(rows_r)}
        man |= {k: {**v, "bench": "robustness"} for k, v in man_r.items()}
    (out / "meta.json").write_text(json.dumps(meta, indent=1))
    (out / "bench_manifest.json").write_text(json.dumps({"budget": budget, "fit_contract": contract, "records": man}, indent=1))
    print(json.dumps(meta["splits"], indent=1))


if __name__ == "__main__":
    import sys
    if sys.argv[1:2] == ["bench"]:
        ap = argparse.ArgumentParser()
        ap.add_argument("cmd")
        ap.add_argument("--names", default="gotestwaf,openappsec,unsw")
        ap.add_argument("--robustness", action="store_true", help="also datasets_ready/robustness_pilot.jsonl")
        ap.add_argument("--out", default="reports/gemma_tpu/data_bench")
        ap.add_argument("--fit-contract", type=int, default=1, help="sensor contract of the trained model's build (v11: 1)")
        a = ap.parse_args()
        bench(a.names.split(","), a.robustness, a.out, contract=a.fit_contract)
        sys.exit(0)
    ap = argparse.ArgumentParser()
    ap.add_argument("--prefix", default="fast_")
    ap.add_argument("--max-tokens", type=int, default=P.MAX_TOKENS)
    ap.add_argument("--src", default="datasets_ready", help="build directory (e.g. datasets_full)")
    ap.add_argument("--out", default="reports/gemma_tpu/data")
    a = ap.parse_args()
    main(a.prefix, a.max_tokens, a.src, a.out)
