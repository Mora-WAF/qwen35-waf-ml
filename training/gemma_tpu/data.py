"""Gemma 4 E2B-it data export for the TPU (JAX/Tunix) run. Runs locally on CPU.

datasets_ready/<prefix>{train,val,test,test_unseen}.jsonl -> reports/gemma_tpu/data/<split>_<L>.npz, one file per
length bucket L (fixed shapes: XLA compiles once per bucket). The user turn is re-serialised with Gemma's control
strings (the build escaped Qwen's), so the model input is the same telemetry the Qwen run saw.

Arrays per bucket (right padding with the pad id):
  tokens [N, L] int32; length [N]; tgt_pos [N, K] (position whose output predicts the target token), tgt_tok [N, K],
  tgt_mask [N, K] (the supervised answer + <turn|>); prompt_len [N] (scoring); label [N] (class index); ids;
  split [N] (the record's own split, e.g. val_calib / val_policy: eval.calibration needs it).
meta.json: classes, the class completion suffixes (scoring), shared prefix length, tokenizer fingerprint, counts.

usage: .venv/bin/python -m training.gemma_tpu.data [--prefix fast_] [--max-tokens 2048]
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


def main(prefix="fast_", max_tokens=P.MAX_TOKENS):
    tok = load_tokenizer()
    ctrl = P.control_strings(tok)
    classes = list(yaml.safe_load(open(ROOT / "config/taxonomy.yaml"))["classes"])
    cidx = {c: i for i, c in enumerate(classes)}
    suf = class_suffixes(tok, classes)
    assert max(map(len, suf)) <= K, f"class completion longer than K={K}"
    out = ROOT / "reports/gemma_tpu/data"
    out.mkdir(parents=True, exist_ok=True)
    meta = {"model": MODEL_ID, "revision": MODEL_REVISION, "classes": classes, "suffixes": suf, "K": K,
            "buckets": list(BUCKETS), "pad_id": tok.pad_token_id,
            "chat_template_sha256": hashlib.sha256(tok.chat_template.encode()).hexdigest(),
            "system_prompt_sha256": P.sha256(P.system_prompt()), "splits": {}}
    for split in ("train", "val", "test", "test_unseen"):
        fp = ROOT / f"datasets_ready/{prefix}{split}.jsonl"
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


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--prefix", default="fast_")
    ap.add_argument("--max-tokens", type=int, default=P.MAX_TOKENS)
    a = ap.parse_args()
    main(a.prefix, a.max_tokens)
