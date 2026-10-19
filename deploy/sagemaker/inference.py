"""SageMaker handler (Hugging Face inference toolkit): the v10 scoring path behind a real-time endpoint.

Same path as the evaluated model: telemetry -> training.prepare messages -> colab_helpers.class_logprobs (summed
completion log-prob of all 41 classes) -> softmax(scores / T) -> eval.policy.decide with the frozen policy.
model_fn loads the pinned base from the Hugging Face Hub, merges the rsLoRA adapter shipped in model.tar.gz and
refuses to start unless 5 test records reproduce the A100 scores (selfcheck.jsonl).

Request:  {"records": [<telemetry object>, ...]}  (or a single telemetry object)
Response: {"results": [{"decision": {...}, "diagnostics": {...}, "top": [[class, p], ...], "latency_s": s}, ...]}
"""
import json
import os
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))  # training/, eval/, config/, schemas/ ship next to this file

import numpy as np  # noqa: E402
import torch  # noqa: E402

from eval import policy as POL  # noqa: E402
from training import colab_helpers as H, prepare as P  # noqa: E402

SELFCHECK_A100_DIFF = 0.05  # probability vs the A100 scores: other GPU, bf16 rounding on uncertain records


def _softmax(s, T):
    z = np.asarray(s, dtype=np.float64) / T
    e = np.exp(z - z.max())
    return e / e.sum()


def model_fn(model_dir, context=None):
    from peft import PeftModel
    from transformers import AutoTokenizer, Qwen3_5ForConditionalGeneration
    adapter = os.path.join(model_dir, "adapter")
    tok = AutoTokenizer.from_pretrained(adapter)
    base = Qwen3_5ForConditionalGeneration.from_pretrained(P.MODEL_ID, revision=P.MODEL_REVISION,
                                                           dtype=torch.bfloat16, device_map="cuda")
    model = PeftModel.from_pretrained(base, adapter).merge_and_unload().eval()
    tax, pol, _ = POL.load(HERE)
    classes = list(tax["classes"])
    assert P.sha256(tok.chat_template) == json.load(open(HERE / "fingerprint.json"))["chat_template_sha256"]
    st = {"model": model, "tok": tok, "tax": tax, "pol": pol, "classes": classes,
          "suf": H.candidate_suffixes(tok, classes), "controls": P.control_strings(tok)}
    checks = [json.loads(line) for line in open(HERE / "selfcheck.jsonl")]
    T = float(pol["temperature"])
    bat = _batch_scores(st, [x["telemetry"] for x in checks])  # the serving path
    seq = [_scores(st, x["telemetry"]) for x in checks]  # the evaluated path

    def gap(a, b):  # (max |p_a - p_b|, same top class, worst class); probabilities, as Part B judges pruning
        pa, pb = _softmax(a, T), _softmax(b, T)
        i = int(np.abs(pa - pb).argmax())
        return float(np.abs(pa - pb).max()), int(pa.argmax()) == int(pb.argmax()), classes[i]
    vs_seq = [gap(b_, s_) for b_, s_ in zip(bat, seq)]
    vs_a100 = [gap(b_, x["scores"]) for b_, x in zip(bat, checks)]
    seq_a100 = [gap(s_, x["scores"]) for s_, x in zip(seq, checks)]
    for name, rows in (("batched vs per-record", vs_seq), ("batched vs A100", vs_a100), ("per-record vs A100", seq_a100)):
        k = max(range(len(rows)), key=lambda i: rows[i][0])
        print(f"[selfcheck] {name}: max prob diff {rows[k][0]:.4f} ({checks[k]['id']}, {rows[k][2]}); "
              f"same top class {sum(r[1] for r in rows)}/{len(rows)}", flush=True)
    fast = {m: m in sys.modules for m in ("fla", "causal_conv1d")}
    try:
        import causal_conv1d  # noqa: F401
        why = "ok"
    except Exception as e:  # the prebuilt wheel must match torch's C++ ABI
        why = f"{type(e).__name__}: {e}"[:200]
    print(f"[selfcheck] kernels {fast}; causal_conv1d import: {why}; torch cxx11 abi "
          f"{torch._C._GLIBCXX_USE_CXX11_ABI}", flush=True)
    # the A100 scores are the evaluated reference: the serving (batched) path must reproduce them. The per-record
    # path is logged only: with the fla kernels its replicated-cache decode drifts more (17 Oct: 0.030 vs batched)
    assert all(r[1] for r in vs_a100) and max(r[0] for r in vs_a100) < SELFCHECK_A100_DIFF, "batched path disagrees with the A100"
    # the default (per-record) path: within the same probability tolerance; its one top-class flip on 17 Oct was the
    # hardest test record, a near tie between two attack classes (both BLOCK)
    assert max(r[0] for r in seq_a100) < SELFCHECK_A100_DIFF, "per-record path disagrees with the A100"
    st["selfcheck_max_diff"] = max(r[0] for r in vs_a100)
    return st


DECODED_VIEWS = os.environ.get("WAF_DECODED_VIEWS", "0") == "1"  # v10 was trained without jwt/base64 views: off


def _enrich(tel):
    return P.add_decoded_views(tel) if DECODED_VIEWS else tel


def _prompt_ids(st, tel):
    msgs = [{"role": "system", "content": P.system_prompt()},
            {"role": "user", "content": P.serialize_telemetry(_enrich(tel), st["controls"])}]
    tok = st["tok"]
    return tok(tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True, enable_thinking=False),
               add_special_tokens=False)["input_ids"]


@torch.no_grad()
def _scores(st, tel):  # the evaluated per-record path (colab_helpers.class_logprobs)
    msgs = [{"role": "system", "content": P.system_prompt()},
            {"role": "user", "content": P.serialize_telemetry(_enrich(tel), st["controls"])}]
    return H.class_logprobs(st["model"], st["tok"], msgs, st["suf"])


@torch.no_grad()
def _logprobs_at(st, seqs, want, max_batch=16):
    """Right-padded batched forward passes (causal, so padding after a sequence never reaches its real tokens);
    the vocabulary projection runs only at the positions in `want` (per sequence: position -> token id or None for
    the full distribution). Returns per sequence {position: log-prob or log-softmax vector}."""
    model, pad = st["model"], st["tok"].pad_token_id
    lm, out = model.model, [None] * len(seqs)
    order = sorted(range(len(seqs)), key=lambda i: len(seqs[i]))
    for s in range(0, len(order), max_batch):
        idx = order[s:s + max_batch]
        L = max(len(seqs[i]) for i in idx)
        ids = torch.full((len(idx), L), pad, device="cuda")
        mask = torch.zeros((len(idx), L), dtype=torch.long, device="cuda")
        for b, i in enumerate(idx):
            ids[b, :len(seqs[i])] = torch.tensor(seqs[i], device="cuda")
            mask[b, :len(seqs[i])] = 1
        hs = lm(input_ids=ids, attention_mask=mask).last_hidden_state
        rows, cols = zip(*[(b, p) for b, i in enumerate(idx) for p in want[i]])
        lp = torch.log_softmax(model.lm_head(hs[list(rows), list(cols)]).float(), -1)
        k = 0
        for b, i in enumerate(idx):
            res = {}
            for p, t in want[i].items():
                res[p] = lp[k] if t is None else lp[k, t].item()
                k += 1
            out[i] = res
    return out


def _batch_scores(st, tels, margin=12.0):
    """class_logprobs for many records at once, same scores (same pruning): pass 1 scores the shared answer prefix
    and the first class token for every record; pass 2 the remaining suffix tokens of each record's kept classes."""
    suf = st["suf"]
    k = 0
    while all(len(s) > k + 1 and s[k] == suf[0][k] for s in suf):
        k += 1
    rest = [s[k:] for s in suf]
    prompts = [_prompt_ids(st, t) for t in tels]
    ctx = [p + suf[0][:k] for p in prompts]
    want1 = [{len(p) - 1 + j: suf[0][j] for j in range(k)} | {len(c) - 1: None} for p, c in zip(prompts, ctx)]
    r1 = _logprobs_at(st, ctx, want1)
    scores, seqs2, want2, owner = [], [], [], []
    for n, (c, res) in enumerate(zip(ctx, r1)):
        first = res.pop(len(c) - 1)
        shared = sum(res.values())
        f0 = [first[r[0]].item() for r in rest]
        keep = [i for i, v in enumerate(f0) if v >= max(f0) - margin]
        scores.append([shared + v for v in f0])
        for i in keep:
            if len(rest[i]) > 1:
                seqs2.append(c + rest[i])
                want2.append({len(c) + j - 1: rest[i][j] for j in range(1, len(rest[i]))})
                owner.append((n, i))
    for (n, i), res in zip(owner, _logprobs_at(st, seqs2, want2) if seqs2 else []):
        scores[n][i] += sum(res.values())
    return scores


def input_fn(body, content_type="application/json"):
    obj = json.loads(body)
    if isinstance(obj, dict) and "records" in obj:
        return obj["records"], obj.get("scoring", "sequential")
    return [obj], "sequential"  # measured 17 Oct (deploy/speed.py, A10G): per-record 0.27 s/record, batched 0.71-1.06
    # (its second pass recomputes the whole prompt for every kept class; a cached-prefix batch is still to write)


def predict_fn(data, st):
    """Each record becomes the prompts the sensor contract allows (P.budget_windows: itself, run-collapsed, or body
    windows); all prompts of the call are scored together; a record's decision comes from its most attack-like
    window (largest probability mass on attack classes)."""
    records, scoring = data
    out, pol, T = [None] * len(records), st["pol"], float(st["pol"]["temperature"])
    benign = [i for i, c in enumerate(st["classes"]) if st["tax"]["classes"][c]["is_benign"]]
    t0 = time.time()
    prompts, owner = [], []
    for n, tel in enumerate(records):  # malformed telemetry fails alone: the policy's own abstain, never a guess
        try:
            ws = P.budget_windows(tel, lambda x: len(_prompt_ids(st, json.loads(json.dumps(x)))))
            if not ws:  # fits no budget even when fitted (5.6% of open-appsec browsing traffic, ~72 cookies): monitor
                out[n] = {"decision": POL.over_budget_decision(pol), "diagnostics": {"reason": "over_budget"},
                          "truncated": True, "windows": 0}
                continue
            prompts += ws
            owner += [n] * len(ws)
        except Exception as e:
            out[n] = {"decision": None, "error": f"{type(e).__name__}: {e}"[:300]}
    if scoring == "sequential":
        sc = [_scores(st, w) for w in prompts]
    else:
        sc = _batch_scores(st, prompts) if prompts else []
    per = (time.time() - t0) / max(1, len(set(owner)))
    best = {}
    for n, w, s in zip(owner, prompts, sc):
        p = _softmax(s, T)
        risk = 1.0 - float(p[benign].sum())
        if n not in best or risk > best[n][0]:
            best[n] = (risk, w, p)
    for n, (_, tel, p) in best.items():
        conf, i = POL.decision_view(p[None], np.isin(np.arange(len(p)), benign), pol.get("act_on", "top_class"))
        conf, i = float(conf[0]), int(i[0])
        raw = json.dumps({"threat_type": st["classes"][i]}, separators=(",", ":"))
        feats = (tel.get("flow") or {}).get("features")
        d, diag = POL.decide(raw, tel.get("mode"), tel.get("context"), conf, st["tax"], pol, feats,
                             (tel.get("http") or {}).get("headers"))
        top = sorted(zip(st["classes"], p.tolist()), key=lambda kv: -kv[1])[:3]
        out[n] = {"decision": d, "diagnostics": diag, "top": [[c, round(v, 6)] for c, v in top],
                  "windows": owner.count(n), "truncated": bool(tel.get("truncated")), "latency_s": round(per, 3)}
    return {"results": out, "scoring": scoring, "batch_s": round(time.time() - t0, 3),
            "selfcheck_max_diff": st["selfcheck_max_diff"]}


def output_fn(pred, accept="application/json"):
    return json.dumps(pred)
