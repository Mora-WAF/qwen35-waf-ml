"""SageMaker handler (Hugging Face inference toolkit): the evaluated scoring path behind a real-time endpoint.

Same path as the evaluated model: telemetry -> training.prepare messages -> summed completion log-prob of every class in the bundle's taxonomy
(colab_helpers.batch_class_logprobs by default, the per-record class_logprobs' scores without prompt recomputation)
-> softmax(scores / T) -> eval.policy.decide_from_probs with the frozen policy.
model_fn loads the pinned base from the Hugging Face Hub, merges the rsLoRA adapter shipped in model.tar.gz and
refuses to start unless 5 test records reproduce the A100 scores (selfcheck.jsonl).

Request:  {"records": [<telemetry object>, ...], "return_probs": false}  (or a single telemetry object)
Response: {"results": [{"decision": {...}, "diagnostics": {...}, "top": [[class, p], ...], "latency_s": s}, ...]}
          return_probs: each result also carries "probs" (the calibrated distribution of the window that decided, in
          taxonomy order) and "temperature", so another member's distribution can be combined offline (eval.ensemble).
serving.json (written by deploy.sh from the bundle): whether the model was trained with the sensor's jwt/base64
decoded views (WAF_DECODED_VIEWS=0/1 overrides it) and which sensor contract (fit_budget version) its build used.
Ensemble package (deploy.sh ENSEMBLE=1, 19 Oct): gemma/ holds the TPU-trained Gemma 4 E2B adapter (lora.npz, run on
HF transformers by training.gemma_tpu.hf_scorer) and its TPU selfcheck records; config/policy.yaml is the ensemble
policy. Each member's distribution uses its own frozen temperature (the policy's ensemble block), their mean is
re-tempered and decided once (eval.policy.ensemble_probs), as eval.public_bench ensemble does offline. model_fn
refuses to start unless both members reproduce their reference scores.
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


def _member_temperature(pol, i):
    """Temperature of ensemble member i (0 = this Qwen model, 1 = Gemma); a single-model policy's own temperature."""
    return float(pol["ensemble"]["members"][i]["temperature"]) if "ensemble" in pol else float(pol["temperature"])


def _load_gemma(gdir, st):
    """The Gemma member of an ensemble package: base from the Hub + the TPU adapter as hooks, checked against its TPU
    scores (same top class, probability difference < SELFCHECK_A100_DIFF)."""
    from training.gemma_tpu import data as GD, hf_scorer as HS
    model, tok = HS.load(str(gdir / "lora.npz"))
    T = _member_temperature(st["pol"], 1)
    checks = [json.loads(line) for line in open(gdir / "selfcheck.jsonl")]
    diff = HS.selfcheck(model, tok, checks, T, st["classes"], tol=SELFCHECK_A100_DIFF)
    print(f"[selfcheck] gemma on HF vs TPU: max prob diff {diff:.4f}, same top class {len(checks)}/{len(checks)}", flush=True)
    return {"model": model, "tok": tok, "suf": GD.class_suffixes(tok, st["classes"]), "T": T, "selfcheck_max_diff": diff}


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
    T = _member_temperature(pol, 0)  # an ensemble policy's own temperature applies to the mean, not to this member
    tels = [x["telemetry"] for x in checks]
    srv = _cached_scores(st, [json.loads(json.dumps(t)) for t in tels])  # the serving path (19 Oct: cached batch)
    bat = _batch_scores(st, [json.loads(json.dumps(t)) for t in tels])  # the 17 Oct batched path (scoring="batched")
    seq = [_scores(st, json.loads(json.dumps(t))) for t in tels]  # the evaluated path

    def gap(a, b):  # (max |p_a - p_b|, same top class, worst class); probabilities, as Part B judges pruning
        pa, pb = _softmax(a, T), _softmax(b, T)
        i = int(np.abs(pa - pb).argmax())
        return float(np.abs(pa - pb).max()), int(pa.argmax()) == int(pb.argmax()), classes[i]
    vs_seq = [gap(b_, s_) for b_, s_ in zip(srv, seq)]
    vs_a100 = [gap(b_, x["scores"]) for b_, x in zip(srv, checks)]
    bat_a100 = [gap(b_, x["scores"]) for b_, x in zip(bat, checks)]
    seq_a100 = [gap(s_, x["scores"]) for s_, x in zip(seq, checks)]
    for name, rows in (("cached vs per-record", vs_seq), ("cached vs A100", vs_a100), ("batched vs A100", bat_a100),
                       ("per-record vs A100", seq_a100)):
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
    # the A100 scores are the evaluated reference: the serving (cached batch) path must reproduce them (Colab L4,
    # 18 Oct: batch 4 same top class 400/400, max probability difference 0.022); the 17 Oct batched path is logged
    assert all(r[1] for r in vs_a100) and max(r[0] for r in vs_a100) < SELFCHECK_A100_DIFF, "cached batch path disagrees with the A100"
    # the default (per-record) path: within the same probability tolerance; its one top-class flip on 17 Oct was the
    # hardest test record, a near tie between two attack classes (both BLOCK)
    assert max(r[0] for r in seq_a100) < SELFCHECK_A100_DIFF, "per-record path disagrees with the A100"
    st["selfcheck_max_diff"] = max(r[0] for r in vs_a100)
    gdir = Path(model_dir) / "gemma"
    if (gdir / "lora.npz").exists():
        assert "ensemble" in pol, "package has a Gemma member but config/policy.yaml is not an ensemble policy"
        st["gemma"] = _load_gemma(gdir, st)
    else:
        assert "ensemble" not in pol, "ensemble policy without its Gemma member (gemma/lora.npz)"
    return st


def _serving():
    f = HERE / "serving.json"
    return json.load(open(f)) if f.exists() else {}


def _decoded_views():
    """Train/serve parity for the sensor's jwt/base64 views: v10 was trained without them, v11 (18 Oct build) with.
    serving.json records what the bundle's training records carry; without it, off (the v10 packages)."""
    if os.environ.get("WAF_DECODED_VIEWS") in ("0", "1"):
        return os.environ["WAF_DECODED_VIEWS"] == "1"
    return bool(_serving().get("decoded_views"))


def _fit_contract():
    """Sensor contract (training.prepare.SENSOR_CONTRACT) of the build the model trained on; without it 1 (v10, v11)."""
    return int(_serving().get("fit_contract", 1))


DECODED_VIEWS, FIT_CONTRACT = _decoded_views(), _fit_contract()


def _enrich(tel):
    return P.add_decoded_views(tel, FIT_CONTRACT) if DECODED_VIEWS else tel


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


CACHED_BATCH = int(os.environ.get("WAF_CACHED_BATCH", 4))  # 18 Oct, Colab L4: 4.17 / 3.95 / 3.71 rec/s at 4 / 8 / 16


@torch.no_grad()
def _cached_scores(st, tels):
    """The serving path (19 Oct): colab_helpers.batch_class_logprobs, same scores and pruning as the per-record
    class_logprobs without recomputing any prompt (left-padded prefill of CACHED_BATCH prompts, class suffixes from each
    record's replicated cache). 18 Oct, L4: 4.17 rec/s at batch 4 vs 3.52 per record."""
    msgs = [[{"role": "system", "content": P.system_prompt()},
             {"role": "user", "content": P.serialize_telemetry(_enrich(t), st["controls"])}] for t in tels]
    return H.batch_class_logprobs(st["model"], st["tok"], msgs, st["suf"], batch=CACHED_BATCH, max_rows=32)


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
        return obj["records"], obj.get("scoring", "cached"), bool(obj.get("return_probs"))
    return [obj], "cached", False
    # scoring: "cached" (default since 19 Oct: batch_class_logprobs), "sequential" (the evaluated per-record path,
    # 0.27 s/record on the A10G, 17 Oct), "batched" (17 Oct: 0.71-1.06 s/record, its second pass recomputes every prompt)


def predict_fn(data, st):
    """Each record becomes the prompts the sensor contract allows (P.budget_windows: itself, run-collapsed, or body
    windows); all prompts of the call are scored together; a record's decision comes from its most attack-like
    window (largest probability mass on attack classes)."""
    records, scoring, return_probs = (*data, False)[:3]
    out, pol, T = [None] * len(records), st["pol"], _member_temperature(st["pol"], 0)
    benign = [i for i, c in enumerate(st["classes"]) if st["tax"]["classes"][c]["is_benign"]]
    t0 = time.time()
    prompts, owner = [], []
    for n, tel in enumerate(records):  # malformed telemetry fails alone: the policy's own abstain, never a guess
        try:
            ws = P.budget_windows(tel, lambda x: len(_prompt_ids(st, json.loads(json.dumps(x)))), contract=FIT_CONTRACT)
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
    elif scoring == "batched":
        sc = _batch_scores(st, prompts) if prompts else []
    else:
        sc = _cached_scores(st, prompts) if prompts else []
    per = (time.time() - t0) / max(1, len(set(owner)))
    best = {}
    for n, w, s in zip(owner, prompts, sc):
        p = _softmax(s, T)
        risk = 1.0 - float(p[benign].sum())
        if n not in best or risk > best[n][0]:
            best[n] = (risk, w, p)
    gp = _gemma_probs(st, records, best) if st.get("gemma") else {}
    for n, (_, tel, p) in best.items():
        member = None
        if st.get("gemma"):  # ensemble: both members' distributions, decided once on the whole record
            if gp[n] is None:  # Gemma's token budget cannot fit it: never guessed (eval.public_bench ensemble)
                out[n] = {"decision": POL.over_budget_decision(pol), "diagnostics": {"reason": "over_budget"},
                          "truncated": True, "windows": owner.count(n)}
                continue
            member, tel = (p, gp[n]), records[n]
            p = POL.ensemble_probs([p, gp[n]], pol)
        d, diag, top = POL.decide_from_probs(p, tel, st["tax"], pol)  # the same path as the offline re-decisions
        out[n] = {"decision": d, "diagnostics": diag, "top": top,
                  "windows": owner.count(n), "truncated": bool(tel.get("truncated")), "latency_s": round(per, 3)}
        if return_probs:
            out[n] |= {"probs": [round(float(v), 7) for v in p],
                       "temperature": float(pol["temperature"]) if member else T}
            if member:
                out[n]["member_probs"] = [[round(float(v), 7) for v in q] for q in member]
    return {"results": out, "scoring": scoring, "batch_s": round(time.time() - t0, 3),
            "selfcheck_max_diff": st["selfcheck_max_diff"], "decoded_views": DECODED_VIEWS, "fit_contract": FIT_CONTRACT}


def _gemma_probs(st, records, best):
    """{record index: Gemma's calibrated distribution of its most attack-like window, or None (fits no budget)}."""
    from training.gemma_tpu import hf_scorer as HS
    g = st["gemma"]
    recs = [{"id": str(n), "telemetry": json.loads(json.dumps(records[n]))} for n in best]
    probs, _ = HS.record_probs(g["model"], g["tok"], recs, st["tax"], g["T"], g["suf"], contract=FIT_CONTRACT)
    return {int(k): v for k, v in probs.items()}


def output_fn(pred, accept="application/json"):
    return json.dumps(pred)
