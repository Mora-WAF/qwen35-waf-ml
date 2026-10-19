"""vLLM vs HF transformers on the same GPU, for the packaged classifier (runs on a Colab VM; driver: colab/run_vllm_bench.sh).

Stages, each in its own process (python vllm_bench.py STAGE), so GPU memory is released between them:
  merge   base model + rsLoRA adapter merged on CPU -> /content/merged (vLLM then serves a plain model)
  hf      HF transformers: the evaluated per-record scorer (colab_helpers.class_logprobs) and the endpoint's batched
          scorer (deploy/sagemaker/inference._batch_scores) and the cached-prefix batch (colab_helpers.batch_class_logprobs)
          on the same records
  vllm    vLLM with prefix caching. prompt_logprobs would skip the prefix cache, so every needed log-prob is one
          single-token request restricted to the target token (allowed_token_ids); vLLM reports the raw log-prob,
          taken before that restriction. Same pruning as class_logprobs (first class token for all 41 classes,
          remaining tokens for classes within 12 nats).
  report  scores vs the A100 run (sample.jsonl) and vs each other, records/s -> /content/vllm_bench.json
Inputs in /content: pkg/ (the SageMaker model.tar.gz: adapter/ + code/), sample.jsonl (telemetry + A100 scores).
"""
import json
import os
import sys
import time

PKG, OUT = "/content/pkg", "/content"
sys.path.insert(0, f"{PKG}/code")
MARGIN = 12.0


def records():
    return [json.loads(line) for line in open(f"{OUT}/sample.jsonl")]


def merge():
    import torch
    from peft import PeftModel
    from transformers import AutoTokenizer, Qwen3_5ForConditionalGeneration
    from training import prepare as P
    base = Qwen3_5ForConditionalGeneration.from_pretrained(P.MODEL_ID, revision=P.MODEL_REVISION, dtype=torch.bfloat16,
                                                           device_map="cpu")
    m = PeftModel.from_pretrained(base, f"{PKG}/adapter").merge_and_unload()
    m.save_pretrained(f"{OUT}/merged")
    AutoTokenizer.from_pretrained(f"{PKG}/adapter").save_pretrained(f"{OUT}/merged")
    print("merged ->", f"{OUT}/merged")


def _prompt_ids(tok, ctrl, tel):
    from training import prepare as P
    msgs = [{"role": "system", "content": P.system_prompt()}, {"role": "user", "content": P.serialize_telemetry(tel, ctrl)}]
    return tok(tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True, enable_thinking=False),
               add_special_tokens=False)["input_ids"]


def hf():
    import torch
    import yaml
    from transformers import AutoTokenizer, Qwen3_5ForConditionalGeneration
    from training import colab_helpers as H, prepare as P
    import inference as I
    tok = AutoTokenizer.from_pretrained(f"{OUT}/merged")
    model = Qwen3_5ForConditionalGeneration.from_pretrained(f"{OUT}/merged", dtype=torch.bfloat16, device_map="cuda").eval()
    classes = list(yaml.safe_load(open(f"{PKG}/code/config/taxonomy.yaml"))["classes"])
    st = {"model": model, "tok": tok, "suf": H.candidate_suffixes(tok, classes), "controls": P.control_strings(tok)}
    recs = records()
    tels = [r["telemetry"] for r in recs]
    I._batch_scores(st, tels[:4]), H.class_logprobs(model, tok, [{"role": "system", "content": P.system_prompt()},
                                                                 {"role": "user", "content": P.serialize_telemetry(tels[0], st["controls"])}], st["suf"])
    torch.cuda.synchronize()
    t0 = time.time()
    seq = [H.class_logprobs(model, tok, [{"role": "system", "content": P.system_prompt()},
                                         {"role": "user", "content": P.serialize_telemetry(t, st["controls"])}], st["suf"])
           for t in tels]
    torch.cuda.synchronize()
    t_seq = time.time() - t0
    t0 = time.time()
    bat = []
    for i in range(0, len(tels), 32):
        bat += I._batch_scores(st, tels[i:i + 32])
    torch.cuda.synchronize()
    t_bat = time.time() - t0
    msgs = [[{"role": "system", "content": P.system_prompt()},
             {"role": "user", "content": P.serialize_telemetry(t, st["controls"])}] for t in tels]
    cached, t_cached = {}, {}
    for b in (4, 8, 16):  # 18 Oct: batch 32 ran out of memory on the L4 (22 GB) in this stage
        try:
            H.batch_class_logprobs(model, tok, msgs[:b], st["suf"], batch=b, max_rows=32)  # warm-up at this shape
            torch.cuda.synchronize()
            t0 = time.time()
            cached[b] = H.batch_class_logprobs(model, tok, msgs, st["suf"], batch=b, max_rows=32)
            torch.cuda.synchronize()
            t_cached[b] = time.time() - t0
            print(f"hf cached batch {b}: {len(tels) / t_cached[b]:.2f} rec/s", flush=True)
        except torch.OutOfMemoryError:
            cached.pop(b, None)
            print(f"hf cached batch {b}: out of memory", flush=True)
            torch.cuda.empty_cache()
    json.dump({"seq": seq, "batched": bat, "t_seq": t_seq, "t_batched": t_bat, "cached": cached, "t_cached": t_cached,
               "gpu": torch.cuda.get_device_name(),
               "kernels": {m: m in sys.modules for m in ("fla", "causal_conv1d")}}, open(f"{OUT}/hf_scores.json", "w"))
    print(f"hf sequential {len(tels) / t_seq:.2f} rec/s, batched {len(tels) / t_bat:.2f} rec/s")


def vllm():
    import yaml
    from transformers import AutoTokenizer
    from vllm import LLM, SamplingParams
    from vllm.inputs import TokensPrompt
    from training import colab_helpers as H, prepare as P
    tok = AutoTokenizer.from_pretrained(f"{OUT}/merged")
    classes = list(yaml.safe_load(open(f"{PKG}/code/config/taxonomy.yaml"))["classes"])
    suf, ctrl = H.candidate_suffixes(tok, classes), P.control_strings(tok)
    k = 0
    while all(len(s) > k + 1 and s[k] == suf[0][k] for s in suf):
        k += 1
    rest = [s[k:] for s in suf]
    firsts = sorted({r[0] for r in rest})
    llm = LLM(model=f"{OUT}/merged", dtype="bfloat16", enable_prefix_caching=True, max_model_len=P.MAX_TOKENS + 64,
              gpu_memory_utilization=0.85, limit_mm_per_prompt={"image": 0, "video": 0}, max_logprobs=1,
              logprobs_mode="raw_logprobs")  # log-probs before allowed_token_ids masks the vocabulary

    def lp(pairs):  # [(prefix ids, target id)] -> raw log P(target | prefix)
        sp = [SamplingParams(max_tokens=1, temperature=0.0, logprobs=0, allowed_token_ids=[t], detokenize=False)
              for _, t in pairs]
        outs = llm.generate([TokensPrompt(prompt_token_ids=p) for p, _ in pairs], sp, use_tqdm=False)
        return [o.outputs[0].logprobs[0][t].logprob for o, (_, t) in zip(outs, pairs)]
    tels = [r["telemetry"] for r in records()]
    lp([(_prompt_ids(tok, ctrl, tels[0]), suf[0][0])])  # warm-up / graph capture
    t0 = time.time()
    prompts = [_prompt_ids(tok, ctrl, t) for t in tels]
    ctx = [p + suf[0][:k] for p in prompts]
    q1 = [(p + suf[0][:j], suf[0][j]) for p in prompts for j in range(k)] + [(c, f) for c in ctx for f in firsts]
    a1 = iter(lp(q1))
    shared = [sum(next(a1) for _ in range(k)) for _ in prompts]
    first = [{f: next(a1) for f in firsts} for _ in ctx]
    scores, q2, owner = [], [], []
    for n, c in enumerate(ctx):
        f0 = [first[n][r[0]] for r in rest]
        keep = [i for i, v in enumerate(f0) if v >= max(f0) - MARGIN]
        scores.append([shared[n] + v for v in f0])
        for i in keep:
            for j in range(1, len(rest[i])):
                q2.append((c + rest[i][:j], rest[i][j]))
                owner.append((n, i))
    for (n, i), v in zip(owner, lp(q2) if q2 else []):
        scores[n][i] += v
    t = time.time() - t0
    json.dump({"scores": scores, "t": t, "requests": len(q1) + len(q2)}, open(f"{OUT}/vllm_scores.json", "w"))
    print(f"vllm {len(tels) / t:.2f} rec/s ({len(q1) + len(q2)} single-token requests)")


def vllm_gen(K=20):
    """One greedy generation per record with the top-K log-probs at every step (prefill once, vLLM batches records).
    Class scores are rebuilt from it: a class whose completion follows the generated tokens gets their log-probs; at
    the first step where it branches off, its token's top-K log-prob (or, outside the top K, the K-th log-prob: an
    upper bound); the rest of its name is taken as certain (log-prob 0). The policy needs the confidence of the
    emitted class, which this gives exactly for the generated class."""
    import yaml
    from transformers import AutoTokenizer
    from vllm import LLM, SamplingParams
    from vllm.inputs import TokensPrompt
    from training import colab_helpers as H, prepare as P
    tok = AutoTokenizer.from_pretrained(f"{OUT}/merged")
    classes = list(yaml.safe_load(open(f"{PKG}/code/config/taxonomy.yaml"))["classes"])
    suf, ctrl = H.candidate_suffixes(tok, classes), P.control_strings(tok)
    llm = LLM(model=f"{OUT}/merged", dtype="bfloat16", enable_prefix_caching=False, max_model_len=P.MAX_TOKENS + 64,
              gpu_memory_utilization=0.85, limit_mm_per_prompt={"image": 0, "video": 0}, max_logprobs=K,
              logprobs_mode="raw_logprobs")
    sp = SamplingParams(max_tokens=max(len(x) for x in suf), temperature=0.0, logprobs=K, detokenize=False)
    tels = [r["telemetry"] for r in records()]
    llm.generate([TokensPrompt(prompt_token_ids=_prompt_ids(tok, ctrl, tels[0]))], sp, use_tqdm=False)  # warm-up
    t0 = time.time()
    outs = llm.generate([TokensPrompt(prompt_token_ids=_prompt_ids(tok, ctrl, t)) for t in tels], sp, use_tqdm=False)
    t = time.time() - t0
    scores = []
    for o in outs:
        gen, steps = o.outputs[0].token_ids, o.outputs[0].logprobs
        row = []
        for s in suf:
            total = 0.0
            for j, tid in enumerate(s):
                if j >= len(steps):
                    break
                lp = steps[j]
                if tid in lp:
                    total += lp[tid].logprob
                else:  # outside the top K: bounded by the K-th log-prob
                    total += min(v.logprob for v in lp.values())
                    break
                if j >= len(gen) or gen[j] != tid:  # branched off the generated path: rest of the name ~ certain
                    break
            row.append(total)
        scores.append(row)
    json.dump({"scores": scores, "t": t}, open(f"{OUT}/vllm_gen_scores.json", "w"))
    print(f"vllm_gen {len(tels) / t:.2f} rec/s")


def fp():
    """The endpoint's own predict_fn (sensor windows + frozen policy, decoded views per the package's serving.json) on
    the records in /content/fp_in.jsonl.gz (open-appsec legitimate requests, or every public benchmark:
    eval.public_bench pack) -> /content/fp_out.jsonl with each record's calibrated distribution (resumable)."""
    import gzip
    import torch
    import yaml  # noqa: F401
    from pathlib import Path as _P
    from transformers import AutoTokenizer, Qwen3_5ForConditionalGeneration
    from eval import policy as POL
    from training import colab_helpers as H, prepare as P
    import inference as I
    tok = AutoTokenizer.from_pretrained(f"{OUT}/merged")
    model = Qwen3_5ForConditionalGeneration.from_pretrained(f"{OUT}/merged", dtype=torch.bfloat16, device_map="cuda").eval()
    tax, pol, _ = POL.load(_P(f"{PKG}/code"))
    classes = list(tax["classes"])
    st = {"model": model, "tok": tok, "tax": tax, "pol": pol, "classes": classes, "selfcheck_max_diff": None,
          "suf": H.candidate_suffixes(tok, classes), "controls": P.control_strings(tok)}
    out_p = f"{OUT}/fp_out.jsonl"
    done = {json.loads(x)["id"] for x in open(out_p)} if os.path.exists(out_p) else set()
    recs = [r for r in map(json.loads, gzip.open(f"{OUT}/fp_in.jsonl.gz", "rt")) if r["id"] not in done]
    t0 = time.time()
    with open(out_p, "a") as fh:
        for i in range(0, len(recs), 16):
            chunk = recs[i:i + 16]
            res = I.predict_fn(([r["telemetry"] for r in chunk], "cached", True), st)["results"]  # the serving path, + probs
            for r, x in zip(chunk, res):
                fh.write(json.dumps({"id": r["id"], **x}) + "\n")
            fh.flush()
            if i % 800 == 0:
                print(f"fp {len(done) + i + len(chunk)} done, {(i + len(chunk)) / (time.time() - t0):.2f} rec/s", flush=True)


def report():
    import numpy as np
    recs = records()
    ref = [r["scores"] for r in recs]
    def opt(name):
        return json.load(open(f"{OUT}/{name}")) if os.path.exists(f"{OUT}/{name}") else None
    hfs, vl, vg = opt("hf_scores.json"), opt("vllm_scores.json"), opt("vllm_gen_scores.json")
    import yaml
    pol = yaml.safe_load(open(f"{PKG}/code/config/policy.yaml"))  # the packaged run's frozen policy (v10: 1.083 / 0.641)
    T, TAU = float(pol["temperature"]), float(pol["threshold"])

    def probs(s):
        z = np.array(s) / T
        e = np.exp(z - z.max())
        return e / e.sum()

    def cmp(a, b):
        kept = [max(abs(x - y) for x, y in zip(sa, sb) if y > max(sb) - MARGIN) for sa, sb in zip(a, b)]
        pd = [float(np.abs(probs(sa) - probs(sb)).max()) for sa, sb in zip(a, b)]
        act = sum(int((probs(sa).max() >= TAU) == (probs(sb).max() >= TAU)) for sa, sb in zip(a, b))
        return {"same_top": sum(int(np.argmax(sa) == np.argmax(sb)) for sa, sb in zip(a, b)), "n": len(a),
                "same_act_or_abstain": act,
                "max_kept_score_diff": round(max(kept), 4), "median_kept_score_diff": round(float(np.median(kept)), 5),
                "max_prob_diff": round(max(pd), 5)}
    n = len(recs)
    rep = {"n": n, "records_per_s": {}, "agreement": {}}
    if hfs:
        rep |= {"gpu": hfs["gpu"], "hf_kernels": hfs["kernels"]}
        rep["records_per_s"] |= {"hf_sequential": round(n / hfs["t_seq"], 2), "hf_batched": round(n / hfs["t_batched"], 2)}
        rep["agreement"] |= {"hf_sequential_vs_a100": cmp(hfs["seq"], ref), "hf_batched_vs_a100": cmp(hfs["batched"], ref)}
        for b, sc in (hfs.get("cached") or {}).items():
            rep["records_per_s"][f"hf_cached_b{b}"] = round(n / hfs["t_cached"][b], 2)
            rep["agreement"][f"hf_cached_b{b}_vs_a100"] = cmp(sc, ref)
    if vl:
        rep["records_per_s"]["vllm_single_token"] = round(n / vl["t"], 2)
        rep["agreement"]["vllm_single_token_vs_a100"] = cmp(vl["scores"], ref)
    if vg:
        rep["records_per_s"]["vllm_generation"] = round(n / vg["t"], 2)
        rep["agreement"]["vllm_generation_vs_a100"] = cmp(vg["scores"], ref)
    json.dump(rep, open(f"{OUT}/vllm_bench.json", "w"), indent=1)
    print(json.dumps(rep, indent=1))


if __name__ == "__main__":
    {"merge": merge, "hf": hf, "vllm": vllm, "vllm_gen": vllm_gen, "fp": fp, "report": report}[sys.argv[1]]()
