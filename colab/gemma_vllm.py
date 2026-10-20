"""The Hub model (hamedgls/waf-gemma4-e2b-v13: merged, text-only) on vLLM vs HF transformers, on a Colab GPU.

Stages, each in its own process (python gemma_vllm.py STAGE) so GPU memory is released between them:
  vllm    vLLM with prefix caching: class scores as single-token requests restricted to the target token (raw log-probs,
          before the restriction; colab/vllm_bench.py's method), same two-pass pruning as hf_scorer.class_scores;
          plus plain greedy generation, whose class must be the scorer's top class on confident records
  hf      HF transformers: hf_scorer.class_scores on the same prompts
  report  both vs the TPU scores of waf/selfcheck.jsonl (hf_scorer.selfcheck's rule: probability difference < 0.05,
          same top class unless the TPU top two are within 0.05), records/s -> /content/gemma_vllm.json
Inputs: HF_TOKEN (private repo), /content/sample.jsonl (telemetry lines for the speed run).
"""
import json
import os
import sys
import time

import numpy as np

REPO, OUT, MARGIN, TOL = os.environ.get("GEMMA_REPO", "hamedgls/waf-gemma4-e2b-v13"), "/content", 12.0, 0.05


def setup():
    from huggingface_hub import snapshot_download
    d = snapshot_download(REPO)
    sys.path.insert(0, f"{d}/waf")
    import yaml
    from transformers import AutoTokenizer
    from training import prepare as P
    from training.gemma_tpu import data as GD
    tok = AutoTokenizer.from_pretrained(d)
    tax = yaml.safe_load(open(f"{d}/waf/config/taxonomy.yaml"))
    contract = json.load(open(f"{d}/waf/serving.json"))["fit_contract"]
    ctrl = P.control_strings(tok)
    checks = [json.loads(line) for line in open(f"{d}/waf/selfcheck.jsonl")]
    sample = [json.loads(line) for line in open(f"{OUT}/sample.jsonl")]
    ids = [GD.prompt_ids(tok, ctrl, c["telemetry"]) for c in checks]
    speed = [GD.prompt_ids(tok, ctrl, P.add_decoded_views(t, contract)) for t in sample]
    speed = [p for p in speed if len(p) <= P.MAX_TOKENS - GD.K]  # the serving budget (longer ones become windows)
    return d, tok, GD.class_suffixes(tok, list(tax["classes"])), ids, speed


def vllm():
    from vllm import LLM, SamplingParams
    from vllm.inputs import TokensPrompt
    d, tok, suf, ids, speed = setup()
    k = 0
    while all(len(s) > k + 1 and s[k] == suf[0][k] for s in suf):
        k += 1
    rest = [s[k:] for s in suf]
    firsts = sorted({r[0] for r in rest})
    llm = LLM(model=d, dtype="bfloat16", enable_prefix_caching=True, max_model_len=2112, gpu_memory_utilization=0.85,
              max_logprobs=1, logprobs_mode="raw_logprobs")

    def lp(pairs):  # [(prefix ids, target id)] -> raw log P(target | prefix)
        sp = [SamplingParams(max_tokens=1, temperature=0.0, logprobs=0, allowed_token_ids=[t], detokenize=False)
              for _, t in pairs]
        outs = llm.generate([TokensPrompt(prompt_token_ids=p) for p, _ in pairs], sp, use_tqdm=False)
        return [o.outputs[0].logprobs[0][t].logprob for o, (_, t) in zip(outs, pairs)]

    def scores(prompts):
        ctx = [p + suf[0][:k] for p in prompts]
        q1 = [(p + suf[0][:j], suf[0][j]) for p in prompts for j in range(k)] + [(c, f) for c in ctx for f in firsts]
        a1 = iter(lp(q1))
        shared = [sum(next(a1) for _ in range(k)) for _ in prompts]
        first = [{f: next(a1) for f in firsts} for _ in ctx]
        out, q2, owner = [], [], []
        for n, c in enumerate(ctx):
            f0 = [first[n][r[0]] for r in rest]
            out.append([shared[n] + v for v in f0])
            for i, v in enumerate(f0):
                if v >= max(f0) - MARGIN:
                    for j in range(1, len(rest[i])):
                        q2.append((c + rest[i][:j], rest[i][j]))
                        owner.append((n, i))
        for (n, i), v in zip(owner, lp(q2) if q2 else []):
            out[n][i] += v
        return out
    sc = scores(ids)
    gen = llm.generate([TokensPrompt(prompt_token_ids=p) for p in ids],
                       SamplingParams(max_tokens=24, temperature=0.0), use_tqdm=False)
    scores(speed[:4])  # warm-up
    t0 = time.time()
    scores(speed)
    t = time.time() - t0
    json.dump({"scores": sc, "generated": [o.outputs[0].text for o in gen], "rec_s": len(speed) / t, "n": len(speed)},
              open(f"{OUT}/vllm_out.json", "w"))
    print(f"vllm {len(speed) / t:.2f} rec/s on {len(speed)} records", flush=True)


def hf():
    import torch
    from transformers import Gemma4ForCausalLM
    d, tok, suf, ids, speed = setup()
    from training.gemma_tpu import hf_scorer as HS  # waf/ is on sys.path after setup()
    model = Gemma4ForCausalLM.from_pretrained(d, dtype=torch.bfloat16, device_map="cuda").eval()
    sc = HS.class_scores(model, ids, suf, tok.pad_token_id)
    HS.class_scores(model, speed[:4], suf, tok.pad_token_id)  # warm-up
    torch.cuda.synchronize()
    t0 = time.time()
    HS.class_scores(model, speed, suf, tok.pad_token_id)
    torch.cuda.synchronize()
    t = time.time() - t0
    json.dump({"scores": sc, "rec_s": len(speed) / t, "n": len(speed)}, open(f"{OUT}/hf_out.json", "w"))
    print(f"hf {len(speed) / t:.2f} rec/s on {len(speed)} records", flush=True)


def report():
    from huggingface_hub import snapshot_download
    import yaml
    d = snapshot_download(REPO)
    T = float(yaml.safe_load(open(f"{d}/waf/config/policy.yaml"))["temperature"])
    classes = list(yaml.safe_load(open(f"{d}/waf/config/taxonomy.yaml"))["classes"])
    checks = [json.loads(line) for line in open(f"{d}/waf/selfcheck.jsonl")]

    def probs(s):
        z = np.asarray(s, np.float64) / T
        e = np.exp(z - z.max())
        return e / e.sum()
    rep = {}
    for name in ("vllm", "hf"):
        r = json.load(open(f"{OUT}/{name}_out.json"))
        rows = []
        for c, s in zip(checks, r["scores"]):
            pg, pt = probs(s), probs(c["scores"])
            top2 = np.sort(pt)[-2:]
            rows.append({"id": c["id"], "diff": round(float(np.abs(pg - pt).max()), 4),
                         "same_top": int(pg.argmax()) == int(pt.argmax()), "tpu_tie": bool(top2[1] - top2[0] < TOL)})
        ok = all(x["diff"] < TOL and (x["same_top"] or x["tpu_tie"]) for x in rows)
        rep[name] = {"pass": ok, "max_diff": max(x["diff"] for x in rows), "rec_s": round(r["rec_s"], 2), "n": r["n"],
                     "rows": rows}
        if name == "vllm":  # greedy generation: the scorer's top class wherever the TPU has a clear winner
            gen = []
            for c, text, x in zip(checks, r["generated"], rows):
                try:
                    g = json.loads(text.split("<turn|>")[0].strip())["threat_type"]
                except Exception:
                    g = None
                gen.append({"id": c["id"], "generated": g, "tpu_top": classes[int(np.argmax(c["scores"]))],
                            "tpu_tie": x["tpu_tie"]})
            rep["vllm_generation"] = {"agree_clear": all(g["generated"] == g["tpu_top"] for g in gen if not g["tpu_tie"]),
                                      "rows": gen}
    json.dump(rep, open(f"{OUT}/gemma_vllm.json", "w"), indent=1)
    print(json.dumps({k: {kk: vv for kk, vv in v.items() if kk != "rows"} for k, v in rep.items()}, indent=1))


if __name__ == "__main__":
    {"vllm": vllm, "hf": hf, "report": report}[sys.argv[1]]()
