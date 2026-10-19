"""The TPU-trained Gemma 4 E2B-it adapter on Hugging Face transformers (GPU, or CPU for tests), scored exactly like the
TPU scorer (train_tunix.score): for the public-benchmark run on one Colab GPU and for an ensemble endpoint.

The adapter (reports/gemma_tpu/colab_full/lora.npz) holds qwix einsum factors. They are re-arranged, never re-fitted,
into nn.Linear factors A [r, in], B [out, r] of the same layer's HF modules:
  attn.q_einsum        a (D, r),    b (r, N, H)      -> self_attn.q_proj   A = a.T,                 B = b.reshape(r, N H).T
  attn.kv_einsum       a (D, r),    b (r, 2, K, H)   -> self_attn.k_proj / v_proj (the fused K/V shares a):
                                                        A = a.T,  B = b[:, 0|1].reshape(r, K H).T
  attn.attn_vec_einsum a (N, H, r), b (r, D)         -> self_attn.o_proj   A = a.reshape(N H, r).T, B = b.T
  mlp.{gate,up,down}_proj kernel a (in, r), b (r, out) -> mlp.*_proj       A = a.T,                 B = b.T
and applied unmerged as forward hooks y += SCALE * B (A x) in the model dtype, as the TPU computed them (bf16 LoRA on
bf16 weights; qwix alpha 8 r / rank r = 8, the rsLoRA scale of the Qwen run). KV-shared layers (15-34 in E2B) have
no k/v projections in either stack; global layers use head_dim 512 in both. A wrong mapping or scale shows at once:
`selfcheck` compares the class distributions with the TPU scores of the same test records.

Class scores: summed completion log-probs of every class answer ({"threat_type":"<class>"} + <turn|>), two passes
as on the TPU: prompt + the shared answer prefix with the full first-token distribution, then the remaining tokens of
the classes within `margin` nats of the best first token (prompt recomputed, right padding, final-logit softcap).
"""
import re

import numpy as np
import torch

SCALE = 8.0
_KEY = re.compile(r"\['layers'\]\[(\d+)\]\['(attn|mlp)'\]\['(\w+)'\]\['(\w+)_lora_(a|b)'\]")


def lora_factors(npz):
    """lora.npz -> {(layer, "self_attn.q_proj" | ...): (A [r, in], B [out, r])} as float32 numpy arrays."""
    z = np.load(npz)
    raw = {}
    for k in z.files:
        m = _KEY.match(k)
        assert m, f"unexpected adapter key {k}"
        layer, _, mod, _, ab = m.groups()
        raw.setdefault((int(layer), mod), {})[ab] = np.asarray(z[k], np.float32)
    out = {}
    for (layer, mod), f in raw.items():
        a, b = f["a"], f["b"]
        r = b.shape[0]
        if mod == "q_einsum":
            out[(layer, "self_attn.q_proj")] = (a.T, b.reshape(r, -1).T)
        elif mod == "kv_einsum":
            out[(layer, "self_attn.k_proj")] = (a.T, b[:, 0].reshape(r, -1).T)
            out[(layer, "self_attn.v_proj")] = (a.T, b[:, 1].reshape(r, -1).T)
        elif mod == "attn_vec_einsum":
            out[(layer, "self_attn.o_proj")] = (a.reshape(-1, a.shape[-1]).T, b.T)
        elif mod in ("gate_proj", "up_proj", "down_proj"):
            out[(layer, f"mlp.{mod}")] = (a.T, b.T)
        else:
            raise ValueError(f"unmapped adapter module {mod}")
    return out


def text_parts(model):
    """(decoder with .layers, lm_head, final-logit softcap) of a Gemma4ForConditionalGeneration or Gemma4ForCausalLM."""
    inner = model.model
    text = getattr(inner, "language_model", inner)
    cfg = model.config.get_text_config() if hasattr(model.config, "get_text_config") else model.config
    return text, model.lm_head, cfg.final_logit_softcapping


def attach(model, factors, scale=SCALE):
    """Forward hooks y += scale * B (A x) on the mapped modules; returns the hook handles. Shapes are checked."""
    text, _, _ = text_parts(model)
    handles = []
    for (layer, path), (A, B) in sorted(factors.items()):
        mod = text.layers[layer]
        for part in path.split("."):
            mod = getattr(mod, part, None)
            assert mod is not None, f"layer {layer} has no {path} (KV-shared layer or another architecture)"
        w = mod.weight
        assert tuple(w.shape) == (B.shape[0], A.shape[1]), f"{layer}.{path}: weight {tuple(w.shape)} vs LoRA {B.shape[0]}x{A.shape[1]}"
        At = torch.tensor(A, dtype=w.dtype, device=w.device)
        Bt = torch.tensor(B, dtype=w.dtype, device=w.device)

        def hook(m, args, out, At=At, Bt=Bt):
            return out + scale * torch.nn.functional.linear(torch.nn.functional.linear(args[0], At), Bt)
        handles.append(mod.register_forward_hook(hook))
    return handles


@torch.no_grad()
def _logprobs_at(model, seqs, want, pad_id, batch=8):
    """Right-padded batched forward passes; want[i] = {position: token id or None (full log-softmax vector)}."""
    text, head, cap = text_parts(model)
    dev = next(model.parameters()).device
    out = [None] * len(seqs)
    order = sorted(range(len(seqs)), key=lambda i: len(seqs[i]))
    for s in range(0, len(order), batch):
        idx = order[s:s + batch]
        L = max(len(seqs[i]) for i in idx)
        ids = torch.full((len(idx), L), pad_id, device=dev)
        mask = torch.zeros((len(idx), L), dtype=torch.long, device=dev)
        for b, i in enumerate(idx):
            ids[b, :len(seqs[i])] = torch.tensor(seqs[i], device=dev)
            mask[b, :len(seqs[i])] = 1
        hs = text(input_ids=ids, attention_mask=mask).last_hidden_state
        rows, cols = zip(*[(b, p) for b, i in enumerate(idx) for p in want[i]])
        lg = head(hs[list(rows), list(cols)]).float()
        if cap:
            lg = torch.tanh(lg / cap) * cap
        lp = torch.log_softmax(lg, -1)
        k = 0
        for b, i in enumerate(idx):
            res = {}
            for p, t in want[i].items():
                res[p] = lp[k] if t is None else lp[k, t].item()
                k += 1
            out[i] = res
    return out


def class_scores(model, prompts, suffixes, pad_id, margin=12.0, batch=8):
    """Summed completion log-prob of every class per prompt (lists of token ids), train_tunix.score's two passes."""
    k = 0
    while all(len(s) > k + 1 and s[k] == suffixes[0][k] for s in suffixes):
        k += 1
    rest = [s[k:] for s in suffixes]
    ctx = [p + suffixes[0][:k] for p in prompts]
    want1 = [{len(p) - 1 + j: suffixes[0][j] for j in range(k)} | {len(c) - 1: None} for p, c in zip(prompts, ctx)]
    scores, seqs2, want2, owner = [], [], [], []
    for n, (c, res) in enumerate(zip(ctx, _logprobs_at(model, ctx, want1, pad_id, batch))):
        first = res.pop(len(c) - 1)
        shared = sum(res.values())
        f0 = [first[r[0]].item() for r in rest]
        scores.append([shared + v for v in f0])
        for i, v in enumerate(f0):
            if (margin is None or v >= max(f0) - margin) and len(rest[i]) > 1:
                seqs2.append(c + rest[i])
                want2.append({len(c) + j - 1: rest[i][j] for j in range(1, len(rest[i]))})
                owner.append((n, i))
    for (n, i), res in zip(owner, _logprobs_at(model, seqs2, want2, pad_id, batch) if seqs2 else []):
        scores[n][i] += sum(res.values())
    return scores


def load(npz, model_id=None, revision=None, device="cuda", dtype=torch.bfloat16):
    """The pinned base (training.gemma_tpu.data MODEL_ID / MODEL_REVISION) with the adapter hooked in, + tokenizer."""
    from transformers import AutoTokenizer, Gemma4ForConditionalGeneration
    from training.gemma_tpu import data as GD
    model_id, revision = model_id or GD.MODEL_ID, revision or GD.MODEL_REVISION
    tok = AutoTokenizer.from_pretrained(model_id, revision=revision)
    model = Gemma4ForConditionalGeneration.from_pretrained(model_id, revision=revision, dtype=dtype, device_map=device).eval()
    attach(model, lora_factors(npz))
    return model, tok


def selfcheck(model, tok, checks, temperature, classes, tol=0.05):
    """checks: [{"telemetry", "scores"}] with the TPU scores of the same records. Asserts the same top class and a
    calibrated-probability difference below tol on every record (the Qwen endpoint's rule); returns the max diff."""
    from training import prepare as P
    from training.gemma_tpu import data as GD
    ctrl = P.control_strings(tok)
    suf = GD.class_suffixes(tok, classes)
    got = class_scores(model, [GD.prompt_ids(tok, ctrl, c["telemetry"]) for c in checks], suf, tok.pad_token_id)

    diffs = []
    for g, c in zip(got, checks):
        pg, pt = _probs(g, temperature), _probs(c["scores"], temperature)
        assert int(pg.argmax()) == int(pt.argmax()), f"Gemma on HF disagrees with the TPU on {c.get('id')}"
        diffs.append(float(np.abs(pg - pt).max()))
    assert max(diffs) < tol, f"Gemma on HF drifts from the TPU scores: max probability difference {max(diffs):.4f}"
    return max(diffs)


def _probs(s, T):
    z = np.asarray(s, np.float64) / T
    e = np.exp(z - z.max())
    return e / e.sum()


def record_probs(model, tok, recs, tax, temperature, suffixes, contract=1, budget=None):
    """Records ({id, telemetry}) -> ({id: calibrated distribution of the most attack-like window, or None when the
    record fits no budget}, sensor manifest): the sensor contract's windows by Gemma's own token budget, decoded views."""
    from training import prepare as P
    from training.gemma_tpu import data as GD
    ctrl = P.control_strings(tok)
    benign = np.array([tax["classes"][c]["is_benign"] for c in tax["classes"]])
    rows, man = GD.bench_rows(recs, lambda tel: GD.prompt_ids(tok, ctrl, tel), budget or P.MAX_TOKENS - GD.K, contract=contract)
    win = {}
    for (wid, _), s in zip(rows, class_scores(model, [ids for _, ids in rows], suffixes, tok.pad_token_id)):
        win.setdefault(wid.rpartition("#")[0], []).append(_probs(s, temperature))
    return {r["id"]: (max(win[r["id"]], key=lambda q: 1.0 - float(q[benign].sum())) if man[r["id"]]["windows"] else None)
            for r in recs}, man


def score_records(model, tok, recs, tax, pol, suffixes, contract=1, budget=None):
    """Benchmark records ({id, telemetry}) -> endpoint-style results, as deploy/sagemaker/inference.predict_fn decides:
    record_probs, then eval.policy.decide_from_probs with Gemma's frozen policy; a record that fits no budget ->
    over_budget."""
    from eval import policy as POL
    T = float(pol["temperature"])
    probs, man = record_probs(model, tok, recs, tax, T, suffixes, contract, budget)
    out = []
    for r in recs:
        m = man[r["id"]]
        if m["windows"] == 0:
            out.append({"id": r["id"], "decision": POL.over_budget_decision(pol), "diagnostics": {"reason": "over_budget"},
                        "truncated": True, "windows": 0})
            continue
        p = probs[r["id"]]
        d, diag, top = POL.decide_from_probs(p, r["telemetry"], tax, pol)
        out.append({"id": r["id"], "decision": d, "diagnostics": diag, "top": top, "windows": m["windows"],
                    "truncated": m["truncated"], "probs": [round(float(v), 7) for v in p], "temperature": T})
    return out


def robustness_scores(model, tok, rob, tax, temperature, suffixes, contract=1, budget=None):
    """Robustness probes ({id, split, kind, label, telemetry}) -> rows of the notebook's score-file format (raw class
    scores of the most attack-like window) for eval.ensemble; probes that fit no budget are left out."""
    from training import prepare as P
    from training.gemma_tpu import data as GD
    ctrl = P.control_strings(tok)
    benign = np.array([tax["classes"][c]["is_benign"] for c in tax["classes"]])
    rows, man = GD.bench_rows(rob, lambda tel: GD.prompt_ids(tok, ctrl, tel), budget or P.MAX_TOKENS - GD.K, contract=contract)
    sc = dict(zip((w for w, _ in rows), class_scores(model, [ids for _, ids in rows], suffixes, tok.pad_token_id)))
    out = []
    for r in rob:
        ws = [sc[f"{r['id']}#{k}"] for k in range(man[r["id"]]["windows"])]
        if ws:
            s = max(ws, key=lambda v: 1.0 - float(_probs(v, temperature)[benign].sum()))
            out.append({"id": r["id"], "split": r["split"], "kind": r.get("kind"), "label": r["label"], "scores": s})
    return out


def selfcheck_records(bundle, scores, out, n_first=5, n_hard=3):
    """Test records of the bundle Gemma trained on, with Gemma's TPU scores (one per input mode, then the hardest):
    the reference selfcheck() needs on the GPU. Same picks as the Qwen endpoint's self-check."""
    import json
    import zipfile
    from deploy.package_meta import pick_selfcheck
    f = open(scores)
    f.readline()
    sc = {r["id"]: r["scores"] for r in map(json.loads, f) if r["split"] == "test"}
    test = [json.loads(line) for line in zipfile.ZipFile(bundle).open("datasets_ready/fast_test.jsonl")]
    picked = pick_selfcheck(test, sc, n_first, n_hard)
    open(out, "w").write("".join(json.dumps(x) + "\n" for x in picked))
    print(f"{len(picked)} selfcheck records -> {out}")


def robustness_input(src, out):
    """datasets_ready/robustness_pilot.jsonl -> {id, split, kind, label, telemetry} lines for the gemma_fp stage."""
    import json
    from training.prepare import deserialize_telemetry
    n = 0
    with open(out, "w") as fh:
        for line in open(src):
            r = json.loads(line)
            fh.write(json.dumps({"id": r["id"], "split": "robustness", "kind": r["metadata"].get("robustness_kind"),
                                 "label": r["metadata"]["threat_type"],
                                 "telemetry": deserialize_telemetry(r["messages"][1]["content"])}) + "\n")
            n += 1
    print(f"{n} robustness probes -> {out}")


if __name__ == "__main__":
    import sys
    if sys.argv[1:2] == ["selfcheck-records"]:  # selfcheck-records BUNDLE_ZIP GEMMA_SCORES OUT
        selfcheck_records(*sys.argv[2:5])
    elif sys.argv[1:2] == ["robustness-input"]:  # robustness-input ROBUSTNESS_JSONL OUT
        robustness_input(*sys.argv[2:4])
