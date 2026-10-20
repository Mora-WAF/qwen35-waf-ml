"""Gemma 4 adapter on Hugging Face (training.gemma_tpu.hf_scorer): the qwix einsum factors re-arranged into linear LoRA
factors must reproduce the einsum deltas, the hooks must equal merged weights, and the two-pass class scorer must
equal brute-force completion log-probs (final-logit softcap, right padding). Tiny random Gemma 4 on CPU."""
import copy

import numpy as np
import pytest
import torch

from training.gemma_tpu import hf_scorer as HS

LAYER_TYPES = ["sliding_attention", "sliding_attention", "full_attention", "sliding_attention", "sliding_attention",
               "full_attention", "sliding_attention", "full_attention"]  # layers 5-7 share K/V, like E2B's 15-34


def _key(layer, grp, mod, kind, ab):
    return f"['layers'][{layer}]['{grp}']['{mod}']['{kind}_lora_{ab}'].value"


def test_einsum_factors_reproduce_the_tpu_deltas(tmp_path):
    rng = np.random.default_rng(0)
    D, r, N, H, K, F = 6, 3, 2, 4, 1, 10
    a_q, b_q = rng.normal(size=(D, r)), rng.normal(size=(r, N, H))
    a_kv, b_kv = rng.normal(size=(D, r)), rng.normal(size=(r, 2, K, H))
    a_o, b_o = rng.normal(size=(N, H, r)), rng.normal(size=(r, D))
    a_g, b_g = rng.normal(size=(D, r)), rng.normal(size=(r, F))
    np.savez(tmp_path / "lora.npz", **{_key(0, "attn", "q_einsum", "w", "a"): a_q, _key(0, "attn", "q_einsum", "w", "b"): b_q,
                                       _key(0, "attn", "kv_einsum", "w", "a"): a_kv, _key(0, "attn", "kv_einsum", "w", "b"): b_kv,
                                       _key(0, "attn", "attn_vec_einsum", "w", "a"): a_o, _key(0, "attn", "attn_vec_einsum", "w", "b"): b_o,
                                       _key(0, "mlp", "gate_proj", "kernel", "a"): a_g, _key(0, "mlp", "gate_proj", "kernel", "b"): b_g})
    f = HS.lora_factors(tmp_path / "lora.npz")
    x, xh = rng.normal(size=D), rng.normal(size=(N, H))

    def lin(name, v):  # the HF side: B (A v)
        A, B = f[(0, name)]
        return B @ (A @ v)
    # the TPU side: the qwix LoRA path of each einsum (q: BTD,NDH->BTNH; kv: BTD,CKDH->CBTKH; o: BTNH,NHD->BTD)
    assert np.allclose(lin("self_attn.q_proj", x), np.einsum("d,dr,rnh->nh", x, a_q, b_q).reshape(-1))
    kv = np.einsum("d,dr,rckh->ckh", x, a_kv, b_kv)
    assert np.allclose(lin("self_attn.k_proj", x), kv[0].reshape(-1)) and np.allclose(lin("self_attn.v_proj", x), kv[1].reshape(-1))
    assert np.allclose(lin("self_attn.o_proj", xh.reshape(-1)), np.einsum("nh,nhr,rd->d", xh, a_o, b_o))
    assert np.allclose(lin("mlp.gate_proj", x), x @ a_g @ b_g)


def _tiny():
    from transformers import Gemma4ForCausalLM, Gemma4TextConfig
    cfg = Gemma4TextConfig(vocab_size=300, hidden_size=32, intermediate_size=64, num_hidden_layers=8, num_attention_heads=2,
                           num_key_value_heads=1, head_dim=16, global_head_dim=32, hidden_size_per_layer_input=8,
                           vocab_size_per_layer_input=300, num_kv_shared_layers=3, use_double_wide_mlp=True,
                           layer_types=LAYER_TYPES, final_logit_softcapping=30.0, sliding_window=8, pad_token_id=0)
    torch.manual_seed(0)
    return Gemma4ForCausalLM(cfg).float().eval()


def _tpu_layout_adapter(model, path, r=4, seed=1):
    """Random factors in the qwix layout for every projection the tiny model has (no k/v in KV-shared layers)."""
    rng, z = np.random.default_rng(seed), {}
    text, _, _ = HS.text_parts(model)
    N = model.config.num_attention_heads
    for i, layer in enumerate(text.layers):
        att, D = layer.self_attn, model.config.hidden_size
        H = att.q_proj.weight.shape[0] // N
        z[_key(i, "attn", "q_einsum", "w", "a")] = rng.normal(0, .3, (D, r))
        z[_key(i, "attn", "q_einsum", "w", "b")] = rng.normal(0, .3, (r, N, H))
        z[_key(i, "attn", "attn_vec_einsum", "w", "a")] = rng.normal(0, .3, (N, H, r))
        z[_key(i, "attn", "attn_vec_einsum", "w", "b")] = rng.normal(0, .3, (r, D))
        if getattr(att, "k_proj", None) is not None:
            Kh = att.k_proj.weight.shape[0] // H
            z[_key(i, "attn", "kv_einsum", "w", "a")] = rng.normal(0, .3, (D, r))
            z[_key(i, "attn", "kv_einsum", "w", "b")] = rng.normal(0, .3, (r, 2, Kh, H))
        for m in ("gate_proj", "up_proj", "down_proj"):
            w = getattr(layer.mlp, m).weight
            z[_key(i, "mlp", m, "kernel", "a")] = rng.normal(0, .3, (w.shape[1], r))
            z[_key(i, "mlp", m, "kernel", "b")] = rng.normal(0, .3, (r, w.shape[0]))
    np.savez(path, **z)
    return path


def test_hooks_equal_merged_weights_and_scorer_equals_brute_force(tmp_path):
    pytest.importorskip("transformers.models.gemma4")
    base = _tiny()
    factors = HS.lora_factors(_tpu_layout_adapter(base, tmp_path / "lora.npz"))
    merged = copy.deepcopy(base)
    text, _, _ = HS.text_parts(merged)
    for (i, path), (A, B) in factors.items():
        mod = text.layers[i]
        for part in path.split("."):
            mod = getattr(mod, part)
        mod.weight.data += HS.SCALE * torch.tensor(B @ A, dtype=mod.weight.dtype)
    hooked = base
    HS.attach(hooked, factors)
    ids = torch.randint(1, 300, (2, 11))
    assert torch.allclose(hooked(input_ids=ids).logits, merged(input_ids=ids).logits, atol=1e-4)
    # class scorer vs brute force: shared answer prefix, classes of different lengths, prompts of different lengths
    suffixes = [[7, 8, 9, 20, 5], [7, 8, 9, 21, 22, 5], [7, 8, 9, 20, 23, 24, 5]]
    prompts = [list(range(30, 30 + n)) for n in (5, 9, 13)]
    got = HS.class_scores(hooked, prompts, suffixes, pad_id=0, margin=None, batch=2)
    for p, row in zip(prompts, got):
        for s, val in zip(suffixes, row):
            seq = torch.tensor([p + s])
            lp = torch.log_softmax(hooked(input_ids=seq).logits[0].float(), -1)  # HF applies the softcap
            want = sum(lp[len(p) - 1 + j, t].item() for j, t in enumerate(s))
            assert abs(val - want) < 1e-3, (val, want)
    with pytest.raises(AssertionError):  # a K/V factor for a KV-shared layer is a mapping error, never silently dropped
        HS.attach(_tiny(), {(6, "self_attn.k_proj"): (np.zeros((4, 32)), np.zeros((32, 4)))})


def test_score_records_decides_like_the_endpoint_on_real_benchmark_records():
    """The gemma_fp stage's code path on a tiny model (Qwen's tokenizer stands in for Gemma's, which needs the Hub):
    windows, max-attack-mass window, Gemma's frozen policy; robustness rows in the score-file format."""
    import json
    from pathlib import Path
    import yaml
    from eval import policy as POL
    from training import colab_helpers as H, prepare as P
    from transformers import Gemma4ForCausalLM, Gemma4TextConfig
    try:
        tok = P.load_tokenizer()
    except Exception as e:
        pytest.skip(f"tokenizer not available: {e}")
    root = Path(__file__).resolve().parent.parent
    tax, _, _ = POL.load()
    pol = yaml.safe_load(open(root / "config/policy_gemma.yaml"))
    classes = list(tax["classes"])
    cfg = Gemma4TextConfig(vocab_size=len(tok), hidden_size=32, intermediate_size=64, num_hidden_layers=8, num_attention_heads=2,
                           num_key_value_heads=1, head_dim=16, global_head_dim=32, hidden_size_per_layer_input=4,
                           vocab_size_per_layer_input=len(tok), num_kv_shared_layers=3, use_double_wide_mlp=True,
                           layer_types=LAYER_TYPES, final_logit_softcapping=30.0, sliding_window=64, pad_token_id=tok.pad_token_id)
    torch.manual_seed(0)
    model = Gemma4ForCausalLM(cfg).float().eval()
    suf = H.candidate_suffixes(tok, classes)
    recs = [json.loads(l) for l in open(root / "reports/benchmarks/gotestwaf.jsonl")][:3]
    recs += [json.loads(l) for l in open(root / "reports/benchmarks/unsw.jsonl")][:2]
    got = HS.score_records(model, tok, recs, tax, pol, suf, contract=1, budget=900)
    ctrl = P.control_strings(tok)
    from training.gemma_tpu import data as GD
    benign = np.array([tax["classes"][c]["is_benign"] for c in classes])
    for r, x in zip(recs, got):
        assert x["id"] == r["id"]
        rows, man = GD.bench_rows([r], lambda tel: GD.prompt_ids(tok, ctrl, tel), 900, contract=1)
        if man[r["id"]]["windows"] == 0:
            assert x["decision"]["rule_hint"] == "over_token_budget"
            continue
        ps = [HS._probs(s, float(pol["temperature"])) for s in HS.class_scores(model, [i for _, i in rows], suf, tok.pad_token_id)]
        p = max(ps, key=lambda q: 1 - q[benign].sum())
        assert np.allclose(x["probs"], p, atol=1e-6) and x["windows"] == len(ps)
        want = POL.decide_from_probs(p, r["telemetry"], tax, pol)[0]  # batch composition moves the last digits only
        assert {k: v for k, v in x["decision"].items() if k != "confidence"} == {k: v for k, v in want.items() if k != "confidence"}
        assert (x["decision"]["confidence"] is None) == (want["confidence"] is None)
        assert x["decision"]["confidence"] is None or abs(x["decision"]["confidence"] - want["confidence"]) < 1e-6
    rob = [{"id": "robust:oot:0001", "split": "robustness", "kind": "out_of_taxonomy", "label": "unknown",
            "telemetry": recs[-1]["telemetry"]}]
    (row,) = HS.robustness_scores(model, tok, rob, tax, float(pol["temperature"]), suf, contract=1, budget=900)
    assert row["split"] == "robustness" and row["label"] == "unknown" and len(row["scores"]) == len(classes)


def test_gemma_only_endpoint_predict_isolates_malformed_records():
    """deploy/sagemaker/inference.py Gemma-only package (deploy.sh GEMMA_ONLY=1): predict_fn scores through
    score_records; a malformed record gets its own error, the others still get decisions."""
    import json
    import sys
    from pathlib import Path
    import yaml
    from eval import policy as POL
    from training import colab_helpers as H, prepare as P
    from transformers import Gemma4ForCausalLM, Gemma4TextConfig
    try:
        tok = P.load_tokenizer()
    except Exception as e:
        pytest.skip(f"tokenizer not available: {e}")
    root = Path(__file__).resolve().parent.parent
    sys.path.insert(0, str(root / "deploy/sagemaker"))
    import inference as INF
    tax, _, _ = POL.load()
    pol = yaml.safe_load(open(root / "config/policy_gemma_v13.yaml"))
    cfg = Gemma4TextConfig(vocab_size=len(tok), hidden_size=32, intermediate_size=64, num_hidden_layers=8, num_attention_heads=2,
                           num_key_value_heads=1, head_dim=16, global_head_dim=32, hidden_size_per_layer_input=4,
                           vocab_size_per_layer_input=len(tok), num_kv_shared_layers=3, use_double_wide_mlp=True,
                           layer_types=LAYER_TYPES, final_logit_softcapping=30.0, sliding_window=64, pad_token_id=tok.pad_token_id)
    torch.manual_seed(0)
    g = {"model": Gemma4ForCausalLM(cfg).float().eval(), "tok": tok, "suf": H.candidate_suffixes(tok, list(tax["classes"]))}
    st = {"tax": tax, "pol": pol, "gemma": g, "gemma_only": True, "selfcheck_max_diff": 0.0}
    tels = [json.loads(l)["telemetry"] for l in open(root / "reports/benchmarks/gotestwaf.jsonl")][:2]
    res = INF.predict_fn(INF.input_fn(json.dumps({"records": [tels[0], "not telemetry", tels[1]]})), st)["results"]
    assert res[1]["decision"] is None and res[1]["error"]
    assert all(r["decision"] is not None and "probs" not in r and "id" not in r for r in (res[0], res[2]))


def test_merged_text_only_model_equals_hooked_multimodal(tmp_path):
    """training.gemma_tpu.merge: LoRA merged into a Gemma4ForConditionalGeneration, saved as a text-only
    Gemma4ForCausalLM and reloaded, gives the hooked model's logits (towers dropped, tied head kept)."""
    from transformers import Gemma4Config, Gemma4ForCausalLM, Gemma4ForConditionalGeneration
    from training.gemma_tpu import merge as MG
    cfg = Gemma4Config(text_config=_tiny().config.to_dict(), vision_config=None, audio_config=None)
    torch.manual_seed(0)
    cg = Gemma4ForConditionalGeneration(cfg).float().eval()
    factors = HS.lora_factors(_tpu_layout_adapter(_tiny(), tmp_path / "lora.npz"))  # same text shapes
    hooked = copy.deepcopy(cg)
    HS.attach(hooked, factors)
    assert MG.merge(cg, factors) == len(factors)
    MG.text_only(cg).save_pretrained(tmp_path / "out")
    lm = Gemma4ForCausalLM.from_pretrained(tmp_path / "out", dtype=torch.float32).eval()
    ids = torch.randint(1, 300, (2, 11))
    assert torch.allclose(lm(input_ids=ids).logits, hooked(input_ids=ids).logits, atol=1e-4)
    assert not any("tower" in k or "embed_vision" in k or "embed_audio" in k for k in lm.state_dict())
