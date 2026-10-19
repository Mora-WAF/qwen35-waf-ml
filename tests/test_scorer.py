"""batch_class_logprobs (left-padded prefill + cached class suffixes) must give the per-record class_logprobs scores.
Tiny random Qwen3.5 on CPU (torch fallback kernels); needs the pinned tokenizer in the local HF cache."""
import pytest
import torch

from training import colab_helpers as H, prepare as P


def _tok():
    try:
        return P.load_tokenizer()
    except Exception as e:  # offline without the cached tokenizer
        pytest.skip(f"tokenizer not available: {e}")


def test_batched_cached_scorer_matches_per_record():
    from transformers import Qwen3_5ForCausalLM, Qwen3_5TextConfig
    tok = _tok()
    torch.manual_seed(0)
    cfg = Qwen3_5TextConfig(vocab_size=len(tok), hidden_size=64, intermediate_size=128, num_hidden_layers=4,
                            layer_types=["linear_attention", "linear_attention", "linear_attention", "full_attention"],
                            num_attention_heads=4, num_key_value_heads=2, head_dim=16, linear_num_key_heads=2,
                            linear_num_value_heads=4, linear_key_head_dim=16, linear_value_head_dim=16,
                            pad_token_id=tok.pad_token_id)
    model = Qwen3_5ForCausalLM(cfg).float().eval()
    classes = ["sqli_classic", "xss_reflected_stored", "benign_standard_traffic", "rce_cmd_injection", "ssti"]
    suf = H.candidate_suffixes(tok, classes)
    texts = ["GET /a?q=1", "POST /login user=admin' OR 1=1--" * 3, "x", "GET /search?q=" + "long " * 40]
    msgs = [[{"role": "system", "content": "classify"}, {"role": "user", "content": t}] for t in texts]
    ref = [H.class_logprobs(model, tok, m, suf, margin=None) for m in msgs]
    # max_rows=3 forces several cache copies per batch; batch=3 mixes lengths (left padding)
    got = H.batch_class_logprobs(model, tok, msgs, suf, margin=1e9, batch=3, max_rows=3)
    for r, g in zip(ref, got):
        assert max(abs(a - b) for a, b in zip(r, g)) < 1e-3, (r, g)


def test_lora_full_set_adapts_the_linear_attention_layers():
    """WAF_LORA=full (default since 19 Oct) puts adapters on the Gated DeltaNet projections; the v11 set leaves them
    frozen. Tiny random Qwen3.5 on CPU, PEFT suffix matching (Unsloth builds a regex over the same names)."""
    peft = pytest.importorskip("peft")
    from transformers import Qwen3_5ForCausalLM, Qwen3_5TextConfig
    cfg = Qwen3_5TextConfig(vocab_size=512, hidden_size=64, intermediate_size=128, num_hidden_layers=4,
                            layer_types=["linear_attention", "linear_attention", "linear_attention", "full_attention"],
                            num_attention_heads=4, num_key_value_heads=2, head_dim=16, linear_num_key_heads=2,
                            linear_num_value_heads=4, linear_key_head_dim=16, linear_value_head_dim=16, pad_token_id=0)

    def adapted(targets):
        torch.manual_seed(0)
        m = peft.get_peft_model(Qwen3_5ForCausalLM(cfg).float(), peft.LoraConfig(r=4, lora_alpha=8, use_rslora=True,
                                                                                 target_modules=targets))
        return m, {n.split(".lora_A")[0] for n, _ in m.named_parameters() if ".lora_A" in n}
    m, full = adapted(H.LORA_TARGETS["full"])
    _, v11 = adapted(H.LORA_TARGETS["v11"])
    linear = {n for n in full if ".linear_attn." in n}
    assert len(linear) == 3 * 5 and not any(".linear_attn." in n for n in v11)  # 3 linear layers x 5 projections
    assert v11 < full and {n.rsplit(".", 1)[1] for n in linear} == set(H.LINEAR_ATTN_TARGETS)
    ids = torch.randint(1, 512, (2, 12))
    m(input_ids=ids, labels=ids).loss.backward()  # trains through the linear-attention adapters
    assert all(p.grad is not None and p.grad.abs().sum() > 0 for n, p in m.named_parameters()
               if ".linear_attn." in n and "lora_B" in n)
