"""Helpers shared by the notebook's Part A (train) and Part B (fresh-process checks).

Part B runs after a kernel restart, so anything both parts need lives here, in the bundle.
"""
import json

import torch

from training import prepare as P

LORA = dict(r=16, lora_alpha=32, use_rslora=True, lora_dropout=0, bias="none",
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
            use_gradient_checkpointing="unsloth")


def dtype_is_bf16():
    # T4 (capability 7.5) reports bf16 support but only emulates it -> fp16 there
    return torch.cuda.is_bf16_supported() and torch.cuda.get_device_capability()[0] >= 8


def load_records(bundle, fname, splits=None):
    out = []
    for line in open(f"{bundle}/datasets_ready/{fname}"):
        r = json.loads(line)
        if splits is None or r["metadata"]["split"] in splits:
            out.append(r)
    return out


def encode(tok, recs):
    ctrl = P.control_strings(tok)
    rows = []
    for r in recs:
        ids, labels = P.tokenize_with_mask(tok, r["messages"])
        P.check_record(tok, r, ctrl, (ids, labels))  # turns, escapes, mask, budget
        rows.append({"input_ids": ids, "labels": labels})
    return rows


def make_collate(pad_id):
    def collate(batch):
        n = max(len(b["input_ids"]) for b in batch)
        ids = torch.full((len(batch), n), pad_id)
        lab = torch.full((len(batch), n), P.IGNORE)
        att = torch.zeros((len(batch), n), dtype=torch.long)
        for i, b in enumerate(batch):
            k = len(b["input_ids"])
            ids[i, :k] = torch.tensor(b["input_ids"])
            lab[i, :k] = torch.tensor(b["labels"])
            att[i, :k] = 1
        return {"input_ids": ids, "labels": lab, "attention_mask": att}
    return collate


EFFECTIVE_BATCH = 16


def batch_for_gpu(total_gb=None):
    """Per-device batch that fits the GPU; gradient accumulation keeps the effective batch at 16.
    T4 peaked at 10.7 GB with batch 1 (fp16, 16-bit weights ~9 GB), so the headroom is activations only."""
    if total_gb is None:
        total_gb = torch.cuda.get_device_properties(0).total_memory / 2**30
    # ponytail: fixed memory tiers, not a probe; raise a tier only after a dry run shows the peak
    return 1 if total_gb < 20 else 2 if total_gb < 30 else 4 if total_gb < 60 else 8


def make_args(out, max_steps=-1, save_every=250, smoke=False, seed=3407, evaluate=True, per_device=None):
    from transformers import TrainingArguments
    bf16 = dtype_is_bf16()
    per_device = per_device or batch_for_gpu()
    assert per_device > 0 and EFFECTIVE_BATCH % per_device == 0, f"per-device batch must divide {EFFECTIVE_BATCH}"
    return TrainingArguments(
        output_dir=out, per_device_train_batch_size=per_device, gradient_accumulation_steps=EFFECTIVE_BATCH // per_device,
        train_sampling_strategy="group_by_length" if per_device > 1 else "random",  # less padding when batch > 1
        num_train_epochs=1, max_steps=max_steps, learning_rate=1e-4, lr_scheduler_type="cosine",
        warmup_steps=0.05,  # transformers v5: a float in [0, 1) is a ratio of total steps
        weight_decay=0.01, max_grad_norm=1.0, optim="adamw_8bit",
        bf16=bf16, fp16=not bf16, logging_steps=1 if smoke else 10,
        eval_strategy="steps" if evaluate else "no", eval_steps=save_every,
        save_strategy="steps", save_steps=save_every, save_total_limit=3,
        seed=seed, data_seed=seed, report_to="none", remove_unused_columns=False)


def generate_label(model, tok, messages, max_new_tokens=24):
    prompt = tok.apply_chat_template(messages[:2], tokenize=False, add_generation_prompt=True,
                                     enable_thinking=False)
    ids = tok(prompt, return_tensors="pt", add_special_tokens=False).to("cuda")
    gen = model.generate(**ids, max_new_tokens=max_new_tokens, do_sample=False, eos_token_id=tok.eos_token_id)
    return tok.decode(gen[0, ids["input_ids"].shape[1]:], skip_special_tokens=True).strip()


# ---- calibrated scoring (spec §10): log P(completion | prompt) for every allowed class ------------------------
def candidate_suffixes(tok, classes):
    """Token ids of each class's full assistant completion (JSON + end of turn), as the training target renders it.
    The prompt/completion boundary is template-fixed, so a dummy prompt gives the same suffixes for every record."""
    msgs = [{"role": "system", "content": "s"}, {"role": "user", "content": "u"}]
    p = tok(tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True, enable_thinking=False),
            add_special_tokens=False)["input_ids"]
    eos, out = tok.convert_tokens_to_ids(tok.eos_token), []
    for c in classes:
        tgt = json.dumps({"threat_type": c}, separators=(",", ":"))
        f = tok(tok.apply_chat_template(msgs + [{"role": "assistant", "content": tgt}], tokenize=False),
                add_special_tokens=False)["input_ids"]
        assert f[:len(p)] == p, "template prefix mismatch"
        s = f[len(p):]
        out.append(s[:max(i for i, t in enumerate(s) if t == eos) + 1])  # through <|im_end|>, as supervised
    return out


@torch.no_grad()
def class_logprobs(model, tok, messages, suffixes, margin=12.0):
    """Summed completion log-probs, one per class: one prefill of prompt + the shared candidate prefix
    ({"threat_type":" ...), then the class suffixes in one batch from a replicated cache.
    Pruning (margin=None: off): a class whose first suffix token is > margin nats below the best first token keeps
    that first-token log-prob, an upper bound of its full score, without the suffix pass. Its softmax mass is then
    < e^-margin x (classes sharing the best first token), ~2e-5 at 12 nats; most records keep a handful of classes."""
    p = tok(tok.apply_chat_template(messages[:2], tokenize=False, add_generation_prompt=True, enable_thinking=False),
            add_special_tokens=False)["input_ids"]
    k = 0
    while all(len(s) > k + 1 and s[k] == suffixes[0][k] for s in suffixes):
        k += 1  # shared prefix: identical for every class, kept in the sum so scores are full log P(completion)
    ctx = torch.tensor([p + suffixes[0][:k]], device="cuda")
    out = model(input_ids=ctx, use_cache=True)
    lp = torch.log_softmax(out.logits[0, len(p) - 1:].float(), -1)  # predicts ctx[len(p):] then the first suffix token
    shared = sum(lp[i, t].item() for i, t in enumerate(suffixes[0][:k]))
    first = lp[-1]
    rest = [s[k:] for s in suffixes]
    f0 = [first[r[0]].item() for r in rest]
    keep = [i for i, v in enumerate(f0) if margin is None or v >= max(f0) - margin]
    scores = [shared + v for v in f0]
    n, L = len(keep), max(len(rest[i]) for i in keep)
    cache = out.past_key_values
    cache.reorder_cache(torch.zeros(n, dtype=torch.long, device="cuda"))  # batch-replicate (linear-attn states too)
    ids = torch.full((n, L), tok.pad_token_id, device="cuda")
    for b, i in enumerate(keep):
        ids[b, :len(rest[i])] = torch.tensor(rest[i], device="cuda")  # right padding: causal, never seen by real tokens
    nxt = torch.log_softmax(model(input_ids=ids, past_key_values=cache, use_cache=True).logits.float(), -1)
    for b, i in enumerate(keep):
        r = rest[i]
        scores[i] = shared + f0[i] + sum(nxt[b, j - 1, r[j]].item() for j in range(1, len(r)))
    return scores


if __name__ == "__main__":  # self-check of the tiers (no GPU needed)
    assert [batch_for_gpu(g) for g in (15, 22.5, 40, 80)] == [1, 2, 4, 8]
    assert all(EFFECTIVE_BATCH % batch_for_gpu(g) == 0 for g in (15, 22.5, 40, 80))
    print("ok")
