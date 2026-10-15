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
        P.check_record(tok, r, ctrl)  # turns, escapes, mask, budget
        ids, labels = P.tokenize_with_mask(tok, r["messages"])
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


def make_args(out, max_steps=-1, save_every=250, smoke=False, seed=3407, evaluate=True):
    from transformers import TrainingArguments
    bf16 = dtype_is_bf16()
    return TrainingArguments(
        output_dir=out, per_device_train_batch_size=1, gradient_accumulation_steps=16,
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
