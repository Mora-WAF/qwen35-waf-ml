"""TPU (PyTorch/XLA) LoRA fine-tune for the Colab TPU v5e-1 runtime. The CUDA notebook needs Unsloth and
bitsandbytes, which do not run on TPU; this is the same recipe without them:

- same base checkpoint and revision, same rsLoRA (r=16, alpha=32) on the language model's attention/MLP
  projections (21.2M trainable, as in the CUDA smoke test), same loss mask (answer + <|im_end|> only, from
  training.prepare), lr 1e-4 cosine with 5% warmup, AdamW wd 0.01, grad clip 1.0, 16 sequences per step;
- fixed-shape length buckets so XLA compiles a few graphs instead of one per sequence length;
- resumable: adapter + optimizer + scheduler + step + data order saved every CKPT_EVERY steps to OUT and,
  if given, copied to DRIVE (free TPU sessions are time-capped).

usage (on the TPU VM, bundle unpacked at /content/bundle):
  python -m training.tpu_train --bundle /content/bundle --prefix fast_ --out /content/run_tpu [--drive DIR]
"""
import argparse
import json
import math
import os
import random
import shutil
import sys
import time

BUCKETS = ((512, 8), (640, 6), (768, 5), (1024, 4), (1536, 2), (2048, 2))  # (padded length, micro-batch); padding 12.7% vs 42.5% with 3 buckets on fast_train
SEQ_PER_STEP = 16
LORA_TARGETS = r".*language_model.*\.(q_proj|k_proj|v_proj|o_proj|gate_proj|up_proj|down_proj)"


def bucket_of(n):
    return next(b for b in BUCKETS if n <= b[0])


def plan_epoch(lengths, seed):
    """Micro-batches of same-bucket rows, shuffled; deterministic given seed (resume replays the plan)."""
    rng = random.Random(seed)
    by = {}
    for i, n in enumerate(lengths):
        by.setdefault(bucket_of(n), []).append(i)
    micro = []
    for (L, mb), idx in by.items():
        rng.shuffle(idx)
        micro += [(L, idx[j:j + mb]) for j in range(0, len(idx), mb)]
    rng.shuffle(micro)
    return micro


def pad(rows, L, pad_id, ignore):
    import torch
    ids = torch.full((len(rows), L), pad_id, dtype=torch.long)
    lab = torch.full((len(rows), L), ignore, dtype=torch.long)
    att = torch.zeros((len(rows), L), dtype=torch.long)
    for i, (x, y) in enumerate(rows):
        ids[i, :len(x)] = torch.tensor(x)
        lab[i, :len(y)] = torch.tensor(y)
        att[i, :len(x)] = 1
    return ids, lab, att


def main(a):
    import torch
    import torch_xla
    import torch_xla.core.xla_model as xm
    from peft import LoraConfig, PeftModel, get_peft_model
    from transformers import AutoModelForImageTextToText, AutoTokenizer, get_cosine_schedule_with_warmup

    sys.path.insert(0, a.bundle)
    from training import prepare as P
    from training.colab_helpers import load_records

    torch.manual_seed(a.seed)
    os.makedirs(a.out, exist_ok=True)
    tok = AutoTokenizer.from_pretrained(P.MODEL_ID, revision=P.MODEL_REVISION)
    gate = json.load(open(f"{a.bundle}/reports/pre_colab_gate.json"))
    assert gate["passed"], "pre-Colab gate failed"
    assert P.sha256(tok.chat_template) == gate["tokenizer_fingerprint"]["chat_template_sha256"], "chat template drift"

    def encode(recs):
        return [P.tokenize_with_mask(tok, r["messages"]) for r in recs]
    t0 = time.time()
    train = encode(load_records(a.bundle, f"{a.prefix}train.jsonl"))
    val = encode(load_records(a.bundle, f"{a.prefix}val.jsonl", {"val_select"})[:a.eval_n])
    print(json.dumps({"train": len(train), "val_select_eval": len(val), "encode_s": round(time.time() - t0)}), flush=True)

    dev = torch_xla.device()
    model = AutoModelForImageTextToText.from_pretrained(P.MODEL_ID, revision=P.MODEL_REVISION, dtype=torch.bfloat16)
    ck = os.path.join(a.out, "ckpt")
    if a.drive and not os.path.exists(os.path.join(ck, "state.json")) and os.path.exists(os.path.join(a.drive, "ckpt", "state.json")):
        shutil.copytree(os.path.join(a.drive, "ckpt"), ck, dirs_exist_ok=True)  # resume from Drive on a new VM
    resume = os.path.exists(os.path.join(ck, "state.json"))
    if resume:
        model = PeftModel.from_pretrained(model, os.path.join(ck, "adapter"), is_trainable=True)
    else:
        model = get_peft_model(model, LoraConfig(r=16, lora_alpha=32, use_rslora=True, lora_dropout=0.0,
                                                 bias="none", target_modules=LORA_TARGETS))
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    model.config.use_cache = False
    model.to(dev)
    params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=a.lr, weight_decay=0.01)
    plan = plan_epoch([len(x) for x, _ in train], a.seed)
    steps_total = math.ceil(len(train) / SEQ_PER_STEP)
    sched = get_cosine_schedule_with_warmup(opt, int(0.05 * steps_total), steps_total)
    state = {"step": 0, "micro": 0, "losses": [], "evals": []}
    if resume:
        state = json.load(open(os.path.join(ck, "state.json")))
        opt.load_state_dict(torch.load(os.path.join(ck, "opt.pt"), map_location="cpu"))
        sched.load_state_dict(torch.load(os.path.join(ck, "sched.pt")))
    print(json.dumps({"resumed": resume, "step": state["step"], "steps_total": steps_total,
                      "micro_batches": len(plan), "trainable": sum(p.numel() for p in params)}), flush=True)

    def evaluate():
        model.eval()
        tot, n = 0.0, 0
        with torch.no_grad():
            for L, idx in plan_epoch([len(x) for x, _ in val], 0):
                ids, lab, att = (t.to(dev) for t in pad([val[i] for i in idx], L, tok.pad_token_id, P.IGNORE))
                tot += model(input_ids=ids, attention_mask=att, labels=lab).loss.detach() * len(idx)
                n += len(idx)
        xm.mark_step()
        model.train()
        return (tot / n).item()

    def save():
        os.makedirs(ck, exist_ok=True)
        model.save_pretrained(os.path.join(ck, "adapter"))  # moves LoRA tensors to CPU for writing
        xm.save(opt.state_dict(), os.path.join(ck, "opt.pt"))
        torch.save(sched.state_dict(), os.path.join(ck, "sched.pt"))
        json.dump(state, open(os.path.join(ck, "state.json"), "w"))
        if a.drive:
            dst = os.path.join(a.drive, "ckpt")
            shutil.copytree(ck, dst + ".part", dirs_exist_ok=True)
            shutil.rmtree(dst, ignore_errors=True)
            os.replace(dst + ".part", dst)

    model.train()
    seen, t_step, tok_count, t_warm = 0, time.time(), 0, None
    while state["micro"] < len(plan):
        L, idx = plan[state["micro"]]
        ids, lab, att = (t.to(dev) for t in pad([train[i] for i in idx], L, tok.pad_token_id, P.IGNORE))
        loss = model(input_ids=ids, attention_mask=att, labels=lab).loss * (len(idx) / SEQ_PER_STEP)
        loss.backward()
        seen += len(idx)
        tok_count += int(att.sum().item()) if state["micro"] % 50 == 0 else 0
        state["micro"] += 1
        if seen >= SEQ_PER_STEP or state["micro"] == len(plan):
            torch.nn.utils.clip_grad_norm_(params, 1.0)
            opt.step(); sched.step(); opt.zero_grad()
            xm.mark_step()
            state["step"] += 1
            seen = 0
            if state["step"] == 3:
                xm.wait_device_ops(); t_warm = time.time(); state["micro_at_warm"] = state["micro"]
            if state["step"] % a.log_every == 0:
                state["losses"].append([state["step"], round(loss.item() * SEQ_PER_STEP / max(1, len(idx)), 4)])
                rate = ((state["micro"] - state.get("micro_at_warm", 0)) / (time.time() - t_warm)) if t_warm else None
                print(json.dumps({"step": state["step"], "of": steps_total, "loss": state["losses"][-1][1],
                                  "micro_per_s": round(rate, 3) if rate else None,
                                  "eta_h": round((len(plan) - state["micro"]) / rate / 3600, 2) if rate else None}), flush=True)
            if state["step"] % a.eval_every == 0:
                state["evals"].append([state["step"], round(evaluate(), 4)])
                print(json.dumps({"eval_step": state["step"], "eval_loss": state["evals"][-1][1]}), flush=True)
            if state["step"] % a.ckpt_every == 0:
                save()
    state["evals"].append([state["step"], round(evaluate(), 4)])
    save()
    exp = os.path.join(a.out, "export")
    model.save_pretrained(exp)
    tok.save_pretrained(exp)
    json.dump({"base_model": P.MODEL_ID, "revision": P.MODEL_REVISION, "steps": state["step"], "evals": state["evals"],
               "losses": state["losses"], "device": "TPU v5e-1 (PyTorch/XLA)", "lora_targets": LORA_TARGETS},
              open(os.path.join(exp, "train_report.json"), "w"), indent=1)
    if a.drive:
        shutil.copytree(exp, os.path.join(a.drive, "export"), dirs_exist_ok=True)
    print(json.dumps({"done": True, "steps": state["step"], "final_eval": state["evals"][-1]}), flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--bundle", default="/content/bundle")
    ap.add_argument("--prefix", default="fast_")
    ap.add_argument("--out", default="/content/run_tpu")
    ap.add_argument("--drive", default=None)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--seed", type=int, default=3407)
    ap.add_argument("--eval-n", type=int, default=1000)
    ap.add_argument("--eval-every", type=int, default=250)
    ap.add_argument("--ckpt-every", type=int, default=100)
    ap.add_argument("--log-every", type=int, default=10)
    main(ap.parse_args())
