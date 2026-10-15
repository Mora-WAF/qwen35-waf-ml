"""Builds colab/Qwen3_5_4B_WAF_FineTune.ipynb from the cells below (keeps the notebook diffable).

usage: python colab/make_notebook.py
"""
import json
from pathlib import Path

import nbformat

MD, PY = "markdown", "code"
CELLS = [
(MD, """# Qwen3.5-4B WAF / NIDS classifier — rsLoRA fine-tune

**Status of this notebook: NOT smoke-tested yet.** Run `MODE="smoke"` on a fresh GPU runtime first; it
writes `gpu_smoke_test.json`, which is the only evidence that counts for the *Notebook smoke-tested* state.

- Input: `colab_bundle_*.zip` (data + helpers + manifests) made by `training/make_bundle.py`.
- Training target: `{"threat_type": "<one of 35 keys>"}` only. Loss on the answer + `<|im_end|>` only.
- 16-bit LoRA (rsLoRA r=16, alpha=32). QLoRA is deliberately not used (Unsloth: not recommended for Qwen3.5).
- Nothing is uploaded or published automatically. Package pins below are **candidates**: this exact set
  has not yet been run together; the smoke report records what actually got installed."""),
(PY, """# ---- configuration ---------------------------------------------------------------------------
MODE = "smoke"            # "smoke" (>=20 optimizer steps) | "fast" (30k subset) | "full" (100k)
BUNDLE = "/content/colab_bundle_pilot.zip"   # upload via the Files pane, or point at Drive
DRIVE_DIR = None          # e.g. "/content/drive/MyDrive/qwen35_waf/run1" -> checkpoints copied here
RUN_RESUME_TEST = True    # smoke only: compare uninterrupted vs save/reload continuation
SEED = 3407
MODEL_ID = "unsloth/Qwen3.5-4B"
MODEL_REVISION = "3764fa359b9082ea5a1e4a5e3ac3aaf6e9671636"
MAX_LEN = 2048"""),
(PY, """# ---- pinned candidate stack (unsloth 2026.10.5 requires transformers<=5.17.0, trl<=1.15.0) -----
!pip install -q "unsloth==2026.10.5" "unsloth_zoo==2026.10.5" "transformers==5.17.0" "trl==1.15.0" "peft==0.21.2" "accelerate==1.15.0" "bitsandbytes==0.50.2" "datasets==5.1.0" pyyaml jsonschema
# If pip reports a torch / CUDA conflict: Runtime -> Restart session, then run from the next cell."""),
(PY, """import json, os, shutil, subprocess, sys, time, math, hashlib, zipfile, platform, random
import torch
assert torch.cuda.is_available(), "No GPU: Runtime -> Change runtime type -> GPU"
props = torch.cuda.get_device_properties(0)
BF16 = torch.cuda.is_bf16_supported()
ENV = {"gpu": props.name, "gpu_mem_gb": round(props.total_memory / 2**30, 1),
       "capability": f"{props.major}.{props.minor}", "dtype": "bfloat16" if BF16 else "float16",
       "disk_free_gb": shutil.disk_usage("/content").free // 2**30, "python": platform.python_version(),
       "torch": torch.__version__, "cuda": torch.version.cuda}
for pkg in ["unsloth", "unsloth_zoo", "transformers", "trl", "peft", "accelerate", "bitsandbytes", "datasets"]:
    try:
        ENV[pkg] = __import__(pkg).__version__
    except Exception as e:
        ENV[pkg] = f"ERROR {e}"
print(json.dumps(ENV, indent=1))
if ENV["gpu_mem_gb"] < 14:
    print("WARNING: <14 GB VRAM. Unsloth cites ~10 GB for 4B bf16 LoRA; 2048-token batches may OOM.")"""),
(PY, """# ---- bundle: unpack + verify checksums, counts, gate ---------------------------------------
if DRIVE_DIR or BUNDLE.startswith("/content/drive"):
    from google.colab import drive; drive.mount("/content/drive")
B = "/content/bundle"
shutil.rmtree(B, ignore_errors=True)
zipfile.ZipFile(BUNDLE).extractall(B)
sys.path.insert(0, B)
info = json.load(open(f"{B}/BUNDLE_INFO.json"))
manifest = json.load(open(f"{B}/datasets_ready/manifest.json"))
gate = json.load(open(f"{B}/reports/pre_colab_gate.json"))
assert gate["passed"] or MODE == "smoke", "pre-Colab gate failed; only smoke runs allowed on this bundle"
for f, meta in manifest["files"].items():
    p = f"{B}/datasets_ready/{f}"
    if os.path.exists(p):
        assert hashlib.sha256(open(p, "rb").read()).hexdigest() == meta["sha256"], f"checksum {f}"
        assert sum(1 for _ in open(p)) == meta["records"], f"count {f}"
from training import prepare as P
from eval.policy import parse_output
import yaml
TAX = yaml.safe_load(open(f"{B}/config/taxonomy.yaml"))["classes"]
FILES = {"smoke": "pilot_train.jsonl", "fast": "train_fast_30k.jsonl", "full": "train_full_100k.jsonl"}
VAL = "pilot_val.jsonl" if MODE == "smoke" else "val_10k.jsonl"
print("bundle ok:", info["prefix"], "gate passed:", gate["passed"])"""),
(PY, """# ---- model (16-bit LoRA, text-only path) -------------------------------------------------------
from unsloth import FastLanguageModel
model, processor = FastLanguageModel.from_pretrained(
    model_name=MODEL_ID, revision=MODEL_REVISION, max_seq_length=MAX_LEN,
    load_in_4bit=False, load_in_16bit=True, full_finetuning=False)
tok = getattr(processor, "tokenizer", processor)   # text tokenizer even if a processor comes back
fp_now = {"chat_template_sha256": P.sha256(tok.chat_template),
          "system_prompt_sha256": P.sha256(P.system_prompt(P.Path(B)))}
fp_ref = gate["tokenizer_fingerprint"]
for k, v in fp_now.items():
    assert v == fp_ref[k], f"{k} mismatch: Colab {v} vs local gate {fp_ref[k]}"
assert tok.eos_token == "<|im_end|>", tok.eos_token
model = FastLanguageModel.get_peft_model(
    model, r=16, lora_alpha=32, use_rslora=True, lora_dropout=0, bias="none",
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    use_gradient_checkpointing="unsloth", random_state=SEED, max_seq_length=MAX_LEN)
trainable = [(n, p.numel()) for n, p in model.named_parameters() if p.requires_grad]
n_train, n_all = sum(n for _, n in trainable), sum(p.numel() for p in model.parameters())
assert not any("visual" in n or "vision" in n for n, _ in trainable), "vision params are trainable"
print(f"trainable {n_train:,} / {n_all:,} ({100*n_train/n_all:.2f}%) in {len(trainable)} tensors")
print("sample trainable modules:", sorted({n.split('.lora')[0].split('.')[-1] for n, _ in trainable}))"""),
(PY, """# ---- data: render with the checkpoint template, mask, verify ---------------------------------
CTRL = P.control_strings(tok)
def load(fname, splits=None):
    out = []
    for line in open(f"{B}/datasets_ready/{fname}"):
        r = json.loads(line)
        if splits is None or r["metadata"]["split"] in splits:
            out.append(r)
    return out
train_recs = load(FILES[MODE])
val_recs = load(VAL, {"val_select"})          # model selection partition ONLY
def encode(recs):
    rows = []
    for r in recs:
        P.check_record(tok, r, CTRL)               # turns, escapes, mask, budget
        ids, labels = P.tokenize_with_mask(tok, r["messages"])
        rows.append({"input_ids": ids, "labels": labels})
    return rows
train_rows, val_rows = encode(train_recs), encode(val_recs)
# mask inspection: supervised span must be exactly the target + <|im_end|>
for r, row in random.Random(SEED).sample(list(zip(train_recs, train_rows)), 5):
    sup = tok.decode([t for t, l in zip(row["input_ids"], row["labels"]) if l != P.IGNORE])
    assert sup == r["messages"][2]["content"] + "<|im_end|>", sup
    print("supervised:", repr(sup), "| prompt tokens masked:", sum(l == P.IGNORE for l in row["labels"]))
print(len(train_rows), "train /", len(val_rows), "val_select rows")"""),
(PY, """# ---- trainer ----------------------------------------------------------------------------------
from transformers import Trainer, TrainingArguments, TrainerCallback
from datasets import Dataset
PAD = tok.pad_token_id
def collate(batch):
    n = max(len(b["input_ids"]) for b in batch)
    ids = torch.full((len(batch), n), PAD); lab = torch.full((len(batch), n), P.IGNORE); att = torch.zeros((len(batch), n), dtype=torch.long)
    for i, b in enumerate(batch):
        k = len(b["input_ids"]); ids[i, :k] = torch.tensor(b["input_ids"]); lab[i, :k] = torch.tensor(b["labels"]); att[i, :k] = 1
    return {"input_ids": ids, "labels": lab, "attention_mask": att}

class Meter(TrainerCallback):
    \"\"\"Token throughput excluding warmup steps; finite-loss check; complete-checkpoint copies to Drive.\"\"\"
    def __init__(self, tokens_per_step, warmup=2):
        self.tps, self.warm, self.t0, self.losses, self.tokens = tokens_per_step, warmup, None, [], 0
    def on_step_end(self, args, state, control, **kw):
        if state.global_step == self.warm: self.t0 = time.time(); self.tokens = 0
        elif state.global_step > self.warm: self.tokens += self.tps
    def on_log(self, args, state, control, logs=None, **kw):
        if logs and "loss" in logs:
            assert math.isfinite(logs["loss"]), f"non-finite loss at step {state.global_step}"
            self.losses.append((state.global_step, logs["loss"]))
    def on_save(self, args, state, control, **kw):
        ck = f"{args.output_dir}/checkpoint-{state.global_step}"
        open(f"{ck}/COMPLETE", "w").write(json.dumps({"step": state.global_step, "time": time.time()}))
        if DRIVE_DIR:
            dst = f"{DRIVE_DIR}/checkpoint-{state.global_step}"
            shutil.copytree(ck, dst + ".part", dirs_exist_ok=True); os.replace(dst + ".part", dst)
    def throughput(self):
        return self.tokens / (time.time() - self.t0) if self.t0 else None

def make_args(out, max_steps=-1, save_every=250):
    return TrainingArguments(
        output_dir=out, per_device_train_batch_size=1, gradient_accumulation_steps=16,
        num_train_epochs=1, max_steps=max_steps, learning_rate=1e-4, lr_scheduler_type="cosine",
        warmup_ratio=0.05, weight_decay=0.01, max_grad_norm=1.0, optim="adamw_8bit",
        bf16=BF16, fp16=not BF16, logging_steps=1 if MODE == "smoke" else 10,
        eval_strategy="steps", eval_steps=save_every, save_strategy="steps", save_steps=save_every,
        save_total_limit=3, seed=SEED, data_seed=SEED, report_to="none", remove_unused_columns=False)
avg_tokens = sum(len(r["input_ids"]) for r in train_rows) / len(train_rows)
STEPS = 25 if MODE == "smoke" else -1
OUT = f"/content/run_{MODE}"
meter = Meter(tokens_per_step=avg_tokens * 16)
trainer = Trainer(model=model, args=make_args(OUT, STEPS, 10 if MODE == "smoke" else 250),
                  train_dataset=Dataset.from_list(train_rows), eval_dataset=Dataset.from_list(val_rows),
                  data_collator=collate, callbacks=[meter])"""),
(PY, """# ---- train (resumes from the newest COMPLETE checkpoint locally or on Drive) -------------------
def latest_complete(*dirs):
    cks = [os.path.join(d, c) for d in dirs if d and os.path.isdir(d) for c in os.listdir(d)
           if c.startswith("checkpoint-") and not c.endswith(".part") and os.path.exists(os.path.join(d, c, "COMPLETE"))]
    return max(cks, key=lambda c: int(c.rsplit("-", 1)[1]), default=None)
resume = latest_complete(OUT, DRIVE_DIR)
print("resuming from", resume) if resume else print("fresh start")
torch.cuda.reset_peak_memory_stats()
t_start = time.time()
result = trainer.train(resume_from_checkpoint=resume)
TRAIN = {"global_steps": trainer.state.global_step, "train_loss": result.training_loss,
         "wall_s": round(time.time() - t_start, 1), "tokens_per_s_excl_warmup": meter.throughput(),
         "peak_mem_gb": round(torch.cuda.max_memory_allocated() / 2**30, 2), "losses": meter.losses,
         "resumed_from": resume}
assert TRAIN["global_steps"] >= 20 or MODE != "smoke", "smoke must complete >= 20 optimizer steps"
print(json.dumps({k: v for k, v in TRAIN.items() if k != "losses"}, indent=1))
if MODE != "smoke":
    n_steps = math.ceil(len(train_rows) / 16)
    print(f"estimate: {n_steps} steps; {n_steps * avg_tokens * 16 / TRAIN['tokens_per_s_excl_warmup'] / 3600:.1f} h at observed throughput")"""),
(PY, """# ---- predictions on val_select (diagnostic only; final test is a separate frozen protocol) ------
FastLanguageModel.for_inference(model)
def predict(recs, n=None):
    out = []
    for r in recs[:n]:
        prompt = tok.apply_chat_template(r["messages"][:2], tokenize=False, add_generation_prompt=True, enable_thinking=False)
        ids = tok(prompt, return_tensors="pt", add_special_tokens=False).to("cuda")
        t = time.time()
        gen = model.generate(**ids, max_new_tokens=24, do_sample=False, eos_token_id=tok.eos_token_id)
        raw = tok.decode(gen[0, ids["input_ids"].shape[1]:], skip_special_tokens=True).strip()
        out.append({"id": r["id"], "gold": r["metadata"]["threat_type"], "raw": raw,
                    "pred": parse_output(raw, TAX), "latency_s": time.time() - t})
    return out
preds = predict(val_recs, 60 if MODE == "smoke" else None)
valid = sum(p["pred"] is not None for p in preds) / len(preds)
acc = sum(p["pred"] == p["gold"] for p in preds) / len(preds)
PRED = {"n": len(preds), "raw_json_valid_rate": valid, "accuracy_diagnostic": acc,
        "latency_p50_s": sorted(p["latency_s"] for p in preds)[len(preds) // 2]}
print(json.dumps(PRED, indent=1)); print([p["raw"] for p in preds[:5]])"""),
(PY, """# ---- export adapter + reload in a FRESH process -----------------------------------------------
EXP = f"/content/export_{MODE}"
model.save_pretrained(EXP); tok.save_pretrained(EXP)
json.dump({"base_model": MODEL_ID, "revision": MODEL_REVISION, "fingerprint": fp_now, "env": ENV,
           "lora": {"r": 16, "alpha": 32, "rslora": True}}, open(f"{EXP}/export_info.json", "w"), indent=1)
sample = [{"messages": r["messages"][:2], "expect": p["raw"]} for r, p in zip(val_recs, preds[:5])]
json.dump(sample, open("/content/reload_sample.json", "w"))
open("/content/reload_check.py", "w").write('''
import json, sys
from unsloth import FastLanguageModel
m, proc = FastLanguageModel.from_pretrained(model_name=sys.argv[1], max_seq_length=2048, load_in_4bit=False, load_in_16bit=True)
tok = getattr(proc, "tokenizer", proc); FastLanguageModel.for_inference(m)
ok = 0; s = json.load(open("/content/reload_sample.json"))
for x in s:
    p = tok.apply_chat_template(x["messages"], tokenize=False, add_generation_prompt=True, enable_thinking=False)
    ids = tok(p, return_tensors="pt", add_special_tokens=False).to("cuda")
    g = m.generate(**ids, max_new_tokens=24, do_sample=False, eos_token_id=tok.eos_token_id)
    ok += tok.decode(g[0, ids["input_ids"].shape[1]:], skip_special_tokens=True).strip() == x["expect"]
print(json.dumps({"reload_match": ok, "of": len(s)}))
''')
r = subprocess.run([sys.executable, "/content/reload_check.py", EXP], capture_output=True, text=True)
RELOAD = json.loads(r.stdout.strip().splitlines()[-1]) if r.returncode == 0 else {"error": r.stderr[-2000:]}
print(RELOAD)"""),
(PY, """# ---- optional: continuation test (uninterrupted N steps vs N/2 + save/reload + N/2) ----------
RESUME = None
if MODE == "smoke" and RUN_RESUME_TEST:
    def run(out, steps, resume=None):
        m2, _ = FastLanguageModel.from_pretrained(model_name=MODEL_ID, revision=MODEL_REVISION, max_seq_length=MAX_LEN, load_in_4bit=False, load_in_16bit=True)
        m2 = FastLanguageModel.get_peft_model(m2, r=16, lora_alpha=32, use_rslora=True, lora_dropout=0, bias="none",
            target_modules=["q_proj","k_proj","v_proj","o_proj","gate_proj","up_proj","down_proj"],
            use_gradient_checkpointing="unsloth", random_state=SEED, max_seq_length=MAX_LEN)
        a = make_args(out, 20, 10); a.eval_strategy = "no"
        t = Trainer(model=m2, args=a, train_dataset=Dataset.from_list(train_rows), data_collator=collate)
        if resume is None and steps < 20:
            t.add_callback(type("Stop", (TrainerCallback,), {"on_step_end": lambda s, a, st, c, **k: setattr(c, "should_training_stop", st.global_step >= steps) or c})())
        t.train(resume_from_checkpoint=resume)
        losses = [h["loss"] for h in t.state.log_history if "loss" in h]
        del t, m2; torch.cuda.empty_cache()
        return losses
    full = run("/content/rt_full", 20)
    half = run("/content/rt_half", 10)
    cont = run("/content/rt_half", 20, resume="/content/rt_half/checkpoint-10")
    diff = max(abs(a - b) for a, b in zip(full[10:20], cont[-10:]))
    RESUME = {"uninterrupted_tail": full[10:20], "continued_tail": cont[-10:], "max_abs_loss_diff": diff,
              "tolerance": 0.05, "pass": diff <= 0.05}
    print(json.dumps(RESUME, indent=1))"""),
(PY, """# ---- smoke report (the evidence for 'Notebook smoke-tested') ---------------------------------
REPORT = {"mode": MODE, "env": ENV, "fingerprint": fp_now, "trainable_params": n_train,
          "train": TRAIN, "predictions": PRED, "reload": RELOAD, "resume_test": RESUME,
          "passed": bool(TRAIN["global_steps"] >= 20 and all(math.isfinite(l) for _, l in TRAIN["losses"])
                         and RELOAD.get("reload_match") == RELOAD.get("of")
                         and (RESUME is None or RESUME["pass"]))}
name = "gpu_smoke_test.json" if MODE == "smoke" else f"train_{MODE}_report.json"
json.dump(REPORT, open(f"/content/{name}", "w"), indent=1)
if DRIVE_DIR: shutil.copy(f"/content/{name}", DRIVE_DIR)
print("passed:", REPORT["passed"], "->", name, "(download it into reports/)")
shutil.make_archive(f"/content/export_{MODE}", "zip", EXP)"""),
]

nb = nbformat.v4.new_notebook()
nb.metadata = {"accelerator": "GPU", "colab": {"provenance": []},
               "kernelspec": {"name": "python3", "display_name": "Python 3"}}
nb.cells = [nbformat.v4.new_markdown_cell(s) if t == MD else nbformat.v4.new_code_cell(s) for t, s in CELLS]
nbformat.validate(nb)
out = Path(__file__).with_name("Qwen3_5_4B_WAF_FineTune.ipynb")
nbformat.write(nb, out)
print(out, len(nb.cells), "cells")
