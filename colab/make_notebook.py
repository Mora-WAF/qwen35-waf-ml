"""Builds colab/Qwen3_5_4B_WAF_FineTune.ipynb from the cells below (keeps the notebook diffable).

usage: python colab/make_notebook.py
"""
import json
from pathlib import Path

import nbformat

MD, PY = "markdown", "code"
CELLS = [
(MD, """# Qwen3.5-4B WAF / NIDS classifier — rsLoRA fine-tune

Two parts. **Part A** trains, predicts and exports, then stops. **Restart the runtime**, then run
**Part B** in that fresh process: adapter reload, continuation test, and `gpu_smoke_test.json` — the
only evidence that counts for the *Notebook smoke-tested* state. Run `MODE="smoke"` first.

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
(PY, """# ---- pinned stack (tested on Colab T4, torch 2.11.0+cu130, 2026-10-15). unsloth 2026.10.5 caps
# transformers<=5.17.0, trl<=1.15.0, datasets<5.0; torch pinned so xformers cannot swap the CUDA build
!pip install -q "torch==2.11.0" "unsloth==2026.10.5" "unsloth_zoo==2026.10.5" "transformers==5.17.0" "trl==1.15.0" "peft==0.21.2" "accelerate==1.15.0" "bitsandbytes==0.50.2" "datasets==4.8.5" "xformers==0.0.35" "torchao==0.18.0" pyyaml jsonschema
# If pip reports a torch / CUDA conflict: Runtime -> Restart session, then run from the next cell."""),
(PY, """import json, os, shutil, subprocess, sys, time, math, hashlib, zipfile, platform, random
import torch
assert torch.cuda.is_available(), "No GPU: Runtime -> Change runtime type -> GPU"
props = torch.cuda.get_device_properties(0)
BF16 = torch.cuda.is_bf16_supported() and props.major >= 8  # T4 (7.5) only emulates bf16 -> fp16
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
from training import colab_helpers as H
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
model = FastLanguageModel.get_peft_model(model, **H.LORA, random_state=SEED, max_seq_length=MAX_LEN)
trainable = [(n, p.numel()) for n, p in model.named_parameters() if p.requires_grad]
n_train, n_all = sum(n for _, n in trainable), sum(p.numel() for p in model.parameters())
assert not any("visual" in n or "vision" in n for n, _ in trainable), "vision params are trainable"
ENV["weights_dtype"] = str(next(p for n, p in model.named_parameters() if "lora" not in n).dtype)
print(f"trainable {n_train:,} / {n_all:,} ({100*n_train/n_all:.2f}%) in {len(trainable)} tensors | base weights {ENV['weights_dtype']}")
print("sample trainable modules:", sorted({n.split('.lora')[0].split('.')[-1] for n, _ in trainable}))"""),
(PY, """# ---- data: render with the checkpoint template, mask, verify ---------------------------------
train_recs = H.load_records(B, FILES[MODE])
val_recs = H.load_records(B, VAL, {"val_select"})          # model selection partition ONLY
train_rows, val_rows = H.encode(tok, train_recs), H.encode(tok, val_recs)
# mask inspection: supervised span must be exactly the target + <|im_end|>
for r, row in random.Random(SEED).sample(list(zip(train_recs, train_rows)), 5):
    sup = tok.decode([t for t, l in zip(row["input_ids"], row["labels"]) if l != P.IGNORE])
    assert sup == r["messages"][2]["content"] + "<|im_end|>", sup
    print("supervised:", repr(sup), "| prompt tokens masked:", sum(l == P.IGNORE for l in row["labels"]))
print(len(train_rows), "train /", len(val_rows), "val_select rows")"""),
(PY, """# ---- trainer ----------------------------------------------------------------------------------
from transformers import Trainer, TrainerCallback
from datasets import Dataset
collate = H.make_collate(tok.pad_token_id)

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

avg_tokens = sum(len(r["input_ids"]) for r in train_rows) / len(train_rows)
STEPS = 25 if MODE == "smoke" else -1
OUT = f"/content/run_{MODE}"
meter = Meter(tokens_per_step=avg_tokens * 16)
args = H.make_args(OUT, STEPS, 10 if MODE == "smoke" else 250, smoke=MODE == "smoke", seed=SEED)
print("optim", args.optim.value, "| clip", args.max_grad_norm, "| warmup", args.warmup_steps, "| fp16", args.fp16, "| bf16", args.bf16)
trainer = Trainer(model=model, args=args, train_dataset=Dataset.from_list(train_rows),
                  eval_dataset=Dataset.from_list(val_rows), data_collator=collate, callbacks=[meter])"""),
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
         "eval_losses": [(h["step"], h["eval_loss"]) for h in trainer.state.log_history if "eval_loss" in h],
         "resumed_from": resume}
assert TRAIN["global_steps"] >= 20 or MODE != "smoke", "smoke must complete >= 20 optimizer steps"
print(json.dumps({k: v for k, v in TRAIN.items() if k != "losses"}, indent=1))
if MODE != "smoke":
    n_steps = math.ceil(len(train_rows) / 16)
    print(f"estimate: {n_steps} steps; {n_steps * avg_tokens * 16 / TRAIN['tokens_per_s_excl_warmup'] / 3600:.1f} h at observed throughput")"""),
(PY, """# ---- predictions on val_select (diagnostic only; final test is a separate frozen protocol) ------
FastLanguageModel.for_inference(model)
preds = []
for r in val_recs[: 60 if MODE == "smoke" else None]:
    t = time.time()
    raw = H.generate_label(model, tok, r["messages"])
    preds.append({"id": r["id"], "gold": r["metadata"]["threat_type"], "raw": raw,
                  "pred": parse_output(raw, TAX), "latency_s": time.time() - t})
PRED = {"n": len(preds), "raw_json_valid_rate": sum(p["pred"] is not None for p in preds) / len(preds),
        "accuracy_diagnostic": sum(p["pred"] == p["gold"] for p in preds) / len(preds),
        "latency_p50_s": sorted(p["latency_s"] for p in preds)[len(preds) // 2]}
print(json.dumps(PRED, indent=1)); print([p["raw"] for p in preds[:5]])"""),
(PY, """# ---- export adapter; Part A ends here ---------------------------------------------------------
EXP = f"/content/export_{MODE}"
model.save_pretrained(EXP); tok.save_pretrained(EXP)
json.dump({"base_model": MODEL_ID, "revision": MODEL_REVISION, "fingerprint": fp_now, "env": ENV,
           "lora": {k: v for k, v in H.LORA.items() if k != "use_gradient_checkpointing"}},
          open(f"{EXP}/export_info.json", "w"), indent=1)
json.dump([{"messages": r["messages"][:2], "expect": p["raw"]} for r, p in zip(val_recs, preds[:5])],
          open("/content/reload_sample.json", "w"))
PART_A = {"mode": MODE, "seed": SEED, "model_id": MODEL_ID, "revision": MODEL_REVISION, "env": ENV,
          "fingerprint": fp_now, "trainable_params": n_train, "train": TRAIN, "predictions": PRED,
          "export_dir": EXP, "bundle_dir": B, "train_file": FILES[MODE]}
json.dump(PART_A, open("/content/part_a.json", "w"), indent=1, default=str)
print("Part A complete -> /content/part_a.json. Now: Runtime -> Restart session, then run Part B.")"""),
(MD, """## Part B — fresh process

**Restart the runtime first** (Runtime → Restart session). The cells below read only files under
`/content`, so they run in a new Python process: that is the reload test the spec asks for, and it frees
the GPU (a 4B model twice does not fit a 15 GB T4)."""),
(PY, """# ---- B1: reload the exported adapter in this fresh process -------------------------------------
import json, sys, os, math, gc, time, shutil, torch
PART_A = json.load(open("/content/part_a.json"))
B, EXP, MODE, SEED = PART_A["bundle_dir"], PART_A["export_dir"], PART_A["mode"], PART_A["seed"]
sys.path.insert(0, B)
assert torch.cuda.memory_allocated() == 0, "not a fresh process: restart the runtime first"
from unsloth import FastLanguageModel
from training import colab_helpers as H, prepare as P
m, proc = FastLanguageModel.from_pretrained(model_name=EXP, max_seq_length=P.MAX_TOKENS,
                                            load_in_4bit=False, load_in_16bit=True)
tok = getattr(proc, "tokenizer", proc)
FastLanguageModel.for_inference(m)
sample = json.load(open("/content/reload_sample.json"))
got = [H.generate_label(m, tok, x["messages"]) for x in sample]
RELOAD = {"reload_match": sum(g == x["expect"] for g, x in zip(got, sample)), "of": len(sample),
          "fingerprint_match": P.sha256(tok.chat_template) == PART_A["fingerprint"]["chat_template_sha256"],
          "weights_dtype": str(next(m.parameters()).dtype)}
print(RELOAD)
del m; gc.collect(); torch.cuda.empty_cache()"""),
(PY, """# ---- B2: continuation test (smoke only): 20 steps straight vs 10 + save/reload + 10 ----------
RESUME = None
if MODE == "smoke":
    from transformers import Trainer, TrainerCallback
    from datasets import Dataset

    class StopAt(TrainerCallback):
        def __init__(self, n): self.n = n
        def on_step_end(self, args, state, control, **kw):
            if state.global_step >= self.n: control.should_training_stop = True
            return control

    def run(out, stop=None, resume=None):
        m2, proc2 = FastLanguageModel.from_pretrained(model_name=PART_A["model_id"], revision=PART_A["revision"],
                                                      max_seq_length=P.MAX_TOKENS, load_in_4bit=False, load_in_16bit=True)
        tok2 = getattr(proc2, "tokenizer", proc2)
        m2 = FastLanguageModel.get_peft_model(m2, **H.LORA, random_state=SEED, max_seq_length=P.MAX_TOKENS)
        rows = H.encode(tok2, H.load_records(B, PART_A["train_file"])[:20 * 16])
        t = Trainer(model=m2, args=H.make_args(out, 20, 10, smoke=True, seed=SEED, evaluate=False),
                    train_dataset=Dataset.from_list(rows), data_collator=H.make_collate(tok2.pad_token_id),
                    callbacks=[StopAt(stop)] if stop else [])
        t.train(resume_from_checkpoint=resume)
        losses = {h["step"]: h["loss"] for h in t.state.log_history if "loss" in h}
        del t, m2; gc.collect(); torch.cuda.empty_cache()
        return losses
    full = run("/content/rt_full")
    run("/content/rt_half", stop=10)
    cont = run("/content/rt_half", resume="/content/rt_half/checkpoint-10")
    steps = [s for s in range(11, 21) if s in full and s in cont]
    diff = max(abs(full[s] - cont[s]) for s in steps)
    RESUME = {"steps_compared": steps, "uninterrupted": [full[s] for s in steps],
              "continued": [cont[s] for s in steps], "max_abs_loss_diff": diff,
              "tolerance": 0.05, "pass": bool(steps) and diff <= 0.05}
    print(json.dumps(RESUME, indent=1))"""),
(PY, """# ---- B3: smoke report (the evidence for 'Notebook smoke-tested') -------------------------------
REPORT = {**PART_A, "reload": RELOAD, "resume_test": RESUME}
REPORT["passed"] = bool(PART_A["train"]["global_steps"] >= 20
                        and all(math.isfinite(l) for _, l in PART_A["train"]["losses"])
                        and RELOAD["reload_match"] == RELOAD["of"] and RELOAD["fingerprint_match"]
                        and (RESUME is None or RESUME["pass"]))
name = "gpu_smoke_test.json" if MODE == "smoke" else f"train_{MODE}_report.json"
json.dump(REPORT, open(f"/content/{name}", "w"), indent=1, default=str)
shutil.make_archive(f"/content/export_{MODE}", "zip", EXP)
print("passed:", REPORT["passed"], "->", name, "(download it into reports/)")"""),
]

nb = nbformat.v4.new_notebook()
nb.metadata = {"accelerator": "GPU", "colab": {"provenance": []},
               "kernelspec": {"name": "python3", "display_name": "Python 3"}}
nb.cells = [nbformat.v4.new_markdown_cell(s) if t == MD else nbformat.v4.new_code_cell(s) for t, s in CELLS]
nbformat.validate(nb)
out = Path(__file__).with_name("Qwen3_5_4B_WAF_FineTune.ipynb")
nbformat.write(nb, out)
print(out, len(nb.cells), "cells")
# CLI runs (`colab exec -f`) cannot restart mid-notebook: also write the two halves
cut = next(i for i, c in enumerate(nb.cells) if c.cell_type == "markdown" and c.source.startswith("## Part B"))
for name, cells in (("partA", nb.cells[:cut]), ("partB", nb.cells[cut:])):
    part = nbformat.v4.new_notebook(metadata=nb.metadata, cells=cells)
    nbformat.write(part, out.with_name(f"{name}.ipynb"))
