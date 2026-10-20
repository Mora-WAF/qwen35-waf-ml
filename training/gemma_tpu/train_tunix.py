"""Gemma 4 E2B-it LoRA fine-tune + class scoring on a TPU v5e-1 with JAX / Tunix / Qwix (runs on the Colab TPU VM).

Inputs in DATA (from training.gemma_tpu.data): <split>_<L>.npz per length bucket + meta.json.
Same recipe as the Qwen run where the stack allows: LoRA r16 on q/k/v/o + gate/up/down, effective scale of rsLoRA
(alpha / sqrt(r) = 8 -> qwix alpha = 8 r), AdamW lr 1e-4 cosine with 5% warmup, wd 0.01, clip 1.0, effective
batch 16 records, 1 epoch. Loss on the supervised answer only, projected at those K positions (skip_lm_head): the
default Tunix loss builds a one-hot over the 262k vocabulary at every position, gigabytes on a 16 GB chip.

  python train_tunix.py probe [STEPS]   compile time, HBM, records/s per bucket -> OUT/probe.json
  python train_tunix.py train           full run, LoRA weights -> OUT/lora.npz (resumes from OUT/ckpt.npz)
  python train_tunix.py score SPLIT...  class log-probs per record -> OUT/SCORES_<split>.jsonl
"""
import dataclasses
import functools
import json
import os
import sys
import time
from pathlib import Path

# 20 Oct: the Colab TPU image ships torch_tpu; tunix -> metrax -> tensorboardX imports torch, whose backend autoload
# rejects the libtpu version string ("0.0.44.1") that the tunix install brings. Torch is only needed on the CPU here.
os.environ.setdefault("TORCH_DEVICE_BACKEND_AUTOLOAD", "0")

import jax
import jax.numpy as jnp
import numpy as np
import optax
import qwix
from flax import nnx

DATA = Path(os.environ.get("GEMMA_DATA", "/content/data"))
OUT = Path(os.environ.get("GEMMA_OUT", "/content/out"))
CKPT_EVERY = int(os.environ.get("GEMMA_CKPT_EVERY", 100))  # optimizer steps between OUT/ckpt.npz saves
LIMIT = int(os.environ.get("GEMMA_LIMIT", 0))  # >0: first N records per split bucket (small end-to-end test)
ONLY = set(Path(os.environ["GEMMA_ONLY"]).read_text().split()) if os.environ.get("GEMMA_ONLY") else None  # id file: rescore
MICRO = {512: 8, 1024: 4, 2048: 2}  # records per micro-batch per bucket (tokens per step ~4k)
EFFECTIVE_BATCH, LR, WARMUP, WD, CLIP = 16, 1e-4, 0.05, 0.01, 1.0
RANK = int(os.environ.get("GEMMA_RANK", 16))
EPOCHS = int(os.environ.get("GEMMA_EPOCHS", 1))
LORA_TARGETS = ".*q_einsum|.*k_einsum|.*kv_einsum|.*attn_vec_einsum|.*gate_proj|.*up_proj|.*down_proj"
if os.environ.get("GEMMA_PLE_LORA"):  # also the per-layer-embedding projections (Gemma E2B's PLE path)
    LORA_TARGETS += "|.*per_layer_input_gate|.*per_layer_projection|.*per_layer_model_projection"
META = json.loads((DATA / "meta.json").read_text())



def load_model():
    """(graphdef, lora fp32 master weights, frozen bf16 weights, mesh). Functional split on purpose: nnx.jit returns
    the whole model state, an undonated 8.7 GB copy per call (OOM on 16 GB HBM, 17.5 GB temporaries at 1 x 512)."""
    from huggingface_hub import snapshot_download
    from tunix.models.gemma4 import model as gemma4_lib, params_safetensors
    path = snapshot_download(META["model"], revision=META["revision"], allow_patterns=["*.json", "*.safetensors"])
    # compute in bf16 (the default f32 upcasts the weights in the graph); remat off while qwix creates the LoRA params:
    # they are created lazily on the dummy call, and nnx.remat(graph_updates=False) would discard them
    cfg = dataclasses.replace(gemma4_lib.ModelConfig.gemma4_e2b_it(), dtype=jnp.bfloat16)
    mesh = jax.make_mesh((1, 1), ("fsdp", "tp"), axis_types=(jax.sharding.AxisType.Auto,) * 2)
    with jax.set_mesh(mesh):
        base = params_safetensors.create_model_from_safe_tensors(path, cfg, mesh, dtype=jnp.bfloat16, text_only=True)
        model = qwix.apply_lora_to_model(base, qwix.LoraProvider(module_path=LORA_TARGETS, rank=RANK, alpha=8.0 * RANK),
                                         rngs=nnx.Rngs(3407), **base.get_model_input())
    for _, m in nnx.iter_modules(model):  # then BLOCK remat for activation memory at 2048 tokens
        c = getattr(m, "config", None)
        if hasattr(c, "remat_config"):
            object.__setattr__(c, "remat_config", gemma4_lib.RematConfig.BLOCK)
    gd, lora, frozen = nnx.split(model, nnx.LoRAParam, ...)
    lora = jax.tree.map(lambda x: x.astype(jnp.float32), lora)
    n = sum(x.size for x in jax.tree.leaves(lora))
    assert n > 0, "no LoRA params created"
    print("lora params", n, flush=True)
    return gd, lora, frozen, mesh


def batch_inputs(tokens, length):
    """positions and causal attention mask from the padding (right padding)."""
    from tunix.sft import utils
    pad = jnp.arange(tokens.shape[1])[None, :] < length[:, None]
    return utils.build_positions_from_mask(pad), utils.make_causal_attn_mask(pad)



def final_logits(model, h):
    """LM head (tied embedding) with f32 accumulation; casting the 262k x 1536 table to f32 would cost 1.5 GB."""
    lg = jnp.einsum("bkd,vd->bkv", h.astype(jnp.bfloat16), model.embedder.input_embedding[...],
                    preferred_element_type=jnp.float32)
    cap = model.config.final_logit_softcap
    return jnp.tanh(lg / cap) * cap if cap else lg


def make_steps(gd):
    """(logprobs_at, grad_step) jitted over a closed-over graphdef (GraphDef is not hashable: no static arg)."""
    def fwd_logprobs(lora, frozen, tokens, length, pos):
        """Full next-token log-softmax at the K positions in `pos`: hidden states only there, then the LM head."""
        model = nnx.merge(gd, jax.tree.map(lambda x: x.astype(jnp.bfloat16), lora), frozen)
        positions, mask = batch_inputs(tokens, length)
        h, _ = model(tokens, positions, None, mask, skip_lm_head=True)
        return jax.nn.log_softmax(final_logits(model, jnp.take_along_axis(h, pos[..., None], axis=1)), -1)

    @functools.partial(jax.jit, donate_argnums=3)
    def grad_step(lora, frozen, b, acc, w):
        """Loss on the supervised answer tokens; returns (loss, acc + w * grad)."""
        def loss_fn(l):
            lp = fwd_logprobs(l, frozen, b["tokens"], b["length"], b["tgt_pos"])
            lp = jnp.take_along_axis(lp, b["tgt_tok"][..., None], -1)[..., 0]
            m = b["tgt_mask"].astype(jnp.float32)
            return -(lp * m).sum() / jnp.maximum(m.sum(), 1.0)
        loss, g = jax.value_and_grad(loss_fn)(lora)
        return loss, jax.tree.map(lambda a, x: a + w * x, acc, g)
    return jax.jit(fwd_logprobs), grad_step


def make_optimizer(total_steps):
    sched = optax.warmup_cosine_decay_schedule(0.0, LR, max(1, int(WARMUP * total_steps)), max(2, total_steps))
    tx = optax.chain(optax.clip_by_global_norm(CLIP), optax.adamw(sched, weight_decay=WD))

    @functools.partial(jax.jit, donate_argnums=(0, 1))
    def apply_step(lora, opt_state, grads):
        upd, opt_state = tx.update(grads, opt_state, lora)
        return optax.apply_updates(lora, upd), opt_state
    return tx, apply_step


def zeros_like(tree):
    return jax.tree.map(jnp.zeros_like, tree)


def flat(tree):
    return {jax.tree_util.keystr(p): np.asarray(x) for p, x in jax.tree_util.tree_flatten_with_path(tree)[0]}


def unflat(like, saved):
    paths, td = jax.tree_util.tree_flatten_with_path(like)
    return jax.tree_util.tree_unflatten(td, [jnp.asarray(saved[jax.tree_util.keystr(p)], x.dtype) for p, x in paths])


def load_split(split):
    out = {}
    for b in META["buckets"]:
        fp = DATA / f"{split}_{b}.npz"
        if fp.exists():
            d = dict(np.load(fp))
            sel = np.isin(d["ids"], list(ONLY)) if ONLY else slice(LIMIT or None)
            out[b] = {k: v[sel] for k, v in d.items()}
    return out


def batch_order(data, rng, epochs=1):
    """[(bucket, record indices)] in random order, one shuffled pass per epoch (epochs never mix)."""
    order = []
    for _ in range(epochs):
        ep = []
        for b, d in data.items():
            idx, mb = rng.permutation(len(d["tokens"])), MICRO[b]
            ep += [(b, idx[i:i + mb]) for i in range(0, len(idx), mb)]
        rng.shuffle(ep)
        order += ep
    return order


def count_steps(order):
    """Optimizer steps train() takes over `order`: micro-batches of 8/4/2 overshoot the effective batch, so this is
    fewer than records / 16 (18 Oct: 2,076, not 2,290; the cosine schedule then never reached its end)."""
    seen = steps = 0
    for _, idx in order:
        seen += len(idx)
        if seen >= EFFECTIVE_BATCH:
            seen, steps = 0, steps + 1
    return steps + (seen > 0)


def micro_batches(data, order, skip=0):
    """(bucket, batch dict, real records) for `order`; each micro-batch has MICRO[bucket] records (the last one is
    padded with zero-mask copies, so shapes never change). `skip`: micro-batches already consumed (resume)."""
    for b, idx in order[skip:]:
        d, mb = data[b], MICRO[b]
        pad = mb - len(idx)
        sel = np.concatenate([idx, np.repeat(idx[:1], pad)]) if pad else idx
        batch = {k: d[k][sel] for k in ("tokens", "length", "tgt_pos", "tgt_tok", "tgt_mask")}
        if pad:
            batch["tgt_mask"][-pad:] = 0
        yield b, {k: jnp.asarray(v) for k, v in batch.items()}, len(idx)



def hbm():
    s = jax.devices()[0].memory_stats() or {}
    return {k: round(s[k] / 2**30, 2) for k in ("bytes_in_use", "peak_bytes_in_use", "bytes_limit") if k in s}



def probe(steps=30):
    """Per bucket: first-call (compile) time and steady micro-steps/s; no weights written."""
    t0 = time.time()
    gd, lora, frozen, mesh = load_model()
    _, grad_step = make_steps(gd)
    rep = {"load_s": round(time.time() - t0, 1), "hbm_after_load_gb": hbm(), "devices": str(jax.devices()), "buckets": {}}
    with jax.set_mesh(mesh):
        tx, apply_step = make_optimizer(1000)
        opt_state = tx.init(lora)
        data = load_split("train")
        for b in sorted(data):
            d, mb = data[b], MICRO[b]
            batch = {k: jnp.asarray(d[k][:mb]) for k in ("tokens", "length", "tgt_pos", "tgt_tok", "tgt_mask")}
            t, losses = time.time(), []
            for i in range(steps + 1):
                loss, acc = grad_step(lora, frozen, batch, zeros_like(lora), 1.0)
                lora, opt_state = apply_step(lora, opt_state, acc)
                losses.append(loss)
                if i == 0:
                    jax.block_until_ready(loss)
                    compile_s, t = time.time() - t, time.time()
            jax.block_until_ready(losses[-1])
            dt = (time.time() - t) / steps
            rep["buckets"][b] = {"micro_batch": mb, "first_step_s": round(compile_s, 1), "step_s": round(dt, 3),
                                 "records_per_s": round(mb / dt, 2), "tokens_per_s": round(mb * b / dt),
                                 "loss_first_last": [round(float(losses[0]), 4), round(float(losses[-1]), 4)],
                                 "hbm_gb": hbm(), "train_records": int(len(d["tokens"]))}
            print(b, rep["buckets"][b], flush=True)
    rep["train_records"] = sum(v["train_records"] for v in rep["buckets"].values())
    rep["estimate_epoch_h"] = round(sum(v["train_records"] / v["records_per_s"] for v in rep["buckets"].values()) / 3600, 2)
    (OUT / "probe.json").write_text(json.dumps(rep, indent=1))
    print(json.dumps(rep, indent=1))


def train():
    gd, lora, frozen, mesh = load_model()
    _, grad_step = make_steps(gd)
    data = load_split("train")
    n = sum(len(d["tokens"]) for d in data.values())
    order = batch_order(data, np.random.default_rng(3407), EPOCHS)
    total = count_steps(order)
    tx, apply_step = make_optimizer(total)
    opt_state = tx.init(lora)
    step, mi, ck = 0, 0, OUT / "ckpt.npz"
    if ck.exists():  # resume: weights, optimizer state and position in the (seeded) micro-batch order
        z = np.load(ck)
        lora = unflat(lora, {k[2:]: z[k] for k in z.files if k.startswith("l:")})
        opt_state = unflat(opt_state, {k[2:]: z[k] for k in z.files if k.startswith("o:")})
        step, mi = int(z["step"]), int(z["micro"])
        print("resumed at step", step, "micro", mi, flush=True)
    log = open(OUT / "train_log.jsonl", "a")
    t0 = time.time()
    with jax.set_mesh(mesh):
        acc, seen, losses = zeros_like(lora), 0, []
        for b, batch, real in micro_batches(data, order, skip=mi):
            loss, acc = grad_step(lora, frozen, batch, acc, real / EFFECTIVE_BATCH)  # grad of the effective-batch mean
            seen, mi = seen + real, mi + 1
            losses.append(loss)
            if seen >= EFFECTIVE_BATCH:
                lora, opt_state = apply_step(lora, opt_state, acc)
                acc, seen, step = zeros_like(lora), 0, step + 1
                if step % 20 == 0:
                    rec = {"step": step, "of": total, "loss": round(float(np.mean([float(x) for x in losses])), 5),
                           "elapsed_s": round(time.time() - t0), "hbm_gb": hbm()}
                    log.write(json.dumps(rec) + "\n"); log.flush(); print(rec, flush=True)
                    losses = []
                if step % CKPT_EVERY == 0:
                    np.savez(OUT / "ckpt.tmp.npz", step=step, micro=mi, **{"l:" + k: v for k, v in flat(lora).items()},
                             **{"o:" + k: v for k, v in flat(opt_state).items()})
                    os.replace(OUT / "ckpt.tmp.npz", ck)
        if seen:
            lora, opt_state = apply_step(lora, opt_state, acc)
            step += 1
        np.savez(OUT / "lora.npz", **flat(lora))
        (OUT / "train_report.json").write_text(json.dumps({"steps": step, "planned_steps": total, "epochs": EPOCHS, "rank": RANK,
                                                           "lora_targets": LORA_TARGETS, "records": n, "wall_s": round(time.time() - t0),
                                                           "hbm_gb": hbm()}, indent=1))
    return gd, lora, frozen, mesh


def score(splits, loaded=None, margin=12.0):
    """Exact completion log-prob per class with the Qwen scorer's pruning: pass 1 scores the shared answer prefix and
    each class's first token; pass 2 the remaining tokens of the classes within `margin` nats of the best first token
    (prompt recomputed: no prefix cache, simple and fixed-shape)."""
    if loaded is None:
        gd, lora, frozen, mesh = load_model()
        lora = unflat(lora, dict(np.load(OUT / "lora.npz")))
    else:
        gd, lora, frozen, mesh = loaded
    logprobs_at, _ = make_steps(gd)
    suf, classes, K = META["suffixes"], META["classes"], META["K"]
    k = 0
    while all(len(s) > k + 1 and s[k] == suf[0][k] for s in suf):
        k += 1
    rest = [s[k:] for s in suf]
    with jax.set_mesh(mesh):
        for split in splits:
            fh = open(OUT / f"SCORES_{split}.jsonl", "w")
            fh.write(json.dumps({"classes": classes}) + "\n")
            for L, d in load_split(split).items():
                mb = MICRO[L]
                for i0 in range(0, len(d["tokens"]), mb):
                    sl = slice(i0, i0 + mb)
                    n = len(d["tokens"][sl])
                    pl = d["prompt_len"][sl]
                    # pass 1: prompt + shared prefix; full first-token distribution at the last position
                    toks = np.full((mb, L), META["pad_id"], np.int32)
                    pos = np.zeros((mb, K), np.int32)
                    for r in range(n):
                        seq = list(d["tokens"][sl][r][:pl[r]]) + suf[0][:k]
                        toks[r, :len(seq)] = seq
                        pos[r, :k + 1] = [pl[r] - 1 + j for j in range(k + 1)]
                    lens = np.array([pl[r] + k for r in range(n)] + [1] * (mb - n), np.int32)
                    lp = np.asarray(logprobs_at(lora, frozen, jnp.asarray(toks), jnp.asarray(lens), jnp.asarray(pos)))
                    rows, pend = [], []
                    for r in range(n):
                        shared = sum(lp[r, j, t] for j, t in enumerate(suf[0][:k]))
                        f0 = [lp[r, k, x[0]] for x in rest]
                        sc = [float(shared + v) for v in f0]
                        rows.append(sc)
                        pend += [(r, c) for c, v in enumerate(f0) if v >= max(f0) - margin and len(rest[c]) > 1]
                    # pass 2: kept classes, prompt + full completion, log-probs of tokens after the first class token
                    for g0 in range(0, len(pend), mb):
                        grp = pend[g0:g0 + mb]
                        # another class's completion can be longer than the record's own: a prompt near the bucket
                        # size then overflows it (out-of-bounds gather = NaN, 53 records on 18 Oct): next bucket up
                        need = max(pl[r] + len(suf[c]) for r, c in grp)
                        L2 = next(x for x in sorted(META["buckets"]) + [need] if x >= need and x >= L)
                        toks2 = np.full((mb, L2), META["pad_id"], np.int32)
                        tp, tt, tm = (np.zeros((mb, K), np.int32) for _ in range(3))
                        lens2 = np.ones(mb, np.int32)
                        for j, (r, c) in enumerate(grp):
                            seq = list(d["tokens"][sl][r][:pl[r]]) + suf[c]
                            toks2[j, :len(seq)] = seq
                            lens2[j] = len(seq)
                            ts = list(range(pl[r] + k + 1, len(seq)))[:K]
                            tp[j, :len(ts)] = [t - 1 for t in ts]
                            tt[j, :len(ts)] = [seq[t] for t in ts]
                            tm[j, :len(ts)] = 1
                        lp2 = np.asarray(logprobs_at(lora, frozen, jnp.asarray(toks2), jnp.asarray(lens2), jnp.asarray(tp)))
                        for j, (r, c) in enumerate(grp):
                            rows[r][c] += float(sum(lp2[j, q, tt[j, q]] for q in range(K) if tm[j, q]))
                    for r in range(n):
                        lab = int(d["label"][sl][r])  # -1: no class (benchmark attack outside the taxonomy, robustness)
                        fh.write(json.dumps({"id": str(d["ids"][sl][r]), "split": str(d["split"][sl][r]), "kind": None,
                                             "label": classes[lab] if lab >= 0 else None, "scores": rows[r]}) + "\n")
            fh.close()
            print("scored", split, flush=True)



if __name__ == "__main__":
    OUT.mkdir(parents=True, exist_ok=True)
    cmd = sys.argv[1]
    if cmd == "probe":
        probe(int(sys.argv[2]) if len(sys.argv) > 2 else 30)
    elif cmd == "train":
        score(["val", "test", "test_unseen"], train())
    elif cmd == "score":
        score(sys.argv[2:])
