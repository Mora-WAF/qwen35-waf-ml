"""The TPU-trained Gemma 4 E2B adapter merged into its base, saved as a text-only Gemma4ForCausalLM (vision and audio
towers dropped: a WAF classifier, not a general model) for the Hugging Face Hub, with the scorer code, policy,
taxonomy and self-check records under waf/ (the Gemma-only endpoint package's code/, so Hub and endpoint run one path).

  GEMMA_ONLY=1 PACKAGE_ONLY=/tmp/gemma.tar.gz deploy/deploy.sh      the endpoint package (adapter, selfcheck, code)
  .venv/bin/python -m training.gemma_tpu.merge /tmp/gemma.tar.gz OUT_DIR
  hf repo create hamedgls/waf-gemma4-e2b-v13 --private && hf upload hamedgls/waf-gemma4-e2b-v13 OUT_DIR

Merge: W += SCALE * B A on every mapped linear, in float32 per layer, cast back to bf16 once (what a float32 merge of
the whole model gives, at bf16 memory: ~11 GB). The saved model is reloaded from OUT_DIR and must pass
hf_scorer.selfcheck against the TPU scores of the package's records (same top class, probability difference < 0.05).
"""
import json
import shutil
import sys
import tarfile
import tempfile
from pathlib import Path

import torch

from training.gemma_tpu import hf_scorer as HS

CARD = """---
license: apache-2.0
base_model: google/gemma-4-E2B-it
library_name: transformers
pipeline_tag: text-generation
tags: [waf, security, classification, gemma4]
---

# waf-gemma4-e2b-v13

Gemma 4 E2B-it fine-tuned (LoRA r16, merged; text-only: vision and audio towers removed) as a web-application /
network threat classifier over 42 classes (`waf/config/taxonomy.yaml`). Input: one request or flow as telemetry
JSON; output: `{{"threat_type":"<class>"}}`. Research model: evaluated on held-out splits of its own build (synthetic
and public-dataset records), not yet on public WAF benchmarks or real-traffic false-positive sets.

## Use

Decisions come from the scorer, not free generation: summed completion log-prob of every class, softmax at the frozen
temperature, then the frozen policy (threshold, abstain rules, flow envelope) in `waf/config/policy.yaml`
(act_on {act_on}, T {T}, threshold {tau}). Greedy generation gives the top class without calibration or abstention.

```python
import json, sys, torch
from huggingface_hub import snapshot_download
from pathlib import Path
d = Path(snapshot_download("{repo}"))
sys.path.insert(0, str(d / "waf"))
from transformers import AutoTokenizer, Gemma4ForCausalLM
from eval import policy as POL
from training.gemma_tpu import data as GD, hf_scorer as HS
tok = AutoTokenizer.from_pretrained(d)
model = Gemma4ForCausalLM.from_pretrained(d, dtype=torch.bfloat16, device_map="cuda").eval()
tax, pol, _ = POL.load(d / "waf")
suf = GD.class_suffixes(tok, list(tax["classes"]))
contract = json.load(open(d / "waf/serving.json"))["fit_contract"]
telemetry = json.loads(open(d / "waf/selfcheck.jsonl").readline())["telemetry"]  # or your sensor's record
print(HS.score_records(model, tok, [{{"id": "1", "telemetry": telemetry}}], tax, pol, suf, contract=contract))
```

`waf/selfcheck.jsonl` holds test records with the TPU scores: `HS.selfcheck` reproduces them on your hardware.

## Results (v13 build, attack-mass policy)

| split | accuracy | attack recall | benign false intervention |
|---|---|---|---|
| test | 0.9874 | 0.9942 | 0.0036 |
| test_unseen (held-out families) | 0.9722 | 0.9878 | 0.0086 |
"""


def merge(model, factors, scale=HS.SCALE):
    """W += scale * B A in float32, cast back to the weight dtype; returns the number of merged linears."""
    text, _, _ = HS.text_parts(model)
    for (layer, path), (A, B) in factors.items():
        mod = text.layers[layer]
        for part in path.split("."):
            mod = getattr(mod, part)
        w = mod.weight
        assert tuple(w.shape) == (B.shape[0], A.shape[1]), f"{layer}.{path}: weight {tuple(w.shape)}"
        delta = torch.tensor(B, dtype=torch.float32, device=w.device) @ torch.tensor(A, dtype=torch.float32, device=w.device)
        w.data = (w.data.float() + scale * delta).to(w.dtype)
    return len(factors)


def text_only(cg):
    """Gemma4ForConditionalGeneration -> Gemma4ForCausalLM sharing its language-model tensors (no copy)."""
    from transformers import Gemma4ForCausalLM
    cfg = cg.config.get_text_config()
    cfg.architectures = ["Gemma4ForCausalLM"]
    with torch.device("meta"):
        lm = Gemma4ForCausalLM(cfg)
    sd = {f"model.{k}": v for k, v in cg.model.language_model.state_dict().items()} | {"lm_head.weight": cg.lm_head.weight}
    lm.load_state_dict(sd, assign=True, strict=True)
    lm.tie_weights()
    return lm


def main(package, out, repo="hamedgls/waf-gemma4-e2b-v13", device="cpu"):
    import yaml
    from transformers import AutoTokenizer, Gemma4ForCausalLM, Gemma4ForConditionalGeneration, GenerationConfig
    from training.gemma_tpu import data as GD
    out = Path(out)
    assert not out.exists(), f"{out} exists"
    with tempfile.TemporaryDirectory() as t:
        tarfile.open(package).extractall(t, filter="data")
        t = Path(t)
        assert not (t / "adapter").exists(), "not a Gemma-only package (deploy.sh GEMMA_ONLY=1)"
        cg = Gemma4ForConditionalGeneration.from_pretrained(GD.MODEL_ID, revision=GD.MODEL_REVISION,
                                                            dtype=torch.bfloat16).to(device).eval()
        print("merged", merge(cg, HS.lora_factors(t / "gemma/lora.npz")), "linears", flush=True)
        text_only(cg).save_pretrained(out)
        del cg
        AutoTokenizer.from_pretrained(GD.MODEL_ID, revision=GD.MODEL_REVISION).save_pretrained(out)
        GenerationConfig.from_pretrained(GD.MODEL_ID, revision=GD.MODEL_REVISION).save_pretrained(out)
        shutil.copytree(t / "code", out / "waf", ignore=shutil.ignore_patterns("inference.py", "requirements.txt"))
        shutil.copy(t / "gemma/selfcheck.jsonl", out / "waf/selfcheck.jsonl")
    pol = yaml.safe_load(open(out / "waf/config/policy.yaml"))
    tax = yaml.safe_load(open(out / "waf/config/taxonomy.yaml"))
    (out / "README.md").write_text(CARD.format(repo=repo, act_on=pol["act_on"], T=pol["temperature"], tau=pol["threshold"]))
    # the files as the Hub will serve them must reproduce the TPU scores
    model = Gemma4ForCausalLM.from_pretrained(out, dtype=torch.bfloat16).to(device).eval()
    tok = AutoTokenizer.from_pretrained(out)
    checks = [json.loads(line) for line in open(out / "waf/selfcheck.jsonl")]
    diff = HS.selfcheck(model, tok, checks, float(pol["temperature"]), list(tax["classes"]))
    print(f"selfcheck: merged text-only model vs TPU, max prob diff {diff:.4f} over {len(checks)} records")


if __name__ == "__main__":
    main(*sys.argv[1:3], **dict(a.split("=", 1) for a in sys.argv[3:]))
