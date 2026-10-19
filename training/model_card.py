"""Model card for a finished Colab run: README.md + enriched export_info.json, written into the run's export zip.

Everything is read from the run's own artifacts (training report, calibration, scores) and from the exact bundle it
trained on (data, taxonomy, system prompt), verified by sha256. Baselines are included only when reports/baselines.json
was computed on the same splits.

usage: python -m training.model_card reports/colab_NAME [--mode fast] [--bundle colab/colab_bundle_fast.zip]
"""
import argparse
import collections
import hashlib
import json
import math
import zipfile
from pathlib import Path

import yaml

from training import colab_helpers as H
from training import prepare as P

ROOT = Path(__file__).resolve().parent.parent


def sha256_file(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def softmax_pred(scores, T):
    s = [x / T for x in scores]
    m = max(s)
    z = sum(math.exp(x - m) for x in s)
    i = max(range(len(s)), key=s.__getitem__)
    return i, math.exp(s[i] - m) / z


def per_class(run, T, classes):
    """Per split and class: (correct, n) from the run's scores at the calibrated temperature."""
    out = collections.defaultdict(lambda: collections.defaultdict(lambda: [0, 0]))
    with open(run / f"SCORES_{MODE}.jsonl") as f:
        next(f)
        for line in f:
            r = json.loads(line)
            if r["split"] in ("test", "test_unseen"):
                i, _ = softmax_pred(r["scores"], T)
                c = out[r["split"]][r["label"]]
                c[0] += classes[i] == r["label"]
                c[1] += 1
    return out


def data_stats(zf):
    split, cls, prov, gen, lic = (collections.Counter() for _ in range(5))
    for fn in sorted(n for n in zf.namelist() if n.startswith(f"datasets_ready/{MODE}_") and n.endswith(".jsonl")):
        for line in zf.open(fn):
            m = json.loads(line)["metadata"]
            split[m["split"]] += 1
            cls[m["threat_type"]] += 1
            prov[m["provenance"]] += 1
            gen[m["generator"]] += 1
            lic[(m.get("source", "?"), m.get("license_ref", "?"))] += 1
    return split, cls, prov, gen, lic


def pct(x):
    return "n/a" if x is None else f"{100 * x:.1f}%"


def card(run, bundle, baselines=None):
    rep = json.load(open(run / f"train_{MODE}_report.json"))
    cal = json.load(open(run / "calibration.json"))
    if sha256_file(bundle) != rep["bundle_sha256"]:
        raise SystemExit(f"{bundle} is not the bundle {rep['bundle_sha256'][:12]} this run trained on (pass --bundle)")
    zf = zipfile.ZipFile(bundle)
    tax = yaml.safe_load(zf.open("config/taxonomy.yaml"))["classes"]
    classes = list(tax)
    sysp = json.loads(zf.open(f"datasets_ready/{MODE}_train.jsonl").readline())["messages"][0]["content"]
    split, cnt, prov, gen, lic = data_stats(zf)
    base = json.load(open(baselines or ROOT / "reports/baselines.json"))
    sizes = collections.Counter({"train": split["train"], "test": split["test"], "test_unseen": split["test_unseen"]})
    if any(base.get("sizes", {}).get(k) != v for k, v in sizes.items()):
        base = None  # computed on another build
    pc = per_class(run, cal["temperature"], classes)
    t, sc = rep["train"], rep["scores"]
    targets = rep.get("lora_target_modules") or H.LORA_TARGETS["v11"]  # runs before 19 Oct did not record it: the v11 set
    acc = {s: sum(c for c, _ in pc[s].values()) / max(1, sum(n for _, n in pc[s].values())) for s in pc}
    by_domain = collections.defaultdict(list)
    for c, v in tax.items():
        by_domain[v["domain"]].append(c)

    L = ["---", f"base_model: {rep['model_id']}", "library_name: peft", "pipeline_tag: text-generation", "language:", "- en",
         "tags:", "- security", "- web-application-firewall", "- intrusion-detection", "- classification", "- lora",
         "- rslora", "- unsloth", f"- base_model:adapter:{rep['model_id']}", "---", "",
         "# Qwen3.5-4B WAF/NIDS threat classifier (rsLoRA adapter)", "",
         f"A LoRA adapter for `{rep['model_id']}` (revision `{rep['revision']}`) that reads one piece of security telemetry "
         f"(an HTTP request, a network flow, or both with context) and returns exactly one of {len(classes)} taxonomy classes "
         "as JSON. It is a candidate second-stage classifier behind a deterministic decision policy, evaluated offline. "
         "**It is not qualified for inline blocking**: see Limitations.", "",
         "## Model details", "",
         f"- Base model: `{rep['model_id']}` @ `{rep['revision']}` (bf16 weights, 16-bit LoRA, no quantisation)",
         f"- Adapter: rank {H.LORA['r']}, alpha {H.LORA['lora_alpha']}, rsLoRA, dropout {H.LORA['lora_dropout']}, "
         f"targets {', '.join(targets)}; {rep['trainable_params']:,} trainable parameters",
         "- Output: `{\"threat_type\": \"<class>\"}` then end of turn; thinking disabled in the chat template",
         f"- Fingerprints: chat template `{rep['fingerprint']['chat_template_sha256'][:16]}…`, system prompt "
         f"`{rep['fingerprint']['system_prompt_sha256'][:16]}…`; training bundle `{rep['bundle_sha256'][:16]}…`",
         "- Language: English telemetry (payloads may contain other languages as data)", "",
         "### Classes", ""]
    for d, cs in sorted(by_domain.items()):
        L.append(f"- **{d}** ({len(cs)}): " + ", ".join(f"`{c}`" for c in cs))
    L += ["", "### Prompt", "", "System prompt (exact):", "", "```", sysp, "```", "",
          "User message: `[<MODE>] <telemetry JSON>`, where MODE is `L7_HTTP`, `L4_FLOW` or `HYBRID_CTX` and the telemetry "
          "follows `schemas/telemetry.schema.json`. Control strings inside telemetry are escaped (`training/prepare.py`), so "
          "payload text cannot break the turn structure.", "",
          "## How to use", "",
          "```python", "from transformers import AutoModelForCausalLM, AutoTokenizer", "from peft import PeftModel",
          f"base = AutoModelForCausalLM.from_pretrained(\"{rep['model_id']}\", revision=\"{rep['revision']}\", torch_dtype=\"bfloat16\")",
          "model = PeftModel.from_pretrained(base, \"<this adapter dir>\")",
          "tok = AutoTokenizer.from_pretrained(\"<this adapter dir>\")",
          "# messages = training.prepare record messages: system prompt + serialized telemetry",
          "# score: training.colab_helpers.class_logprobs -> softmax(scores / temperature) -> eval.policy.decide", "```", "",
          f"For decisions use calibrated class probabilities, not free generation: temperature **{cal['temperature']}**, "
          f"threshold tau **{cal['tau']}**, and the benign flow envelope frozen in `config/policy.yaml`. "
          "`eval.policy.decide` maps a class to ALLOW / BLOCK / ABSTAIN with fixed reason codes, and abstains on malformed "
          "output, unsupported mode, missing required context, low confidence, or a background-class flow outside the "
          "benign envelope.", "",
          "## Training data", "",
          f"{sum(split.values()):,} records ({', '.join(f'{k} {v:,}' for k, v in sorted(split.items()))}), all {len(cnt)} "
          f"classes in every evaluation split, smallest class {min(cnt.values())}. Splits are by lineage group (seed "
          "clusters, attacker campaigns, benign twins share a group); `test_unseen` holds out whole scenario families, flow "
          "variants and sources, so it measures generalisation rather than template recall.", "",
          "- Provenance: " + ", ".join(f"{k} {v:,}" for k, v in prov.most_common()),
          "- Generators: " + ", ".join(f"{k} {v:,}" for k, v in gen.most_common()),
          "- Observed sources: CIC-IDS2017, CIC-IDS2018, CIC-DDoS2019, CIC-Bell-DNS-EXF-2021 and NF-ToN-IoT v2 flows (labels the "
          "telemetry contradicts are dropped); public payload lists (SecLists, PayloadsAllTheThings, FuzzDB and others) "
          "and OWASP CRS regression fixtures wrapped into requests. Synthetic: deterministic flow profiles and LLM-written "
          "scenario families with context-matched benign twins, blind-reviewed by Haiku and cross-checked by a second "
          "model family (Codex, Sonnet or Gemini); disputed records are quarantined.",
          "- Records by source and licence reference:", ""]
    L += [f"  - `{s}` ({l}): {n:,}" for (s, l), n in lic.most_common()]
    L += ["", "### Training procedure", "",
          f"- One epoch, {t['global_steps']:,} optimizer steps, effective batch {H.EFFECTIVE_BATCH}, AdamW 8-bit, lr 1e-4 "
          f"cosine with 5% warmup, weight decay 0.01, max grad norm 1.0, max {P.MAX_TOKENS} tokens, seed {rep['seed']}; "
          "loss on the assistant completion only",
          f"- Hardware: {rep['env']['gpu']} ({rep['env']['dtype']}), {t['wall_s'] / 60:.0f} min, "
          f"{t['tokens_per_s_excl_warmup']:,.0f} tokens/s, peak {t['peak_mem_gb']} GB; torch {rep['env']['torch']}, "
          f"transformers {rep['env']['transformers']}, peft {rep['env']['peft']}, trl {rep['env']['trl']}, unsloth "
          f"{rep['env']['unsloth']}",
          f"- Mean train loss {t['train_loss']:.4f}; eval loss " + " → ".join(f"{l:.4f} (step {s})" for s, l in t["eval_losses"]),
          f"- Reload check {rep['reload']['reload_match']}/{rep['reload']['of']}; cached scorer vs full forward pass max "
          f"|Δ| {sc['max_abs_diff_vs_bruteforce']:.1e}", "",
          "## Evaluation", "",
          "| split | n | accuracy | SVM baseline accuracy | attack recall* | benign false intervention* | abstention* |",
          "|---|---|---|---|---|---|---|"]
    for s in ("test", "test_unseen"):
        pol, b = cal[s]["policy"], (base or {}).get("tfidf_svm", {}).get(s, {})
        L.append(f"| {s} | {cal[s]['reliability']['n']:,} | {pct(acc.get(s))} | {pct(b.get('accuracy'))} | "
                 f"{pct(pol['attack_intervention_recall'])} | {pct(pol['benign_false_intervention'])} | {pct(pol['abstention_rate'])} |")
    L += ["", "\\* under the frozen policy (tau and flow envelope); abstained attacks go to log-and-monitor, not allow.", "",
          f"Calibration: ECE {cal['test']['reliability']['ece']} (test), {cal['test_unseen']['reliability']['ece']} (unseen). "
          + (f"OWASP CRS PL1 regexes on the same HTTP records: attack-vs-benign accuracy "
             f"{pct(base['crs_pl1']['test']['attack_vs_benign_accuracy'])} (test), "
             f"{pct(base['crs_pl1']['test_unseen']['attack_vs_benign_accuracy'])} (unseen)." if base else
             "Baselines: not recomputed for this build."), "",
          "Robustness probes (outside the closed set; pass = an acceptable verdict):", ""]
    for k, v in cal.get("robustness", {}).get("by_kind", {}).items():
        L.append(f"- {k}: pass {pct(v['pass'])}, abstained {pct(v['abstained'])} (n={v['n']})" if isinstance(v, dict) else f"- {k}: {v}")
    L += ["", "Per-class accuracy on `test_unseen` (held-out families/variants/sources):", "", "| class | correct / n |", "|---|---|"]
    for c, (a, n) in sorted(pc["test_unseen"].items(), key=lambda kv: kv[1][0] / kv[1][1]):
        L.append(f"| `{c}` | {a}/{n} ({pct(a / n)}) |")
    L += ["", "## Intended use", "",
          "Offline scoring, shadow mode beside an existing WAF/NIDS, or a second-stage classifier for traffic another control "
          "already flagged, always behind `eval.policy` (recommendations only; it executes nothing).", "",
          "## Limitations and risks", "",
          "- Not deployment-qualified: never validated on production traffic; most records are synthetic or wrapped public "
          "payloads, and labels rest on generator intent plus LLM review agreement, which filters noise but is not ground truth.",
          "- Closed set: out-of-taxonomy traffic is pulled toward the benign background class; the flow envelope abstains "
          "on part of it (see robustness), not all.",
          "- SYN flood, UDP amplification and HTTP/2 rapid reset have no usable observed captures (synthetic only); flow "
          "records carry no source port.",
          f"- Latency about {rep['predictions']['latency_p50_s']:.1f} s per request (generation, A100): unsuitable inline "
          "without batching or a pre-filter.",
          "- Context-dependent classes (JWT verifier flaws, DOM XSS, smuggling, BOLA/BFLA) need the context blocks the "
          "policy requires; without them the policy abstains.", "",
          "## Provenance", "",
          f"Training report, calibration and scores: `reports/{run.name}/`. Pipeline, spec and progress log: `docs/PLAN.md`."]
    info = {"base_model": rep["model_id"], "revision": rep["revision"], "fingerprint": rep["fingerprint"],
            "bundle_sha256": rep["bundle_sha256"], "env": rep["env"], "lora": {**H.LORA, "target_modules": targets},
            "training": {"global_steps": t["global_steps"], "effective_batch": H.EFFECTIVE_BATCH, "learning_rate": 1e-4,
                         "epochs": 1, "max_tokens": P.MAX_TOKENS, "seed": rep["seed"], "train_loss": t["train_loss"],
                         "eval_losses": t["eval_losses"], "wall_s": t["wall_s"]},
            "data": {"splits": dict(split), "classes": dict(cnt), "provenance": dict(prov)},
            "taxonomy": classes, "system_prompt": sysp,
            "policy": {"temperature": cal["temperature"], "tau": cal["tau"], "flow_envelope": cal.get("flow_envelope")},
            "evaluation": {s: {"accuracy": acc.get(s), "policy": cal[s]["policy"], "ece": cal[s]["reliability"]["ece"]}
                           for s in ("test", "test_unseen")} | {"robustness": cal.get("robustness")}}
    return "\n".join(L) + "\n", info


def write(run, bundle, baselines=None):
    md, info = card(run, bundle, baselines)
    z = run / f"export_{MODE}.zip"
    tmp = z.with_suffix(".tmp")
    with zipfile.ZipFile(z) as src, zipfile.ZipFile(tmp, "w", zipfile.ZIP_DEFLATED) as dst:
        for item in src.infolist():
            if item.filename not in ("README.md", "export_info.json"):
                dst.writestr(item, src.read(item.filename))
        dst.writestr("README.md", md)
        dst.writestr("export_info.json", json.dumps(info, indent=1, default=str))
    tmp.replace(z)
    (run / "MODEL_CARD.md").write_text(md)
    print(f"{z}: README.md ({len(md.splitlines())} lines) + export_info.json; copy at {run / 'MODEL_CARD.md'}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("run")
    ap.add_argument("--mode", default="fast")
    ap.add_argument("--bundle", default=None)
    ap.add_argument("--baselines", default=None, help="baselines.json of the run's build (default reports/baselines.json)")
    a = ap.parse_args()
    MODE = a.mode
    write(ROOT / a.run if not Path(a.run).is_absolute() else Path(a.run),
          Path(a.bundle) if a.bundle else ROOT / f"colab/colab_bundle_{MODE}.zip", a.baselines)
