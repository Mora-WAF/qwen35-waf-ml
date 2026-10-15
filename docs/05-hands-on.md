# 5. Hands-on walkthrough

[Learning index](README.md) · Previous: [Evaluation](04-evaluation-and-policy.md) · [Glossary](glossary.md)

Run these commands from `qwen35-waf-ml/`. The first three exercises need only the existing local environment; they do not download a model, build datasets, or start a GPU runtime.

## Exercise 1: inspect the stored build

```bash
.venv/bin/python - <<'PY'
import json
from collections import Counter
from pathlib import Path

stats = json.loads(Path('datasets_ready/dataset_stats_report.json').read_text())
splits = Counter()
for counts in stats['split_counts'].values():
    splits.update(counts)
manifest_path = Path('datasets_ready/manifest.json')
if manifest_path.exists():
    manifest = json.loads(manifest_path.read_text())
    assert sum(splits.values()) == manifest['total']
else:
    print('Manifest absent: a build is incomplete; these statistics may be historical.')
print('classes:', len(stats['classes_present']))
print('records in statistics:', sum(splits.values()))
print('splits:', dict(sorted(splits.items())))
print('provenance:', stats['provenance'])
PY
```

Expected for the documented snapshot: 41 classes and 42,565 records. After rebuilding, read the new reports instead of expecting these historical counts. Can you explain why `fast_val.jsonl` has three logical partitions?

## Exercise 2: observe shadow-mode abstention

```bash
.venv/bin/python - <<'PY'
from eval.policy import decide, load, validate_decision

tax, policy, schema = load()
decision, diagnostic = decide(
    '{"threat_type":"benign_standard_traffic"}',
    mode='L7_HTTP', context={}, confidence=None,
    tax=tax, pol=policy,
)
validate_decision(decision, tax, schema)
assert decision['verdict'] == 'ABSTAIN'
assert diagnostic['reason'] == 'uncalibrated'
print(decision)
print(diagnostic)
PY
```

This uses no model: it supplies an illustrative raw answer to the existing wrapper. The tentative benign class survives only in diagnostics. Null confidence deliberately keeps this example in shadow mode even if a fitted configuration exists later.

## Exercise 3: test grouping with a tiny inert fixture

```bash
.venv/bin/python - <<'PY'
from data_pipeline.dedup_and_split import assign_splits, lineage_overlap

records = [
    {'metadata': {'threat_type': label, 'lineage_group': f'family-{i}'}}
    for i in range(12)
    for label in ('sqli_classic', 'benign_hard_negative_dev')
]
assign_splits(records)
assert not lineage_overlap(records)
assert len({r['metadata']['split'] for r in records}) > 1
for i in range(12):
    pair = [r for r in records if r['metadata']['lineage_group'] == f'family-{i}']
    assert len({r['metadata']['split'] for r in pair}) == 1
print('All 12 attack/benign pairs stayed together.')
PY
```

This is a fixture for the split helper, not a complete record suitable for training. Try a row-level random assignment mentally: why would paired examples in different splits weaken the exam?

## Existing local checks

```bash
.venv/bin/python -m pytest -q tests
```

Tests exercise logic and fixtures. They are separate from rebuilding the data, rerunning the data gate, or executing the Colab notebook.

## Rebuild workflow: changes derived artifacts

The following sequence is a **reference**, not a prerequisite for reading the guides. It requires the source cache, accepted scenario outputs, and cached/downloadable tokenizer. It rewrites derived files and reports; preserve any snapshot you need first. `build` removes the previous manifest before writing, so an interrupted build cannot retain a stale passable manifest.

```bash
# Existing source ingestion / normalization / scenario review must be complete first.
.venv/bin/python -m data_pipeline.build --quota-frac 0.25 --prefix fast_
.venv/bin/python -m data_pipeline.robustness
.venv/bin/python -m tests.validate_pre_colab --prefix fast_
.venv/bin/python -m eval.shortcut_audit --prefix fast_
.venv/bin/python -m eval.coverage --prefix fast_
.venv/bin/python -m eval.baselines --prefix fast_
.venv/bin/python -m training.make_bundle --prefix fast_
```

Run these individually and stop if a command fails. Do not bypass a failed gate to package a training run. Follow [CLAUDE.md](../CLAUDE.md) and [PLAN.md](PLAN.md) for source ingestion and review; raw scenario output is irreplaceable and must be versioned rather than overwritten.

After a real training/reload/scoring run, download `SCORES_fast.jsonl` into `reports/`. This command writes `reports/calibration.json` but leaves policy configuration unchanged:

```bash
.venv/bin/python -m eval.calibration reports/SCORES_fast.jsonl
```

Inspect the report and evaluation protocol before freezing a policy. Adding `--write-policy` changes `config/policy.yaml`; fitting parameters is not proof that the specification's acceptance targets passed. Smoke/dry-run score samples are diagnostic and do not qualify a final policy.

## Explain it back

You understand the core design when you can explain why a published payload inside a generated request is hybrid, why twins share a split, why user tokens are ignored by the loss, why Part B needs a fresh process, and why a valid model label can still produce `ABSTAIN`.
