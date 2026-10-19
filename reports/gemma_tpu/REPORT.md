# Gemma 4 E2B-it on a TPU v5e-1 — report (18–19 October 2026)

Fine-tuning Gemma 4 E2B-it (2B effective parameters) as the 41-class WAF/NIDS classifier on a Colab TPU v5e-1, the
follow-up experiments, and a data bug that affects the Qwen models too. Every number below comes from files in
`reports/gemma_tpu/` (and `reports/colab_a100v11/` for Qwen v11).

## Summary

- **Gemma 4 E2B matches Qwen3.5-4B v11 on accuracy** (same data build): test_unseen 0.9802 vs 0.9800, test 0.9885 vs
  0.9898. It trains in 45 minutes on a TPU v5e-1; Qwen needs an A100 (82 min).
- **Gemma acts on fewer attacks** (test_unseen recall 0.974 vs 0.984): it is more often unsure between two similar
  attack classes. An "attack mass" decision rule (attack vs benign first, then the class) lifts it to 0.983 with
  fewer false blocks (0.0010). It does not help Qwen.
- **Two epochs overfit**: test improves (0.9913, best of all models), test_unseen gets worse (0.9770). Keep one epoch.
- **Data bug found, affects both models**: real CIC-IDS2017 port scans (0/500) and SSH/FTP brute force (0/470) are
  classified benign by Gemma and by Qwen v11, and the deployed v11 policy ALLOWs them. Cause: the split put all
  CIC-2017 attack flows outside train while 941 CIC-2017 benign flows were in train. A fix is built
  (`data_pipeline/block_split.py`, `datasets_cicfix/`) but not trained (Colab budget).

## Results

Calibrated on val_calib (temperature) and val_policy (threshold tau), same procedure as the Qwen runs.
Recall = attacks blocked; benign FI = benign records wrongly blocked; hard-neg FI = developer/search hard negatives
wrongly blocked.

| Model | Rule | tau | test acc | test recall | test benign FI | unseen acc | unseen NLL | unseen recall | unseen benign FI | unseen hard-neg FI |
|---|---|---|---|---|---|---|---|---|---|---|
| Gemma, 1 epoch | top class | 0.759 | 0.9885 | 0.9904 | 0.0026 | **0.9802** | **0.066** | 0.9739 | 0.0031 | 0.0 |
| Gemma, 1 epoch | attack mass | 0.877 | 0.9885 | 0.9882 | 0.0016 | **0.9802** | **0.066** | 0.9830 | **0.0010** | 0.0 |
| Gemma, 2 epochs | top class | 0.714 | **0.9913** | **0.9941** | 0.0011 | 0.9770 | 0.074 | 0.9755 | 0.0051 | 0.0016 |
| Gemma, 2 epochs | attack mass | 0.942 | **0.9913** | 0.9911 | **0.0** | 0.9770 | 0.074 | 0.9771 | 0.0031 | 0.0 |
| Qwen v11 (A100) | top class | 0.772 | 0.9898 | 0.9926 | 0.0026 | 0.9800 | 0.071 | **0.9841** | 0.0041 | 0.0 |
| Qwen v11 (A100) | attack mass | 0.896 | 0.9898 | 0.9908 | 0.0021 | 0.9800 | 0.071 | 0.9814 | 0.0031 | 0.0 |

test_unseen (held-out attack families and sources never seen in training) is the score that matters.

Real CIC-IDS2017 flows in val_select (top class right / total):

| Class | Gemma 1 epoch | Gemma 2 epochs | Qwen v11 |
|---|---|---|---|
| port scan | 0 / 500 | 0 / 500 | 0 / 500 (500 ALLOW) |
| credential stuffing / brute force | 0 / 470 | 0 / 470 | 0 / 470 (468 ALLOW) |
| slowloris | 10 / 185 | 0 / 185 | 149 / 185 |
| benign | 60 / 60 | 60 / 60 | 60 / 60 |

## What was done

1. **Small test, then the real run** (TPU v5e-1, JAX 0.11 + Tunix + qwix, LoRA r16, same recipe as Qwen):
   64 records per length bucket for 12 steps (train, checkpoint, resume, scoring all checked), then one epoch on all
   36,630 train records: 2,076 steps in 44.5 min, loss 0.535 → 0.0017, then scoring of val / test / test_unseen.
2. **Trainer fixes needed on the way** (`training/gemma_tpu/train_tunix.py`): LoRA weights were discarded by Tunix's
   rematerialisation (applied with remat off, then switched on); each step copied the whole 8.7 GB model (17.8 GB
   needed on a 16 GB chip; now only loss and LoRA gradients leave the step, 3.8-4.4 GB); bf16 compute; resumable
   checkpoints; the learning-rate schedule now counts the real number of steps (it ended at 91% before).
3. **Scorer bug**: 53 records got NaN scores when a long class name overflowed the length bucket; fixed and rescored
   (top-class predictions unchanged).
4. **Attack-mass rule** (`eval.policy.decision_view`, `eval.calibration --act-on attack_mass`): offline only. The
   deployed scorer and `config/policy.yaml` are unchanged, and `--write-policy` refuses it.
5. **2-epoch run** (`colab_ep2/`): 4,166 steps in 86 min, then scoring.
6. **Qwen v11 on the CIC-2017 flows** (L4 GPU, the endpoint's own scorer and policy): confirmed the data bug.
7. **Fix for the data bug** (`data_pipeline/block_split.py`): orders each affected class by capture time; the
   earlier 80% go to train (924 flows), the latest 20% stay held out (231). test and test_unseen are byte-identical
   to the original build, so a model trained on `datasets_cicfix/` compares directly. Gemma export ready
   (`data_cicfix/`, `colab/gemma_data_cicfix.tgz`).
8. **Full-quota build** (`datasets_full/`, local only): 121,789 records, 2.3x the fast build; 17 of 41 classes are
   short of quota because the LLM-written classes do not scale with the quota. Not trained.

Stopped or skipped to save Colab units: `ple32` (LoRA rank 32 + Gemma's per-layer-embedding projections) and
`sched1` (1-epoch reference with the fixed schedule), each stopped after ~20 min; `cicfix` and the full-data run were
not started.

## Colab spend

Balance 128.08 → 98.49 compute units over the whole session (~30 units, including the other session's A100 v11 and
L4 runs while they overlapped). No Colab session is running.

## Recommended next steps

1. **Train on `datasets_cicfix/`** (Gemma: `training/gemma_tpu/run_tpu.sh cicfix reports/gemma_tpu/colab/gemma_data_cicfix.tgz`,
   ~1.6 h, ~4.5 units) and apply the same fix to the next Qwen build. Until then the deployed v11 lets real port
   scans and brute force with CIC-2017-shaped telemetry through.
2. Fix the split rule itself: match benign and attack flows per source **per split**, not only over the whole build.
3. Keep Gemma at one epoch. If Gemma is to be deployed, use the attack-mass rule (it needs a scorer change first).
4. The full-quota build is only worth training once the LLM-written classes have more data.

## Files

| Path | What |
|---|---|
| `reports/gemma_tpu/colab_full/` | 1-epoch run: adapter `lora.npz`, logs, scores (+ `rescore/` for the 53 NaN records) |
| `reports/gemma_tpu/colab_ep2/` | 2-epoch run: adapter, logs, scores, both calibrations |
| `reports/gemma_tpu/calibration*.json`, `qwen11_calibration_attack_mass.json` | calibrations (top class / attack mass) |
| `reports/gemma_tpu/qwen11_valselect_cic2017.results.jsonl` | Qwen v11 decisions on the CIC-2017 flows |
| `training/gemma_tpu/run_tpu.sh` | one-command TPU run driver (new session → train → score → download → stop) |
| `data_pipeline/block_split.py`, `datasets_cicfix/` | the CIC-2017 fix and its build (jsonl not in git) |
| `docs/PLAN.md`, `docs/03-training.md` | progress log entries and the TPU how-to |
