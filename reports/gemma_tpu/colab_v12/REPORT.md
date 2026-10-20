# Gemma 4 E2B-it on the v12 build (TPU v5e-1, 20 October 2026)

First model trained on the v12 data: 42 classes, 54,747 records, including the CIC campaign split fix and tonight's
CRS 921/931/933/934 rounds. Same recipe as the 18 Oct Gemma run: LoRA r16, 1 epoch, AdamW 1e-4 cosine, effective
batch 16. Files: `SCORES_fast.jsonl`, `calibration_{top_class,attack_mass}.json`, `../eval_v12.json`
(`../eval_v12.py`), `train_log.jsonl`, `train_report.json`, `lora.npz`.

## Results

The test sets are not the same records as on 18 Oct. This build adds the 42nd class, moves real CIC flows by the
campaign split, and draws more seeded payload-list records into test. Compare per class, not record by record.

| Run | Rule | T / tau | test acc | test recall | test benign FI | unseen acc | unseen NLL | unseen recall | unseen benign FI | unseen hard-neg FI |
|---|---|---|---|---|---|---|---|---|---|---|
| v12 | top class | 1.053 / 0.678 | 0.9822 | 0.9874 | 0.0080 | 0.9738 | 0.086 | 0.9716 | 0.0051 | 0.0 |
| v12 | attack mass | 1.053 / 0.725 | 0.9822 | 0.9921 | 0.0080 | 0.9738 | 0.086 | 0.9790 | 0.0046 | 0.0016 |
| 18 Oct (v11 build) | top class | 1.038 / 0.759 | 0.9885 | 0.9904 | 0.0026 | 0.9802 | 0.066 | 0.9739 | 0.0031 | 0.0 |
| 18 Oct (v11 build) | attack mass | 1.038 / 0.877 | 0.9885 | 0.9882 | 0.0016 | 0.9802 | 0.066 | 0.9830 | 0.0010 | 0.0 |

**Fixed: the CIC-2017 data bug.** These are the real held-out late-20% flows in val_select; the 18 Oct numbers
were over all held-out flows.

| Real CIC-2017 flows | v12 | 18 Oct Gemma | Qwen v11 |
|---|---|---|---|
| port scan | **96 / 100** | 0 / 500 | 0 / 500 |
| SSH/FTP brute force | **94 / 94** | 0 / 470 | 0 / 470 |
| slowloris | **32 / 37** (5 → http flood) | 10 / 185 | 149 / 185 |
| HTTP flood (new class) | 74 / 74 | - | - |
| benign | 76 / 78 (2 → slowloris) | 60 / 60 | 60 / 60 |

## Where the headline numbers dropped

1. **Real CIC-2018 benign flows called slowloris: 17 in test.** These are new in test; the campaign split put
   187 CIC-2018 benign flows there. The test benign false intervention of 0.008 matches these ~17 records. These
   would be false blocks, the first real-traffic false positives the split exposed. Fix: more CIC-2018 benign
   flows in train (the split currently matches benign 1:1 per source with that source's attacks), or slowloris
   variants that overlap these flows' features in train.
2. **http_flood_app_dos held-out variant: 33 / 50 on test_unseen.** 17 go to benign_high_rate_legit. The held-out
   variant is the paced flood (200-600 requests / min), which overlaps the benign `proxy_nat_fanin` lookalike
   (100-1,200 / min) by design, and train has no paced example. This is a design decision to revisit: either give
   train some paced variants or accept that a paced flood is indistinguishable from NAT fan-in in one window.
3. **sqli_classic: 196 / 232 (18 Oct 210 / 232).**
   - 18 go to sqli_obfuscated and 11 the other way (the known oracle pair; same verdict).
   - 14 go to benign classes, all seeded payload-list records (`sqli_classic_httpwrap`). These are attack misses;
     the 18 Oct run had 3.
4. **ldap_xpath_injection on test: 83 / 95.** All 12 errors are seeded payload-list records (→ rce 5, → sqli 4), none
   from tonight's Sol/Gemini rounds. This build has 95 ldap records in test vs 62.
5. Benign ↔ benign moves (standard / hard-negative dev / search): about 30 across test and test_unseen. They
   cannot cause a block. Some are tonight's twins that `sn10` relabelled to hard-negative dev.

Improved or unchanged: web_cache_poisoning 95/96 (88), jwt 141/141 (139/142), ssti 50/50, rce test 125/125,
upload 68/68, http2 rapid reset 80/81 (78). New CRS-round classes on test: deserialization 71/72, ssrf 106/107,
path traversal 129/130, smuggling and upload without errors.

## Verdict

v12 Gemma is **model-evaluated**, not better than the 18 Oct model on aggregate metrics. It fixes the real-traffic bug
that mattered most: port scans and brute force were ALLOWed by v11. The new errors are concentrated and
explainable:
- 17 CIC-2018 benign flows called slowloris;
- the paced-flood holdout;
- seeded SQLi payload lists.

Recommended before the next train:
1. Add CIC-2018 benign flows to train.
2. Add paced http-flood variants to train, or drop that holdout.
3. Look at the 14 sqli_classic → benign payloads.

Qwen was not trained (free Colab, CUDA only).

## Run notes

- The first launch crashed before step 1. The Colab TPU image now ships `torch_tpu`, which rejects libtpu `0.0.44.1`
  on import. Fixed in `train_tunix.py` with `TORCH_DEVICE_BACKEND_AUTOLOAD=0`. Logs are in `failed_0052/`.
- Training took 2,225 steps in 52 min, loss 0.004–0.007 at the end, HBM 9.4 GB.
- The free VM was reclaimed at 02:43, during test_unseen scoring, and the first account then got
  `Service Unavailable` for a new TPU. The adapter and val/test scores were copied from that account's Drive mirror
  (rclone remote `colabold`, read-only). test_unseen was scored on a second free Colab account in score-only mode,
  with the adapter uploaded and checksum-verified. `run_tpu.sh` can now also take the adapter from the Drive
  mirror (`DRIVE=1 GEMMA_SCORE=...`, no `GEMMA_LORA`).
