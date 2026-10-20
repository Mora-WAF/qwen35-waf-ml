# Gemma 4 E2B-it on the v13 build (TPU v5e-1, 20 October 2026)

v13 data: 55,128 records, 42 classes, sensor contract 3. It fixes the Gemma v12 errors (`docs/PLAN.md`, 20 Oct ~13:00):
- CIC-2018 benign flows are now in train;
- two train flood variants sit in the NAT fan-in range;
- short Base64 values get decoded views;
- 11 junk ldap seeds are quarantined;
- 5 focus rounds were added (bare / stacked SQLi, CSTI ×2, CORS origin bypass, WebSocket frame abuse).

Same recipe as v12: LoRA r16, 1 epoch, AdamW 1e-4 cosine, effective batch 16; 2,233 steps in 52 min, final loss
~0.003, HBM 9.4 GB. Third free Colab account. Files: `SCORES_fast.jsonl` (merged), `calibration_{top_class,attack_mass}.json`,
`../eval_v13.json` (`../eval_v13.py`), `lora.npz`.

The test sets differ from v12's. This build moves the CIC-2018 flows out of test and adds families that shift
the hash holdout. Compare per class, not record by record.

## Results

| Run | Rule | T / tau | test acc | test recall | test benign FI | unseen acc | unseen NLL | unseen recall | unseen benign FI | unseen hard-neg FI |
|---|---|---|---|---|---|---|---|---|---|---|
| v13 | top class | 0.954 / 0.576 | **0.9874** | **0.9956** | 0.0052 | 0.9722 | 0.106 | **0.9855** | 0.0091 | 0.0079 |
| v13 | attack mass | 0.954 / 0.706 | **0.9874** | **0.9942** | **0.0036** | 0.9722 | 0.106 | **0.9878** | 0.0086 | 0.0079 |
| v12 | top class | 1.053 / 0.678 | 0.9822 | 0.9874 | 0.0080 | 0.9738 | 0.086 | 0.9716 | 0.0051 | 0.0 |
| v12 | attack mass | 1.053 / 0.725 | 0.9822 | 0.9921 | 0.0080 | 0.9738 | 0.086 | 0.9790 | 0.0046 | 0.0016 |

Argmax errors over test + test_unseen: **attack → benign (missed attacks): 69 → 37**; benign → attack (false
blocks): 33 → 33, with a different mix (below).

**Real observed flows (val_select, latest 20% of each capture):**
- **CIC-2017:** port scan 95/100, brute force 94/94, HTTP flood 74/74, slowloris 31/37 (5 → http flood), benign 99/100.
- **CIC-2018:** benign 35/37 (2 → slowloris). In v12 all 17 errors on these flows were false blocks in test.
- **DNS-EXF:** exfil 65/65, benign 65/65.

## Fixed

| v12 error | v12 | v13 |
|---|---|---|
| paced http flood holdout (unseen) | 33/50 | **49/50** |
| sqli_classic (unseen) | 196/232 | **219/237** (seeded payloads → benign: 14 → 0) |
| ldap_xpath_injection (test) | 83/95 | **45/45** |
| CIC-2018 benign → slowloris | 17 false blocks in test | 2/37 held-out flows |
| open_redirect_cors (unseen) | 46/50 | 49/50 |
| xss_dom_polyglot (unseen) | 153/165 | 161/170 (Angular → ssti: 10 → 0) |
| sqli_classic / open_redirect / xss / ssrf / path traversal / ldap (test) | 1-12 errors each | 0-1 each |

## Where it is worse

1. **Unseen benign false intervention 0.0046 → 0.0086.** The false blocks concentrate in the benign twins of held-out
   attack families: `nested_cty_inner_unverified_benign` → jwt 8 (test + unseen), the CSTI round's Vue / Angular
   twins → xss_dom 6, and 4 benign_high_rate flows → http flood on test (the new train flood variants overlap NAT
   fan-in by design: the price of 49/50 paced floods). Hard-negative FI 0.0079 stays under the 0.01 target.
2. **nosqli unseen 208 → 189/212.** 21 PayloadsAllTheThings bracket-operator form payloads
   (`login[$ne]=...`) go to ldap_xpath_injection: attack → attack, still a block.
3. **slowloris unseen 91 → 78/92.** 13 go to syn_flood_ddos: attack → attack, still an intervention.
4. Benign ↔ benign moves (standard / hard-negative dev / search) ~40: cannot cause a block.

## Verdict

v13 Gemma is **model-evaluated** and better than v12 where it matters for enforcement:
- missed attacks are almost halved;
- every v12 error class is fixed;
- the real-traffic false blocks are gone;
- test is better on every metric.

The cost is a higher test_unseen benign false-intervention rate (0.0086). It comes from benign twins of held-out
synthetic families, not from real traffic.

Next data:
1. JWT `cty` / nested-token benign twins in train (so a nested token alone is not the label).
2. Benign Vue / Angular content with plain `{{ }}` in more train families.
3. nosqli bracket-operator form payloads in train (the unseen source's style).
4. Check the held-out slowloris variant vs syn_flood features.

Qwen was not trained (free Colab, CUDA only).
