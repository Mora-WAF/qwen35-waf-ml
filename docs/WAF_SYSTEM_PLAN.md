# Gemma WAF System Plan

Proposal, 19 October 2026. The diagrams in `assets/waf-system/` are renders of the editable originals.

## Summary

We put OWASP CRS 4 on the Coraza engine inline, embedded in our own sensor behind an Envoy proxy, as the always-on layer. The fine-tuned Gemma 4 E2B classifier is a second opinion on the traffic where rules alone decide badly. The model runs inline only for requests the rules flag or score as borderline. It also runs out of band on mirrored traffic and on network flows.

Every layer is a mature open-source project, except four thin services we write ourselves:

1. **waf-sensor**: turns a request into the model's telemetry JSON and adds trusted route context. It also runs Coraza and CRS.
2. **waf-model**: wraps the repo's existing scorer (`training/gemma_tpu/hf_scorer.py` plus `eval.policy.decide_from_probs`) behind a gRPC API.
3. **Decision combiner**: merges the CRS anomaly score and the model's decision into one action.
4. **Flow featurizer**: maps Zeek, Suricata or NetFlow records onto the model's flow vocabulary (`config/features.yaml`).

Why we need the model at all:

- **CRS blocks good traffic.** In the repo's CRS cross-check, CRS rules fired on 49% of developer hard negatives and 12% of ordinary benign requests (`reports/crs_oracle.json`).
- **CRS misses attack types.** It caught 19% of deserialization and 25% of path-traversal records.
- **Gemma does better on unseen data.** On held-out attack families and sources it blocks 98.3% of attacks and wrongly blocks 0.10% of benign requests (`reports/gemma_tpu/REPORT.md`).
- **Some classes need context rules cannot express**: BOLA/BFLA, mass assignment, prompt injection and MCP tool abuse.

**The model is not ready to enforce yet.**

- It lets real CIC-IDS2017 port scans (0 of 500 caught) and SSH/FTP brute force (0 of 470) through. That is a split bug, fixed in the build but not yet retrained.
- Its GPU latency is unmeasured.
- Its false-block rate on real legitimate traffic is unmeasured.
- 4.2% of real browsing requests do not fit its 2,048-token budget until sensor contract 2 is trained in.

Phase 0 of the rollout closes these gaps before the model touches any traffic decision.

## Goals, non-goals and service levels

The system must block web attacks with fewer false blocks than CRS alone. It must never take the protected site down when a WAF component fails.

**Goals**

- Cover all 37 attack classes of the taxonomy (`config/taxonomy.yaml`), across four domains: 13 L7 injection, 11 API/protocol, 4 AI/modern and 9 network-flow classes.
- Cut CRS false positives on developer content, search text and many-cookie browsing traffic. Let the model overrule CRS only where the evidence supports it.
- Act on classes that need trusted context, such as BOLA, BFLA, mass assignment, JWT bypass, prompt injection and MCP abuse. That context comes from the gateway and route configuration, never from the client.
- Keep every verdict explainable: a fixed reason code, the CRS rule IDs, and the model's class and calibrated confidence. The model never writes free text into a decision.
- Close the loop: reviewed false positives and misses flow back into the repo's data pipeline, the next training run, and gated promotion.

**Non-goals**

- **Volumetric DDoS at L3/L4.** SYN floods and UDP amplification stay with the upstream provider or CDN. The model's flow classes only detect them and feed IP-level actions.
- **Response inspection by the model.** It reads requests and flows only. CRS outbound rules stay available but off at launch.
- **Rules or text written by the model.** It picks one of 41 labels by scoring; it never generates.
- **Full bot management.** The challenge action is a proof-of-work page, not device fingerprinting.
- **Production use of the Qwen3.5 v11 model.** It stays an offline comparison and an optional ensemble member.

**Service levels.** The targets are proposals. The repo column shows what exists today.

| Measure | Target | Repo evidence today |
| --- | --- | --- |
| Added latency, rules layer (Envoy + Coraza + CRS), body ≤ 64 KB | p99 ≤ 10 ms; sensor CPU sized from Phase 0 load tests | Not measured for our stack. CrowdSec's Coraza-based CRS added +33 ms mean and +58 ms p90 per GET at 400 concurrent users (an out-of-process setup), so this target needs proof |
| Added latency, inline model path | p99 ≤ 200 ms, hard deadline 250 ms, then rules-only fallback | Gemma not measured on GPU. Qwen v11: 0.21 s median, 0.33 s p90 on an A10G |
| Share of requests sent to the model inline | ≤ 5% | Depends on CRS gray-band tuning in shadow |
| False blocks on real legitimate traffic (enforced classes) | ≤ 0.1% of requests | v10: 0.57% on open-appsec's 10,380 legitimate requests. Gemma: not measured |
| False blocks, benign held-out set | ≤ 0.2% | Gemma: 0.10% (test_unseen, attack-mass rule) |
| Attack recall on held-out families and sources | ≥ 98% | Gemma: 98.3%; Qwen v11 + Gemma ensemble: 99.0% |
| Requests too long for the model | ≤ 1% | 4.2% under contract 1. Contract 2 is written, not trained |
| Data-plane availability | 99.95% monthly; WAF faults degrade to rules-only, never to an outage | n/a |
| Model-path availability | 99.5% monthly; its outage only removes the second opinion | n/a |

Failure behaviour:

- When the model path fails or misses its deadline, the request gets the rules-only verdict.
- When the sensor itself fails, Envoy lets traffic through by default (fail-open).
- Routes marked critical, such as admin or payment, can fail closed. That list is a per-route choice the application owners make.

## At a glance

![Architecture: inline, out-of-band and learning planes](assets/waf-system/architecture.png)

A request crosses only the top band. Gemma is called inline only when CRS flags the request, scores it as borderline, or the route requires the model; a timeout leaves the rules-only verdict. Everything below the top band runs off the request path.

## Building blocks

Every layer below is a maintained open-source project with a permissive or well-understood license. We write only the four services named in the summary. Versions are the latest stable releases on 19 October 2026, read from each project's GitHub releases.

| Layer | Choice | Version · license | Why | Not chosen, and why |
| --- | --- | --- | --- | --- |
| Edge proxy | [Envoy](https://github.com/envoyproxy/envoy) via [Envoy Gateway](https://github.com/envoyproxy/gateway) | 1.40 / 1.9.2 · Apache-2.0, CNCF | ext_proc is stable and has a no-latency shadow mode. Rate limits, `jwt_authn` and OpenTelemetry logs are built in | ingress-nginx: retired in March 2026, no more security fixes. [HAProxy](https://github.com/haproxy/haproxy) 3.4 + [coraza-spoa](https://github.com/corazawaf/coraza-spoa) 0.9: kept as the fallback data plane |
| Rule engine | [OWASP Coraza](https://github.com/corazawaf/coraza), embedded in waf-sensor | 3.8.1 · Apache-2.0 | A Go library that states full CRS v4 compatibility. No Wasm overhead | [coraza-proxy-wasm](https://github.com/corazawaf/coraza-proxy-wasm): last release July 2025, open issues report 5–10× slowdowns. [ModSecurity v3](https://github.com/owasp-modsecurity/ModSecurity) 3.0.17: mature, but tied to nginx or Apache |
| Rules | [OWASP CRS](https://github.com/coreruleset/coreruleset) LTS line | 4.25.2 · Apache-2.0 | Anomaly scoring, paranoia levels, exclusion plugins. Security fixes until Q3 2027 | CRS 4.30 (latest, not LTS): staging only |
| Classifier | Gemma 4 E2B-it + our LoRA | Apache-2.0 | Trained and evaluated: 98.3% recall, 0.10% benign false blocks on held-out data | Qwen3.5-4B v11: comparison and optional ensemble member |
| Model runtime | [Transformers](https://github.com/huggingface/transformers), then [vLLM](https://github.com/vllm-project/vllm) | 5.17 / 0.32.0 · Apache-2.0 | Transformers has the repo's parity tests; vLLM is fast once labels are single tokens | TGI: archived. SGLang: gaps in Gemma 4 LoRA |
| GPU platform | Kubernetes with the [NVIDIA GPU Operator](https://github.com/NVIDIA/gpu-operator), [KEDA](https://github.com/kedacore/keda), [Karpenter](https://github.com/kubernetes-sigs/karpenter) | 26.7.2 / 2.21.0 / 1.15.0 · Apache-2.0 | Driver, device plugin and DCGM in one operator; KEDA (CNCF graduated) scales on queue depth | KServe 0.21: only if the platform team already runs it. SageMaker: Phase 0 bridge |
| IP reputation and bans | [CrowdSec](https://github.com/crowdsecurity/crowdsec) local API + community blocklist | 1.8.1 · MIT | Shared blocklist (free tiers of 3,000 and about 15,000 IPs); time-limited decisions our scorer can write | Its AppSec WAF also runs CRS on Coraza, but we embed Coraza ourselves |
| Rate limiting | [envoyproxy/ratelimit](https://github.com/envoyproxy/ratelimit) + Valkey | commit-pinned · Apache-2.0; Valkey BSD-3-Clause | One bucket per client IP across all Envoy pods; has a shadow mode | Redis: relicensed away from BSD in 2024; Valkey is the BSD-licensed fork under the Linux Foundation |
| Challenge page | [Anubis](https://github.com/TecharoHQ/anubis) | 1.28.1 · MIT | Mature proof-of-work proxy, frequent releases | ALTCHA: a library, not a service. mCaptcha: work in progress (AGPL) |
| Network sensors | [Zeek](https://github.com/zeek/zeek), [Suricata](https://github.com/OISF/suricata) + ET Open, [GoFlow2](https://github.com/netsampler/goflow2) | 9.0 LTS · BSD-3; 8.0.7 · GPL-2.0 (rules MIT); 2.2.7 · BSD-3 | Measured fields that map onto the model's flow features | CICFlowMeter: unmaintained since 2022. NFStream: lacks the NetFlow v2 set. nProbe: commercial |
| Streaming | [Apache Kafka](https://github.com/apache/kafka) | 4.3.1 · Apache-2.0 | KRaft only (no ZooKeeper); ClickHouse reads it directly with its Kafka table engine | Redpanda: BSL license |
| Telemetry | [OpenTelemetry Collector](https://github.com/open-telemetry/opentelemetry-collector-releases), [Prometheus](https://github.com/prometheus/prometheus), Jaeger | 0.163.1; 3.13 LTS · Apache-2.0 | Envoy sends OTLP access logs natively; standard metrics and traces | Loki: not needed beside ClickHouse |
| Decision store, dashboards | [ClickHouse](https://github.com/ClickHouse/ClickHouse), [Grafana](https://github.com/grafana/grafana) | 26.8 LTS · Apache-2.0; 13.2 · AGPL-3.0 (internal use) | Fast analytics over billions of decision events | [OpenSearch](https://github.com/opensearch-project/OpenSearch) 3.9 or [Wazuh](https://github.com/wazuh/wazuh) 4.14: an optional SIEM for the SOC |
| Labelling and MLOps | [Label Studio](https://github.com/HumanSignal/label-studio), [MLflow](https://github.com/mlflow/mlflow), [Evidently](https://github.com/evidentlyai/evidently) | 1.23.2; 3.17.1; 0.7.23 · Apache-2.0 | Text labelling; registry aliases (champion, challenger); drift reports | Argilla: maintenance only since 2025. MLflow stages: deprecated, aliases replace them |
| Supply chain and identity | [cosign](https://github.com/sigstore/cosign), [Syft](https://github.com/anchore/syft), [Grype](https://github.com/anchore/grype), [Kyverno](https://github.com/kyverno/kyverno), [OpenBao](https://github.com/openbao/openbao), [cert-manager](https://github.com/cert-manager/cert-manager) | 3.1.3; 1.55.0; 0.121.0; 1.19.1 · Apache-2.0; 2.7.1 · MPL-2.0; 1.21 · Apache-2.0 | Signed images, SBOMs, scans, admission checks, secrets, mTLS certificates | Trivy: compromised in March 2026. SPIRE 1.15: if workload identity beyond mTLS is needed |
| Testing | [GoTestWAF](https://github.com/wallarm/gotestwaf), [go-ftw](https://github.com/coreruleset/go-ftw), [open-appsec WAF comparison](https://github.com/openappsec/waf-comparison-project) | 0.5.10 · MIT; 2.6.1 · Apache-2.0; corpus Apache-2.0 / MIT | Attack suites, CRS regression tests, a benign corpus of 1,040,242 requests from 185 sites and 73,924 attacks | — |
| Comparison baselines (not the base) | [open-appsec](https://github.com/openappsec/openappsec), [BunkerWeb](https://github.com/bunkerity/bunkerweb), [SafeLine](https://github.com/chaitin/SafeLine) | 1.1.36; 1.6.16 · AGPL-3.0; 9.4.2 · GPL-3.0 | Benchmarked against in Phase 0 | open-appsec's production model has its own license and cannot be retrained. BunkerWeb is bound to nginx. SafeLine's detector ships as an image without source |

## Inline HTTP path

![Inline request path: 5 steps, 1 decision](assets/waf-system/request-path.png)

"Model needed" means one of three things: CRS would block, some rules fired below the threshold, or the route catalog marks the route model-required.

**Envoy settings** (Envoy 1.40 through Envoy Gateway 1.9, `EnvoyExtensionPolicy` with `extProc`):

- **Processing mode.** Request headers `SEND`; request body `BUFFERED_PARTIAL`, which sends up to the route's limit (64 KiB by default). Plain `BUFFERED` errors when the body passes the buffer limit. Response headers and bodies are `SKIP` at launch.
- **`message_timeout`: 300 ms.** The default is 200 ms, below the model deadline plus CRS time. waf-sensor enforces the 250 ms model deadline itself and always answers in time with at least the rules-only verdict.
- **`failure_mode_allow: true`.** The default (false) turns a sensor failure into a 500 for the client. Routes marked critical fail closed by their own policy.
- **Shadow phase: `observability_mode`** (Envoy Gateway `shadowMode`). Envoy sends headers and a streamed body without pausing the request and ignores the answer, so Phase 1 adds no latency. It works only with a streamed body or none.
- **Verdicts.** Block is an `ImmediateResponse` 403 whose body carries the request ID. Rate limit is a 429 with `Retry-After`. Challenge is a 302 to the proof-of-work page. Allow continues, and dynamic metadata carries the decision to the access log.
- **Rate limits.** Local token buckets plus the global rate-limit service (`envoyproxy/ratelimit`, backed by Valkey), with one bucket per client IP. That service has had no versioned release since v1.4.0, so its image is pinned by commit digest.
- **IP bans.** waf-sensor pulls decisions from the CrowdSec local API and checks them before CRS. There is no first-party Envoy bouncer; the community one uses ext_authz.

**CRS settings** (Coraza 3.8.1 embedded in waf-sensor; CRS 4.25 LTS, supported until Q3 2027. CRS 4.30 is the newest, but production stays on the LTS line):

- **Anomaly scoring** with the default inbound threshold of 5.
- **Phase 1** runs `SecRuleEngine DetectionOnly` with a detection paranoia level of 2.
- **Phase 2** turns the engine on. Following the CRS guidance, the threshold starts high and steps down to 5, and the blocking paranoia level goes from 1 to 2.
- **`tx.early_blocking` stays off** until tuning is done: it hides the later-phase matches tuning needs.
- **Exclusions** come from the official CRS 4 exclusion plugins (WordPress, Nextcloud, Drupal and others) where those applications run. Our own runtime exclusions (`ctl:ruleRemoveTargetById`) go before CRS.
- **go-ftw** regression-tests every exclusion. Its quantitative mode counts false positives per rule and paranoia level on a benign corpus, and gates CI against a baseline.
- **Coraza 3.8's `SecArgumentsLimit`** (default 1,000) and the body limits bound the work per request.

## Sensor and telemetry contract

The sensor only captures raw fields, within byte limits. Every transform the model saw in training runs inside waf-model, from the same Python code the training build uses. That gives one implementation, versioned with each model package, so serving cannot drift from training. The transforms are decoded views, run collapsing, budget fitting and windows.

**Fields** (`schemas/telemetry.schema.json`). The model input carries telemetry only. A missing value is an absent key, never a zero.

| Field | Production source | Rule |
| --- | --- | --- |
| `schema_version`, `mode` | Sensor | `L7_HTTP`; `HYBRID_CTX` when `context` is attached; `L4_FLOW` on the flow path |
| `decision_stage`, `observed_until` | Sensor | `request_headers` or `request_headers_body`. `observed_until` is the fixed token `request_received`, as canonicalised in training (`subagents/orchestrator.py`) |
| `http.method`, `http.target` | Envoy `:method`, `:path` | Raw path and query as received, never decoded or normalised |
| `http.protocol` | Envoy stream info | Canonical tokens `HTTP/1.1`, `HTTP/2`, `HTTP/3` |
| `http.headers` | ext_proc header map | Ordered pairs with duplicates kept; pseudo-headers left out |
| `http.query`, `http.cookies` | Parsed by waf-model from target and `Cookie` | Ordered pairs with duplicates kept, raw values |
| `http.body`, `body_encoding` | ext_proc buffered body | Up to 64 KiB per route by default. `utf8`, or `base64` for non-UTF-8 bytes. `truncated: true` past the cap |
| `context.edge` | TLS inspector, IP-to-ASN and geo lookup | `tls_version`, `client_asn_type`, `geo`: the neutral keys training uses (`data_pipeline/generate.py`) |
| `context.route` | Route catalog | `auth`, `rate_limit_tier` |
| `history.client_requests_60s` | Valkey counter per client | Only when it is known at decision time |

**Budget fitting** (`training/prepare.py`, contract 2 from the next build):

- Every prompt fits 2,048 tokens. Cookie values are capped first, then header values, the body and the URL, in four levels.
- The last level keeps the first 48 cookies and records `cookies_truncated`.
- Runs of more than 64 identical characters collapse to 64, which defeats padding evasion.
- Long bodies are split into at most 8 windows, head and tail always included. Each window is scored, and the most attack-like one decides.
- A request that still fits no budget gets ABSTAIN with reason `over_token_budget`, so the rules-only verdict stands.

**Decoded views.** The model sees JWT header and payload JSON as encoded (so duplicate claims stay visible), and Base64 values that decode to printable text. URL-encoding layers are decoded too. Each view records its decoder and depth (at most 4). The package's `serving.json` switches views on or off and sets the sensor contract. Both follow the training build, so a v10 or v11 package keeps contract 1.

**Trusted context.** Twelve classes act only when a named context key is present (`required_context` in `config/policy_gemma.yaml`). Without it they abstain, by design. The keys come from one versioned **route catalog**: OpenAPI files with `x-waf-*` extensions, owned by the application teams, kept in git and loaded by the sensor by Envoy route name.

| Class | Context key | Production source |
| --- | --- | --- |
| `bfla_admin_function_probing` | `caller_role`, `route_required_role` | Verified JWT claim (Envoy `jwt_authn`); the role the route requires, from the catalog |
| `bola_idor_enumeration` | `authz_context` | Subject and object ownership the application's authorization layer exposes |
| `bopla_mass_assignment` | `object_schema` | OpenAPI request schema of the route |
| `jwt_auth_bypass` | `verifier_config` | Issuer, allowed algorithms and JWKS from the `jwt_authn` provider |
| `http_request_smuggling` | `parser_stack` | Declared proxy chain per listener, for example `envoy -> gunicorn` |
| `open_redirect_cors` | `browser_or_cors_config` | The route's CORS policy and redirect allowlist |
| `web_cache_poisoning` | `cache_config` | CDN or cache key configuration of the route |
| `xss_dom_polyglot`, `csv_formula_injection` | `sink_context` | Catalog: how the field is rendered or exported |
| `llm_prompt_injection` | `trust_boundary` | Catalog: which inputs reach an LLM, and with what tools |
| `mcp_agentic_tool_abuse` | `tool_manifest` | The MCP server's registered tool list |
| `race_condition_toctou_burst` | `route_semantics` | Catalog: for example single-use redemption |

**Gap to close before those classes enforce.** The context and history keys in training were written by the generator models, not by a sensor. Before Phase 3, pin a context dictionary in the style of `config/features.yaml`. Then check that the sensor's keys and value shapes occur in training, and retrain on sensor-shaped context where they do not. The client never supplies context: a header saying "I am admin" is request data, not context.

## Model serving

We start on the repo's own Hugging Face scorer, the only path with a parity check against the TPU scores. vLLM comes after the next training run switches to single-token class labels. Then all 41 class scores come from one forward pass, which vLLM serves well.

**The model.** [Gemma 4 E2B-it](https://raw.githubusercontent.com/huggingface/blog/main/gemma4.md), released 2 April 2026 under Apache-2.0.

- 2.3B effective parameters, 5.1B counting the per-layer embeddings.
- 35 layers; the last 20 reuse another layer's keys and values.
- 128K context; we use 2,048 tokens.
- Our adapter: LoRA rank 16, trained with Tunix on a TPU v5e-1 and stored as `lora.npz` in the qwix layout. `training/gemma_tpu/hf_scorer.py` maps it onto the Hugging Face layers. A test proves the hooked model equals the merged weights.

**How it scores today** (`hf_scorer.class_scores`):

1. **Pass 1** reads the prompt plus the answer prefix all classes share. It gives every class's first-token log-probability.
2. **Pass 2** completes the multi-token class names, but only for classes within 12 nats of the best. It recomputes the whole prompt for each class it keeps.
3. **The decision.** Summed log-probabilities go through a softmax at T = 1.038391, then the attack-mass rule and `decide_from_probs`.

**Speed path.** Each step must pass the startup self-check and the release gates before it serves.

| Step | When | Change | Why it should help |
| --- | --- | --- | --- |
| 1 | Phase 0 | Transformers 5.17 (the repo pin), bf16, text-only. LoRA merged at load. Pass 2 reuses the prompt's KV cache. Micro-batches of up to 8 requests, collected over a few milliseconds | The same KV reuse made Qwen's scorer 4.27 instead of 3.56 records/s on an L4, with all 400 top classes unchanged |
| 2 | Next training run | One reserved token per class, so pass 2 disappears. Served by [vLLM](https://github.com/vllm-project/vllm) 0.32 (Gemma 4 with LoRA since 0.19) through its single-token scoring endpoint or classify runner, with the shared system prompt prefix-cached | One forward pass instead of two. vLLM skips its prefix cache for prompt-logprob requests, so multi-token scoring gains little there (the repo saw the same with Qwen) |
| 3 | Optional | FP8 weights and activations (supported on the L4) or Google's QAT W4A16 base | Less memory and faster matrix maths. Scores change, so recalibrate and rerun every gate |

**Runtimes considered**

| Runtime | Gemma 4 E2B + LoRA | Verdict |
| --- | --- | --- |
| Hugging Face transformers 5.17 | Yes; parity tested on CPU (`tests/test_gemma_hf.py`) | Phase 0 default |
| vLLM 0.32 (Apache-2.0) | Yes | After single-token labels |
| SGLang 0.5.22 | Gemma 4 since 0.5.11. LoRA is off for the text model. The multimodal one assumes one MLP width, which E2B's double-wide shared layers may break | Not now |
| Hugging Face TGI | Archived on 21 March 2026 | Avoid |
| llama.cpp, OpenVINO, ONNX Runtime GenAI | All support E2B. The llama.cpp server returns no prompt log-probabilities | CPU fallback, only after single-token labels |

**Deployment.**

- **Two pools**: inline (low latency) and async (batches of 16 to 32, fed from Kafka), so an async backlog never delays an inline call.
- **One pod per NVIDIA L4** (`g6.xlarge`: 24 GB GPU, 4 vCPU, 16 GiB). The vLLM recipe asks for a 24 GB GPU for E2B in bf16.
- **Kubernetes stack**: the NVIDIA GPU Operator, Karpenter for GPU nodes, and KEDA scaling on queue depth.
- **Inline replicas**: at least two per availability zone.
- **Avoid T4 GPUs**: no bf16, and Gemma 4 in fp16 is untested.

**Package and startup.**

- **The package**: `lora.npz` (sha256), `config/policy_gemma.yaml`, `serving.json` (sensor contract, decoded views), the tokenizer fingerprint and `selfcheck.jsonl`. `deploy/package_meta.py` builds this for Qwen and needs a Gemma variant.
- **Base weights**: the pinned revision, mirrored to an internal bucket; no Hub access at runtime.
- **Startup**: load, merge and self-check. Ready only if every check record keeps its top class and moves less than 0.05 in probability against the recorded TPU scores.

**API.** gRPC `Score(telemetry, route_mode, deadline_ms)` returns the decision, the 41 probabilities and the time spent per stage. Health, readiness and Prometheus metrics sit on a side port.

**Managed alternative.** The repo's SageMaker path (`deploy/deploy.sh`, Gemma via `ENSEMBLE=1`) works today. It costs $1.1267 per hour for `ml.g6.xlarge` against $0.8048 for EC2 `g6.xlarge`, 40% more. The AWS account has no EC2 GPU quota yet, so SageMaker is the Phase 0 bridge until that quota exists.

**Qwen as a second member** raises held-out recall from 0.983 to 0.990 offline (`config/policy_ensemble.yaml`). It also doubles GPU cost and has never run on a GPU. Not for launch.

## Decision layer

At launch CRS decides alone. The model gains authority one class and one direction at a time, and each step is earned with shadow data. The combiner is a pure function of three inputs, frozen in a `combiner.yaml` and pinned by a truth-table test.

**Inputs**

- **CRS result**: the inbound anomaly score, the matched rule IDs and their paranoia levels.
- **Model decision**: from `eval.policy.decide_from_probs`. That is a verdict (ALLOW, BLOCK, RATE_LIMIT, CHALLENGE or ABSTAIN), the class, the domain, a calibrated confidence and a fixed reason code. Gemma's frozen policy (`config/policy_gemma.yaml`):
  - temperature 1.038391
  - attack-mass rule, threshold 0.877
  - the flow envelope and the required-context table
  - a known scanner User-Agent turns a benign verdict into log-and-monitor, never a block
- **Route policy** from the catalog: the enforcement mode (shadow, log or enforce), the fail mode, and whether the model is called inline.

**When the model is called inline.** Only for requests where its answer can change the action:

1. CRS would block (score at or above the threshold). The model confirms, or later may overrule.
2. Gray band: some rules fired, below the threshold. The model may escalate.
3. Routes the catalog marks model-required, such as admin, GraphQL, LLM and MCP endpoints, where the context classes live.

Everything else is allowed inline and sampled for asynchronous scoring.

**Truth table** (final state, Phase 4; T = CRS threshold, τ = model threshold):

| CRS | Model | Action | Reason code |
| --- | --- | --- | --- |
| Score ≥ T | Attack class, confidence ≥ τ | Block | `crs_<rule>+ml_<class>` |
| Score ≥ T | Benign at ≥ τ veto (stricter, e.g. 0.99), and every matched rule is on the veto list | Allow, logged for review | `ml_veto_<rule family>` |
| Score ≥ T | Anything else: lower confidence, abstain, timeout | Block (CRS stands) | `crs_<rule>` |
| 0 < score < T | Attack class, enforced for this class | The class action | `ml_<class>` |
| 0 < score < T | Anything else | Allow, logged | `none` |
| 0, route model-required | Attack class, enforced, required context present | The class action | `ml_<class>` |
| 0, not called inline | (async) | Allow; async scoring may add an IP-level action later | `none` |

- **The veto list starts empty.** A CRS rule family joins only when two conditions hold on reviewed shadow data: the model's benign verdicts on that family's matches are right at ≥ 99.5% precision, and the family is not protocol enforcement or request smuggling.
- **Per-class thresholds** (`class_thresholds`) are fitted on shadow data for the classes that turn out to be noisy, then frozen like `τ`.

**Class actions** (proposal; each needs its own evaluation, as the policy file says):

| Class group | Action when enforced |
| --- | --- |
| L7 injection (13 classes), path traversal, SSRF, smuggling, webshell upload | Block, 403 |
| BOLA, BFLA, mass assignment, JWT bypass, GraphQL abuse, prompt injection, MCP abuse | Block, only with the required context present |
| `credential_stuffing_bruteforce` | Rate limit (`class_actions` with eligibility flag `rate_limit_eligible`) |
| `tls_ja4_ua_spoofing` | Challenge: a mismatch alone is not malicious |
| `slowloris_slowhttp`, `http2_rapid_reset_continuation` | Envoy's own timeouts and HTTP/2 limits act inline; the model adds an IP-level rate limit |
| Flow classes: scan, beaconing, DNS exfiltration, SYN flood, UDP amplification | Out of band only: an IP decision (ban or limit for a set time) or an alert |
| Scanner User-Agent on benign traffic | Log and monitor, tag `tier0_scanner_<tool>` |

**What the client sees**

- **Block**: a 403 with a request ID and a "report a wrong block" link, which feeds the review queue.
- **Challenge**: a redirect to a proof-of-work page.
- **Rate limit**: a 429 with `Retry-After` from the Envoy rate-limit service.

**Where it runs.** The combiner runs in waf-sensor, so the fallback still works when waf-model is down. The same truth-table fixture is replayed in Python over shadow logs, so a proposed change to `combiner.yaml` is measured before it ships.

## Network flow path

Flows are scored out of band and never sit on the request path. The model's nine flow-capable classes feed IP-level decisions and alerts.

- **Zeek** is the primary flow source.
- **NetFlow/IPFIX** from routers, collected by GoFlow2, covers segments without a tap.
- **Suricata** adds signature alerts, joined to flows by Community ID.

**Sources** (versions as of the date above):

| Source | What it gives the model (`config/features.yaml`) | Gaps |
| --- | --- | --- |
| [Zeek](https://github.com/zeek/zeek) 9.0 LTS, `conn.log` as JSON | `duration`; payload bytes per direction (`orig_bytes`, `resp_bytes`, the CIC measurement layer); IP-layer bytes and packets (`orig_ip_bytes`, `orig_pkts`, the NetFlow layer; needs `use_conn_size_analyzer`); ports and protocol; `conn_state` and `history`; native `community_id` | No min/max packet length. `tcp_flags_bitmask` must be derived from the `history` letters, with a documented and tested mapping |
| Zeek `dns.log` | Query names per window, for `qname_len_mean`, `qname_entropy_bits_per_char`, `parent_domain_top_share` | Windowing is done in the featurizer |
| Router NetFlow v9 / IPFIX through [GoFlow2](https://github.com/netsampler/goflow2) 2.2.7 (JSON to Kafka) | Packets and bytes per direction (one-way records; the featurizer pairs them), millisecond timestamps, OR of TCP flags. Same semantics as the NF datasets, so the envelope's zero-duration rule applies | Min/max packet length only if the exporter's template carries them. Training already drops them on 15% of synthetic flows |
| [Suricata](https://github.com/OISF/suricata) 8.0 EVE `flow` and `alert` | Per-direction packets and bytes, `age`, hex `tcp_flags`; signature alerts from ET Open (MIT-licensed rules) | Used as alert context and as a review-queue trigger, not as the model's main input |

**Flow featurizer (we build).**

- It is a Kafka consumer in Python. It reuses the repo's own source mappings (`data_pipeline/flows.py`), so production features mean what training features meant.
- It recomputes `pkts_per_s` and `bytes_per_s` from duration, and leaves them out when the duration is zero.
- It builds `conn_count_60s` and `distinct_dst_ports_60s` per source and destination over the clock minute, the same windows training used.
- A missing feature stays an absent key, never a zero.
- It writes `L4_FLOW` telemetry with `observed_until` set to `full_flow` or `fixed_window`.
- **One measurement layer per source, recorded on every record.** Training mixed CIC payload bytes and NetFlow IP bytes, and the repo documents that mix rather than reconciling it. Shadow data must show which layer scores better on our traffic.

**Volume control.** Flows outnumber HTTP requests by far, and the model scores tens of records per second per GPU. Three rules keep that in reach:

- The model scores one record per source and destination window, not one per connection.
- Windows matching a Suricata alert, a high port fan-out or a high connection rate are always scored.
- Of the rest, a stratified sample is scored.

**Decisions.**

- Gemma's frozen policy applies the flow envelope. A flow outside the benign training range abstains (`out_of_distribution_flow`), and zero durations get millisecond interval semantics (`eval.policy.outside_envelope`).
- **External sources**: scan, brute force and flood verdicts become CrowdSec decisions, a ban or rate limit for a set time, which Envoy enforces.
- **Internal sources**: beaconing and DNS exfiltration verdicts become SOC alerts, never automatic bans.

**Known limits to fix before flows enforce.**

- **Missed CIC-2017 scans and brute force.** The current adapter catches none of the real CIC-IDS2017 port scans and SSH/FTP brute force. That is the split bug fixed in the next build.
- **Unknown traffic looks benign.** Out-of-taxonomy flow probes mostly land in benign at high confidence, and v10 abstained on only 64% of them. Phase 1 refits the flow envelope on our own benign traffic.
- **Synthetic-only features.** JA4 consistency, beacon interval variation and the HTTP/2 reset features exist only in synthetic training data. Until a production source is defined, `tls_ja4_ua_spoofing`, `c2_beaconing_anomaly` and `http2_rapid_reset_continuation` stay detection-only. Envoy's own HTTP/2 limits handle rapid reset inline.

## Observability and data

Every verdict is one decision event: who decided, on what evidence, with which model version, and how long each stage took. Raw request content is kept only in a short-lived, encrypted review store.

**Pipeline**

- **Metrics.** Prometheus scrapes:
  - Envoy stats
  - waf-sensor: CRS score histogram, model calls, deadline misses, fallbacks
  - waf-model: latency per stage (fit, scoring, policy), queue depth, batch size
  - NVIDIA's DCGM exporter for the GPUs
  - Kafka consumer lag

  Alertmanager pages.
- **Logs.** Envoy's OpenTelemetry access logger sends OTLP to the OpenTelemetry Collector. The collector writes decision events to Kafka, and ClickHouse loads them.
- **Traces.** OTel traces follow a sampled share of requests through Envoy, waf-sensor and waf-model, into Jaeger.
- **Dashboards.** Grafana on ClickHouse and Prometheus. OpenSearch or Wazuh is an optional SIEM for the SOC; Wazuh reads Suricata's EVE JSON natively.

**Decision event** (one per request; fields are illustrative):

```json
{"request_id": "…", "ts": "…", "route": "api.orders.update", "client": "hmac:3f9c…",
 "crs": {"score": 8, "rules": ["942100", "942190"], "paranoia": 2},
 "model": {"called": true, "threat_type": "sqli_classic", "confidence": 0.97, "verdict": "BLOCK",
           "rule_hint": "ml_sqli_classic", "version": "champion@sha256:…", "windows": 1, "truncated": false},
 "final": {"action": "drop_request", "reason": "crs_942100+ml_sqli_classic", "mode": "enforce"},
 "latency_ms": {"crs": 1.4, "model_queue": 6, "model_score": 88, "total": 97}}
```

**Retention and privacy** (OWASP Logging Cheat Sheet: no session IDs, tokens or passwords in logs):

| Data | Kept for | Treatment |
| --- | --- | --- |
| Decision events | 90 days in detail, 13 months as daily rollups | Client IP replaced by a keyed hash rotated monthly; no header or body values |
| Raw client IP for active bans | Ban duration + 7 days | Only in CrowdSec and the rate-limit store |
| Telemetry for the review queue | 30 days | Redacted first: `Authorization`, cookie values, `Set-Cookie`, JWTs and key-shaped strings (AWS keys and the like) become keyed hashes. Encrypted at rest, every access logged |
| Labelled training records | Life of the dataset | Redacted values only, with lineage and provenance as the repo requires |

IP addresses are treated as personal data. Legal review of the retention table is an open item.

**Dashboards and alerts**

| Signal | Alert when |
| --- | --- |
| Block rate per route | Above 3× its 7-day baseline for 15 minutes |
| CRS-versus-model disagreement rate | Above 2× baseline for a day |
| Model abstentions by reason | Above 2× baseline, or `over_token_budget` above 1% |
| Model deadline misses | Above 1% of model calls for 5 minutes |
| GPU utilisation and queue depth | Queue near its bound for 5 minutes (scale-out lag) |
| Kafka consumer lag (async scorer, featurizer) | Above 5 minutes |
| Startup self-check | Any failure; the pod must not serve |

## Feedback loop and retraining

Every disagreement becomes a candidate label. A new model ships only after it beats the current one on the repo's gates and on a replay of real traffic. The loop reuses the repo's pipeline end to end: build, gate, TPU training, calibration, frozen policy.

**What enters review** (daily caps keep the queue workable):

1. Every "report a wrong block" from the block page.
2. A sample of CRS-versus-model disagreements, in both directions.
3. A sample of model abstentions: low confidence, out-of-envelope flow, over budget.
4. Every async detection at or above τ on traffic that CRS allowed.
5. A random 0.1% of allowed traffic, the baseline for missed attacks and for drift.

**Labelling.**

- One Label Studio project shows the 41 classes with their scope and exclusion text from `config/taxonomy.yaml`.
- Two reviewers label each item, and disagreements go to adjudication.
- Pre-labels may come from the repo's cross-model review (`subagents.xmodel_run both`), on redacted records only. Agreement filters noise; it is not ground truth.
- The repo's rules apply: reviewers read every item (no scripts for labels), and answer keys stay locked during external runs.
- A user's report never becomes a label by itself. An attacker can report their own blocked payload.

**Into the data build.**

- Reviewed records enter as a new source, `prod_feedback`, in `config/sources.yaml`, with redacted values. The lineage group is the client network or site, so one actor never spans splits.
- The latest 20% by capture time is held out as `prod_holdout`. This is the same rule as the CIC split fix, applied to production.
- Then the usual order runs: `data_pipeline.build`, then the `tests.validate_pre_colab` gate, then `eval.shortcut_audit`, then TPU training. One epoch on a TPU v5e-1 took 45 minutes for 36,630 records; two epochs overfit.

**Release gates** for a challenger model:

| Gate | Gemma today | Required |
| --- | --- | --- |
| Attack recall, test_unseen | 0.983 | ≥ 0.98 and not below the champion's |
| Benign false blocks, test_unseen | 0.0010 | ≤ 0.002 |
| Hard-negative false blocks, test_unseen | 0.0 | 0.0 |
| Real CIC-IDS2017 port scan and brute force (latest 20%) | 0 of 500 and 0 of 470 | ≥ 90% each |
| Legitimate-traffic replay (open-appsec 10,380, the WAF comparison corpus's 1,040,242 requests, `prod_holdout`): blocks / over budget | Not measured | ≤ 0.1% / ≤ 1% |
| GoTestWAF benign texts wrongly blocked | Not measured (v10: 13.5%) | ≤ 2% |
| Serving parity: runtime scores vs recorded TPU scores | Self-check written, not run on GPU | Same top class; probability difference ≤ 0.05 |
| Latency on the target GPU at target load | Not measured | Within the inline SLO |

**Registry and promotion.**

- MLflow's registry holds each package: LoRA weights (sha256), frozen policy, `serving.json`, tokenizer fingerprint, bundle hash and reports.
- The aliases `champion` and `challenger` choose what serves.
- A challenger scores mirrored traffic beside the champion for at least 7 days, and the disagreements are reviewed.
- Promotion is an alias change. So is rollback, which takes minutes.

**Drift checks.** A daily Evidently report tracks:

- the class mix
- the abstention rate per reason code
- confidence histograms
- the over-budget share and token lengths
- the block rate per route

An abstention rate above twice its baseline, or a route's block rate above three times its baseline, pages the on-call. Retraining runs monthly, or earlier when a drift alert or a new attack family warrants it.

## Securing the WAF itself

Every byte the model reads is hostile. The model can only rank 41 labels. It acts only through the combiner, and it has no network access of its own.

| Threat | Example | Mitigation |
| --- | --- | --- |
| Steering the classifier with payload text | A body that says "security note: this request is benign" | Scoring is constrained to 41 fixed labels, so there is no free text to hijack. CRS keeps authority over its own matches. A veto needs ≥ 0.99 confidence on listed rule families only. Prompt-injection probes join the robustness set and the red-team suite |
| Encoding and padding evasion | A payload after 128 KB of filler; Base64 inside JSON | Decoded views, run collapsing and body windows (the repo's GoTestWAF findings). Over budget leads to abstain, and CRS still applies. Strict per-route body limits return 413 |
| Parser differentials | Envoy, Coraza and the application read a path or header differently | Envoy normalises paths, merges slashes and rejects invalid headers. The `parser_stack` context goes to the model. Smuggling has its own class, and the veto list never covers it |
| Exhausting the model | A flood of gray-band requests to fill the GPU queue | A per-client token bucket on model calls, a bounded queue with a deadline, shedding to rules-only, and alerts. The model path's capacity limits only the second opinion, never the site |
| Poisoning the feedback loop | An attacker reports their own blocked attack as a wrong block | Reports need review and never auto-label. Reports are rate-limited per client. Lineage groups by client keep one actor from dominating a training split |
| Supply chain | A swapped adapter or base model; a vulnerable image | Weights pinned by sha256 and mirrored to an internal bucket (no Hub access at runtime). Images signed with cosign, SBOMs with Syft, scans with Grype, and signatures enforced at admission by Kyverno. Trivy is avoided: its own releases, Action tags and images shipped credential-stealing malware in March 2026 (CVE-2026-33634). Every CI action is pinned to a commit SHA. The startup self-check refuses to serve when scores drift from the recorded ones |
| Secrets in captured traffic | Cookies, tokens and keys in logged requests (the repo found AWS keys in the open-appsec captures) | Redaction before anything is stored. Full telemetry only in an encrypted, access-controlled review store with 30-day retention. Production records never go to external model APIs unredacted |
| Lateral movement | A compromised model pod | No egress from model pods, mTLS between services, least-privilege service accounts, and GPU nodes in their own pool |

**Red team before each phase gate.** Before every phase gate the stack faces three test sets, with the results recorded in the PLAN progress log:

- GoTestWAF
- the CRS regression tests (go-ftw)
- the repo's robustness set: out-of-taxonomy and insufficient-context probes, 100 per kind from the next build

A bypass that one layer misses but the other catches is a result. A bypass both miss is a release blocker.

## Capacity and cost

GPU cost depends almost entirely on the scorer's per-GPU throughput. Single-token labels are the difference between about 26 and 4 L4 GPUs for 100 model calls per second. Gemma's own throughput on a GPU has not been measured; Phase 0 measures it first.

**Model calls per second** = peak requests/s × inline share (≤ 5%) × mean windows per request + async HTTP sample + flow windows scored.

For example, 1,000 requests/s at peak gives 1,000 × 5% × 1.1 = 55 inline calls. Add a 1% async sample (10/s) and about 20 flow windows/s, and the total is about 85 calls/s. That is close to the 100 calls/s the table sizes for.

| Per-GPU throughput, NVIDIA L4 | Basis | L4 GPUs for 100 calls/s, incl. 1 spare | Per month, EC2 `g6.xlarge` | Per month, SageMaker `ml.g6.xlarge` | Per million calls, EC2 |
| --- | --- | --- | --- | --- | --- |
| 4 calls/s | Qwen v11, Hugging Face scorer, measured on an L4 (3.56 to 4.27/s) | 26 | $15,275 | $21,385 | $55.89 |
| 12 calls/s | Qwen v11, vLLM generation, measured on an L4 (12.07/s) | 10 | $5,875 | $8,225 | $18.63 |
| 40 calls/s | Gemma with single-token labels: a target, not a measurement | 4 | $2,350 | $3,290 | $5.59 |

The prices are on-demand in us-east-1, from the AWS Price List API: EC2 `g6.xlarge` $0.8048/h, SageMaker `ml.g6.xlarge` $1.1267/h. A month is 730 hours. Savings plans and spot capacity (async pool only) lower these figures.

**Sizing rules**

- **Inline pool**: sized for peak at the p99 target, two replicas per zone at least.
- **Async pool**: sized for the daily average. It may lag by up to the 5-minute alert threshold, scales on Kafka lag, and can use spot instances.
- **Sensor and Envoy CPU**: Coraza with CRS costs CPU on every request, and no published number fits our setup. The Phase 0 load test sets replica counts from measured CPU per request, at our real body-size mix.
- **Training**: one Gemma epoch takes 45 minutes on a TPU v5e-1. A monthly retrain is a small line item next to serving.

## Rollout

![Rollout: 5 phases, 4 gates](assets/waf-system/rollout.png)

Every phase can be rolled back by a config change alone: the CRS mode, the route's enforcement mode, or the model alias. A failed gate extends its phase; it never skips one.

## Risks and open questions

The largest risk is speed: if Gemma cannot meet 200 ms at p99 on an L4, it stays a second opinion out of band, and the inline tier is CRS alone. Every other risk has a fallback that keeps the site up and CRS in charge.

| Risk | Early signal | Fallback |
| --- | --- | --- |
| Gemma too slow inline | Phase 0 p99 above 200 ms at target load | Call the model inline only to confirm CRS blocks. Retrain with single-token labels. Otherwise run Gemma async only |
| Real-traffic false blocks above 0.1% | Replay corpora; reviewed shadow decisions | The class stays log-only. Per-class thresholds. Hard-negative data rounds from reviewed production false positives |
| A retrained model regresses (two epochs already overfit once) | Release gates; 7-day challenger shadow | Keep the champion. Rollback is an alias change |
| Production context keys differ from training | Context-dictionary check before Phase 3 | Context classes keep abstaining, a safe default, until retrained on sensor-shaped context |
| Coraza + CRS costs more CPU or latency than planned | Phase 0 load test | Per-route body limits, more sensor replicas, early blocking after tuning. Fallback data plane: HAProxy + coraza-spoa, or nginx + ModSecurity v3 |
| CRS bypasses | CRS security advisories (4.30 fixed three bypasses) | Apply LTS security releases within a week; go-ftw in CI; Gemma reviews the gray band |
| An attacker floods the model path | Deadline misses, queue depth | Per-client model-call budget; shedding to rules-only. Never an outage |
| No GPU capacity | The AWS account has 0 EC2 GPU quota; `ml.g6.xlarge` had no capacity on 17 October | Request quota now. SageMaker bridge with the g6-then-g5 fallback `deploy/deploy.sh` already has |
| Lost training runs (a free Colab TPU run vanished on 19 October) | — | Paid TPU or GPU capacity for production retrains, with checkpoint resume (`training/gemma_tpu/run_tpu.sh`) |
| Gaps in the taxonomy: no application-DoS class; thin CVE-specific data (5.8 training records per KEV CVE for command injection) | GoTestWAF; `reports/kev_coverage.json` | CRS and Envoy limits cover these; the taxonomy decisions below |
| Retention not approved | Legal review | Shorter retention; no raw IPs outside active bans |

**Open questions for the team**

- [ ] Kubernetes with Envoy Gateway (this plan's assumption), or VMs with standalone Envoy?
- [ ] Peak requests per second and body-size mix per site. These size the sensor CPU and the GPUs.
- [ ] Which routes fail closed, and who owns the route catalog for each application?
- [ ] Request EC2 GPU quota, or stay on SageMaker?
- [ ] Legal sign-off on the retention table.
- [ ] Taxonomy decisions already open in the repo:
  - an HTTP-flood / application-DoS class
  - whether `csv_formula_injection` needs formula syntax or only a trigger character
  - more CSE-CIC-IDS2018 capture days
  - CRS 921/931/933/934 fixtures

## Sources

The external facts come from the projects' GitHub sources and the AWS Price List API, read on 19 October 2026. Project sites outside GitHub were unreachable from the research environment: coreruleset.org, envoyproxy.io, huggingface.co, docs.vllm.ai, zeek.org and others. Project versions are linked in the Building blocks table.

**Repository** (`mora-waf/qwen35-waf-ml`, branch `master`)

- `docs/PLAN.md`: the specification and progress log, with latency, real-traffic false positives and benchmark findings.
- `reports/gemma_tpu/REPORT.md`, `config/policy_gemma.yaml`: Gemma results and its frozen policy.
- `reports/ensemble/REPORT.json`: Qwen v11 + Gemma ensemble.
- `reports/crs_oracle.json`: the CRS cross-check on the training data.
- `reports/vllm_bench_L4.json`, `reports/endpoint_speed.json`: scorer throughput on an L4 and an A10G.
- `config/taxonomy.yaml`, `config/features.yaml`, `schemas/telemetry.schema.json`, `schemas/decision.schema.json`: classes, flow features and record shapes.
- `training/prepare.py`, `training/gemma_tpu/hf_scorer.py`, `eval/policy.py`, `subagents/orchestrator.py`, `data_pipeline/generate.py`: the sensor contract, scorer, decision rules and canonical context.

**Proxy, rules and engine**

- [Envoy ext_proc API](https://raw.githubusercontent.com/envoyproxy/envoy/main/api/envoy/extensions/filters/http/ext_proc/v3/ext_proc.proto) and [processing modes](https://raw.githubusercontent.com/envoyproxy/envoy/main/api/envoy/extensions/filters/http/ext_proc/v3/processing_mode.proto)
- [Envoy Gateway ext_proc types](https://raw.githubusercontent.com/envoyproxy/gateway/v1.9.2/api/v1alpha1/ext_proc_types.go)
- CRS docs: [anomaly scoring](https://raw.githubusercontent.com/coreruleset/documentation/main/content/2-how-crs-works/2-1-anomaly_scoring/index.md), [plugins](https://raw.githubusercontent.com/coreruleset/documentation/main/content/4-about-plugins/4-1-plugins.md), [LTS policy](https://raw.githubusercontent.com/coreruleset/coreruleset/main/SECURITY.md), [v4.30.0 release](https://github.com/coreruleset/coreruleset/releases/tag/v4.30.0)
- [Coraza recommended config](https://raw.githubusercontent.com/corazawaf/coraza/main/coraza.conf-recommended)
- coraza-proxy-wasm performance: [issue #262](https://github.com/corazawaf/coraza-proxy-wasm/issues/262), [issue #316](https://github.com/corazawaf/coraza-proxy-wasm/issues/316)
- [CrowdSec AppSec benchmark](https://raw.githubusercontent.com/crowdsecurity/crowdsec-docs/main/crowdsec-docs/docs/appsec/benchmark.md) and [blocklist tiers](https://raw.githubusercontent.com/crowdsecurity/crowdsec-docs/main/crowdsec-docs/docs/central_api/blocklist.md)
- [ingress-nginx retirement statement](https://raw.githubusercontent.com/kubernetes/website/main/content/en/blog/_posts/2026/ingress-nginx-statement.md)

**Model and serving**

- [Gemma 4 launch post](https://raw.githubusercontent.com/huggingface/blog/main/gemma4.md): Apache-2.0, sizes, context
- [Gemma 4 reference code](https://raw.githubusercontent.com/google-deepmind/gemma/main/gemma/gm/nn/gemma4/_gemma4.py): layers, KV sharing
- [vLLM v0.19.0](https://github.com/vllm-project/vllm/releases/tag/v0.19.0): Gemma 4 support
- [vLLM sampling parameters](https://raw.githubusercontent.com/vllm-project/vllm/main/vllm/sampling_params.py): prompt log-probs skip the prefix cache
- [vLLM Gemma 4 recipe](https://raw.githubusercontent.com/vllm-project/recipes/main/Google/Gemma4.md)
- [vLLM quantisation hardware table](https://raw.githubusercontent.com/vllm-project/vllm/main/docs/features/quantization/README.md)
- [TGI archive notice](https://github.com/huggingface/text-generation-inference)
- AWS Price List API, us-east-1: [SageMaker](https://pricing.us-east-1.amazonaws.com/offers/v1.0/aws/AmazonSageMaker/20261016211645/us-east-1/index.json), [EC2](https://pricing.us-east-1.amazonaws.com/offers/v1.0/aws/AmazonEC2/current/us-east-1/index.json)

**Network sensors, data and security**

- [Zeek conn.log fields](https://raw.githubusercontent.com/zeek/zeek/master/scripts/base/protocols/conn/main.zeek)
- [Suricata EVE format](https://raw.githubusercontent.com/OISF/suricata/main-8.0.x/doc/userguide/output/eve/eve-json-format.rst)
- [Trivy advisory GHSA-69fq-xp46-6x23](https://github.com/aquasecurity/trivy/security/advisories/GHSA-69fq-xp46-6x23)
- [OWASP Logging Cheat Sheet](https://raw.githubusercontent.com/OWASP/CheatSheetSeries/master/cheatsheets/Logging_Cheat_Sheet.md)
- [open-appsec WAF comparison project](https://github.com/openappsec/waf-comparison-project)
