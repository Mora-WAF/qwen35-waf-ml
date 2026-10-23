# Gemma WAF System Plan

Proposal, 19 October 2026, revised after review the same day, and again on 20 October 2026 (Gemma moved to a second tier; see Revision 20 October). The diagrams in `assets/waf-system/` are renders of the editable originals and still show the 19 October design with Gemma inline.

## Revision 20 October 2026: Gemma is a second tier

**Decision.** The trained Gemma 4 E2B model runs on vLLM as an asynchronous second tier. Nothing on the request path waits for it. The first build of this plan (`waf-system/`, same repository tree) measured it on a Colab L4 (300 `test_unseen` records):

| Path | Records/s | Latency |
| --- | --- | --- |
| Hugging Face transformers, adapter hooked in (`GemmaScorer`), batch 1 / 4 / 8 / 16 | 1.19 / 1.08 / 1.04 / 1.12 | batch 1: p50 659 ms, p95 1.8 s, p99 5.5 s |
| Hugging Face, text-only `Gemma4ForCausalLM` (repo scorer) | 1.60 | n/a |
| vLLM 0.30.0, merged model, same two-pass method (repo script) | 4.38 | n/a |
| vLLM 0.30.0 through the worker's `VllmScorer`, batch 1 / 16 / 64 records | 3.49 / 4.79 / 4.83 | batch 1: p50 270 ms, p95 441 ms, p99 686 ms |
| Same, fast request pattern (no retrain), batch 1 / 64 | 5.39 / 6.51 | batch 1: p50 172 ms, p99 559 ms; verdicts identical to the exact method on 300 records |

Parity holds on both paths (maximum probability difference to the TPU scores 0.0118 on HF and 0.0245 on vLLM; limit 0.05). On vLLM a single record takes 270 ms at the median and 686 ms at p99, and throughput saturates near 4.8 records/s because each record needs many small scoring requests. The p99 target of 200 ms for an inline model is out of reach with multi-token labels; whether single-token labels bring it within reach is a Phase 0 measurement. At 4.4 to 4.8 records/s a GPU cannot carry the planning inline share. Detection quality is identical on both runtimes (94.9% of attacks blocked, no false blocks, 3.3% abstain on the sample).

**What changed in this plan**

- The sensor queues gray-band requests, pre-filter hits, every CRS block and a random sample on the capped Valkey Stream. A worker scores them in batches and writes verdicts. BLOCK verdicts on the 15 block classes become evidence against the client. Three within ten minutes challenge the client for an hour; ten ban it for four hours (`config/decisions.yaml`). The sensor enforces these decisions on later requests. A single verdict never acts on an IP.
- The model can no longer stop a first request. CRS, its tuning and virtual patches carry that job.
- Parked, because they need a model fast enough for the request path: Phase 3 per-request class blocking, the Phase 4 veto, and the 250 ms inline deadline. The code keeps them behind `WAF_MODEL_INLINE`, off by default.
- Single-token labels are an optimisation of the async tier's cost, no longer a gate to inline use.
- The model-required route flag now means "queue every request of this route".

**Findings from the build that correct this plan** (`waf-system/docs/FINDINGS.md`): Envoy 1.40 does not keep the raw request URI when it normalises the path; Coraza cannot inspect XML attributes (224 of 229 CRS regression failures); the policy threshold is 0.706, not 0.877; a sensor outage on a fail-closed route returns 504, not 503. Each is repeated where it applies below.

**Replay result (21 October, real model).** 10,465 queued requests from the open-appsec and GoTestWAF corpora scored by Gemma v13 on vLLM (`waf-system/reports/replay/REPORT.md`). Of 4,206 attacks, CRS alone blocks 85.2% inline; the model confirms 93% of those, catches 119 more asynchronously (2.8% of attacks, 19% of what CRS misses) and never sees 497 (11.8%), which carry no trigger. On legitimate traffic the model says BLOCK for 0.9 to 1.4% of requests and 15 to 27% of requests are queued (CRS tuning moves that number far more than anything else). With the BLOCK-share rule added below, simulated legitimate clients are almost never challenged (1 of 106 at 100 requests each, 0 of 11 at 1,000) while attackers are challenged after a median of 3 requests. CRS alone blocks 17.0% of legitimate requests untuned (6.7% with two standard exclusions) and costs 100 to 900 ms on cookie-heavy requests; the Phase 2 gate cannot pass without tuning.

**Built and tested by 20 October** (all with the fake scorer unless stated): combiner truth table, pre-filter, route catalog and its CI check, Coraza + CRS 4.25.2 engine (4,575 of 4,804 CRS regression tests pass at paranoia level 4), ext_proc server, the second-tier loop through real Envoy (attacks queued, scored, corroborated into a challenge that is then enforced), flow featurizer, redaction and review sampler, benchmark harness. Gemma v13 on vLLM and on HF was run on an L4. Not built: report-a-block endpoint, go-ftw in CI, `dns.log` featurizer, per-client queue caps, CrowdSec adapter, residential-ASN exemption.

## Summary

OWASP CRS 4 on the Coraza engine checks every request inline. It is embedded in our own sensor behind an Envoy proxy. The fine-tuned Gemma 4 E2B classifier is a second tier on vLLM, never on the request path. The sensor queues, on a capped Valkey Stream:

- requests CRS scores in the gray band;
- requests a cheap deterministic pre-filter marks as suspect (including every XML body);
- every request CRS would block, and every request of a route the catalog marks model-required;
- a random 0.1% sample of the rest.

Network flows join the same queue. A worker scores in batches and writes verdicts for review and dashboards. Corroborated BLOCK verdicts become per-client decisions (challenge, then ban) that the sensor enforces. The model cannot allow a request CRS blocked or lower a CRS score.

Every layer is a maintained open-source project. We write four pieces; the full list with owners is in Components and owners:

1. **waf-sensor** (Go): Coraza + CRS, the client-decision check (Valkey today, CrowdSec later), request capture, the pre-filter, the queue publisher and the decision combiner, a library inside it.
2. **waf-model** (Python): the async worker (Valkey Stream in; verdicts and per-client decisions out) with a vLLM scorer that follows the repo's scoring method (`training/gemma_tpu/hf_scorer.py`, `colab/gemma_vllm.py`) and `eval.policy.decide_from_probs`. An inline gRPC `Score` service exists and is off.
3. **Flow featurizer**: maps Zeek, Suricata or NetFlow records onto the model's flow features (`config/features.yaml`).
4. **Feedback tooling**: redaction, review sampling into Label Studio, and the "report a wrong block" endpoint.

**What Gemma must prove before it acts**

- **Fewer false blocks than CRS.** The repo's approximate CRS cross-check (regex rules only, 143 compiled) saw at least one rule fire on 49% of developer hard negatives and 12% of benign records. Those records are synthetic and built to resemble attacks, and "a rule fired" is not an anomaly-score block. Phase 0 measures real Coraza + CRS 4.25 at paranoia levels 1 and 2 on the same records and on real legitimate traffic.
- **Detection CRS lacks.** On held-out synthetic data Gemma blocks 98.3% of attacks and wrongly blocks 0.10% of benign records (`reports/gemma_tpu/REPORT.md`). The comparison with CRS is fair only once both run on the same records.
- **What it will not do.** It never stops a first request: it answers seconds to minutes later, and a client is acted on only after corroboration. A one-shot exploit CRS misses is the job of virtual patches (Operations).

**Gemma is not ready to enforce yet**

- It lets real CIC-IDS2017 port scans (0 of 500) and SSH/FTP brute force (0 of 470) through. That split bug is fixed, and `http_flood_app_dos` (CIC-2017 `DoS Hulk`, `DoS GoldenEye` and web-port `DDoS` at `conn_count_60s ≥ 200`, plus `proxy_nat_fanin` and `crawler_burst` benign lookalikes) is built and gated in the 42-class v12 dataset (`53,794` records), but not trained yet.
- **Training and serving decode and parse requests differently.** Training data carries URL (depth 1–2), HTML-entity, form, JSON-unicode and chunked decoded views, and URL-decodes `http.query` and `http.cookies` values once (`data_pipeline/generate.py`); the serving path (`training.prepare.add_decoded_views`) produces only JWT and Base64 views, drops duplicate JSON keys (`json.loads` without `object_pairs_hook`), and needs HTTP/2 multi-`cookie` header joining.
- **Too slow for the request path, unmeasured on real traffic.** On an L4 it scores 1.19 records/s on Hugging Face (batch 1: p50 659 ms, p99 5.5 s) and 4.38 on vLLM 0.30. Its false-decision rate on real traffic and its share of over-budget requests are still unknown. The Qwen models put 4.2 to 5.6% of real browsing requests over budget under sensor contract 1.

Phase 0 (8 weeks) fixes these. It retrains once, with single-token pretrained vocabulary labels: now a cost optimisation of the async tier (far fewer scoring requests per record), not a gate to inline use. No model decision is enforced before Phase 3.

## Goals, non-goals and service levels

The system must block web attacks with fewer false blocks than CRS alone. A WAF fault must never silently remove inspection.

**Goals**

- Detect all 38 attack classes of the taxonomy (`config/taxonomy.yaml`; 37 in the current 1-epoch checkpoint, plus `http_flood_app_dos` in the gated 42-class v12 build): 13 L7 injection, 11 API/protocol, 4 AI/modern and 10 network-flow classes. Act on the 15 L7/API/AI classes that need no dynamic context once each passes its gate, as corroborated per-client decisions (Appendix A).
- Cut CRS false positives on developer content, search text and many-cookie browsing traffic through tuning and exclusions, using the model's disagreements as the review queue. Overruling CRS inline (the veto) is parked: it needs a model fast enough for the request path.
- Keep every verdict explainable: a fixed reason code, the CRS rule IDs, the model's class and calibrated confidence, and the model version. The model never writes free text into a decision.
- Close the loop: reviewed false positives and misses flow back into the repo's data pipeline, the next training run and gated promotion.

**Non-goals**

- **Volumetric DDoS at L3/L4.** SYN floods and UDP amplification stay with the upstream provider or CDN. The model's flow classes only raise alerts for them, because their source addresses are spoofed or belong to reflectors.
- **Authorization decisions.** BOLA stays with the application, which alone knows object ownership. BFLA belongs to the gateway's RBAC on verified roles, and mass assignment to schema validation. The model flags these classes; it does not enforce them.
- **Response inspection by the model.** It reads requests and flows only. CRS outbound rules stay available but off at launch.
- **Per-request blocking by the model.** It answers after the request has been served. Its output is evidence against a client, never a verdict on one request.
- **Rules or text written by the model.** It ranks 42 fixed labels (38 attack + 4 benign; 41 in the current checkpoint); it never generates.
- **Full bot management.** The challenge action is a proof-of-work page, not device fingerprinting.
- **Production use of the Qwen3.5 v11 model.** It stays an offline comparison and an optional ensemble member.

**Service levels.** These are proposed targets; the right column shows the evidence the repo holds today.

| Measure | Target | Evidence today |
| --- | --- | --- |
| Added latency, rules layer (Envoy + Coraza + CRS) | p99 ≤ 10 ms; sensor CPU sized from Phase 0 load tests | Measured 20 Oct on a shared laptop: small benign request 10 ms p50 / 26 ms p99 through Envoy + sensor at saturation (Envoy alone 0.9 / 2.2 ms). Legitimate browsing requests with 5 to 10 KB of cookies cost 100 to 900 ms of CRS time with the default rules (replay p99 156 ms; 77 ms with cookies dropped as a rule target). The target does not hold untuned |
| Time from request to async verdict | p95 ≤ 60 s; alert at 5 minutes | Not measured under load. One L4 scores 4.38 records/s on vLLM. No model latency on the request path |
| Share of requests queued for the model | Set from Phase 1 shadow data. Planning figure: 2% of traffic after per-client caps, plus a 0.1% sample | Unknown. The 12% CRS fire rate above comes from synthetic records |
| Wrong actions on real legitimate traffic (CRS blocks and client decisions) | ≤ 0.1% of requests blocked; no legitimate client banned; reviewed challenges counted per class | CRS alone, replayed 20 Oct on 10,380 open-appsec legitimate requests: 17.0% blocked at PL1, 12.4% with `text/plain` allowed, 6.7% with cookies also dropped as a target; 52.8% at PL2. Gemma decisions: see Revision 20 October |
| False blocks, benign held-out set | ≤ 0.2% | Gemma: 0.10% (test_unseen, attack-mass rule) |
| Attack recall on held-out families and sources | ≥ 98% | Gemma: 98.3%; Qwen v11 + Gemma ensemble: 99.0% |
| Requests too long for the model | ≤ 1% | Qwen v10/v11, contract 1: 4.2 to 5.6%. Gemma v13: 0 of 300 on `test_unseen` (L4 run). Contract 2 is built in the 42-class v12 dataset (476 cookie-truncated records) |
| Data-plane availability | 99.95% monthly | The sensor runs with at least 3 replicas per zone, because fail-closed routes depend on it (Failure behaviour) |
| Async tier availability | 99.5% monthly; an outage stops new verdicts and decisions and leaves inline inspection untouched | n/a |

All failure behaviour is collected in one table, in the Failure behaviour section.

## At a glance

![Architecture: inline, out-of-band and learning planes](assets/waf-system/architecture.png)

A request crosses only the top band. *The picture shows the 19 October design with Gemma inline; since 20 October the model box is the asynchronous second tier fed by the queue, and it answers only through verdict streams and per-client decisions.* Everything below the top band runs off the request path.

## Building blocks

Every layer is a maintained open-source project with a permissive or well-understood license. To keep Day-1 operations small, we reuse Valkey, ClickHouse, Git/S3 and Hugging Face transformers instead of standing up separate Kafka, MLflow, Evidently or vLLM services before traffic demands them. Versions and the alternatives we rejected are in Appendix B.

| Layer | Choice | License | Why |
| --- | --- | --- | --- |
| Edge proxy | [Envoy](https://github.com/envoyproxy/envoy) via [Envoy Gateway](https://github.com/envoyproxy/gateway) | Apache-2.0, CNCF | ext_proc is stable and has a shadow mode with no added latency. Rate limits, `jwt_authn` and OpenTelemetry logs are built in |
| Rule engine | [OWASP Coraza](https://github.com/corazawaf/coraza), embedded in waf-sensor | Apache-2.0 | Go library compatible with CRS v4. No Wasm overhead |
| Rules | [OWASP CRS](https://github.com/coreruleset/coreruleset), LTS line | Apache-2.0 | Anomaly scoring, paranoia levels, exclusion plugins. Security fixes until Q3 2027 |
| Classifier | Gemma 4 E2B-it + our LoRA | Apache-2.0 | Trained and evaluated in this repo |
| Model runtime | [vLLM](https://github.com/vllm-project/vllm) 0.30 on the merged text-only model (`hamedgls/waf-gemma4-e2b-v13`), prefix caching, single-token requests restricted to the target token | Apache-2.0 | 4.38 records/s on an L4 against 1.19 to 1.60 for Hugging Face transformers (20 Oct). Self-check difference to the TPU scores 0.0245 (limit 0.05). Transformers with the adapter hooked in stays as the reference and parity path |
| GPU platform | Kubernetes with the [NVIDIA GPU Operator](https://github.com/NVIDIA/gpu-operator), [KEDA](https://github.com/kedacore/keda), [Karpenter](https://github.com/kubernetes-sigs/karpenter) | Apache-2.0 | One operator installs driver, device plugin and DCGM. KEDA (CNCF graduated) scales on queue depth |
| IP reputation and client decisions | [CrowdSec](https://github.com/crowdsecurity/crowdsec) local API + community blocklist (later); today Valkey keys `waf:decision:<client>` with a TTL | MIT; BSD-3-Clause | Shared blocklist, and time-limited decisions the async worker writes. The sensor reads them through one interface, so the store can change |
| Rate limiting and async queue | [envoyproxy/ratelimit](https://github.com/envoyproxy/ratelimit) + [Valkey](https://github.com/valkey-io/valkey) (counters + Valkey Streams) | Apache-2.0; BSD-3-Clause | One Valkey cluster backs global per-IP rate limits, `client_requests_60s` counters and the `waf.telemetry.restricted` stream (`XADD`/`XREADGROUP`) for the async scorer. Kafka is deferred until >10k req/s |
| Challenge page | [Anubis](https://github.com/TecharoHQ/anubis) | MIT | Mature proof-of-work proxy |
| Network sensors | [Zeek](https://github.com/zeek/zeek) on taps; [GoFlow2](https://github.com/netsampler/goflow2) only where taps cannot exist | BSD-3 | One primary tap sensor matching CIC-IDS2017 payload byte semantics. Suricata is optional (only if the SOC already runs it) |
| Telemetry | [OpenTelemetry Collector](https://github.com/open-telemetry/opentelemetry-collector-releases), [Prometheus](https://github.com/prometheus/prometheus), Jaeger | Apache-2.0 | Envoy sends OTLP access logs natively; OTel Collector exports decision events directly to ClickHouse |
| Decision store, drift and dashboards | [ClickHouse](https://github.com/ClickHouse/ClickHouse), [Grafana](https://github.com/grafana/grafana) | Apache-2.0; AGPL-3.0 (internal use) | Analytics over every decision event, plus scheduled SQL views for daily model drift alerts (replacing a separate Evidently service) |
| Labelling and model registry | [Label Studio](https://github.com/HumanSignal/label-studio); signed S3 bundles + Git-tracked `champion.yaml` | Apache-2.0 | Text labelling; `deploy/package_meta.py` tarballs in S3 with `cosign` signatures and a Git-tracked alias file replace running an MLflow server |
| Supply chain and secrets | [cosign](https://github.com/sigstore/cosign), [Syft](https://github.com/anchore/syft), [Grype](https://github.com/anchore/grype), [Kyverno](https://github.com/kyverno/kyverno), [OpenBao](https://github.com/openbao/openbao), [cert-manager](https://github.com/cert-manager/cert-manager) | Apache-2.0; OpenBao MPL-2.0 | Signed images and model packages, SBOMs, scans, admission checks, secrets, mTLS |
| Testing | [GoTestWAF](https://github.com/wallarm/gotestwaf), [go-ftw](https://github.com/coreruleset/go-ftw), [open-appsec WAF comparison corpus](https://github.com/openappsec/waf-comparison-project) | MIT; Apache-2.0 | Attack suites, CRS regression tests, a large legitimate-traffic corpus |

## Components and owners

Every running piece has one owning role. Names replace the roles once they are assigned (an open question).

| Component | What it does | Built or run | Owner |
| --- | --- | --- | --- |
| Envoy, Envoy Gateway | Edge proxy, TLS, rate limits, `jwt_authn`, ext_proc | Run | Platform team |
| waf-sensor | Coraza + CRS, client-decision check, capture, pre-filter, queue publisher, decision combiner (a library inside it) | Build, Go | Security engineering |
| waf-model async worker | vLLM scoring and policy. Reads the Valkey Stream, writes verdicts and corroborated per-client decisions; the only writer of decisions | Build, Python | ML team |
| Flow featurizer | Zeek (and router NetFlow where untapped) records to flow telemetry | Build, Python | ML team with network operations |
| Route catalog and its CI check | `x-waf-*` route metadata, schema-checked on every change | Build | Each application team; schema by security engineering |
| Redaction library | Strips secrets before every store except the review store | Build | Security engineering |
| Review sampler and Label Studio importer | Picks review items, imports them, exports labels to the repo | Build | ML team |
| Report-a-block endpoint | Public form keyed by request ID; rate-limited, behind the challenge | Build | Security engineering |
| Labelling | Two reviewers plus adjudication | People | SOC analysts and ML team |
| CrowdSec, ratelimit + Valkey (including Streams), ClickHouse, OTel Collector, Prometheus, Grafana | Shared infrastructure | Run | Platform team |
| Zeek (and GoFlow2 on untapped links) | Network sensors | Run | Network operations and SOC |
| Signed S3 model bucket, `champion.yaml` and ClickHouse drift SQL | Registry and drift checks | Build and run | ML team |
| Kill switch, virtual patches, runbooks | Emergency controls | Run | Security engineering on-call |

## Inline HTTP path

![Inline request path: 5 steps, 1 decision](assets/waf-system/request-path.png)

**What goes to the model queue** (nothing waits for the answer), highest priority first:

1. **A model-required route**, as the catalog marks it: every request of the route is queued.
2. **A pre-filter hit**, even at CRS score 0. These are cheap, deterministic triggers, and their share of traffic is measured in shadow:
   - a decoded view that shows payload syntax: Base64 text containing shell, SQL or HTML metacharacters, a JWT, or a second URL decoding that reveals them (plain Base64 JSON cookies and nested redirect URLs do not count; on 10,521 legitimate requests that cut the two triggers from 9.9% and 9.1% to 1.7% and 0.1%)
   - a serialized-object or XML DOCTYPE marker
   - an XML body: Coraza has no XPath, so CRS rules aimed at XML attributes never fire (224 of the 229 failures in the CRS regression suite at paranoia level 4)
   - a GraphQL body
   - a body on an admin route
3. **The gray band**: some rules fired, below the CRS threshold.
4. **Every CRS block**, to find the clients behind it and to review CRS false positives.
5. **A random 0.1% sample** of everything else (the baseline for missed attacks and drift).

Publishing never blocks a request: the stream is capped (`MAXLEN`), the publish buffer drops when full, and every drop is counted. Per-client caps on queued requests and body windows (so one client cannot fill the queue) and shedding lowest priority first are specified but not built. The timeout action of a model-required route (`timeout_action`) applies only if an inline model is enabled.

**Envoy settings** (Envoy 1.40 through Envoy Gateway 1.9; one `EnvoyExtensionPolicy` with `extProc` per route class):

- **Co-location.** `waf-sensor` runs as a sidecar in the Envoy pod over a Unix domain socket (`unix:///var/run/waf/sensor.sock`), avoiding loopback TCP/mTLS overhead and saving 1–3 ms at p99 on every request (HTTP/2 with mTLS applies when deployed as a separate service).
- **Filter chain order.** `original_ip_detection` → `local_ratelimit` → `ratelimit` (Valkey) → `decompressor` (with a max inflation-ratio cap, or `415` on routes that do not accept request `Content-Encoding`, so compressed bodies cannot slip past Coraza and Gemma to be decompressed by the application) → `jwt_authn` (`allow_missing_or_failed: true`, writing verified claims and failure reason to `dynamic_metadata`) → `ext_proc` (`waf-sensor`) → `router`.
- **Path normalisation.** Envoy normalises `:path` (`normalize_path: true`, `merge_slashes: true`, `path_with_escaped_slashes_action: REJECT_REQUEST`) before route matching, so `/public/../admin` cannot select the `/public` route policy. Envoy 1.40 does **not** keep the raw URI (tested 20 October: `GET /a/../b` reaches ext_proc as `:path=/b` with no `x-envoy-original-path`). The sensor, CRS and the model see the normalised path, as does the upstream; query strings stay raw. Training targets that carry raw `../` are a known distribution shift, to settle before the retrain: accept it, or normalise the training targets.
- **Request body.** Request headers `SEND`; request body `BUFFERED`, with the route's accepted body limit as the buffer limit. A larger body is rejected with 413, so no byte reaches the application uninspected.
- **Upload routes are the one exception.** They use `BUFFERED_PARTIAL`: the first 64 KiB is inspected, and file parts must have an allowed content type. The catalog names the uninspected remainder as an accepted risk, route by route.
- **`message_timeout`: 300 ms.** The sensor answers after CRS alone (1 to 5 ms in local tests); 300 ms is a ceiling that keeps a stuck sensor from holding requests. If an inline model is ever enabled, the sensor enforces its own 250 ms model deadline inside this.
- **Fail closed by default.** Only routes on the catalog's fail-open list (public GET/HEAD, static assets, health checks) fail open, through a per-route `ExtProcPerRoute` override with `failure_mode_allow: true` (tested: static and health routes answer 200 with the sensor down). A sensor outage on a fail-closed route returns 504 in Envoy 1.40, not 503. Every fail-open pass is counted and alerted (Failure behaviour).
- **Shadow phase.**
  - `observability_mode` (Envoy Gateway `shadowMode`) runs on all traffic and adds no latency, but it supports only a streamed body.
  - A detection-only canary on 5% of traffic therefore runs the real `BUFFERED` settings, so the enforcement path is tested before Phase 2.
- **Client IP.**
  - It comes from Envoy's original-IP detection, with the trusted hop count set to the real chain (CDN, load balancer), or from the PROXY protocol on the load balancer. Never from the raw `X-Forwarded-For`.
  - IPv6 clients are grouped by /64 for rate limits and bans.
- **Verdicts.**
  - Block: `ImmediateResponse` 403 with the request ID.
  - Rate limit: 429 with `Retry-After`.
  - Challenge: 302 to the Anubis page on browser navigation requests (`GET`/`HEAD` with `Sec-Fetch-Mode: navigate` or `Accept: text/html`); state-changing (`POST`/`PUT`/`PATCH`/`DELETE`) or non-HTML API requests get 429 (or 403) so bodies and API clients are never redirected. `waf-sensor` verifies Anubis's signed clearance cookie (bound to `/24` or `/64`, User-Agent hash and TTL) so solved browsers do not loop; a clearance cookie never overrides a `BLOCK` verdict.
  - Allow: the request continues, and dynamic metadata carries the decision to the access log.
- **Rate limits.** Local token buckets run before ext_proc, so a flood is cut before it reaches the sensor. Global per-IP buckets use `envoyproxy/ratelimit` with Valkey. Its image is pinned by digest, as it has had no versioned release since v1.4.0.
- **Client decisions.** waf-sensor is the only enforcer. It reads the decisions the async worker wrote: Valkey keys with a TTL today, the CrowdSec local API in stream mode later. A lookup is cached for 1 second per client and has a 20 ms ceiling; on a store error a 5-second circuit breaker answers "no decision", so a store outage adds no latency and never blocks anyone. A ban is checked before any inspection. Floods at L3/L4 are the upstream provider's job.

**CRS settings** (Coraza 3.8.1 in waf-sensor; CRS 4.25 LTS, supported until Q3 2027; CRS 4.30, the newest, runs in staging only):

- **Anomaly scoring** with the default inbound threshold of 5.
- **Body limits.** `SecRequestBodyLimit` equals the route's limit, with `SecRequestBodyLimitAction Reject` (never `ProcessPartial`).
- **Argument and parser limits.** Coraza 3.8's `SecArgumentsLimit` is 1,000 by default, which stops parsing further arguments unless a rule blocks on overflow. `coraza.conf` enables engine blocking rules `200001`–`200007` (`REQBODY_ERROR`, `MULTIPART_STRICT_ERROR`, `PCRE_LIMITS_EXCEEDED`, and argument-limit overflow) so malformed bodies and extra arguments are never skipped silently; Phase 0 verifies Coraza's behaviour with go-ftw.
- **Rollout.** Phase 1 runs `SecRuleEngine DetectionOnly` with a detection paranoia level of 2. Phase 2 turns the engine on: the threshold starts high and steps down to 5, and the blocking paranoia level goes from 1 to 2.
- **Early blocking.** `tx.early_blocking` stays off until tuning is done: it hides the later-phase matches tuning needs.
- **Cookies and content types first.** The replay found the false blocks and most of the CPU in two places: analytics beacons posting `text/plain` (add it to `tx.allowed_request_content_type`) and tracking cookies (Akamai bot manager, OneTrust) matching injection rules. Phase 1 writes exclusions that drop `REQUEST_COOKIES` as a target of the SQLi, RCE, XSS, LFI, generic and PHP rule families, tested against the attack corpora (no detection loss on open-appsec or GoTestWAF, which carry no cookie attacks: add cookie-borne cases to the regression set before shipping). Envoy caps request header size so a 10 KB cookie cannot cost 900 ms.
- **Exclusions** come from the official CRS 4 plugins (WordPress, Nextcloud, Drupal and others) where those applications run. Our runtime exclusions (`ctl:ruleRemoveTargetById`) go before CRS.
- **Regression tests.** go-ftw tests every exclusion. Its quantitative mode counts false positives per rule and paranoia level on a benign corpus and gates CI against a baseline.

## Failure behaviour

No failure silently removes inspection from a fail-closed route. The trade-off is plain: a sensor outage makes fail-closed routes unavailable. That is why the sensor runs with at least 3 replicas per zone, with disruption budgets and canary deploys.

| Component or event | Behaviour | Per-route override | Alert |
| --- | --- | --- | --- |
| waf-sensor unreachable, or slower than the 300 ms timeout | Fail-closed routes: 504 (Envoy 1.40). Fail-open routes: the request passes uninspected and is counted | The catalog's fail-open list | Fail-open passes above 0.1% of a route's requests for 5 minutes |
| waf-sensor CPU saturated | Local rate limits shed first; the sensor's admission cap returns a fast 503 on fail-closed routes | — | CPU above 70% for 10 minutes; the autoscaler adds replicas |
| Async worker or GPU pool down, or lagging | Inline inspection is unchanged. No new verdicts or decisions; existing decisions expire by TTL | — | Stream lag above 5 minutes |
| A model package fails its startup self-check | The pod never becomes ready; the previous version keeps serving | — | Any failure |
| Decision store unreachable (Valkey today, CrowdSec later) | The sensor treats it as "no decision" and a circuit breaker skips lookups for 5 seconds; no latency is added | — | Any lookup error |
| Rate-limit or Valkey service down | Global limits and async Valkey Streams pause; local rate limits still apply; telemetry events buffer in the sensor up to a bound, then drop and are counted | — | Any error or dropped event |
| ClickHouse or OTel Collector down | Inline traffic is unaffected. OTel Collector queues decision events on disk up to a bound | — | Exporter queue growth or any dropped event |
| Route catalog change is invalid | CI rejects it; the sensor keeps the last good version | — | Rejected deploy |
| Kill switch engaged | As set by on-call (Operations) | — | Always pages |

## Sensor and telemetry contract

The sensor captures raw fields within byte limits, and waf-model applies every transform the model saw in training. Today training and serving do not share one implementation:

- **Decoded views.** Training (`data_pipeline/generate.py`) writes URL (depth 1 to 2), HTML-entity, form, JSON-unicode and chunked views; serving (`training.prepare.add_decoded_views`, used by the SageMaker endpoint) writes only JWT and Base64 views.
- **Query and cookie values.** `data_pipeline/generate.py` URL-decodes `http.query` (`parse_qsl(..., keep_blank_values=True)`) and `http.cookies` (`unquote`) values once, while keeping `http.target` and `http.headers` raw.
- **Duplicate JSON keys and HTTP/2 cookies.** `training/prepare.py` (`_http_values`) calls `json.loads` without `object_pairs_hook`, dropping earlier duplicate keys (`{"q":"<attack>","q":"safe"}`), and `data_pipeline/generate.py` reads only the first `Cookie` header whereas HTTP/2 (RFC 9113 §8.2.3) allows multiple `cookie` headers.

Phase 0 moves every parser and decoder into one module used by the generator, the build and waf-model (with `object_pairs_hook=list` for JSON bodies and `"; "` joining across all `cookie` headers), and adds a parity test on captured ext_proc requests.

**Fields** (`schemas/telemetry.schema.json`). The model input carries telemetry only. A missing value is an absent key, never a zero.

| Field | Production source | Rule |
| --- | --- | --- |
| `schema_version`, `mode` | Sensor | `L7_HTTP`; `HYBRID_CTX` when `context` is attached; `L4_FLOW` on the flow path |
| `decision_stage`, `observed_until` | Sensor | `request_headers` or `request_headers_body`. `observed_until` is the fixed token `request_received`, as canonicalised in training (`subagents/orchestrator.py`) |
| `http.method`, `http.target` | Envoy `:method`, `:path` | Path as normalised by Envoy (the raw URI is not available), query raw; never decoded further |
| `http.protocol` | Envoy stream info | Canonical tokens `HTTP/1.1`, `HTTP/2`, `HTTP/3` |
| `http.headers` | ext_proc header map | Rendered the way training renders them: `Host` first (from `:authority` on HTTP/2), training's name case, duplicates kept, other pseudo-headers left out. Envoy lower-cases names, so the parity test checks this |
| `http.query`, `http.cookies` | Parsed by waf-model from raw `http.target` and all `Cookie` headers joined with `"; "` | Ordered pairs, duplicates kept, values URL-decoded once (`parse_qsl` / `unquote`), matching `data_pipeline/generate.py` |
| `http.body`, `body_encoding` | ext_proc buffered body | The whole body up to the route's limit; larger bodies were rejected earlier. `utf8`, or `base64` for non-UTF-8 bytes |
| `context.edge` | TLS inspector, IP-to-ASN and geo lookup | `tls_version`, `client_asn_type`, `geo`: the neutral keys training uses (`data_pipeline/generate.py`) |
| `context.route` | Route catalog | `auth`, `rate_limit_tier` |
| `history.client_requests_60s` | Valkey counter per client | Only when it is known at decision time |

**How much of a request the model sees** (`training/prepare.py`, contract 2 from the next build):

- **Budget.** Every prompt fits 2,048 tokens. Cookie values are capped first, then header values (512 down to 64 characters), the body (4,000 down to 400) and the URL (1,024 down to 128), over four levels. The last level keeps the first 48 cookies and records `cookies_truncated`.
- **Run collapsing.** Runs of more than 64 identical characters collapse to 64. That defeats single-character padding only; varied filler stays.
- **Windows.** A long body is cut into windows of 3,000 characters, at most 8, head and tail always included. On a 64 KiB body the model therefore sees about 24,000 characters in 8 evenly spaced pieces. CRS sees the whole body, and the combiner never lets a partial view overrule CRS.
- **Over budget.** A request that still fits no budget gets ABSTAIN with reason `over_token_budget`; the rules-only verdict stands.

**Trusted context.** Twelve classes need a named context key (`required_context` in `config/policy_gemma.yaml`) and abstain without it. In the current dataset (`datasets_ready/fast_train.jsonl`), only scenario records—not seeded records—carry `parser_stack`, `sink_context` or `browser_or_cors_config`, and `eval/calibration.py` did not check `required_context` when computing `test_unseen` recall. Phase 0 backfills static listener/catalog keys onto seeded records and enforces `required_context` inside `eval/calibration.py`. All twelve classes stay log-only until two things hold:

- a context dictionary (in the style of `config/features.yaml`) shows that the keys and value shapes the sensor emits occur in training;
- a retrain on sensor-shaped context has passed its gates.

Training contexts were written by generator models, not by a sensor. The client never supplies context: a header saying "I am admin" is request data.

| Class | Context key | Production source |
| --- | --- | --- |
| `bfla_admin_function_probing` | `caller_role`, `route_required_role` | Verified JWT claim (`jwt_authn`); the role the route requires, from the catalog. Gateway RBAC enforces |
| `bola_idor_enumeration` | `authz_context` | None before the request reaches the application, which alone knows ownership. Out of enforcement scope |
| `bopla_mass_assignment` | `object_schema` | OpenAPI request schema of the route. Schema validation enforces |
| `jwt_auth_bypass` | `verifier_config` | Issuer, algorithms and JWKS of the `jwt_authn` provider, which also rejects bad tokens itself |
| `http_request_smuggling` | `parser_stack` | Declared proxy chain per listener, e.g. `envoy -> gunicorn` |
| `open_redirect_cors` | `browser_or_cors_config` | The route's CORS policy and redirect allowlist |
| `web_cache_poisoning` | `cache_config` | CDN or cache key configuration |
| `xss_dom_polyglot`, `csv_formula_injection` | `sink_context` | Catalog: how the field is rendered or exported |
| `llm_prompt_injection` | `trust_boundary` | Catalog: which inputs reach an LLM, with what tools |
| `mcp_agentic_tool_abuse` | `tool_manifest` | The MCP server's registered tool list |
| `race_condition_toctou_burst` | `route_semantics` | Catalog: e.g. single-use redemption |

## Model serving

Gemma 4 E2B-it with our LoRA runs as the asynchronous second tier: a worker reads the restricted Valkey Stream in batches, scores on vLLM, and writes verdicts and corroborated per-client decisions. Measured on a Colab L4 on 20 October (300 `test_unseen` records): vLLM 0.30 scores 4.38 records/s, Hugging Face transformers 1.19 to 1.60. The Revision 20 October table has the full numbers. Capacity is planned from the vLLM figure.

**The model.** [Gemma 4 E2B-it](https://raw.githubusercontent.com/huggingface/blog/main/gemma4.md), released 2 April 2026 under Apache-2.0.

- 2.3B effective parameters, 5.1B counting the per-layer embeddings.
- 35 layers; the last 20 reuse another layer's keys and values.
- 128K context; we use 2,048 tokens.
- Our adapter: LoRA rank 16, trained with Tunix on a TPU v5e-1 and stored as `lora.npz` in the qwix layout. For vLLM it is merged into the base and exported text-only to the Hub bundle `hamedgls/waf-gemma4-e2b-v13` (`training/gemma_tpu/merge.py`). The bundle is self-contained: weights, the repo code the scorer imports (`waf/`), taxonomy, the frozen Gemma policy, `serving.json` (sensor contract) and the 8-record self-check. The Hugging Face reference path (`training/gemma_tpu/hf_scorer.py`) loads `Gemma4ForConditionalGeneration` with the unused vision and audio towers and attaches 205 Python `register_forward_hook` closures; `tests/test_gemma_hf.py` proves the hooked model equals the merged weights in `float32` on CPU. Merging in `float32` before casting to `bfloat16` avoids rounding of small LoRA deltas.

**How a record is scored** (the same method on both runtimes; vLLM: `colab/gemma_vllm.py`, ported to the worker's `VllmScorer`):

1. **Windows.** The sensor contract cuts the record into prompts that fit the token budget (Sensor and telemetry contract); the most attack-like window decides.
2. **Pass 1** reads the prompt plus the answer prefix all classes share. It gives every class's first-token log-probability. On vLLM each of these is a one-token request restricted to the target token (`allowed_token_ids`, raw log-probs) over a prefix-cached prompt.
3. **Pass 2** completes the multi-token class names, but only for classes within 12 nats of the best.
4. **The decision.** Summed log-probabilities go through a softmax at the policy temperature (0.953563 for v13), then the attack-mass rule and `decide_from_probs` with the frozen policy `config/policy_gemma_v13.yaml` (threshold 0.705984).

**Speed path.** Each step must pass the release gates, including full parity, before it serves.

| Step | Status | Change | Evidence and condition |
| --- | --- | --- | --- |
| 1 | Measured 20 Oct | vLLM 0.30 on the merged model, prefix caching, single-token restricted requests | 4.38 records/s on an L4; self-check difference 0.0245; generation agrees with the scorer on clear records |
| 1b | Measured 20 Oct | Fewer requests with the same scores: drop the shared answer-prefix tokens, one top-64 request for pass 1 (exact fallback when an uncertain prompt could change the ranking), deduplicate pass 2; optionally skip pass 2 for the 26 classes whose first token is unique | 6.51 records/s (1.5x), 7.74 with the optional skip (1.8x); 0 of 300 decisions changed; maximum probability difference to the exact method 0.026 and 0.043; self-check to the TPU scores 0.0163 and 0.0245. Default: the first; the skip waits for a larger comparison |
| 2 | Phase 0, optional; worth little after step 1b unless the cost target is missed | Retrain with one distinct pretrained vocabulary token per class (unused `<|reserved_*|>` tokens cannot be used: `train_tunix.py` freezes the tied 262,144 × 1,536 embedding table, leaving them untrained), plus a cached static system prefix | One scoring request per window instead of the many of pass 1 and pass 2. The saving is an estimate until measured; recalibrate and rerun every gate. The 19 October plan counted 40 records/s per L4 for this: a target, not a measurement |
| 3 | Optional | FP8 weights and activations (supported on the L4), or Google's QAT W4A16 base | Scores change: recalibrate and rerun every gate |
| 4 | Only if steps 1 to 3 miss the cost target | A distilled encoder tier trained on the 55k records and reviewed production labels | Not started. It could also serve inline, which Gemma cannot |

vLLM 0.32 is the newest release; 0.30 is the one validated here. A retest on 0.32 belongs in Phase 0. On Qwen v10, vLLM's generation path moved probabilities by up to 0.249 and changed 2 of 400 act-or-abstain decisions; the single-token restricted method used here is the one that passed its self-check on Gemma, and the release parity gate decides again for every package.

**Runtimes considered**

| Runtime | Gemma 4 E2B + LoRA | Verdict |
| --- | --- | --- |
| vLLM 0.30 (merged model, single-token restricted requests) | Yes; self-check passed on an L4 | Production choice for the async tier: 2.7× the transformers throughput |
| Hugging Face transformers 5.17 (`Gemma4ForCausalLM` or the adapter hooked in) | Yes; parity tested on CPU (`tests/test_gemma_hf.py`) and on an L4 | Reference and parity path; 1.19 to 1.60 records/s |
| SGLang 0.5.22 | LoRA off for the text model; the multimodal model assumes one MLP width, which E2B's double-wide shared layers may break | Not now |
| Hugging Face TGI | Archived on 21 March 2026 | Avoid |
| llama.cpp, OpenVINO, ONNX Runtime GenAI | E2B supported; the llama.cpp server returns no prompt log-probabilities | CPU fallback after single-token labels |

**Deployment.**

- **One async pool**, one pod per NVIDIA L4 (`g6.xlarge`: 24 GB GPU, 4 vCPU, 16 GiB). Batches of 16 to 32 from Valkey Streams (`XREADGROUP`); an entry a crashed worker left unprocessed is reclaimed after 60 seconds.
- **Kubernetes stack**: the NVIDIA GPU Operator, Karpenter for GPU nodes, and KEDA scaling on Valkey Stream lag.
- **Floor**: 2 replicas, one per zone. The pool may lag up to the 5-minute alert and may run on spot instances.
- **No T4s**: they have no bf16, and Gemma 4 in fp16 is untested.

**Package, parity and startup.**

- **Package contents**: the merged weights, the frozen policy, `serving.json` (sensor contract, decoded views), the tokenizer fingerprint and the self-check records. The Hub bundle above has them; `deploy/package_meta.py` builds the Qwen variant.
- **Signing**: the package is signed with cosign and verified at load.
- **Base weights**: a pinned revision mirrored to an internal bucket. No Hub access at runtime.
- **Release parity gate**: scores on all of test and test_unseen (10,288 records in the 42-class build; 10,257 in v11), with the same decision on at least 99.9% of records. Any flip must sit within 0.05 of τ.
- **Startup smoke test**: the 8-record self-check (same top class, probability shift under 0.05) runs at every start. The pod is ready only if it passes.

**Interfaces.**

- **Queue**: the sensor writes to `waf.telemetry.restricted` one entry per request: `id`, `client` (the IP, IPv6 grouped by /64), `route`, `trigger`, `crs_score`, `crs_rules` and the raw `telemetry` JSON. The worker normalises it (query and cookie parsing, Host first) and scores.
- **Outputs**: `waf.verdicts` (every scored item), `waf:evidence:<client>` (sorted set of request IDs, one window long), `waf:decision:<client>` (the action, with a TTL) and `waf.decisions.audit`.
- **Inline gRPC** `Score(telemetry, pool, deadline_ms)` returns the decision, the class probabilities and the time per stage. It exists for a future inline-capable model and is off (`WAF_MODEL_INLINE`).
- Health, readiness and Prometheus metrics sit on a side port (not built).

**SageMaker today.**

- **What exists.** `deploy/deploy.sh` deploys Qwen v11, or Qwen v11 and Gemma together (`ENSEMBLE=1`, `ml.g6e.xlarge` at $2.6054/h), or Gemma alone (`GEMMA_ONLY=1`, `ml.g6.xlarge`, hosted on transformers, not vLLM). The ensemble package has never been deployed.
- **Why it still matters.** The AWS account has no EC2 GPU quota, so SageMaker is the bridge until it does. A Gemma-only endpoint on `ml.g6.xlarge` costs $1.1267/h, against $0.8048/h for EC2 `g6.xlarge`. A vLLM worker on SageMaker is not built.

**Qwen as a second member** raises held-out recall to 0.990 offline (`config/policy_ensemble.yaml`). It doubles GPU cost, and the ensemble has never run on a GPU. Not for launch.

## Decision layer

CRS decides every request alone. From Phase 3 the model's verdicts earn authority over clients, one class at a time, never over a single request; each step is earned on reviewed shadow data. The combiner is a pure function of four inputs, frozen in a signed `combiner.yaml` and pinned by a truth-table test (`sensor/combiner`).

**Inputs**

- **CRS result**: inbound anomaly score, matched rule IDs and their paranoia levels, and the variables each rule matched. A match of an engine error rule (`200002`–`200005`: body parse error, strict multipart, argument overflow) counts as a score at the threshold.
- **Client decision**: the action the async worker wrote for this client (`block`, `challenge`, `rate_limit`, or none), read from the decision store.
- **Model decision (inline only, off)**: from `eval.policy.decide_from_probs`: a verdict, the class, a calibrated confidence and a fixed reason code, plus truncated or not, number of windows and cookies dropped. The async worker applies the same frozen policy (`config/policy_gemma_v13.yaml`) to every verdict it writes:
  - temperature 0.953563
  - the attack-mass rule, threshold τ 0.705984 (the 0.877 of the 19 October plan belonged to the earlier policy)
  - the flow envelope and the required-context table
  - a known scanner User-Agent turns a benign verdict into log-and-monitor, never a block
- **Route policy** from the catalog: the enforcement mode (`enforce` or `detect`), the fail mode, whether the route is model-required, and the action on a model timeout (inline only).

**The combiner by phase** (T = CRS threshold). Every phase queues the requests listed under "What goes to the model queue".

| Phase | CRS ≥ T | Gray band, pre-filter hit, model-required route | Client with a decision |
| --- | --- | --- | --- |
| 1 Shadow | Logged (DetectionOnly) | Allow; logged | Logged |
| 2 CRS enforces | Block | Allow | Logged |
| 3 Client decisions | Block | Allow; the client's verdicts corroborate into a decision | Ban: block before any inspection. Challenge or rate limit: applied to a request that would otherwise be allowed. A CRS block still wins |
| 4 Inline (parked) | Block, or allow when every veto condition holds | Class action if the model says an enforced attack class at ≥ τ, else allow | As Phase 3 |

Phases 3 (per-request class actions) and 4 of the 19 October plan need an inline model. The code keeps them behind `WAF_MODEL_INLINE`, tested with a fake model, off by default.

**How verdicts become decisions** (`config/decisions.yaml`, signed and reviewed like the combiner):

- Only a BLOCK verdict on one of the 15 block classes counts as evidence. Log-only classes, ABSTAIN and ALLOW never do.
- Evidence is a sorted set of request IDs per client, one window (10 minutes) long, so a redelivered queue entry counts once.
- BLOCK verdicts must also be at least 30% (`min_share`) of everything scored for the client in the window. Without it, a busy legitimate client collects 3 false BLOCKs in minutes (replay: 16 of 106 clients with 100 requests each); with it, 1 of 106.
- 3 pieces of evidence challenge the client for 1 hour; 10 ban it for 4 hours. A decision is never downgraded; a stronger one replaces it.
- A challenge comes first because shared NAT and mobile addresses carry real users. Exempting residential and mobile ASNs from bans is specified for flows and not built for HTTP.
- Phase 3 per class: at least 600 reviewed decisions for the class with no error, first at the challenge stage, then at the ban stage.

**Veto conditions (parked).** They apply only with an inline-capable model. All must hold, or CRS stands:

1. The model says benign at 0.99 or more.
2. Every matched rule is on the veto list.
3. The model saw the whole request: not truncated, one window, no cookies dropped.
4. Every variable a CRS rule matched appears in the model input unabridged, and decoded where CRS decoded it. Until the decoder parity test exists, the sensor reports this as false, so no veto can fire.

**Joining the veto list (parked).** The list starts empty. A rule family joins after at least 600 reviewed matches with no wrong veto: at 0 errors in 600 that gives a 95% upper bound on the error rate of 0.5%. Engine body/parser error rules (`200001`–`200007`), protocol-enforcement rules (`920xxx`) and smuggling/protocol-attack rules (`921xxx`) never join; the combiner config loader rejects them.

**Abstaining for missing context** (a Phase 0 policy change). Today, if the attack-mass rule picks an attack class whose required context is missing, the policy abstains, even at attack mass 0.99. Payload text that tips the top class to `llm_prompt_injection` could exploit that. The new rule:

- Take the best attack class whose context is present.
- If no such class exists but attack mass is at least τ, the verdict is "suspicious". The worker records it (`suspicious`) for review; it is not evidence against the client. Inline it would challenge in the gray band.
- Probes that wrap real payloads in context-steering text join the robustness gate. v11 passed 100% of the existing classifier-injection probes.

**Class actions.** Appendix A lists all 38 attack classes with their action:

- **Challenge, then ban (client decisions) from Phase 3**, each class after its own gate: 15 classes needing no context. These are 11 injection classes plus path traversal, SSRF, GraphQL abuse and prototype pollution.
- **Brute force and HTTP application floods** are rate-limited, then challenged (and banned after corroboration for floods).
- **External port scans** get time-limited bans after corroboration.
- **Log only**: the 12 context classes, WebSocket/gRPC abuse and the classes that rest on synthetic-only features.
- **Alert only**: SYN floods and UDP amplification, whose sources are spoofed or are reflectors.
- **SOC alerts**: internal beaconing and DNS exfiltration.

**Per-class thresholds** (`class_thresholds`) are fitted on shadow data for noisy classes, then frozen like τ.

**What the client sees.**

- **Block**: a 403 with a request ID and a "report a wrong block" link. A banned client gets it on every request for the ban's TTL.
- **Challenge**: a 302 redirect to the Anubis proof-of-work page on browser navigation `GET`/`HEAD` requests, or a 429 on non-navigation API and state-changing requests.
- **Rate limit**: a 429 with `Retry-After`.

**Where it runs, and how it changes.**

- The combiner is a library in waf-sensor, so inspection works when the async tier is down.
- A change to `combiner.yaml`, `decisions.yaml`, the veto list or `class_thresholds` goes through a reviewed pull request with two approvals.
- CI replays shadow logs through the change, with the same gates a model must pass.
- The signed result is verified at load.

## Network flow path

Flows are scored out of band, one completed connection at a time, the same unit the model was trained on. Their verdicts become time-limited IP decisions for a few classes and SOC alerts for the rest; nothing on the request path waits for them.

- **Zeek** is the single primary flow sensor on tapped links (matching CIC-IDS2017 payload byte semantics directly).
- **NetFlow/IPFIX** from routers, collected by GoFlow2, is used only on network segments where a packet tap cannot exist.
- **Suricata** is optional: joined by Community ID only if the SOC already runs it for ET Open alerts, rather than deploying a second tap daemon for the WAF alone.

**Sources** (versions in Appendix B):

| Source | What it gives the model (`config/features.yaml`) | Gaps |
| --- | --- | --- |
| Zeek `conn.log` as JSON (primary) | `duration`; payload bytes per direction (the CIC measurement layer); IP-layer bytes and packets (the NetFlow layer; needs `use_conn_size_analyzer`); ports and protocol; `conn_state` and `history`; native `community_id` | No min/max packet length. `tcp_flags_bitmask` must be derived from the `history` letters, with a documented and tested mapping |
| Zeek `dns.log` (primary) | Query names, for the DNS window features | Query-name length and entropy are synthetic-only in training; only `parent_domain_top_share` has real (CIC-Bell-DNS-EXF) training values |
| Router NetFlow v9 / IPFIX through GoFlow2 (untapped links only) | Packets and bytes per direction (one-way records the featurizer pairs), millisecond timestamps, OR of TCP flags: the NF datasets' semantics, so the envelope's zero-duration rule applies | Min/max packet length only if the exporter template carries them; training already drops them on 15% of synthetic flows |
| Suricata EVE `flow` and `alert` (optional, if SOC runs it) | Per-direction packets and bytes, `age`, hex `tcp_flags`; ET Open signature alerts (MIT-licensed rules) | Optional alert context and review trigger, not deployed solely for the WAF |

**Flow featurizer (we build).**

- **One record per completed connection** (`full_flow`), like `data_pipeline/flows.py` builds for training, with `conn_count_60s` and `distinct_dst_ports_60s` for the same source and destination over the clock minute.
- **Same mappings as training.** It reuses the repo's source mappings, so a production feature means what the training feature meant.
- **Derived rates.** It recomputes `pkts_per_s` and `bytes_per_s`, and leaves them out when the duration is zero.
- **Missing values** stay absent keys, never zeros.
- **One measurement layer per source**, recorded on every record. Training mixed CIC payload bytes and NetFlow IP bytes, so shadow data must show which layer scores better on our traffic.

**Volume control.** The model scores tens of records per second per GPU, so it scores a selection of connections:

- every connection whose minute counts pass fan-out or rate thresholds (and any connection matching a Suricata alert where Suricata runs);
- a stratified sample of the rest, with a cap per source.

**Decisions.**

- **Envelope.** Gemma's frozen policy applies the flow envelope: a flow outside the benign training range abstains (`out_of_distribution_flow`). Zero durations get millisecond interval semantics (`eval.policy.outside_envelope`).
- **External port scans.** A CrowdSec ban for 4 hours, after at least 3 corroborating verdicts within 10 minutes. Residential and mobile ASNs get a challenge instead, because many users share those addresses.
- **External brute force and HTTP application floods.** A per-IP rate limit, then a challenge (and a CrowdSec ban after corroboration for floods), with the same corroboration rule.
- **SYN flood and UDP amplification.** Alerts only. The sources are spoofed or are reflectors, so a ban would hit victims, and an L7 ban cannot stop an L3/L4 flood. Mitigation is upstream.
- **Internal hosts.** Beaconing and DNS exfiltration from internal hosts become SOC alerts, never automatic bans.
- **One enforcer.** waf-sensor enforces every IP decision. Only the async scorer writes them.

**Known limits to fix before flows enforce.**

- **CIC-2017 misses.** The current adapter catches none of the real CIC-IDS2017 port scans and SSH/FTP brute force; that is the split bug fixed in the 42-class v12 build.
- **Unknown traffic looks benign.** Out-of-taxonomy probes mostly land in benign. Qwen abstained on 64% of them (v10) and 84% (v11); Gemma has no robustness scores yet. Phase 1 refits the flow envelope on our own benign traffic.
- **Synthetic-only features.** JA4 consistency, beacon intervals, HTTP/2 resets and DNS query-name statistics exist only in synthetic training data. The classes that rest on them stay log-only or alert-only until a production source is defined and the model is retrained on it.

## Observability and data

Every verdict is one decision event: who decided, on what evidence, with which model version, and how long each stage took. Each data store has exactly one redaction rule, set in the retention table below.

**Pipeline**

- **Metrics.** Prometheus scrapes:
  - Envoy stats, including ext_proc errors and fail-open passes
  - waf-sensor: CRS score histogram, pre-filter hits, model calls, deadline misses, fallbacks
  - waf-model: latency per stage, queue depth, batch size
  - NVIDIA's DCGM exporter for the GPUs
  - Valkey Stream consumer lag (and OTel Collector exporter queue depth)

  Alertmanager pages.
- **Logs and async queue.** Envoy's OpenTelemetry access logger sends OTLP to the OpenTelemetry Collector, which writes decision events (`waf.decisions`, without request values) directly to ClickHouse via its native ClickHouse exporter. For full telemetry of model-called and sampled requests (`waf.telemetry.restricted`), `waf-sensor` pushes to a capped Valkey Stream (`XADD` with `MAXLEN ~`, TLS and Valkey ACLs) read only by the async scorer and the review sampler (`XREADGROUP`). That reuses the Valkey cluster already run for `envoyproxy/ratelimit` and avoids operating a Kafka cluster until sustained traffic exceeds ~10,000 requests/s.
- **Traces.** OTel traces follow a sampled share of requests through Envoy, waf-sensor and waf-model, into Jaeger.
- **Dashboards.** Grafana on ClickHouse and Prometheus. OpenSearch or Wazuh is an optional SIEM for the SOC.

**Decision event** (one per request; fields are illustrative). The model's verdict is not in it: it arrives later on `waf.verdicts` and joins by `request_id`.

```json
{"request_id": "…", "route": "api.orders.update",
 "crs": {"score": 3, "rules": ["942190"]},
 "prefilter": ["graphql_body"], "async": "gray_band",
 "final": {"action": "allow", "reason": "allow", "phase": 3},
 "latency_ms": {"crs": 1.4, "total": 1.5}}
```

A verdict on `waf.verdicts`: `{"id", "client", "route", "trigger", "crs_score", "verdict", "threat_type", "confidence", "rule_hint", "reason", "windows", "truncated", "score_ms", "model_version"}`. A decision on `waf.decisions.audit`: `{"client", "action", "ttl_s", "evidence", "class", "model_version"}`.

**Retention and redaction, one rule per store.** The OWASP Logging Cheat Sheet rules out session IDs, tokens and passwords in logs.

| Store | Content | Kept for | Who reads it |
| --- | --- | --- | --- |
| Valkey Streams `waf.verdicts` and `waf.decisions.audit` | Verdict and decision fields only (the client key is the grouped IP) | 7 days, then ClickHouse rollups | SOC, security engineering, ML team |
| Valkey Stream `waf.telemetry.restricted` | Full telemetry of sampled/async items; TLS in transit, encrypted volume | 24 hours (capped stream) | Async scorer and review sampler only (Valkey ACLs) |
| ClickHouse (`waf.decisions`) | Decision events, no request values; client IP as a keyed hash rotated monthly | 90 days in detail, 13 months as daily rollups | SOC, security engineering, ML team |
| CrowdSec, rate-limit store | Raw client IP | Ban duration + 7 days | waf-sensor, async scorer |
| Review store | Full telemetry of review items; encrypted, every access logged | 30 days | Reviewers |
| Training data in the repo | Reviewed records. Attack-bearing values are kept. Credential-shaped strings (AWS keys, session cookies, JWT signatures) become format-preserving surrogates of the same length and alphabet, so the evidence keeps its shape | Life of the dataset | ML team |
| External model APIs | Nothing from production | — | — |

IP addresses are treated as personal data. Legal review of this table is an open item.

**Dashboards and alerts**

| Signal | Alert when |
| --- | --- |
| Fail-open passes per route | Above 0.1% of the route's requests for 5 minutes |
| Block rate per route | Above 3× its 7-day baseline for 15 minutes |
| CRS-versus-model disagreement rate | Above 2× baseline for a day |
| Model abstentions by reason | Above 2× baseline, or `over_token_budget` above 1% |
| Async verdict lag | p95 above 60 s for 5 minutes |
| GPU queue depth | Near its bound for 5 minutes |
| Valkey Stream consumer lag | Above 5 minutes |
| Startup self-check, signature check | Any failure |

## Feedback loop and retraining

Every disagreement becomes a candidate label. A new model ships only after it beats the current one on the release gates below. The loop reuses the repo's pipeline end to end: build, gate, TPU training, calibration and a frozen policy.

**What enters review** (daily caps keep the queue workable):

1. Every "report a wrong block" from the block page. The endpoint is rate-limited and sits behind the challenge.
2. A sample of CRS-versus-model disagreements, in both directions.
3. A sample of model abstentions: low confidence, missing context, out-of-envelope flows, over budget.
4. Every async detection at or above τ on traffic CRS allowed.
5. A random 0.1% of allowed traffic, the baseline for missed attacks and drift.

**Labelling.**

- One Label Studio project shows the 42 classes (38 attack + 4 benign) with their scope and exclusions from `config/taxonomy.yaml`.
- Two reviewers label each item; disagreements go to adjudication.
- Pre-labels come only from in-house models (Gemma, Qwen). Production records never go to external model APIs.
- The repo's rules apply: reviewers read every item (no scripts for labels), and agreement filters noise; it is not ground truth.
- A user's report never becomes a label by itself.

**Into the data build.**

- Reviewed records enter as a new source, `prod_feedback`, in `config/sources.yaml`, with values handled as the retention table says.
- **Lineage.** The lineage group is the client network or site, so one actor never spans splits. Whole groups are assigned by the time they were first seen: groups first seen in the latest 20% of the period form `prod_holdout`.
- Then the usual order: `data_pipeline.build`, the `tests.validate_pre_colab` gate, `eval.shortcut_audit`, TPU training. One epoch took 45 minutes for 36,630 records (38,530 in the 42-class build); two epochs overfit.

**Release gates** for any model package. These are offline; phase gates are in Rollout.

| Gate | Gemma today | Required |
| --- | --- | --- |
| Attack recall, test_unseen | 0.983 | ≥ 0.98 and not below the champion |
| Benign false blocks, test_unseen | 0.0010 | ≤ 0.002 |
| Hard-negative false blocks, test_unseen | 0.0 | 0.0 |
| Real CIC-IDS2017 port scans, brute force and HTTP floods, the latest 20% held out (305 flows: 231 port scan/brute force + 74 flood) | Not measured on the fixed build; 0 of 970 before the fix | ≥ 90% |
| Attack flows from a source absent from training (e.g. NF-UNSW-NB15 attack rows) | Not measured | Threshold agreed in Phase 0 and recorded in the progress log |
| Legitimate replay through the worker and the decision policy: open-appsec 10,380, the WAF comparison corpus (1,040,242 requests), `prod_holdout` from Gate 1 on | Not measured | Blocks ≤ 0.1%; no legitimate client challenged more than once per replay window; over budget ≤ 1% |
| GoTestWAF benign texts wrongly blocked | Not measured (v10: 13.5%) | ≤ 2% |
| Robustness: out-of-taxonomy abstention; classifier-injection and context-steering probes | No Gemma scores yet (v11: 84%; 100%) | ≥ 84%; 100% |
| Serving parity on all of test and test_unseen (10,288 records in the 42-class build; 10,257 in v11) | Not run | Same decision on ≥ 99.9%; any flip within 0.05 of τ |
| Throughput on the target GPU (vLLM, L4) and async verdict lag at the planned queue rate | 4.38 records/s (20 Oct, 300 records, two-pass labels); lag not measured | Lag p95 ≤ 60 s at the Phase 1 queue rate; capacity table updated with the measured rate |

**Minimum evidence.** Any precision gate on reviewed production actions needs at least 600 reviewed actions with no error. That gives a 95% upper bound on the error rate of 0.5%. A class that cannot reach 600 stays log-only.

**Registry and promotion.**

- **Registry.** Signed model bundles (`deploy/package_meta.py` tarballs verified with `cosign`) live in an immutable S3 bucket—no separate MLflow server to run.
- **Aliases.** A Git-tracked `champion.yaml` (and `challenger.yaml`) pins the active bundle's `sha256` digest. Moving it requires a pull request with two approvals, and CI verifies the signature and release gates before applying.
- **Shadow first.** A challenger scores mirrored traffic beside the champion for at least 7 days, and the disagreements are reviewed.
- **Rollback** is a Git revert of `champion.yaml` (or Kubernetes deployment rollout undo) and takes minutes.

**Drift checks.** Scheduled ClickHouse SQL views (alerted through Grafana and Prometheus—no separate Evidently service) track the daily class mix, abstention rate per reason, confidence quantiles, over-budget share, token lengths and block rate per route against a 7-day baseline. An abstention rate above twice its baseline, or a route's block rate above three times its baseline, pages the on-call. Retraining runs monthly, or earlier on a drift alert or a new attack family.

## Securing the WAF itself

Every byte the model reads is hostile. The model only ranks 42 labels (41 in the current checkpoint), acts only through the combiner, and has no network access of its own. Everything that can weaken blocking (model packages, policy, combiner, exclusions, aliases) is signed, reviewed by two people and replayed before it applies.

| Threat | Example | Mitigation |
| --- | --- | --- |
| Steering the classifier with payload text | Prompt-injection text around a real payload | Scoring is constrained to 42 fixed labels. The model cannot allow a request CRS blocked or lower a CRS score: it only adds evidence against a client, and a single verdict acts on nothing. Missing context leads to "suspicious" (reviewed), not to a block. Classifier-injection and context-steering probes sit in the release gates |
| Encoding and padding evasion | A payload after 128 KB of filler; Base64 inside JSON; compressed request body | Over-limit bodies are rejected with 413; unexpected `Content-Encoding` is rejected with 415 or decompressed with a ratio cap before `ext_proc`. Decoders are shared by training and serving (including duplicate JSON keys via `object_pairs_hook=list` and all HTTP/2 `cookie` headers). Run collapsing handles single-character filler. CRS reads the whole body; the model's partial view never overrules it |
| Parser differentials | Envoy, Coraza and the application read a path or header differently | Envoy normalises `:path`, merges slashes and rejects escaped slashes before route matching (Envoy 1.40 does not preserve the raw URI, so the model and CRS see the normalised path; the upstream receives the same one). Invalid headers and ambiguous framing are rejected. The veto list never covers engine body-error (`200001`–`200007`), protocol (`920xxx`) or smuggling (`921xxx`) rules |
| Exhausting the model | A flood of pre-filter or gray-band requests | Local rate limits before ext_proc. The queue is capped, publishing never blocks a request, and every drop is counted. Per-client caps on queued requests and priority shedding are specified, not built. The async tier can lag without any effect on inline inspection |
| Exhausting the sensor | CPU-heavy requests to push the sensor past its timeout | Local rate limits before ext_proc, body and argument limits, an admission cap. Fail-closed routes answer 503; they never pass uninspected |
| Getting someone else challenged or banned | An attacker sends attack payloads through a shared NAT or proxy address, or from a spoofed-looking path, so that real users behind it are challenged | Challenge first (a person passes it), a ban needs 10 corroborating verdicts, every decision has a TTL and an audit record, decisions are sampled into review, and the kill switch clears them. Residential and mobile ASN exemption is not built |
| Poisoning the feedback loop | An attacker reports their own blocked attack as a wrong block | Reports need review and never auto-label. Reports are rate-limited and challenged. Lineage by client keeps one actor from dominating a split |
| Tampering with promotion or config | Moving the `champion` alias; adding a rule to the veto list | Signed packages verified at load. Two-person approval for `champion.yaml` moves and config changes. The same replay gates as a model. Audit log |
| Supply chain | A swapped adapter or base model; a vulnerable image | Weights pinned by sha256 and mirrored internally. Images and model packages signed with cosign. SBOMs with Syft, scans with Grype, signatures enforced at admission by Kyverno. Trivy is avoided: its releases, Action tags and images shipped credential-stealing malware in March 2026 (CVE-2026-33634). CI actions pinned to commit SHAs |
| Secrets in captured traffic | Cookies, tokens and keys in requests (the repo found AWS keys in the open-appsec captures) | One redaction rule per store (Observability). Full telemetry only in the restricted Valkey Stream and the review store |
| Control plane and lateral movement | A stolen credential that writes CrowdSec bans or reads the restricted stream | Only the async scorer's credential may write CrowdSec decisions. Valkey ACLs per key prefix and stream. Label Studio and Grafana behind SSO with roles. No egress from model pods; mTLS between services |

**Red team before each phase gate.** The stack faces three test sets, and the results go into the PLAN progress log:

- GoTestWAF
- the CRS regression tests (go-ftw)
- the repo's robustness set: out-of-taxonomy, insufficient-context, classifier-injection and context-steering probes, 100 per kind

A bypass one layer misses but the other catches is a result. A bypass both miss is a release blocker until a virtual patch covers it.

## Operations and emergency controls

On-call can switch the model off, put CRS in detection-only mode, or deploy an emergency rule within hours, without a retrain. Each control is a signed config change, logged, and it pages when used.

| Control | Effect | Used when |
| --- | --- | --- |
| Kill switch 1: model off | Stops queueing and ignores client decisions; rules-only verdicts everywhere. Existing decisions expire by TTL, or are flushed | Model drift, bad package, GPU outage, a wave of wrong decisions |
| Kill switch 2: CRS detection-only, per route or global | Nothing blocks on that scope; everything is logged | A false-block storm |
| Kill switch 3: sensor bypass for a route | The route fails open; every pass is counted | Sensor outage on a route that must stay up; a deliberate, logged decision |
| Virtual patch | An emergency rule in a `virtual-patches` CRS plugin (CRS reserves IDs 9,500,000 to 9,999,999 for plugins), tested with go-ftw, deployed within 4 hours | A new exploit that CRS and the model both miss. Removed once covered |

**Runbooks** for the security-engineering on-call: false-block storm, sensor overload, async lag, wrong client decisions, drift alert, bypass report, kill switch use and recovery.

**Edge placement.**

- **TLS** terminates at Envoy, or at a CDN in front of it. With a CDN, the client IP (and any TLS fingerprint) come from the CDN's headers through the trusted hop count.
- **HTTP/3** stays off at launch unless the CDN terminates it.

**WebSocket and gRPC.** ext_proc sees the WebSocket upgrade request and unary gRPC bodies, but not WebSocket frames or streamed messages. So `websocket_grpc_abuse` detection is limited to what the upgrade or unary request shows, and that class stays log-only.

**Changes.** The route catalog, `combiner.yaml`, the policy, the veto list and the CRS exclusions change only through reviewed pull requests, with CI replay and signing.

## Capacity and cost

The model is off the request path, so GPU cost is set by the queue rate and the per-GPU throughput, and a lag of a few minutes is acceptable. Gemma's throughput on an L4 is measured: 1.19 records/s on Hugging Face and 4.38 on vLLM (20 October). Phase 0 measures the single-token variant; Phase 1 shadow data replaces every planning figure below.

**Model calls per second** = peak requests/s × queued share × mean windows per request + random sample + flow records scored.

The queued share is the gray band, pre-filter hits, CRS blocks and the model-required routes, minus what the per-client cap (20 per client per 10 minutes) and the "client already has a decision" rule hold back. The replay measured it on real legitimate browsing traffic (open-appsec, 10,521 requests): **26.6%** at blocking PL1 untuned (CRS blocks 23.9%, pre-filter 2.6%), **15.0%** with the `text/plain` and cookie exclusions of the Findings. The 2% of the 19 October plan was a guess, and CRS tuning decides this number before anything else does.

For example, take 1,000 requests/s at peak. At a 15% share: 1,000 × 15% × 1.1 = 165 calls, plus the 0.1% sample (1/s) and about 20 flow records/s, about 186 calls/s: 44 L4s on vLLM including one spare, about $26k a month on EC2. At the 2% the table below assumes, the total is about 41 calls/s. The table's rows are per-GPU throughputs; read its GPU column for 41 calls/s and scale it.

| Per-GPU throughput, NVIDIA L4 | Basis | L4 GPUs for 41 calls/s (load + 1 spare, floor of 2) | Per month, EC2 `g6.xlarge` | Per month, SageMaker `ml.g6.xlarge` | Per million calls, EC2, at full use |
| --- | --- | --- | --- | --- | --- |
| 1.19 records/s | Gemma on Hugging Face (`GemmaScorer`, batch 1), measured on an L4, 20 Oct | 36 | $21,150 | $29,610 | $187.86 |
| 4.38 records/s | Gemma on vLLM 0.30, measured on an L4, 20 Oct (the worker's scorer reached 4.83 at batch 64); the planning figure | 11 | $6,463 | $9,047 | $51.04 |
| 40 records/s | Gemma with single-token labels and a cached prefix: a target, not a measurement | 3 | $1,763 | $2,467 | $5.59 |

The prices are on-demand in us-east-1, from the AWS Price List API: EC2 `g6.xlarge` $0.8048/h, SageMaker `ml.g6.xlarge` $1.1267/h. A month is 730 hours. The last column assumes full use and no spare. The async pool tolerates spot capacity, which lowers these figures.

**Sizing rules**

- **Async pool**: sized for the daily average of the queue rate. It may lag up to the 5-minute alert threshold, scales on Valkey Stream lag, can use spot instances, and never drops below 2 replicas (one per zone). The inline pool of the 19 October plan is gone.
- **Levers if the cost is too high**: a lower queued share (tighter triggers, per-client caps, stop queueing a client that already has a decision), a lower sample rate, the single-token retrain, quantisation, and the distilled-encoder tier (Model serving, step 4).
- **Sensor and Envoy CPU**: Coraza with CRS costs CPU on every request, and no published number fits our setup. In local tests CRS took 1 to 5 ms per request; the Phase 0 load test, at our real body-size mix, sets replica counts from measured CPU per request. The floor is 3 sensor replicas per zone.
- **Training**: one Gemma epoch takes 45 minutes on a TPU v5e-1, a small line item next to serving.

## Rollout

![Rollout: 5 phases, 4 gates](assets/waf-system/rollout.png)

The durations are proposals. Every phase rolls back by config alone: the CRS mode, the route mode, the model alias or a kill switch. A failed gate extends its phase; it never skips one.

| Gate | Criteria | Evidence | Owner |
| --- | --- | --- | --- |
| 0 | Every offline release gate passes on the model package, including full parity and throughput on the target GPU (vLLM). Real Coraza + CRS 4.25 baseline measured on the same sets. Sensor CPU and latency met in a load test. GPU quota, or the SageMaker bridge, in place | Release-gate and load-test reports | ML team, security engineering |
| 1 | 14 days of shadow data. The 5% buffered canary runs without errors. `prod_holdout` built. Reviewed shadow false blocks counted per class. CRS exclusions written and go-ftw tested | Shadow dashboards, review counts | Security engineering |
| 2 | CRS false blocks on target for 14 days at blocking paranoia level 1, then 2. Per-class model precision measured in log-only mode | ClickHouse reports | Security engineering |
| 3, per class | At least 600 reviewed client decisions for the class with no error: first at the challenge stage, then at the ban stage. No robustness regression | Review records | ML team |
| 4 | Flows: CIC and unseen-source gates passed, corroboration rules tested. The veto and inline class actions are parked until an inline-capable model exists | Review records, release gates | Security engineering, SOC |

## Risks and open questions

The largest risk is that Gemma never earns enforcement: too many wrong client decisions on real traffic, too few reviewed decisions per class, or a GPU bill the queue rate cannot justify. Then it stays a reviewer's tool out of band, and CRS with virtual patches carries the request path, as it does from day one. Every risk below has a fallback that keeps inspection on.

| Risk | Early signal | Fallback |
| --- | --- | --- |
| Gemma too slow inline | Measured 20 Oct: L4 p50 659 ms at batch 1 on Hugging Face; 4.38 records/s on vLLM | Decided: second tier, asynchronous only. A distilled encoder is the route back to inline use |
| Async lag or cost above plan | Stream lag, GPU count against the measured 4.38 records/s | Per-client caps and stop-queueing-after-decision, a lower sample rate, spot capacity, single-token retrain, quantisation, the distilled encoder |
| The model only sees what the triggers select | Replay: 79% of the 624 attacks CRS misses have no trigger (plain traversal and SQLi) | A cheap screening tier in front of Gemma: a distilled encoder scoring every request and queueing what it flags (Model serving, step 4). Broadening the trigger instead would queue 29% of legitimate traffic |
| CRS untuned blocks or times out legitimate traffic | Replay: 17.0% of legitimate requests blocked at PL1, 300 ms of CRS time for a 6 KB cookie header | Cookie and content-type exclusions, header size cap, tuning before Phase 2; the Phase 2 gate cannot pass without them |
| The model cannot stop a one-shot attack | Bypass reports, GoTestWAF, red-team results | CRS tuning, virtual patches (4 hours), challenge on repeated probing |
| Single-token labels lose accuracy | Release gates after the Phase 0 retrain | Keep the two-pass method on vLLM; accept a larger GPU pool |
| Decoder parity not reached | Parity test on captured ext_proc requests | Train only on views the serving path produces, then rebuild and retrain |
| Real-traffic false blocks above 0.1% | Replay corpora; reviewed shadow | The class stays log-only; per-class thresholds; hard-negative rounds from reviewed false positives |
| Rare classes never reach 600 reviewed actions | Review counts per class | They stay log-only; CRS and virtual patches cover them |
| A retrained model regresses (two epochs overfit once) | Release gates; 7-day challenger shadow | Keep the champion; rollback is an alias change |
| Sensor outage takes fail-closed routes down | Replica health, error budget | 3+ replicas per zone, canary deploys; kill switch 3 for a route that must stay up |
| Coraza + CRS costs more CPU or latency than planned | Phase 0 load test | Per-route body limits, more replicas, early blocking after tuning. Fallback data planes: HAProxy + coraza-spoa, or nginx + ModSecurity v3 |
| CRS bypasses | CRS security advisories (4.30 fixed three) | LTS security releases within a week; go-ftw in CI; virtual patches |
| An attacker floods the model path | Deadline misses, queue depth | Priorities, per-client caps, shedding to rules-only |
| No GPU capacity | 0 EC2 GPU quota today; `ml.g6.xlarge` had no capacity on 17 October | Request quota now; SageMaker bridge with the g6-then-g5 fallback in `deploy/deploy.sh` |
| Lost training runs (a free Colab TPU run vanished on 19 October) | — | Paid TPU or GPU capacity, with checkpoint resume (`training/gemma_tpu/run_tpu.sh`) |
| Thin CVE-specific data (5.8 training records per KEV CVE for command injection) and uncalibrated `http_flood_app_dos` | GoTestWAF; `reports/kev_coverage.json`; Phase 0 gates | CRS, Envoy limits and virtual patches; the taxonomy decisions below |
| Retention not approved | Legal review | Shorter retention; no raw IPs outside active bans |

**Open questions for the team**

- [ ] Kubernetes with Envoy Gateway (this plan's assumption), or VMs with standalone Envoy?
- [ ] Peak requests per second and body-size mix per site. These size the sensor CPU and the GPUs.
- [ ] Per route: the body limit, the fail-open list, model-required or not, and the timeout action.
- [ ] Names for the owner roles in Components and owners, and an on-call rota.
- [ ] CDN in front or not. This decides where TLS terminates, how the client IP is derived, and HTTP/3.
- [ ] Request EC2 GPU quota, or stay on SageMaker?
- [ ] Raw path: accept the path as Envoy normalises it (the raw URI is not available), or normalise the training targets before the retrain?
- [ ] Build the distilled-encoder tier (Model serving, step 4) now, or only after Phase 0 measures the single-token retrain?
- [ ] CrowdSec adapter for client decisions, and the source of residential and mobile ASN data for the ban exemption.
- [ ] Legal sign-off on the retention table.
- [ ] Taxonomy and data items in the repo (`docs/PLAN.md`):
  - `http_flood_app_dos` is built and gated in the 42-class v12 bundle (`53,794` records, 747 flood records); needs Phase 0 training and calibration (`csv_formula_injection` was decided on 20 Oct: an unescaped leading trigger character is sufficient)
  - CRS 921/931/933/934 fixtures: focus template `data/focus/crs_921_931_933_934.yaml` (rounds `vsn61`–`vsn72`) is ready for user focus text and generation
  - more CSE-CIC-IDS2018 capture days (note: CIC-2018 rows lack IP columns, so they cannot supply per-source window counts for `http_flood_app_dos`)

## Appendix A: class actions

All 38 attack classes with their action once they pass their gates. For the HTTP classes the action is a per-client decision after corroboration (`config/decisions.yaml`), never a verdict on one request. Context keys come from `config/policy_gemma.yaml`; benign classes always allow.

| Class | Domain | Required context | Action | Note |
| --- | --- | --- | --- | --- |
| `sqli_classic` | L7 injection | — | Challenge, then ban from Phase 3 | — |
| `sqli_obfuscated` | L7 injection | — | Challenge, then ban from Phase 3 | — |
| `nosqli` | L7 injection | — | Challenge, then ban from Phase 3 | — |
| `xss_reflected_stored` | L7 injection | — | Challenge, then ban from Phase 3 | — |
| `xss_dom_polyglot` | L7 injection | `sink_context` | Log only | Until the context parity check and a gated retrain |
| `rce_cmd_injection` | L7 injection | — | Challenge, then ban from Phase 3 | — |
| `ssti` | L7 injection | — | Challenge, then ban from Phase 3 | — |
| `xxe` | L7 injection | — | Challenge, then ban from Phase 3 | — |
| `deserialization` | L7 injection | — | Challenge, then ban from Phase 3 | — |
| `crlf_header_injection` | L7 injection | — | Challenge, then ban from Phase 3 | — |
| `ldap_xpath_injection` | L7 injection | — | Challenge, then ban from Phase 3 | — |
| `csv_formula_injection` | L7 injection | `sink_context` | Log only | Leading trigger character suffices (decided 20 Oct); log-only until context parity check and gated retrain |
| `malicious_file_upload_webshell` | L7 injection | — | Challenge, then ban from Phase 3 | — |
| `path_traversal_lfi` | API/protocol | — | Challenge, then ban from Phase 3 | — |
| `ssrf_cloud_metadata` | API/protocol | — | Challenge, then ban from Phase 3 | — |
| `http_request_smuggling` | API/protocol | `parser_stack` | Log only | Envoy rejects ambiguous framing |
| `graphql_abuse` | API/protocol | — | Challenge, then ban from Phase 3 | — |
| `jwt_auth_bypass` | API/protocol | `verifier_config` | Log only | Envoy `jwt_authn` verifies tokens |
| `bola_idor_enumeration` | API/protocol | `authz_context` | Log only | Out of scope: the application decides ownership |
| `open_redirect_cors` | API/protocol | `browser_or_cors_config` | Log only | Until the context parity check and a gated retrain |
| `websocket_grpc_abuse` | API/protocol | — | Log only | ext_proc sees the upgrade request and unary bodies, not frames or streams |
| `bopla_mass_assignment` | API/protocol | `object_schema` | Log only | Schema validation enforces |
| `bfla_admin_function_probing` | API/protocol | `caller_role`, `route_required_role` | Log only | Gateway RBAC on verified roles enforces |
| `race_condition_toctou_burst` | API/protocol | `route_semantics` | Log only | Until the context parity check and a gated retrain |
| `llm_prompt_injection` | AI/modern | `trust_boundary` | Log only | Until the context parity check and a gated retrain |
| `prototype_pollution` | AI/modern | — | Challenge, then ban from Phase 3 | — |
| `web_cache_poisoning` | AI/modern | `cache_config` | Log only | Until the context parity check and a gated retrain |
| `mcp_agentic_tool_abuse` | AI/modern | `tool_manifest` | Log only | Until the context parity check and a gated retrain |
| `syn_flood_ddos` | Network flow | — | Alert only | Source addresses can be spoofed; mitigation is upstream |
| `udp_amplification` | Network flow | — | Alert only | Sources are reflectors, not attackers; mitigation is upstream |
| `slowloris_slowhttp` | Network flow | — | Envoy timeouts inline; per-IP rate limit after corroboration | — |
| `credential_stuffing_bruteforce` | Network flow | — | Per-IP rate limit, then challenge | Corroboration: 3 verdicts in 10 minutes |
| `port_scan_recon` | Network flow | — | External source banned 4 h via CrowdSec; challenge for residential and mobile ASNs | Corroboration: 3 verdicts in 10 minutes; needs the CIC fix |
| `c2_beaconing_anomaly` | Network flow | — | SOC alert (internal hosts) | Beacon-interval feature is synthetic-only |
| `dns_tunneling_exfil` | Network flow | — | SOC alert (internal hosts) | Query-name length and entropy are synthetic-only |
| `tls_ja4_ua_spoofing` | Network flow | — | Log only | JA4 is not in the telemetry contract; synthetic-only feature |
| `http2_rapid_reset_continuation` | Network flow | — | Log only; Envoy's HTTP/2 limits act inline | HTTP/2 features are synthetic-only |
| `http_flood_app_dos` | Network flow / Hybrid | — | Per-IP rate limit, then challenge; CrowdSec ban after corroboration | Built and gated in the 42-class v12 bundle (CIC-2017 DoS Hulk, GoldenEye and web-port DDoS ≥ 200 conn/min); needs Phase 0 retrain |

## Appendix B: versions and alternatives

The latest stable releases on 19 October 2026, read from each project's GitHub releases, and the options we rejected or deferred.

| Layer | Version | Not chosen or deferred, and why |
| --- | --- | --- |
| Envoy, Envoy Gateway | 1.40.0; 1.9.2 | ingress-nginx: retired in March 2026, no more security fixes. [HAProxy](https://github.com/haproxy/haproxy) 3.4 + [coraza-spoa](https://github.com/corazawaf/coraza-spoa) 0.9: kept as the fallback data plane |
| Coraza | 3.8.1 | [coraza-proxy-wasm](https://github.com/corazawaf/coraza-proxy-wasm): last release July 2025; open issues report 5–10× slowdowns. [ModSecurity v3](https://github.com/owasp-modsecurity/ModSecurity) 3.0.17: mature but tied to nginx or Apache |
| OWASP CRS | 4.25.2 LTS | 4.30.0, the newest, is not LTS: staging only |
| vLLM (production); Transformers (`Gemma4ForCausalLM`, reference) | vLLM 0.30.0 (validated); 0.32.0 is the newest, retest in Phase 0. Transformers 5.17 (repo pin) | TGI: archived in March 2026. SGLang 0.5.22: Gemma 4 LoRA gaps. vLLM's generation path shifted Qwen v10 probabilities by up to 0.249; the restricted single-token method passed Gemma's self-check |
| NVIDIA GPU Operator, KEDA, Karpenter | 26.7.2; 2.21.0; 1.15.0 | KServe 0.21: only if the platform team already runs it |
| CrowdSec | 1.8.1 | Its AppSec WAF also runs CRS on Coraza, but we embed Coraza ourselves. Free blocklist tiers: 3,000 and about 15,000 IPs |
| envoyproxy/ratelimit, Valkey (counters + Streams) | Pinned by commit digest; Valkey 9.0 | Redis: relicensed away from BSD in 2024; Valkey is the BSD-licensed fork. Kafka 4.3.1: deferred until sustained traffic exceeds ~10k req/s; Redpanda: BSL license |
| Anubis | 1.28.1 | ALTCHA: a library, not a service. mCaptcha: work in progress (AGPL) |
| Zeek (and GoFlow2 on untapped links) | 9.0 LTS; 2.2.7 | Suricata 8.0.7: optional (joined only if the SOC already runs it). CICFlowMeter: unmaintained since 2022. NFStream: lacks the NetFlow v2 feature set. nProbe: commercial |
| OpenTelemetry Collector, Prometheus | 0.163.1; 3.13 LTS | Loki: not needed beside ClickHouse |
| ClickHouse, Grafana | 26.8 LTS; 13.2 | Evidently 0.7.23: replaced by scheduled ClickHouse SQL drift views + Grafana alerts. [OpenSearch](https://github.com/opensearch-project/OpenSearch) 3.9 or [Wazuh](https://github.com/wazuh/wazuh) 4.14 as an optional SIEM |
| Label Studio; S3 + `cosign` + Git `champion.yaml` | 1.23.2 | MLflow 3.17.1: replaced by signed S3 bundles + Git-tracked `champion.yaml` (two-person PR). Argilla: maintenance only since 2025 |
| cosign, Syft, Grype, Kyverno, OpenBao, cert-manager | 3.1.3; 1.55.0; 0.121.0; 1.19.1; 2.7.1; 1.21 | Trivy: compromised in March 2026. SPIRE 1.15: if workload identity beyond mTLS is needed |
| GoTestWAF, go-ftw | 0.5.10; 2.6.1 | — |
| Comparison baselines | open-appsec 1.1.36; BunkerWeb 1.6.16; SafeLine 9.4.2 | Benchmarked against, not used as the base. open-appsec's production model has its own license and cannot be retrained. BunkerWeb is bound to nginx (AGPL). SafeLine's detector ships as an image without source |

## Sources

The external facts come from the projects' GitHub sources and the AWS Price List API, read on 19 October 2026. Project sites outside GitHub were unreachable from the research environment: coreruleset.org, envoyproxy.io, huggingface.co, docs.vllm.ai, zeek.org and others. Project versions are linked in the Building blocks table.

**Repository** (`mora-waf/qwen35-waf-ml`, branch `master`)

- `waf-system/` (sibling directory, built 20 October): the sensor, the async worker, tests and the benchmark harness. `waf-system/docs/FINDINGS.md`, `waf-system/reports/bench_L4.json`, `waf-system/reports/gemma_vllm_L4.json` and `waf-system/reports/crs_regress.json` hold the measurements this revision rests on.
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
