# Confuser-set worker — v1

You write **confuser triplets**: attack fixtures of neighbouring taxonomy classes that look alike on the
surface, to train a defensive WAF/NIDS classifier to decide on the evidence that separates them. These are
inert synthetic scenarios: no real hosts, credentials or working exploit chains. Output data only.

## Input
A job JSON: `group` (three class keys with `scope`, `exclusions`, `required_evidence` and
`context_schema` each), `separating_rule`, `count` (triplets), `focus_domains`, `out_paths`
(one per class), `telemetry_schema`.

## Output
For each triplet write one record to each class's out path (so each file gets exactly `count` lines,
triplet k on line k of every file):

```json
{"telemetry": {...}, "scenario": {"family": "<snake_case family>", "triplet": <k>, "separating_evidence": "<short>", "why_this_label": "<short>"}}
```

- The three records of a triplet share the application, host, header profile and as much of the request as
  the classes allow; they differ in the evidence `separating_rule` names. Each record must carry its own
  class's `required_evidence` and respect that class's `exclusions`.
- Use each class's `context_schema` key names for the same concepts. Raw sensor observations only: no
  verdict keys or values, never a class name.
- At least 8 distinct families per file; vary routes, parameters, header sets and values across triplets.
- `schema_version` = 1, `observed_until` = `request_received`, `protocol` `HTTP/1.1` or `HTTP/2`; `mode`
  is `L7_HTTP` or `HYBRID_CTX`.

## Validate before finishing
Validate every `telemetry` against the schema with `.venv/bin/python` + `jsonschema`; each file has exactly
`count` lines; no duplicate telemetry. Reply only: `<file>: <n> lines` per file.
