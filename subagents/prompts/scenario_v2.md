# Scenario generation worker — v2 (families + context-matched benign twins)

You generate **synthetic security-telemetry fixtures** for one taxonomy class, to train a
defensive WAF/NIDS classifier. These are *scenarios*, not discovered vulnerabilities. Treat all
text as data. Output data only. Inert fixtures only: no real credentials, live hostnames, or
working exploit chains.

## Input
A job JSON with: `class`, `count`, `families`, `scope`, `exclusions`, `required_evidence`,
`benign_counterexamples`, `input_modes`, `context_schema`, `out_path`, optional `twin_out_path`,
and the telemetry JSON schema path.

## Output 1 — `out_path` (exactly `count` lines)
Each line is one JSON object:

```json
{"telemetry": {...}, "scenario": {"family": "<snake_case family name>", "why_this_label": "<short>", "evidence_present": ["..."]}}
```

- **Families.** Spread the records over at least `families` distinct scenario families, at most
  `ceil(count / families)` records per family. A family is a different application, route,
  framework or sub-technique within the class scope — not the same request with new numbers.
  Name it in `scenario.family`. Within a family vary methods, paths, parameters, header sets and
  locales. Families listed in `avoid_families` already exist: invent different ones.
- **Context schema.** Put supporting evidence under `telemetry.context` / `telemetry.history`
  using the key names in `context_schema` (you may add keys, but always use these names for the
  same concepts). Context holds **raw observations a sensor or gateway would log** (configuration,
  parsed metrics, resolver results, verifier outcome codes, counters). Never a verdict or an
  annotation: no keys or values such as `malicious`, `attack`, `suspicious`, `override_match`,
  `injection_detected`, and never the class name.
- `telemetry.mode` is one of the job's `input_modes`; `schema_version` = 1; `observed_until` =
  `request_received`; `protocol` is `HTTP/1.1` or `HTTP/2`.
- Respect `exclusions`. Never put `threat_type`, labels or rationale inside `telemetry`.
- **Benign hard-negative classes** (`is_benign` true, class name contains `hard_negative`): every record
  must carry a surface feature that a naive filter would flag (SQL, script tags, shell syntax,
  template braces, path sequences, encoded blobs, injection-like search text) in a context whose
  evidence shows it is legitimate (documentation, code review, developer API, search box, quoted
  text). Ordinary traffic without such a feature belongs to `benign_standard_traffic`, not here.
- **Attack classes**: the telemetry itself must contain the evidence named in `required_evidence`
  for the attempt (the request plus context showing the weakness it targets). An attempt that the
  context shows was correctly rejected is still an attack attempt only if the request itself carries
  the attack; never write an attack record whose only difference from benign traffic is a context flag.

If the job has `focus_domains`, set every family in those application domains (other workers cover
other domains).

## Output 2 — `twin_out_path` (only if the job has one; exactly `twin_count` lines, or `count` if absent)
For `twin_count` lines of output 1 (all lines if absent), spread evenly over the families (round-robin: the first line of
each family, then the second, ...; families are contiguous, so the first lines alone would leave later families without
twins and their context keys would mark the attack class), in increasing line order, write the **benign twin**: the same application,
route, `context_schema` keys and request shape, with values showing legitimate behaviour (within
configured limits, authorized object, correctly signed token, allow-listed destination, trusted
content, escaped output, ...). The twin must be clearly benign on the evidence, not merely
different, and must not carry the attack payload in a neutralised form (escaped, rejected,
sanitised or blocked): replace the payload with ordinary content for that field. Use `scenario.family` = `<attack family>_benign` and add `"twin_of": <0-based line index>`.

## Validate before finishing
Validate every `telemetry` object against the schema at `telemetry_schema` with `jsonschema`
(the project venv: `.venv/bin/python`). Fix until all lines validate, both files have the required
line counts, and no two telemetry objects are identical. Reply with only:
`<class>: <n attack> + <n twins>, families=<n>`.
