"""Deterministic offline decision wrapper (spec §7).

Model output -> application-facing decision object. Recommendations only; this module
executes nothing. Abstains on malformed output, unsupported mode, missing evidence,
missing/insufficient calibrated confidence, or ineligible alternative actions.
"""
import json
import math
import re
from pathlib import Path

import jsonschema
import yaml

ROOT = Path(__file__).resolve().parent.parent
VERDICT_ACTION = {"ALLOW": "pass", "BLOCK": "drop_request", "RATE_LIMIT": "rate_limit_ip",
                  "CHALLENGE": "challenge_captcha", "ABSTAIN": "log_and_monitor"}


def load(root=ROOT):
    tax = yaml.safe_load(open(root / "config/taxonomy.yaml"))
    pol = yaml.safe_load(open(root / "config/policy.yaml"))
    schema = json.load(open(root / "schemas/decision.schema.json"))
    return tax, pol, schema


def parse_output(raw, classes):
    """Strict parse of the model's {"threat_type": ...}. No repair; returns key or None."""
    try:
        obj = json.loads(raw)
    except (TypeError, ValueError):
        return None
    if not isinstance(obj, dict) or set(obj) != {"threat_type"} or obj["threat_type"] not in classes:
        return None
    return obj["threat_type"]


# Time resolution of the flow exporter: a recorded duration of 0 means "shorter than this", not zero. CICFlowMeter
# records microseconds, NetFlow v5/v9 / IPFIX milliseconds (NF-UNSW-NB15: 1,216 of 4,000 benign flows have duration 0
# and, against the CIC-fitted lower bound 2e-06, all of them abstained). The coarsest supported exporter is the default;
# policy key `flow_envelope_resolution` overrides it ({} = compare recorded values exactly, the behaviour before 19 Oct).
ENVELOPE_RESOLUTION = {"duration_s": 1e-3}
IMPLIED_RATES = {"pkts_per_s": "pkts_total", "bytes_per_s": "bytes_total"}  # flows.py omits rates at duration 0


def outside_envelope(features, env, resolution=None):
    """True when the flow cannot lie inside the frozen [lo, hi] box. Bounds are cast to float: PyYAML reads an exponent
    without a decimal point (the frozen `2e-06`) as a string. A duration recorded as 0 at resolution r is a true duration
    in [0, r): it is outside only when that whole interval is (r <= lo), and the rates it leaves out are then above
    total / r, outside when that bound already exceeds the rate's upper bound (a large burst in under r)."""
    bounds = {k: (float(lo), float(hi)) for k, (lo, hi) in env.items()}
    r = float((ENVELOPE_RESOLUTION if resolution is None else resolution).get("duration_s") or 0)
    if r and features.get("duration_s") == 0:
        if "duration_s" in bounds and r <= bounds["duration_s"][0]:
            return True
        if any(rate in bounds and rate not in features and total in features and features[total] / r >= bounds[rate][1]
               for rate, total in IMPLIED_RATES.items()):
            return True
        features = {k: v for k, v in features.items() if k != "duration_s"}
    return any(k in features and not lo <= features[k] <= hi for k, (lo, hi) in bounds.items())


def decision_view(P, benign, act_on="top_class"):
    """(confidence, class index) per row of calibrated probabilities P [n, classes]; `benign` [classes] bool.
    top_class: the argmax class and its probability (the deployed rule). attack_mass: attack vs benign first
    (summed probabilities), then the top class of the winning side, confidence = that side's mass: a request split
    between two attack classes (sqli_classic 0.5 / sqli_obfuscated 0.4) is still a confident attack."""
    import numpy as np
    P, benign = np.asarray(P), np.asarray(benign, bool)
    if act_on == "top_class":
        return P.max(1), P.argmax(1)
    assert act_on == "attack_mass", act_on
    A = P[:, ~benign].sum(1)
    att = np.flatnonzero(~benign)[P[:, ~benign].argmax(1)]
    ben = np.flatnonzero(benign)[P[:, benign].argmax(1)]
    return np.where(A >= 0.5, A, 1 - A), np.where(A >= 0.5, att, ben)


def ensemble_log_mean(member_probs):
    """log of the mean of the members' calibrated class distributions [members, ..., classes] (each member tempered
    with its own frozen temperature): the ensemble's analogue of one model's summed class log-probs (eval.ensemble)."""
    import numpy as np
    return np.log(np.maximum(np.asarray(member_probs, dtype=np.float64).mean(0), 1e-300))


def ensemble_probs(member_probs, pol):
    """Calibrated class distribution under an ensemble policy (`ensemble:` block, written by eval.ensemble): the mean
    of the members' calibrated distributions, in the block's member order, re-tempered with the policy's own
    `temperature` (fitted on val_calib like a single model's)."""
    import numpy as np
    assert len(member_probs) == len(pol["ensemble"]["members"]), "one distribution per ensemble member"
    z = ensemble_log_mean(member_probs) / float(pol["temperature"])
    e = np.exp(z - z.max(-1, keepdims=True))
    return e / e.sum(-1, keepdims=True)


def scanner_tool(headers, pol):
    """Tier-0: the scanner token in the User-Agent (config/policy.yaml scanner_user_agents), or None."""
    ua = next((str(v).lower() for k, v in headers or [] if str(k).lower() == "user-agent"), "")
    return next((t for t in pol.get("scanner_user_agents") or [] if re.search(rf"\b{re.escape(t)}\b", ua)), None)


def decide(raw, mode, context, confidence, tax, pol, flow_features=None, headers=None):
    """Returns (decision, diagnostics). `confidence` is the calibrated probability of the
    emitted class, or None when calibration is unavailable. `flow_features`: telemetry flow.features, if any.
    `headers`: telemetry http.headers, for the tier-0 scanner rule (a benign verdict becomes log_and_monitor)."""
    classes, rc = tax["classes"], pol["reason_codes"]
    cls = parse_output(raw, classes)
    diag = {"tentative_class": cls, "reason": None}
    tool = scanner_tool(headers, pol)
    if tool:
        diag["scanner_user_agent"] = tool

    def abstain(reason):
        diag["reason"] = reason
        return {"verdict": "ABSTAIN", "domain": "UNKNOWN", "threat_type": "unknown",
                "confidence": None, "action": "log_and_monitor", "rule_hint": rc[reason]}, diag

    context = context or {}
    if cls is None:
        return abstain("malformed_output")
    spec = classes[cls]
    if mode not in spec["input_modes"]:
        return abstain("unsupported_mode")
    if any(context.get(k) is None for k in pol["required_context"].get(cls, [])):
        return abstain("missing_evidence")
    threshold = pol["class_thresholds"].get(cls, pol["threshold"])
    if confidence is None or threshold is None or not math.isfinite(confidence):
        return abstain("uncalibrated")
    if confidence < threshold:
        return abstain("low_confidence")
    # the background class absorbs out-of-taxonomy floods/scans with p ~ 1: outside the benign envelope, abstain
    if cls == "benign_standard_traffic" and flow_features and outside_envelope(
            flow_features, pol.get("flow_envelope") or {}, pol.get("flow_envelope_resolution")):
        return abstain("out_of_envelope")

    if spec["is_benign"] and tool:  # tier-0: a scanner is not passed as ordinary traffic, nor blocked on the name
        diag["reason"] = "scanner_user_agent"  # (the schema allows RATE_LIMIT only with an attack class: monitor)
        return {"verdict": "ABSTAIN", "domain": "UNKNOWN", "threat_type": "unknown", "confidence": None,
                "action": "log_and_monitor", "rule_hint": f"tier0_scanner_{re.sub(r'[^a-z0-9_]', '_', tool)}"[:48]}, diag
    if spec["is_benign"]:
        verdict, hint = "ALLOW", rc["allow"]
    else:
        verdict, hint = "BLOCK", f"ml_{cls}"  # fixed mapping, never model text
        alt = pol["class_actions"].get(cls)
        if alt:
            if context.get(alt["eligibility"]) is not True:
                return abstain("ineligible_action")
            verdict = alt["verdict"]
    diag["reason"] = "decided"
    return {"verdict": verdict, "domain": spec["domain"], "threat_type": cls,
            "confidence": float(confidence), "action": VERDICT_ACTION[verdict],
            "rule_hint": hint}, diag


def decide_from_probs(p, tel, tax, pol):
    """The serving decision for one request from its calibrated class distribution `p` (taxonomy order): the frozen
    decision view, then decide() with the request's mode, context, flow features and headers. Used by the endpoint
    (deploy/sagemaker/inference.predict_fn) and by the offline re-decisions (eval.public_bench), so both stay one path.
    Returns (decision, diagnostics, top 3 [[class, p], ...])."""
    import numpy as np
    p = np.asarray(p, dtype=np.float64)
    # the taxonomy only appends classes (19 Oct: http_flood_app_dos): a v10 / v11 model's distribution covers a prefix
    classes = list(tax["classes"])[:len(p)]
    assert len(classes) == len(p), f"distribution has {len(p)} classes, taxonomy {len(tax['classes'])}"
    benign = np.array([tax["classes"][c]["is_benign"] for c in classes])
    conf, i = decision_view(p[None], benign, pol.get("act_on", "top_class"))
    raw = json.dumps({"threat_type": classes[int(i[0])]}, separators=(",", ":"))
    d, diag = decide(raw, tel.get("mode"), tel.get("context"), float(conf[0]), tax, pol,
                     (tel.get("flow") or {}).get("features"), (tel.get("http") or {}).get("headers"))
    top = sorted(zip(classes, p.tolist()), key=lambda kv: -kv[1])[:3]
    return d, diag, [[c, round(v, 6)] for c, v in top]


def over_budget_decision(pol):
    """A request the sensor cannot fit into the token budget is never guessed: log and monitor (18 Oct)."""
    return {"verdict": "ABSTAIN", "domain": "UNKNOWN", "threat_type": "unknown", "confidence": None,
            "action": "log_and_monitor", "rule_hint": pol["reason_codes"].get("over_budget", "over_token_budget")}


def validate_decision(d, tax, schema):
    """Schema + taxonomy domain consistency. Raises on failure."""
    jsonschema.validate(d, schema)
    if d["threat_type"] != "unknown" and tax["classes"][d["threat_type"]]["domain"] != d["domain"]:
        raise ValueError(f"domain mismatch: {d}")
