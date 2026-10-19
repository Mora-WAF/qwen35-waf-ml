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


def outside_envelope(features, env):
    """True when any enveloped flow feature lies outside the frozen [lo, hi] range. Bounds are cast to float:
    PyYAML reads an exponent without a decimal point (the frozen `2e-06`) as a string."""
    return any(k in features and not float(lo) <= features[k] <= float(hi) for k, (lo, hi) in env.items())


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
    if cls == "benign_standard_traffic" and flow_features and outside_envelope(flow_features, pol.get("flow_envelope") or {}):
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


def over_budget_decision(pol):
    """A request the sensor cannot fit into the token budget is never guessed: log and monitor (18 Oct)."""
    return {"verdict": "ABSTAIN", "domain": "UNKNOWN", "threat_type": "unknown", "confidence": None,
            "action": "log_and_monitor", "rule_hint": pol["reason_codes"].get("over_budget", "over_token_budget")}


def validate_decision(d, tax, schema):
    """Schema + taxonomy domain consistency. Raises on failure."""
    jsonschema.validate(d, schema)
    if d["threat_type"] != "unknown" and tax["classes"][d["threat_type"]]["domain"] != d["domain"]:
        raise ValueError(f"domain mismatch: {d}")
