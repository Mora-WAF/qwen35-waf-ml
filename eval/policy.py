"""Deterministic offline decision wrapper (spec §7).

Model output -> application-facing decision object. Recommendations only; this module
executes nothing. Abstains on malformed output, unsupported mode, missing evidence,
missing/insufficient calibrated confidence, or ineligible alternative actions.
"""
import json
import math
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


def decide(raw, mode, context, confidence, tax, pol):
    """Returns (decision, diagnostics). `confidence` is the calibrated probability of the
    emitted class, or None when calibration is unavailable."""
    classes, rc = tax["classes"], pol["reason_codes"]
    cls = parse_output(raw, classes)
    diag = {"tentative_class": cls, "reason": None}

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


def validate_decision(d, tax, schema):
    """Schema + taxonomy domain consistency. Raises on failure."""
    jsonschema.validate(d, schema)
    if d["threat_type"] != "unknown" and tax["classes"][d["threat_type"]]["domain"] != d["domain"]:
        raise ValueError(f"domain mismatch: {d}")
