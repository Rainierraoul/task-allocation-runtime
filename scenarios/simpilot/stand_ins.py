"""Scripted stand-ins for the model tiers and for the person, for Scenario 5.

These are deterministic functions, not language models. They exist so the
runtime's allocation, abstention, escalation and policy logic can be tested
without a model API. Each returns what a model is asked to return (see
MODEL_SYSTEM_PROMPT): decided, payload, confidence, reason and framings.
Swap in AnthropicBackend or PiBackend to run the same task models on real
models.
"""

from __future__ import annotations

import re
from typing import Any

from dcarr_runtime.executors import Proposal

_NUMBER_WORDS = {"one": "1", "two": "2", "three": "3", "four": "4"}


def _normalise(callsign: str) -> str:
    words = callsign.lower().strip().split()
    return " ".join(_NUMBER_WORDS.get(w, w) for w in words)


def _candidates(addressee: str) -> list[str]:
    """'Aircraft two or three' -> ['aircraft 2', 'aircraft 3']."""
    text = _normalise(addressee)
    m = re.match(r"(\D+?)\s+(\d+)((?:\s+or\s+\d+)*)$", text)
    if not m:
        return [text]
    stem = m.group(1)
    numbers = [m.group(2)] + re.findall(r"\d+", m.group(3))
    return [f"{stem} {n}" for n in numbers]


def _parse(transmission: str) -> dict[str, Any]:
    addressee, _, rest = transmission.partition(",")
    distance = re.search(r"(\d+)\s*(?:nautical miles|nm)", rest.lower())
    instruction = "join_tanker" if "tanker" in rest.lower() else "other"
    return {
        "addressee_raw": addressee,
        "candidates": _candidates(addressee),
        "instruction": instruction,
        "report_at_nm": float(distance.group(1)) if distance else None,
    }


# ---------------------------------------------------------------- T1


def parse_command_small(context: dict[str, Any]) -> dict[str, Any]:
    parsed = _parse(context["observations"]["transmission"])
    ambiguous = len(parsed["candidates"]) > 1
    return {
        "decided": True,
        "payload": {
            "addressee": parsed["candidates"][0],
            "instruction": parsed["instruction"],
            "report_at_nm": parsed["report_at_nm"],
            "acknowledged": True,
        },
        "confidence": 0.4 if ambiguous else 0.95,
        "reason": "addressee unclear" if ambiguous else "matches the join-tanker instruction",
    }


def compose_call_small(context: dict[str, Any]) -> dict[str, Any]:
    obs = context["observations"]
    if obs["channel_state"] != "clear":
        return {"decided": False, "reason": f"the channel is {obs['channel_state']}"}
    primary = context["orientation"]["comms_plan"]["primary"]
    return {
        "decided": True,
        "payload": {
            "call_status": "made",
            "message": "Wedgetail, aircraft 2, 20 miles from the tanker.",
            "frequency": primary,
        },
        "confidence": 0.95,
        "reason": "channel clear; standard range report",
    }


# ---------------------------------------------------------------- T2


def parse_command_reasoning(context: dict[str, Any]) -> dict[str, Any]:
    parsed = _parse(context["observations"]["transmission"])
    ours = set(context["orientation"]["own_callsigns"])
    matches = [c for c in parsed["candidates"] if c in ours]
    if len(matches) != 1:
        return {
            "decided": False,
            "reason": f"{len(matches)} of our aircraft could be the addressee",
            "framings": [{"option": f"treat as {c}", "case_for": "named in the call"} for c in matches],
        }
    return {
        "decided": True,
        "payload": {
            "addressee": matches[0],
            "instruction": parsed["instruction"],
            "report_at_nm": parsed["report_at_nm"],
            "acknowledged": True,
        },
        "confidence": 0.85,
        "reason": f"of {parsed['candidates']}, only {matches[0]} is under our control",
    }


def resolve_call_reasoning(context: dict[str, Any]) -> dict[str, Any]:
    obs = context["observations"]
    plan = context["orientation"]["comms_plan"]
    if obs["channel_state"] == "in_use":
        return {
            "decided": True,
            "payload": {"call_status": "deferred", "message": "Retry in 30 seconds", "frequency": plan["primary"]},
            "confidence": 0.8,
            "reason": "the channel is busy; radio procedure is to wait and retry",
        }
    if obs["channel_state"] == "jammed" and obs["jamming_inject_scheduled"]:
        return {
            "decided": False,
            "reason": "the jamming matches a scheduled exercise inject, so the response is a training-design choice",
            "framings": [
                {"option": "hold the call and let the inject play out", "case_for": "the trainee practises recovering communications"},
                {"option": f"switch to the backup channel ({plan['backup']})", "case_for": "keeps the tanker join on schedule"},
                {"option": "relay through aircraft 1", "case_for": "tests the trainee's handling of relayed calls"},
            ],
        }
    return {
        "decided": True,
        "payload": {
            "call_status": "made",
            "message": "Wedgetail, aircraft 2, on backup, 20 miles from the tanker.",
            "frequency": plan["backup"],
        },
        "confidence": 0.8,
        "reason": "unplanned jamming; the comms plan names a backup channel",
    }


HANDLERS = {
    "parse_command_small": parse_command_small,
    "compose_call_small": compose_call_small,
    "parse_command_reasoning": parse_command_reasoning,
    "resolve_call_reasoning": resolve_call_reasoning,
}

# ---------------------------------------------------------------- person


class ScriptedPerson:
    """A White Force member's answers, fixed in advance. Records what they were shown."""

    ANSWERS: dict[str, tuple[dict[str, Any], str]] = {
        "DEV": (
            {"choice": "comply", "rationale": "Initial employment trainees practise the basic join first."},
            "Comply for initial employment trainees when no deviation inject is scheduled.",
        ),
        "CALL": (
            {
                "call_status": "made",
                "message": "Wedgetail, aircraft 2, on backup, 20 miles from the tanker.",
                "frequency": "Channel 7",
            },
            "During a scheduled jamming inject, keep the tanker join on schedule via the backup channel.",
        ),
    }

    def __init__(self) -> None:
        self.requests: list[dict[str, Any]] = []
        self.notices: list[dict[str, Any]] = []
        self.approvals: list[dict[str, Any]] = []

    def decide(self, request: dict[str, Any]) -> Proposal:
        self.requests.append(request)
        task_id = request["task_id"]
        if task_id not in self.ANSWERS:
            raise AssertionError(f"unexpected decision request for {task_id}")
        payload, rationale = self.ANSWERS[task_id]
        return Proposal(
            decided=True,
            payload=dict(payload),
            reason="scripted White Force answer",
            make_policy=True,
            policy_rationale=rationale,
        )

    def approve(self, request: dict[str, Any]) -> bool:
        self.approvals.append(request)
        return True

    def veto(self, notice: dict[str, Any]) -> bool:
        self.notices.append(notice)
        return False
