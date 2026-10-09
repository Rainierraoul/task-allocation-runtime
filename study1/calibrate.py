"""Held-out-only confidence and router calibration; freeze before testing."""
from __future__ import annotations

import json
import math
from pathlib import Path

from dcarr_runtime.executors import ModelExecutor
from dcarr_runtime.handoff import validate
from .harness import build, task_models
from .mock import MockBackend
from .oracle import eligible_tasks, gold, score
from .router import MockRouter


def clean_contexts(instance):
    """Teacher-forced reference trajectory using tools and scripted candidates.

    Only declared observations/orientation are returned. No answer key or
    White Force answer is sent to a diagnostic model or learned router.
    """
    mock = MockBackend()
    mock.start("reference-context")
    engine, _ = build(instance, "routed", mock, router=MockRouter(), router_threshold=0.0)
    engine.run()
    contexts = {c["task_id"]: c for c in mock.contexts if "lower_tier_result" not in c}
    expected = gold(instance)
    eligible = eligible_tasks(instance, expected)
    return {task: ctx for task, ctx in contexts.items() if task in ("ACK", "CALL") and task in eligible}


def diagnostic_items(instances, repetitions=5, phase="diagnostic"):
    """Every (run ID, instance, task, context, repetition, tier) to diagnose, in a fixed order."""
    for instance in instances:
        for task, context in clean_contexts(instance).items():
            for repetition in range(repetitions):
                for tier in ("T1", "T2"):
                    yield f"{phase}/{instance['id']}/{task}/{repetition}/{tier}", instance, task, context, repetition, tier


def diagnostic_row(item, paid, models=None):
    """One diagnostic call and its scored record. Independent of every other item."""
    run_id, instance, task, context, repetition, tier = item
    models = models or task_models()
    paid.start(run_id)
    response, tin, tout = paid.call(tier, context, "diagnostic")
    payload = response.get("payload") if isinstance(response, dict) else None
    conf = response.get("confidence") if isinstance(response, dict) else None
    valid_conf = type(conf) in (float, int) and math.isfinite(conf) and 0 <= conf <= 1
    valid_flag = isinstance(response, dict) and type(response.get("decided")) is bool
    handoff_ok = not validate(payload, models.get(task).hand_off)
    return {"run_id": run_id, "instance_id": instance["id"], "task": task, "tier": tier,
            "repetition": repetition, "empirical": all(r["empirical"] for r in paid.last_rows),
            "candidate_correct": handoff_ok and score(task, payload, gold(instance)[task], instance),
            "model_abstained": not valid_flag or response.get("decided") is False,
            "valid_confidence": valid_conf, "confidence": conf if valid_conf else None,
            "handoff_valid": handoff_ok, "response": response,
            "input_tokens": tin, "output_tokens": tout}


def diagnostic_rows(instances, paid, repetitions=5, phase="diagnostic", pool_map=map):
    """Diagnostic records in item order. pool_map may run the independent calls in parallel."""
    models = task_models()
    return pool_map(lambda item: diagnostic_row(item, paid, models), list(diagnostic_items(instances, repetitions, phase)))


def choose_threshold(rows, target_error=0.05, minimum_accepted=10):
    available = [r for r in rows if not r["model_abstained"] and r["valid_confidence"] and r["handoff_valid"]]
    cutoffs = sorted({0.0, 1.0} | {r["confidence"] for r in available})
    for cutoff in cutoffs:
        accepted = [r for r in available if r["confidence"] >= cutoff]
        errors = sum(not r["candidate_correct"] for r in accepted)
        if len(accepted) >= minimum_accepted and errors / len(accepted) <= target_error:
            return {"threshold": cutoff, "accepted": len(accepted), "errors": errors,
                    "observed_error": errors / len(accepted), "target_met": True}
    # A value above the possible confidence range means explicitly requiring
    # review at this tier. It is a disabled gate, not a probability threshold.
    return {"threshold": 1.000001, "accepted": 0, "errors": 0, "observed_error": None, "target_met": False}


def calibrate(instances, paid, router, materials_sha, config_sha, pool_map=map):
    rows = list(diagnostic_rows(instances, paid, phase="heldout", pool_map=pool_map))
    details, thresholds = {}, {}
    for task in ("ACK", "CALL"):
        details[task], thresholds[task] = {}, {}
        for tier in ("T1", "T2"):
            result = choose_threshold([r for r in rows if r["task"] == task and r["tier"] == tier])
            details[task][tier], thresholds[task][tier] = result, result["threshold"]
    router_scores = [router.score(ctx) for instance in instances for ctx in clean_contexts(instance).values()]
    ordered = sorted(router_scores)
    middle = len(ordered) // 2
    router_threshold = (ordered[middle - 1] + ordered[middle]) / 2 if len(ordered) % 2 == 0 else ordered[middle]
    return {"materials_sha256": materials_sha, "config_sha256": config_sha,
            "empirical": all(r["empirical"] for r in rows), "thresholds": thresholds,
            "threshold_details": details, "target_error": 0.05, "minimum_accepted": 10,
            "router_threshold": router_threshold, "router_target_strong_fraction": 0.5,
            "router_observed_strong_fraction": sum(s >= router_threshold for s in ordered) / len(ordered),
            "diagnostics": rows}
