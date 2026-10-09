"""Four conditions share task specifications, material, scoring, and actuation."""
from __future__ import annotations

import copy
import time
from pathlib import Path

from dcarr_runtime import Engine, EventLog, ModelExecutor, PersonExecutor, PolicyStore, Tier, ToolExecutor, load_task_models
from dcarr_runtime.allocation import allocate
from dcarr_runtime.engine import ActionNotPerformed, DesignFault
from dcarr_runtime.executors import DecisionContext, Proposal, model_context
from dcarr_runtime.handoff import validate
from dcarr_runtime.metamodel import Observation, Orientation
from scenarios.simpilot.tools import ACTUATORS, SENSORS, TOOLS

from .backend import ExecutorBackend
from .instances import make_world, orientation
from .oracle import CONTROL_CALLSIGN, call_reference, gold, score_run

# Allowed values of each enum hand-off field, shown to models (protocol v1.1).
ENUM_VALUES = [
    ("ACK", "instruction", ("join_tanker",)),
    ("DEV", "choice", ("comply", "misinterpret")),
    ("CALL", "call_status", ("made", "deferred")),
    ("FLY", "plan_choice", ("comply", "misinterpret", "not_applicable")),
]

MODEL_FILE = Path(__file__).parents[1] / "scenarios/simpilot/task_models.yaml"
CONDITIONS = ("single", "decomposed", "routed", "allocated")


class InvalidCandidate(RuntimeError):
    pass


def aircraft_state(w, _):
    return vars(w.aircraft["aircraft 2"]).copy()


def tanker_state(w, _):
    return {**vars(w.tanker), "x": w.tanker.x, "y": w.tanker.y}


def task_models():
    ms = load_task_models(MODEL_FILE)
    ms.version = "study1-1.0"
    ms.scenario = "Synthetic tanker join at a variable ordered distance"
    for task_id in ("HDG", "WAIT"):
        tm = ms.get(task_id)
        known = {o.name for o in tm.observations}
        for name in ("aircraft2_state", "tanker_state"):
            if name not in known:
                tm.observations.append(Observation(name=name, value_type="position", source="tool"))
    call = ms.get("CALL")
    call.observations.append(Observation(name="report_at_nm", value_type="number", source="model"))
    call.orientations.append(Orientation(name="exercise_plan", kind="intent"))
    call.decision.policy_features.append("report_at_nm")
    ms.get("ACK").decision.rule += " Normalize number words. If no controlled aircraft is addressed, acknowledge=false and retain the named addressee and ordered distance."
    ms.get("FLY").plan = "ACK; if acknowledged=false stop, else DEV, LOC, SEL, HDG, CFG, WAIT, CALL"
    # Allowed enum values belong to the hand-off specification, so the runtime, diagnostics,
    # calibration and analysis all validate against the same values (protocol v1.1).
    for task, field, allowed in ENUM_VALUES:
        ms.get(task).hand_off.enum_values[field] = list(allowed)
    # All machine decisions need an executor in the single-tier ablation.
    for tm in ms.task_models:
        if tm.id != "DEV":
            tm.executors[Tier.T2] = f"strong:{tm.id}"
        if tm.id in ("ACK", "CALL"):
            tm.executors[Tier.T1] = f"small:{tm.id}"
    return ms


class StudyPerson:
    """Simulated authority and recovery; not a research participant."""
    def __init__(self, instance, policies_enabled):
        self.instance, self.policies_enabled = instance, policies_enabled
        self.answers = gold(instance)

    def decide(self, request):
        task = request["task_id"]
        if task == "DEV":
            # The White Force plan answers DEV for every instance, including a misaddressed
            # call that a model wrongly acknowledged and so carried past ACK.
            payload = {"choice": self.instance["white_force_plan"]["deviation"], "rationale": "Scripted exercise plan"}
        elif task not in self.answers:
            raise ValueError(f"no reference recovery for {task}: the instruction was not addressed to aircraft 2")
        else:
            payload = copy.deepcopy(self.answers[task])
        if task == "CALL":
            distance = request["situation"]["observations"]["report_at_nm"]
            payload = call_reference(self.instance, distance)
        # Keep human hand-offs concise while leaving numerical fields intact.
        if task == "FLY":
            payload["summary"], payload["next"] = "Synthetic join status", "Await control"
        return Proposal(decided=True, payload=payload, reason="Scripted exercise plan or reference recovery",
                        make_policy=self.policies_enabled and task in ("DEV", "CALL"),
                        policy_rationale="Only the exact declared situation and intent")

    def approve(self, request):
        return True

    def veto(self, notice):
        return False


class StudyEngine(Engine):
    def __init__(self, *args, condition, paid, router=None, router_threshold=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.condition, self.paid = condition, paid
        self.router, self.router_threshold = router, router_threshold

    def _decide(self, tm, ctx, alloc, features, fkey):
        if self.condition == "allocated" or tm.id == "DEV" or (self.condition == "routed" and tm.id not in ("ACK", "CALL")):
            return super()._decide(tm, ctx, alloc, features, fkey)
        tier = Tier.T2
        if self.condition == "routed":
            rate = self.router.score(model_context(ctx))
            tier = Tier.T2 if rate >= self.router_threshold else Tier.T1
            self.log.emit("routed", task=tm.id, score=rate, threshold=self.router_threshold, tier=tier.value)
        backend = ExecutorBackend(self.paid, tier.value, mode="forced")
        p = ModelExecutor(tier, backend).decide(ctx)
        self.log.emit("call", task=tm.id, tier=tier.value, by=p.decided_by,
                      tokens_in=p.tokens_in, tokens_out=p.tokens_out, confidence=p.confidence)
        problems = validate(p.payload, tm.hand_off)
        if problems:
            self.log.emit("invalid_candidate", task=tm.id, problems=problems)
            raise InvalidCandidate(f"{tm.id}: invalid forced candidate")
        p.decided = True
        self.log.emit("decided", task=tm.id, tier=tier.value, by=p.decided_by)
        return p, tier


def build(instance, condition, paid, policies=None, thresholds=None, router=None, router_threshold=None, run_id="run"):
    ms, w, ori = task_models(), make_world(instance), orientation(instance)
    ori["intercept_method"] = (
        "Earliest lead intercept. x,y NM; speed knots; heading 0 north, 90 east. "
        "Aircraft position after t seconds adds speed*t/3600*(sin(heading),cos(heading)). "
        "Tanker position: (cx+r*sin(angle+speed*t/3600/r),cy+r*cos(angle+speed*t/3600/r)). "
        "Solve distance to predicted tanker = aircraft speed*t/3600."
    )
    ms.get("WAIT").decision.rule += " Use the declared states and current aircraft heading to find the first five-second sample at or below report_at_nm; output elapsed minutes and that distance."
    # Protocol v1.1: every model input carries what its scoring requires.
    ori["comms_plan"] = {**ori["comms_plan"], "control_callsign": CONTROL_CALLSIGN}
    ms.get("CALL").decision.rule += (
        " If the channel is busy, defer the call. If the primary channel is jammed and no jamming inject is"
        " scheduled, make the call on the backup channel."
        " A made report gives control's callsign, then 'aircraft 2', then the ordered reporting distance"
        " (report_at_nm) in NM from the tanker, for example '<control callsign>, aircraft 2, 25 NM from the tanker.'"
        " A deferred call says it will hold, wait, retry or defer.")
    sensors = {**SENSORS, "aircraft2_state": aircraft_state, "tanker_state": tanker_state}
    actuators = dict(ACTUATORS)

    def ack_act(w, payload, ctx):
        if payload["acknowledged"]:
            w.radio_log.append((w.time_s, ori["comms_plan"]["primary"], f"{payload['addressee']}, wilco"))

    def call_act(w, payload, ctx):
        if payload["call_status"] == "made":
            w.radio_log.append((w.time_s, payload["frequency"], payload["message"]))
        valid_channel = ori["comms_plan"]["backup"] if w.channel_state == "jammed" else ori["comms_plan"]["primary"]
        w.control_acknowledged = payload["call_status"] == "made" and w.channel_state != "in_use" and payload["frequency"] == valid_channel

    actuators.update(transmit_acknowledgement=ack_act, transmit_call=call_act)
    person = StudyPerson(instance, condition == "allocated")
    thresholds = {(task, Tier(tier)): value for task, row in (thresholds or {}).items() for tier, value in row.items()}
    log = EventLog(run_id)
    engine = StudyEngine(model_set=ms, executors={Tier.T0: ToolExecutor(TOOLS),
                         Tier.T1: ModelExecutor(Tier.T1, ExecutorBackend(paid, "T1")),
                         Tier.T2: ModelExecutor(Tier.T2, ExecutorBackend(paid, "T2"))},
                         person=PersonExecutor(person, "scripted White Force"),
                         policies=policies if policies is not None else PolicyStore(), log=log,
                         sensors=sensors, orientation_store=ori, actuators=actuators, world=w,
                         plan_guard=lambda parent, done: "ACK" not in done or done["ACK"]["acknowledged"],
                         confidence_thresholds=thresholds, condition=condition, paid=paid,
                         router=router, router_threshold=router_threshold)
    return engine, log


def run_instance(instance, condition, paid, run_id, **kwargs):
    paid.start(run_id)
    engine, log = build(instance, condition, paid, run_id=run_id, **kwargs)
    router_start = getattr(kwargs.get("router"), "seconds", 0.0)
    start = time.perf_counter()
    status = "completed"
    if condition == "single":
        # Include all scenario facts and task specs, excluding the private plan
        # answers, split labels and generator answer key.
        context = {"transmission": instance["transmission"], "world": instance["world"],
                   "orientation": engine.orientation_store,
                   "task_models": [tm.model_dump(mode="json", exclude={"executors", "allocation_override"}) for tm in engine.model_set.task_models]}
        results, tin, tout = paid.call("T2", context, "monolithic")
        if not isinstance(results, dict):
            results = {}
        for task, payload in list(results.items()):
            if task not in {tm.id for tm in engine.model_set.task_models}:
                status = "failed_candidate_or_action"
                continue
            if payload is not None and validate(payload, engine.model_set.get(task).hand_off):
                results[task] = None
                status = "failed_candidate_or_action"
        if any(task not in results or results[task] is None for task in ("ACK", "FLY")):
            status = "failed_candidate_or_action"
        log.emit("call", task="ALL", tier="T2", tokens_in=tin, tokens_out=tout)
        for task, payload in results.items():
            log.emit("hand_off", task=task, payload=payload)
    else:
        try:
            engine.run()
        except (InvalidCandidate, DesignFault, ActionNotPerformed, KeyError, TypeError, ValueError, AssertionError) as exc:
            status = "failed_candidate_or_action"
            log.emit("study_run_failed", error_type=type(exc).__name__, detail=str(exc))
        results = engine.results
    elapsed = time.perf_counter() - start
    scored = score_run(instance, results)
    rows = paid.last_rows
    timing = paid.run_timing() if hasattr(paid, "run_timing") else {
        "failed_attempts": 0, "retry_count": 0, "unknown_timing_attempts": 0,
        "known_failed_latency_s": 0.0, "prior_known_failed_latency_s": 0.0}
    # Current-attempt failures are already in elapsed. Only add failures saved
    # before this invocation and successful calls replayed from the journal.
    completion_time = None if timing["unknown_timing_attempts"] else (
        elapsed + timing["prior_known_failed_latency_s"]
        + sum(r["latency_s"] for r in rows if r.get("replayed")))
    return {"run_id": run_id, "instance_id": instance["id"], "condition": condition, "status": status,
            "empirical": all(row["empirical"] for row in rows), "accuracy": sum(scored.values()) / len(scored),
            "correct_by_task": scored, "results": results, "metrics": log.metrics(),
            "input_tokens": sum(r["input_tokens"] for r in rows), "output_tokens": sum(r["output_tokens"] for r in rows),
            "max_context_tokens": max((r["input_tokens"] for r in rows), default=0),
            "cost_usd": sum(r["cost_usd"] for r in rows),
            "time_s": completion_time,
            **{k: timing[k] for k in ("failed_attempts", "retry_count",
                                     "unknown_timing_attempts", "known_failed_latency_s")},
            "router_s": getattr(kwargs.get("router"), "seconds", 0.0) - router_start,
            "events": log.events}
