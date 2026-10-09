import copy

import pytest

from dcarr_runtime import PolicyStore, ScriptedBackend, Tier
from dcarr_runtime.executors import Proposal
from dcarr_runtime.handoff import estimate_tokens
from scenarios.simpilot import run as simpilot
from scenarios.simpilot.stand_ins import HANDLERS, ScriptedPerson


def run(variant, policies=None, backend=None, person=None):
    policies = policies if policies is not None else PolicyStore()
    engine, world, person, log = simpilot.build(variant, policies, backend=backend, person=person)
    result = engine.run()
    return result, world, person, log


def kinds(log, kind, task=None):
    return [e for e in log.of(kind) if task is None or e.get("task") == task]


# ------------------------------------------------------------- clean run


def test_clean_run_meets_the_goal():
    result, world, person, log = run("clean")
    assert result["goal_met"] is True
    assert 19.0 <= result["distance_nm"] <= 20.0
    assert world.control_acknowledged
    assert world.aircraft["aircraft 2"].heading_deg != 90.0  # heading was adjusted


def test_clean_run_asks_people_only_the_normative_question():
    _, _, person, log = run("clean")
    assert [r["task_id"] for r in person.requests] == ["DEV"]
    m = log.metrics()
    assert m["decisions_by_tier"] == {"T0": 6, "T1": 2, "T3": 1}
    assert m["abstentions"] == 0


def test_a_stored_policy_answers_the_same_situation_next_time():
    policies = PolicyStore()
    run("clean", policies)
    assert len(policies) == 1
    _, _, person, log = run("clean", policies)
    assert person.requests == []
    assert [e["task"] for e in log.of("policy_applied")] == ["DEV"]


def test_policy_store_survives_a_restart(tmp_path):
    path = tmp_path / "policies.json"
    run("clean", PolicyStore(path))
    _, _, person, _ = run("clean", PolicyStore(path))
    assert person.requests == []


def test_a_different_situation_goes_back_to_people():
    policies = PolicyStore()
    run("clean", policies)
    engine, world, person, log = simpilot.build("clean", policies)
    world.trainee_level = "lfe_proficient"
    engine.run()
    assert [r["task_id"] for r in person.requests] == ["DEV"]


# --------------------------------------------------- abstain and escalate


def test_ambiguous_addressee_escalates_from_small_to_reasoning_model():
    result, _, _, log = run("ambiguous")
    assert [e["tier"] for e in kinds(log, "abstain", "ACK")] == ["T1"]
    assert [(e["frm"], e["to"]) for e in kinds(log, "escalate", "ACK")] == [("T1", "T2")]
    assert [e["tier"] for e in kinds(log, "decided", "ACK")] == ["T2"]
    assert result["goal_met"] is True


def test_jammed_channel_reaches_people_with_options_then_becomes_policy():
    policies = PolicyStore()
    _, _, person, log = run("jammed", policies)
    assert [e["tier"] for e in kinds(log, "abstain", "CALL")] == ["T1", "T2"]
    call_request = next(r for r in person.requests if r["task_id"] == "CALL")
    assert len(call_request["options"]) == 3
    assert len(call_request["why_you"]) >= 3  # allocation reason plus two abstentions

    _, _, person2, log2 = run("jammed", policies)
    assert person2.requests == []
    assert sorted(e["task"] for e in log2.of("policy_applied")) == ["CALL", "DEV"]


def test_deterministic_rejection_skips_tiers_without_executors():
    class Person(ScriptedPerson):
        ANSWERS = {
            **ScriptedPerson.ANSWERS,
            "LOC": ({"tanker": "SHELL 1", "tanker_found": True, "orbit": "reported by control"}, ""),
        }

    engine, world, person, log = simpilot.build("clean", PolicyStore(), person=Person())
    world.tanker.callsign = "SHELL 9"  # not the callsign in the briefing
    engine.run()
    assert [e["tier"] for e in kinds(log, "abstain", "LOC")] == ["T0"]
    assert [e["tier"] for e in kinds(log, "skip", "LOC")] == ["T1", "T2"]
    assert "LOC" in [r["task_id"] for r in person.requests]


def test_hand_off_over_budget_counts_as_abstention():
    handlers = dict(HANDLERS)

    def verbose(context):
        out = HANDLERS["parse_command_small"](context)
        out["payload"]["addressee"] = "aircraft 2 " + "(repeated) " * 40
        return out

    handlers["parse_command_small"] = verbose
    _, _, _, log = run("clean", backend=ScriptedBackend(handlers))
    reasons = [e["reason"] for e in kinds(log, "abstain", "ACK")]
    assert len(reasons) == 1 and "exceeds the budget" in reasons[0]
    assert [e["tier"] for e in kinds(log, "decided", "ACK")] == ["T2"]


# ------------------------------------------------------ the typed boundary


def test_model_tier_cannot_make_a_normative_decision():
    from dcarr_runtime.metamodel import AllocationOverride, Cynefin

    engine, world, person, log = simpilot.build("clean", PolicyStore())
    dev = engine.model_set.get("DEV")
    # Give the normative task a reasoning-model executor and start its ladder at T2,
    # so a model gets the chance to decide it.
    dev.executors[Tier.T2] = "eager_model"
    dev.cynefin = Cynefin.COMPLICATED
    dev.allocation_override = AllocationOverride(tier=Tier.T2, reason="test", set_by="test")
    backend = engine.executors[Tier.T2].backend
    backend.handlers["eager_model"] = lambda ctx: {
        "decided": True,
        "payload": {"choice": "misinterpret", "rationale": "more interesting"},
        "confidence": 0.99,
    }
    engine.run()
    assert [e["task"] for e in log.of("boundary")] == ["DEV"]
    assert [e["tier"] for e in kinds(log, "decided", "DEV")] == ["T3"]


# ------------------------------------------------------- context and hand-off


def test_model_tiers_see_only_declared_fields():
    seen = {}

    class Recording(ScriptedBackend):
        def respond(self, name, context):
            seen[name] = copy.deepcopy(context)
            return super().respond(name, context)

    run("clean", backend=Recording(HANDLERS))
    call_ctx = seen["compose_call_small"]
    assert set(call_ctx["observations"]) == {"channel_state", "jamming_inject_scheduled", "distance_nm"}
    assert set(call_ctx["orientation"]) == {"radio_procedure", "comms_plan", "exercise_injects"}
    assert "transmission" not in str(call_ctx)  # ACK's input never reaches CALL


def test_what_people_receive_is_fused_and_within_budget():
    result, _, _, log = run("clean")
    root = [e for e in log.of("hand_off") if e["task"] == "FLY"][0]
    assert root["to"] == "people" and root["fusion_level"] == "L2"
    assert root["tokens"] <= root["budget"] == 90
    assert estimate_tokens(result) == root["tokens"]
