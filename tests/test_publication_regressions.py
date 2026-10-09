import copy
import math

import pytest

from dcarr_runtime import PolicyStore, ScriptedBackend
from dcarr_runtime.engine import ActionNotPerformed, DesignFault
from dcarr_runtime.metamodel import TaskModelSet
from scenarios.simpilot.run import build
from scenarios.simpilot.stand_ins import HANDLERS, ScriptedPerson


@pytest.mark.parametrize("refusal", ["approval", "veto"])
def test_refused_action_cannot_be_handed_off_as_completed(refusal):
    class Refusing(ScriptedPerson):
        def approve(self, request):
            return request["task_id"] != "CALL"

        def veto(self, notice):
            return notice["task_id"] == "CALL"

    engine, world, _, log = build("clean", PolicyStore(), person=Refusing())
    engine.model_set.get("CALL").action.reversible = refusal != "approval"
    with pytest.raises(ActionNotPerformed, match="CALL"):
        engine.run()
    assert not world.control_acknowledged
    assert "CALL" not in engine.results
    assert "FLY" not in engine.results
    assert not any(e.get("task") == "CALL" for e in log.of("hand_off"))


def test_missing_actuator_is_not_reported_as_executed():
    engine, _, _, log = build("clean", PolicyStore())
    del engine.actuators["transmit_call"]
    with pytest.raises(DesignFault, match="no actuator"):
        engine.run()
    assert not any(e.get("task") == "CALL" for e in log.of("acted"))


@pytest.mark.parametrize("changed", ["distance", "orientation", "version"])
def test_policy_does_not_cross_its_scoped_situation(changed):
    policies = PolicyStore()
    engine, _, _, _ = build("jammed", policies)
    engine.run()
    second, world, _, log = build("jammed", policies)
    if changed == "distance":
        world.transmission = world.transmission.replace("20 nautical", "30 nautical")
    elif changed == "orientation":
        second.orientation_store = copy.deepcopy(second.orientation_store)
        second.orientation_store["comms_plan"]["backup"] = "Channel 9"
    else:
        second.model_set.version = "new-design"
    second.run()
    assert not any(e["task"] == "CALL" for e in log.of("policy_applied"))


@pytest.mark.parametrize("bad", ["high", True, -0.1, 1.1, float("nan")])
def test_invalid_model_confidence_abstains_instead_of_crashing(bad):
    def handler(ctx):
        response = HANDLERS["parse_command_small"](ctx)
        response["confidence"] = bad
        return response

    engine, _, _, log = build("clean", PolicyStore(), backend=ScriptedBackend({**HANDLERS, "parse_command_small": handler}))
    engine.run()
    assert any(e["task"] == "ACK" and e["tier"] == "T1" for e in log.of("abstain"))


def test_unknown_parent_and_cycle_are_rejected():
    engine, _, _, _ = build("clean", PolicyStore())
    raw = engine.model_set.model_dump(mode="json")
    raw["task_models"][0]["parent"] = "missing"
    with pytest.raises(ValueError, match="unknown parent"):
        TaskModelSet.model_validate(raw)
    raw = engine.model_set.model_dump(mode="json")
    raw["task_models"][0]["parent"] = "FLY"
    raw["task_models"][0]["subtasks"].append("FLY")
    with pytest.raises(ValueError, match="cycle"):
        TaskModelSet.model_validate(raw)
