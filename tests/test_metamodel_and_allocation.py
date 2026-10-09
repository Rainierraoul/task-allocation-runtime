from pathlib import Path

import pytest
import yaml

from dcarr_runtime import allocate, load_task_models
from dcarr_runtime.metamodel import ActionLevel, TaskModel, TaskModelSet, Tier

SIMPILOT = Path(__file__).parents[1] / "scenarios" / "simpilot" / "task_models.yaml"


def raw():
    return yaml.safe_load(SIMPILOT.read_text())


def task(**changes) -> TaskModel:
    """A minimal valid task model, with fields replaced as given."""
    base = {
        "id": "X",
        "name": "x",
        "goal": "x",
        "objectives": ["O1"],
        "description": {"action": "a", "object": "o", "output": "p", "tooling": "t", "context": "c"},
        "task_type": "functional",
        "control_level": "rule",
        "cynefin": "clear",
        "observations": [{"name": "f", "value_type": "enum"}],
        "decision": {"rule": "r", "kind": "epistemic", "describability": "plain_language"},
        "action": {"name": "act", "reversible": True},
        "impact": {"distance": 1, "failure_breaks_objective": False},
        "hand_off": {"fusion_level": "L1", "fields": {"v": "text"}, "max_tokens": 20},
    }
    for key, value in changes.items():
        if isinstance(value, dict) and isinstance(base.get(key), dict):
            base[key] = {**base[key], **value}
        else:
            base[key] = value
    return TaskModel.model_validate(base)


# ------------------------------------------------------------ metamodel


def test_simpilot_task_models_load():
    ms = load_task_models(SIMPILOT)
    assert ms.root == "FLY"
    assert len(ms.task_models) == 9


def test_objective_without_task_model_is_rejected():
    data = raw()
    data["objectives"].append({"id": "O3", "text": "Manage aircraft 1"})
    with pytest.raises(ValueError, match="DCARR Step 4"):
        TaskModelSet.model_validate(data)


def test_task_model_without_objective_is_rejected():
    data = raw()
    data["task_models"][1]["objectives"] = []
    with pytest.raises(ValueError, match="maps to no objective"):
        TaskModelSet.model_validate(data)


def test_subtask_must_name_its_parent():
    data = raw()
    data["task_models"][1]["parent"] = None
    with pytest.raises(ValueError, match="parent should be"):
        TaskModelSet.model_validate(data)


def test_normative_decision_needs_policy_features():
    with pytest.raises(ValueError, match="must name the observations"):
        task(decision={"kind": "normative"})


# ----------------------------------------------------------- allocation


@pytest.mark.parametrize(
    "changes, tier, clause",
    [
        ({"decision": {"describability": "not_describable"}}, Tier.T3, "A"),
        ({"decision": {"kind": "normative", "policy_features": ["f"]}}, Tier.T3, "B"),
        ({"cynefin": "chaotic"}, Tier.T3, "C"),
        ({"decision": {"describability": "state_machine"}}, Tier.T0, "D"),
        ({"decision": {"describability": "algorithm"}, "cynefin": "complicated"}, Tier.T0, "D"),
        ({"cynefin": "complex"}, Tier.T2, "E"),
        ({}, Tier.T1, "F"),
        ({"control_level": "skill"}, Tier.T1, "F"),
        ({"control_level": "knowledge"}, Tier.T2, "G"),
        ({"task_type": "procedural"}, Tier.T2, "G"),
    ],
)
def test_allocation_clauses(changes, tier, clause):
    a = allocate(task(**changes))
    assert (a.tier, a.clause) == (tier, clause)


def test_clause_order_normative_beats_describable():
    # A normative choice stays with people even if it could be written as an algorithm.
    a = allocate(task(decision={"kind": "normative", "policy_features": ["f"], "describability": "algorithm"}))
    assert a.tier == Tier.T3 and a.clause == "B"


@pytest.mark.parametrize(
    "changes, level",
    [
        ({"cynefin": "complex"}, ActionLevel.PROPOSE_ONLY),
        ({"action": {"reversible": False}}, ActionLevel.ACT_WITH_APPROVAL),
        ({"impact": {"distance": 0, "failure_breaks_objective": True}}, ActionLevel.ACT_UNLESS_VETOED),
        ({"impact": {"distance": 0, "failure_breaks_objective": False}}, ActionLevel.ACT_AND_REPORT),
        (
            {"impact": {"distance": 0, "failure_breaks_objective": True}, "hand_off": {"to": "people"}},
            ActionLevel.ACT_AND_REPORT,
        ),
        ({"decision": {"describability": "not_describable"}}, ActionLevel.PERSON_ACTS),
    ],
)
def test_action_levels(changes, level):
    assert allocate(task(**changes)).action_level == level


def test_override_is_recorded():
    a = allocate(task(allocation_override={"tier": "T2", "reason": "small model failed trials", "set_by": "SV"}))
    assert a.tier == Tier.T2
    assert any("override" in r for r in a.reasons)


def test_simpilot_allocation_matches_the_worked_example():
    ms = load_task_models(SIMPILOT)
    tiers = {tm.id: allocate(tm).tier for tm in ms.task_models}
    assert tiers == {
        "FLY": Tier.T0,
        "ACK": Tier.T1,
        "DEV": Tier.T3,
        "LOC": Tier.T0,
        "SEL": Tier.T0,
        "HDG": Tier.T0,
        "CFG": Tier.T0,
        "WAIT": Tier.T0,
        "CALL": Tier.T1,
    }
