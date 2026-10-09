"""M2: the OODA metamodel.

Every task model (M1) is an instance of these classes, and every run of a
task model (M0) is recorded in the event log. The levels follow the thesis's
metamodel stack (Vajda, 2023, Figure 'oodamm'): M2 OODA metamodel, M1 task
models, M0 instance models.

Field names follow the DCARR steps they come from, so a task model written
in a Describe workshop can be loaded without translation:

    Step 4   objectives        each task model maps to at least one objective
    Step 5   subtasks          HTA decomposition
    Step 5a  describability    'not_describable' is the Step 5a exit
    Step 6   description       action, object, output, tooling, context
    Step 7   observations      entities taken from the description
    Step 9   describability    'state_machine' means a DFA quintuple exists
    Step 12  value_type        value types for observations
    Step 17  orientations      orientation entities as queryable data
    Step 18  decision.rule     the decision written as an algorithm
    Step 19  task_type         functional or procedural
    Step 22  cynefin           Cynefin domain of the task
    Step 23  impact            distance from the objective; failure effect

Fields added for allocation (not in DCARR): control_level, decision.kind,
action.reversible, hand_off and executors.
"""

from __future__ import annotations

import json
from enum import Enum
from typing import Literal, Optional

from pydantic import BaseModel, ConfigDict, Field, model_validator


class Tier(str, Enum):
    """Who or what makes a task model's decision."""

    T0 = "T0"  # deterministic tools
    T1 = "T1"  # small models
    T2 = "T2"  # reasoning models
    T3 = "T3"  # people


TIER_LABELS = {
    Tier.T0: "deterministic tools",
    Tier.T1: "small models",
    Tier.T2: "reasoning models",
    Tier.T3: "people",
}

TIER_ORDER = [Tier.T0, Tier.T1, Tier.T2, Tier.T3]


class Describability(str, Enum):
    """How far the decision can be written down (DCARR Steps 5a, 6, 9, 18)."""

    STATE_MACHINE = "state_machine"      # a DFA quintuple exists (Step 9)
    ALGORITHM = "algorithm"              # an algorithm over declared inputs (Step 18)
    PLAIN_LANGUAGE = "plain_language"    # No adequate formal specification supplied in this scope
    NOT_DESCRIBABLE = "not_describable"  # Step 5a exit: cannot describe or decompose


class TaskType(str, Enum):
    """DCARR Step 19."""

    FUNCTIONAL = "functional"  # given an observable state, apply a rule set
    PROCEDURAL = "procedural"  # invokes other tasks; several orientations and decisions


class ControlLevel(str, Enum):
    """Rasmussen's levels of cognitive control, as modelled in C-OODA (Breton, 2008)."""

    SKILL = "skill"
    RULE = "rule"
    KNOWLEDGE = "knowledge"


class Cynefin(str, Enum):
    """Cynefin domain of the task as a competent practitioner meets it (Step 22)."""

    CLEAR = "clear"
    COMPLICATED = "complicated"
    COMPLEX = "complex"
    CHAOTIC = "chaotic"


class DecisionKind(str, Enum):
    """Epistemic: establishes what is the case. Normative: chooses what ought to happen."""

    EPISTEMIC = "epistemic"
    NORMATIVE = "normative"


class ActionLevel(str, Enum):
    """Levels of automation for the Action stage (after Sheridan and Verplank, 1978)."""

    ACT_AND_REPORT = "act_and_report"
    ACT_UNLESS_VETOED = "act_unless_vetoed"
    ACT_WITH_APPROVAL = "act_with_approval"
    PROPOSE_ONLY = "propose_only"
    PERSON_ACTS = "person_acts"


class FusionLevel(str, Enum):
    """What a hand-off carries, on the JDL fusion levels and the matching SA levels.

    L0  signal     raw tool output or text
    L1  object     identified entities and states          (SA level 1, perception)
    L2  situation  relations between entities and the goal (SA level 2, comprehension)
    L3  impact     what happens next and what it means     (SA level 3, projection)
    """

    L0 = "L0"
    L1 = "L1"
    L2 = "L2"
    L3 = "L3"


FUSION_ORDER = [FusionLevel.L0, FusionLevel.L1, FusionLevel.L2, FusionLevel.L3]


class _Strict(BaseModel):
    model_config = ConfigDict(extra="forbid", use_enum_values=False)


class Objective(_Strict):
    id: str
    text: str


class Description(_Strict):
    """Answers to DCARR Step 6."""

    action: str = Field(description="What action is performed? (verb)")
    object: str = Field(description="To whom or what is the action performed?")
    output: str = Field(description="What does the action produce, or why is it needed?")
    tooling: str = Field(description="Processes, guides, equipment and tools used")
    context: str = Field(description="When the task happens in the scenario (temporality)")


class Observation(_Strict):
    name: str
    value_type: Literal["bool", "number", "text", "enum", "entity", "position"]
    source: Literal["tool", "model", "person"] = "tool"
    description: str = ""


class Orientation(_Strict):
    """An orientation entity: knowledge the decision needs besides the observations."""

    name: str
    kind: Literal["data", "procedure", "intent", "policy"]
    description: str = ""


class Decision(_Strict):
    rule: str = Field(description="The decision as an algorithm or in plain language")
    kind: DecisionKind
    describability: Describability
    abstain_when: list[str] = Field(default_factory=list)
    confidence_threshold: Optional[float] = Field(
        default=None,
        ge=0.0,
        le=1.0,
        description="Set by people. Empty means the default for the task's impact.",
    )
    policy_features: list[str] = Field(
        default_factory=list,
        description="Observation names that key a stored policy for a normative decision",
    )


class Action(_Strict):
    name: str
    reversible: bool
    effect: str = ""


class Impact(_Strict):
    """DCARR Step 23."""

    distance: int = Field(ge=0, description="Steps from the objective; 0 is on the critical path")
    failure_breaks_objective: bool


class HandOff(_Strict):
    """What a task model passes up when it finishes."""

    to: Literal["parent", "people"] = "parent"
    fusion_level: FusionLevel
    fields: dict[str, Literal["bool", "number", "text", "enum", "list"]]
    max_tokens: int = Field(gt=0)
    # Allowed values of enum fields. When declared, models are shown them and
    # the validator rejects any other value.
    enum_values: dict[str, list[str]] = Field(default_factory=dict)


class AllocationOverride(_Strict):
    tier: Tier
    reason: str
    set_by: str


class TaskModel(_Strict):
    id: str
    name: str
    goal: str
    objectives: list[str] = Field(default_factory=list)
    parent: Optional[str] = None
    subtasks: list[str] = Field(default_factory=list, description="HTA plan: run in this order")
    plan: str = ""
    description: Description
    task_type: TaskType
    control_level: ControlLevel
    cynefin: Cynefin
    observations: list[Observation] = Field(default_factory=list)
    orientations: list[Orientation] = Field(default_factory=list)
    decision: Decision
    action: Action
    impact: Impact
    frequency: Optional[int] = Field(default=None, ge=0, description="Occurrences per run (Step 21)")
    hand_off: HandOff
    executors: dict[Tier, str] = Field(
        default_factory=dict,
        description="Registered executor name for each tier that can make this decision",
    )
    allocation_override: Optional[AllocationOverride] = None

    @model_validator(mode="after")
    def _normative_needs_policy_key(self) -> "TaskModel":
        if self.decision.kind == DecisionKind.NORMATIVE and not self.decision.policy_features:
            raise ValueError(
                f"{self.id}: a normative decision must name the observations that key its policy"
            )
        known = {o.name for o in self.observations}
        missing = [f for f in self.decision.policy_features if f not in known]
        if missing:
            raise ValueError(f"{self.id}: policy features {missing} are not declared observations")
        return self


class TaskModelSet(_Strict):
    """An M1 document: one scenario, its objectives and its task models."""

    scenario: str
    version: str
    objectives: list[Objective]
    root: str
    task_models: list[TaskModel]

    @model_validator(mode="after")
    def _check_structure(self) -> "TaskModelSet":
        ids = [t.id for t in self.task_models]
        dupes = {i for i in ids if ids.count(i) > 1}
        if dupes:
            raise ValueError(f"duplicate task model ids: {sorted(dupes)}")
        by_id = {t.id: t for t in self.task_models}
        if self.root not in by_id:
            raise ValueError(f"root {self.root!r} is not a task model")
        objective_ids = {o.id for o in self.objectives}
        # DCARR Step 4: each goal maps to at least one objective, and each objective
        # has at least one goal.
        covered: set[str] = set()
        for t in self.task_models:
            if not t.objectives:
                raise ValueError(f"{t.id}: maps to no objective (DCARR Step 4)")
            unknown = set(t.objectives) - objective_ids
            if unknown:
                raise ValueError(f"{t.id}: unknown objectives {sorted(unknown)}")
            covered |= set(t.objectives)
            for s in t.subtasks:
                if s not in by_id:
                    raise ValueError(f"{t.id}: unknown subtask {s!r}")
                if by_id[s].parent != t.id:
                    raise ValueError(f"{s}: parent should be {t.id!r}")
            if t.parent is not None and t.parent not in by_id:
                raise ValueError(f"{t.id}: unknown parent {t.parent!r}")
            if t.parent is not None and t.id not in by_id[t.parent].subtasks:
                raise ValueError(f"{t.id}: not listed in its parent's subtasks")
        visiting: set[str] = set()
        visited: set[str] = set()

        def visit(task_id: str) -> None:
            if task_id in visiting:
                raise ValueError(f"cycle in task hierarchy at {task_id!r}")
            if task_id in visited:
                return
            visiting.add(task_id)
            for child in by_id[task_id].subtasks:
                visit(child)
            visiting.remove(task_id)
            visited.add(task_id)

        visit(self.root)
        if visited != set(by_id):
            raise ValueError(f"task models unreachable from root: {sorted(set(by_id) - visited)}")
        if by_id[self.root].parent is not None:
            raise ValueError("root must not have a parent")
        uncovered = objective_ids - covered
        if uncovered:
            raise ValueError(f"objectives with no task model: {sorted(uncovered)} (DCARR Step 4)")
        return self

    def get(self, task_id: str) -> TaskModel:
        for t in self.task_models:
            if t.id == task_id:
                return t
        raise KeyError(task_id)


def json_schema() -> dict:
    """The M2 metamodel as JSON Schema, for editors and other tools."""
    return TaskModelSet.model_json_schema()


if __name__ == "__main__":
    print(json.dumps(json_schema(), indent=2))
