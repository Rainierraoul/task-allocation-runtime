"""Design-time allocation: which tier makes each task model's decision, and how
freely the Action stage may act on it.

The rule is ordered: the first clause that applies sets the tier. Each clause
records its letter and reason so a reviewer can trace every allocation.

    A  Decision cannot be described or decomposed (DCARR Step 5a exit)  -> people
    B  Decision is normative (a choice about what ought to happen)      -> people,
       unless a stored policy covers the situation at run time
    C  Cynefin domain is chaotic                                         -> people
    D  Decision reaches a state machine or an algorithm (Steps 9, 18)    -> deterministic tools
    E  Plain-language decision in a complex domain                       -> reasoning models,
       as decision support only (the Action stage proposes)
    F  Plain-language, functional task at the skill or rule level        -> small models
    G  Any other plain-language decision (procedural or knowledge-based) -> reasoning models

Clauses A-D restate the thesis: the ordered, describable domains are the ones
deterministic automation can take, and the rest stayed with White Force. Clauses
E-G are new. They allocate the plain-language decisions the thesis ruled out of
scope, which language models can now make.

The Action stage level follows Sheridan and Verplank's scale, the thesis's
Cynefin mapping (complex -> decision support) and the rule that reversible,
low-consequence actions act and report:
    person decides                                   -> person_acts
    complex domain                                   -> propose_only
    irreversible action                              -> act_with_approval
    hand-off goes to people (the hand-off reports)   -> act_and_report
    on the critical path and failure breaks the
    objective                                        -> act_unless_vetoed
    otherwise                                        -> act_and_report
"""

from __future__ import annotations

from dataclasses import dataclass, field

from .metamodel import (
    ActionLevel,
    ControlLevel,
    Cynefin,
    DecisionKind,
    Describability,
    TaskModel,
    TaskType,
    Tier,
)


@dataclass(frozen=True)
class Allocation:
    tier: Tier
    action_level: ActionLevel
    clause: str
    reasons: list[str] = field(default_factory=list)


def allocate(tm: TaskModel) -> Allocation:
    d = tm.decision
    reasons: list[str] = []

    if d.describability == Describability.NOT_DESCRIBABLE:
        tier, clause = Tier.T3, "A"
        reasons.append("A: the decision cannot be described or decomposed (DCARR Step 5a)")
    elif d.kind == DecisionKind.NORMATIVE:
        tier, clause = Tier.T3, "B"
        reasons.append("B: normative decision; people decide unless a stored policy covers it")
    elif tm.cynefin == Cynefin.CHAOTIC:
        tier, clause = Tier.T3, "C"
        reasons.append("C: chaotic domain; people act first")
    elif d.describability in (Describability.STATE_MACHINE, Describability.ALGORITHM):
        tier, clause = Tier.T0, "D"
        reasons.append(f"D: the decision is written as a {d.describability.value.replace('_', ' ')}")
    elif tm.cynefin == Cynefin.COMPLEX:
        tier, clause = Tier.T2, "E"
        reasons.append("E: plain-language decision in a complex domain; decision support only")
    elif tm.task_type == TaskType.FUNCTIONAL and tm.control_level in (
        ControlLevel.SKILL,
        ControlLevel.RULE,
    ):
        tier, clause = Tier.T1, "F"
        reasons.append("F: functional task at the skill or rule level, in plain language")
    else:
        tier, clause = Tier.T2, "G"
        reasons.append(
            f"G: {tm.task_type.value} task at the {tm.control_level.value} level, in plain language"
        )

    if tm.allocation_override is not None:
        o = tm.allocation_override
        reasons.append(f"override to {o.tier.value} by {o.set_by}: {o.reason}")
        tier = o.tier

    if tier == Tier.T3:
        level = ActionLevel.PERSON_ACTS
    elif tm.cynefin == Cynefin.COMPLEX:
        level = ActionLevel.PROPOSE_ONLY
    elif not tm.action.reversible:
        level = ActionLevel.ACT_WITH_APPROVAL
        reasons.append("action is irreversible: act only with approval, unless a policy covers it")
    elif tm.hand_off.to == "people":
        level = ActionLevel.ACT_AND_REPORT
        reasons.append("the hand-off to people is the report")
    elif tm.impact.distance == 0 and tm.impact.failure_breaks_objective:
        level = ActionLevel.ACT_UNLESS_VETOED
        reasons.append("on the critical path and failure breaks the objective: act unless vetoed")
    else:
        level = ActionLevel.ACT_AND_REPORT

    return Allocation(tier=tier, action_level=level, clause=clause, reasons=reasons)


def default_threshold(tm: TaskModel) -> float:
    """Confidence a model tier must reach before it may decide.

    Thresholds are normative: people set them. This default applies only when
    the task model leaves the threshold empty. Higher stakes raise it.
    """
    if tm.decision.confidence_threshold is not None:
        return tm.decision.confidence_threshold
    if tm.impact.distance == 0 or tm.impact.failure_breaks_objective:
        return 0.9
    return 0.7
