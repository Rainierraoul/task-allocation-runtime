"""The runtime: runs a task model set, one OODA cycle per task model.

For each task model, depth first in HTA plan order:

  Observe   read declared observations from sensors or from subtask hand-offs
  Orient    fetch the declared orientation entities, and nothing else
  Decide    apply a stored policy if one covers the situation; otherwise start
            at the allocated tier and escalate on each abstention:
              T0 abstains when its state machine rejects the input
              T1 and T2 abstain when they say so, fall below the confidence
                 threshold, return an invalid hand-off, or try to make a
                 normative decision with no covering policy
              T3 (people) always decides, and may store the decision as a policy
  Act       at the allocated level: act and report, act unless vetoed,
            act with approval, or propose only
  Hand off  pass the declared fields, at the declared fusion level and within
            the token budget, to the parent task model or to people

Every step writes to the event log (M0).
"""

from __future__ import annotations

from typing import Any, Callable, Optional

from .allocation import Allocation, allocate, default_threshold
from .boundary import blocks_machine_decision
from .eventlog import EventLog
from .executors import DecisionContext, PersonExecutor, Proposal, judgement_request
from .handoff import estimate_tokens, validate
from .metamodel import ActionLevel, DecisionKind, TIER_ORDER, TaskModel, TaskModelSet, Tier
from .policy import PolicyStore, feature_key

Sensor = Callable[[Any, dict[str, dict[str, Any]]], Any]
Actuator = Callable[[Any, dict[str, Any], DecisionContext], None]


class DesignFault(RuntimeError):
    """A deterministic tool or a person's answer broke its own hand-off spec."""


class ActionNotPerformed(RuntimeError):
    """Stop the plan when approval or a veto prevents a required action."""


class Engine:
    def __init__(
        self,
        model_set: TaskModelSet,
        executors: dict[Tier, Any],
        person: PersonExecutor,
        policies: PolicyStore,
        log: EventLog,
        sensors: Optional[dict[str, Sensor]] = None,
        orientation_store: Optional[dict[str, Any]] = None,
        actuators: Optional[dict[str, Actuator]] = None,
        world: Any = None,
        plan_guard: Optional[Callable[[TaskModel, dict[str, dict[str, Any]]], bool]] = None,
        confidence_thresholds: Optional[dict[tuple[str, Tier], float]] = None,
    ):
        self.model_set = model_set
        self.executors = executors
        self.person = person
        self.policies = policies
        self.log = log
        self.sensors = sensors or {}
        self.orientation_store = orientation_store or {}
        self.actuators = dict(actuators or {})
        self.world = world
        self.plan_guard = plan_guard
        self.confidence_thresholds = confidence_thresholds or {}
        # Hand-offs completed so far in this run, in order. Later task models in
        # the HTA plan read their declared observations from here, so data flows
        # between siblings through named fields only.
        self.results: dict[str, dict[str, Any]] = {}

    # ------------------------------------------------------------------ run

    def run(self) -> dict[str, Any]:
        self.log.emit("run_start", scenario=self.model_set.scenario, policies_in_store=len(self.policies))
        payload = self.run_task(self.model_set.root)
        self.log.emit("run_end")
        return payload

    def run_task(self, task_id: str) -> dict[str, Any]:
        tm = self.model_set.get(task_id)
        self.log.emit("task_start", task=tm.id, name=tm.name)

        subtask_results = {}
        for s in tm.subtasks:
            if self.plan_guard is not None and not self.plan_guard(tm, subtask_results):
                self.log.emit("plan_stopped", task=tm.id, before=s)
                break
            subtask_results[s] = self.run_task(s)
        observations = self._observe(tm, subtask_results)
        orientations = {o.name: self.orientation_store.get(o.name) for o in tm.orientations}

        alloc = allocate(tm)
        self.log.emit(
            "allocated",
            task=tm.id,
            tier=alloc.tier.value,
            clause=alloc.clause,
            action_level=alloc.action_level.value,
            reasons=alloc.reasons,
        )

        features = {n: observations.get(n) for n in tm.decision.policy_features}
        # A policy is a decision within a particular task design and intent.
        # Canonical strings make structured orientation values hashable.
        if features:
            import json
            features["__design"] = self.model_set.scenario + ":" + self.model_set.version
            features["__orientation"] = json.dumps(orientations, sort_keys=True, default=str)
        fkey = feature_key(features, list(features)) if features else None
        policy = self.policies.match(tm.id, fkey) if fkey is not None else None
        if policy is not None:
            problems = validate(policy.decision, tm.hand_off)
            if problems:
                self.log.emit("policy_rejected", task=tm.id, problems=problems)
                policy = None

        ctx = DecisionContext(
            task=tm,
            observations=observations,
            orientations=orientations,
            subtask_results=subtask_results,
            policy=policy,
            reasons_so_far=list(alloc.reasons),
            world=self.world,
        )

        if policy is not None:
            final = Proposal(
                decided=True,
                payload=dict(policy.decision),
                reason=f"policy set by {policy.set_by} on {policy.set_on}",
                decided_by="policy",
            )
            final_tier = Tier.T0
            self.log.emit("policy_applied", task=tm.id, set_by=policy.set_by, features=features)
            self.log.emit("decided", task=tm.id, tier=Tier.T0.value, by="policy")
        else:
            final, final_tier = self._decide(tm, ctx, alloc, features, fkey)

        if not self._act(tm, ctx, alloc, final, final_tier, policy_applied=policy is not None):
            self.log.emit("run_stopped", task=tm.id, reason="required action not performed")
            raise ActionNotPerformed(f"{tm.id}: required action not performed")

        payload = final.payload or {}
        self.log.emit(
            "hand_off",
            task=tm.id,
            to=tm.hand_off.to,
            fusion_level=tm.hand_off.fusion_level.value,
            tokens=estimate_tokens(payload),
            budget=tm.hand_off.max_tokens,
            payload=payload,
        )
        self.results[tm.id] = payload
        return payload

    # -------------------------------------------------------------- observe

    def _observe(self, tm: TaskModel, subtask_results: dict[str, dict[str, Any]]) -> dict[str, Any]:
        values: dict[str, Any] = {}
        for o in tm.observations:
            if o.name in self.sensors:
                values[o.name] = self.sensors[o.name](self.world, subtask_results)
                continue
            found = [r[o.name] for r in subtask_results.values() if isinstance(r, dict) and o.name in r]
            if not found:
                found = [r[o.name] for r in self.results.values() if isinstance(r, dict) and o.name in r]
            if found:
                values[o.name] = found[-1]
            else:
                values[o.name] = None
                self.log.emit("missing_observation", task=tm.id, observation=o.name)
        return values

    # --------------------------------------------------------------- decide

    def _ask_person(
        self,
        tm: TaskModel,
        ctx: DecisionContext,
        features: dict[str, Any],
        fkey: Optional[tuple],
    ) -> Proposal:
        # Design/version and canonical orientation are internal match keys,
        # not additional implementation details for a person to inspect.
        request = judgement_request(ctx, {k: v for k, v in features.items() if not k.startswith("__")})
        self.log.emit(
            "person_request",
            task=tm.id,
            tokens_shown=estimate_tokens(request),
            why=ctx.reasons_so_far,
        )
        p = self.person.decide(ctx, request)
        problems = validate(p.payload, tm.hand_off)
        if problems:
            raise DesignFault(f"{tm.id}: person's answer fails its hand-off spec: {problems}")
        self.log.emit("decided", task=tm.id, tier=Tier.T3.value, by=p.decided_by)
        if p.make_policy and fkey is not None:
            self.policies.add(
                tm.id,
                fkey,
                p.payload or {},
                set_by=self.person.name,
                rationale=p.policy_rationale,
            )
            self.log.emit("policy_set", task=tm.id, features=features, decision=p.payload)
        return p

    def _decide(
        self,
        tm: TaskModel,
        ctx: DecisionContext,
        alloc: Allocation,
        features: dict[str, Any],
        fkey: Optional[tuple],
    ) -> tuple[Proposal, Tier]:
        threshold = default_threshold(tm)
        ladder = TIER_ORDER[TIER_ORDER.index(alloc.tier):]

        for i, tier in enumerate(ladder):
            if tier == Tier.T3:
                return self._ask_person(tm, ctx, features, fkey), Tier.T3

            executor = self.executors.get(tier)
            if executor is None or not executor.can_decide(tm):
                self.log.emit("skip", task=tm.id, tier=tier.value, reason="no executor at this tier")
                continue

            p = executor.decide(ctx)
            threshold = self.confidence_thresholds.get((tm.id, tier), default_threshold(tm))
            if tier in (Tier.T1, Tier.T2):
                self.log.emit(
                    "call",
                    task=tm.id,
                    tier=tier.value,
                    by=p.decided_by,
                    tokens_in=p.tokens_in,
                    tokens_out=p.tokens_out,
                    confidence=p.confidence,
                )

            reason: Optional[str] = None
            if not p.decided:
                reason = p.reason or "abstained"
            else:
                block = blocks_machine_decision(tm.decision.kind == DecisionKind.NORMATIVE, has_policy=False)
                if block:
                    reason = block
                    self.log.emit("boundary", task=tm.id, tier=tier.value, reason=block)
                elif tier in (Tier.T1, Tier.T2) and (p.confidence is None or p.confidence < threshold):
                    reason = f"confidence {p.confidence} below threshold {threshold}"
                else:
                    problems = validate(p.payload, tm.hand_off)
                    if problems and tier == Tier.T0:
                        raise DesignFault(f"{tm.id}: tool output fails its hand-off spec: {problems}")
                    if problems:
                        reason = "hand-off failed: " + "; ".join(problems)

            if reason is None and alloc.action_level == ActionLevel.PROPOSE_ONLY:
                self.log.emit("proposal", task=tm.id, tier=tier.value, payload=p.payload)
                ctx.lower_proposals.append(p)
                ctx.reasons_so_far.append(f"{tier.value} proposes; complex domain, so people decide (clause E)")
                return self._ask_person(tm, ctx, features, fkey), Tier.T3

            if reason is None:
                self.log.emit("decided", task=tm.id, tier=tier.value, by=p.decided_by, confidence=p.confidence)
                return p, tier

            self.log.emit("abstain", task=tm.id, tier=tier.value, reason=reason)
            ctx.lower_proposals.append(p)
            ctx.reasons_so_far.append(f"{tier.value} abstained: {reason}")
            self.log.emit("escalate", task=tm.id, frm=tier.value, to=ladder[i + 1].value)

        raise AssertionError("the ladder always ends with people")

    # ------------------------------------------------------------------ act

    def _act(
        self,
        tm: TaskModel,
        ctx: DecisionContext,
        alloc: Allocation,
        final: Proposal,
        final_tier: Tier,
        policy_applied: bool,
    ) -> bool:
        level = alloc.action_level
        payload = final.payload or {}
        if final_tier == Tier.T3:
            level = ActionLevel.PERSON_ACTS
        elif policy_applied:
            level = ActionLevel.ACT_AND_REPORT

        if level == ActionLevel.ACT_WITH_APPROVAL:
            request = {
                "task_id": tm.id,
                "task": tm.name,
                "action": tm.action.name,
                "payload": payload,
                "reason": final.reason,
            }
            self.log.emit("approval_request", task=tm.id, tokens_shown=estimate_tokens(request))
            granted = self.person.responder.approve(request)
            self.log.emit("approval", task=tm.id, granted=granted)
            if not granted:
                self.log.emit("not_acted", task=tm.id, reason="approval refused")
                return False
        elif level == ActionLevel.ACT_UNLESS_VETOED:
            notice = {"task_id": tm.id, "task": tm.name, "action": tm.action.name, "payload": payload}
            self.log.emit("notice", task=tm.id, tokens_shown=estimate_tokens(notice))
            if self.person.responder.veto(notice):
                self.log.emit("not_acted", task=tm.id, reason="vetoed")
                return False

        actuator = self.actuators.get(tm.action.name)
        if actuator is None:
            raise DesignFault(f"{tm.id}: no actuator registered for {tm.action.name!r}")
        actuator(self.world, payload, ctx)
        self.log.emit("acted", task=tm.id, action=tm.action.name, level=level.value, by_tier=final_tier.value)
        return True
