"""Executors: the things that make a decision at each tier.

T0  ToolExecutor    registered Python functions (state machines, algorithms)
T1  ModelExecutor   a small model behind a backend
T2  ModelExecutor   a reasoning model behind a backend
T3  PersonExecutor  a person, through a responder (a console, a situation
                    board, or a scripted responder in tests)

Model tiers see only what their task model declares: its goal, its rule, its
observations and orientation entities, its subtasks' hand-offs and any
proposal passed up from the tier below. They never see the run's transcript.

Backends:
  ScriptedBackend    deterministic stand-ins for tests and worked examples;
                     not language models
  AnthropicBackend   the Anthropic Messages API (needs the anthropic package
                     and ANTHROPIC_API_KEY); untested in this repository
  PiBackend          the pi agent harness in RPC mode; untested, and the
                     field names for streamed text are read defensively
"""

from __future__ import annotations

import json
import math
import subprocess
from dataclasses import dataclass, field
from typing import Any, Callable, Optional, Protocol

from .boundary import NormativePolicy
from .handoff import estimate_tokens
from .metamodel import TaskModel, Tier

MODEL_SYSTEM_PROMPT = """\
You make one decision inside a larger task. You are given only what this decision needs.
Reply with one JSON object and nothing else, with these keys:
  "decided": true or false
  "payload": an object with exactly the hand-off fields listed, or null if you do not decide
  "confidence": a number from 0 to 1: how likely your decision is correct
  "reason": one sentence
  "framings": when you do not decide, a list of options, each {"option": ..., "case_for": ...}
Set "decided" to false when any abstain condition holds, when the information given is not
enough, or when the choice is about what ought to happen rather than what is the case.
People make those choices. Keep the payload within the token budget."""


@dataclass
class DecisionContext:
    task: TaskModel
    observations: dict[str, Any]
    orientations: dict[str, Any]
    subtask_results: dict[str, dict[str, Any]]
    policy: Optional[NormativePolicy] = None
    lower_proposals: list["Proposal"] = field(default_factory=list)
    reasons_so_far: list[str] = field(default_factory=list)
    world: Any = None


@dataclass
class Proposal:
    decided: bool
    payload: Optional[dict[str, Any]] = None
    confidence: Optional[float] = None
    reason: str = ""
    framings: list[dict[str, Any]] = field(default_factory=list)
    make_policy: bool = False
    policy_rationale: str = ""
    decided_by: str = ""
    tokens_in: int = 0
    tokens_out: int = 0


class Executor(Protocol):
    tier: Tier

    def can_decide(self, task: TaskModel) -> bool: ...

    def decide(self, ctx: DecisionContext) -> Proposal: ...


# ----------------------------------------------------------------------------
# T0: deterministic tools

ToolFn = Callable[[DecisionContext], Proposal]


class ToolExecutor:
    tier = Tier.T0

    def __init__(self, tools: dict[str, ToolFn]):
        self.tools = tools

    def can_decide(self, task: TaskModel) -> bool:
        return task.executors.get(Tier.T0) in self.tools

    def decide(self, ctx: DecisionContext) -> Proposal:
        p = self.tools[ctx.task.executors[Tier.T0]](ctx)
        p.decided_by = f"tool:{ctx.task.executors[Tier.T0]}"
        return p


# ----------------------------------------------------------------------------
# T1 and T2: models


def model_context(ctx: DecisionContext) -> dict[str, Any]:
    """Everything a model tier sees for one decision, and nothing else."""
    tm = ctx.task
    out: dict[str, Any] = {
        "task_id": tm.id,
        "task": tm.name,
        "goal": tm.goal,
        "decision_rule": tm.decision.rule,
        "abstain_when": tm.decision.abstain_when,
        "observations": ctx.observations,
        "orientation": ctx.orientations,
        "subtask_results": ctx.subtask_results,
        "hand_off_fields": tm.hand_off.fields,
        "hand_off_token_budget": tm.hand_off.max_tokens,
    }
    if tm.hand_off.enum_values:
        out["hand_off_allowed_values"] = tm.hand_off.enum_values
    if ctx.lower_proposals:
        last = ctx.lower_proposals[-1]
        out["lower_tier_result"] = {
            "decided": last.decided,
            "payload": last.payload,
            "confidence": last.confidence,
            "reason": last.reason,
        }
    return out


class Backend(Protocol):
    label: str

    def respond(self, executor_name: str, context: dict[str, Any]) -> tuple[dict[str, Any], int, int]: ...


class ScriptedBackend:
    """Deterministic stand-ins keyed by executor name. Not a language model."""

    label = "scripted stand-in"

    def __init__(self, handlers: dict[str, Callable[[dict[str, Any]], dict[str, Any]]]):
        self.handlers = handlers

    def respond(self, executor_name: str, context: dict[str, Any]) -> tuple[dict[str, Any], int, int]:
        response = self.handlers[executor_name](context)
        tokens_in = estimate_tokens(MODEL_SYSTEM_PROMPT) + estimate_tokens(context)
        return response, tokens_in, estimate_tokens(response)

    def has(self, executor_name: str) -> bool:
        return executor_name in self.handlers


def _parse_json_object(text: str) -> dict[str, Any]:
    """The model's answer: the first complete JSON object with a "decided" field (protocol v1.1).

    Each opening brace is tried in turn, so prose before the answer, even prose with a stray
    brace, and text after it are ignored. A reply with no such object, as in the single-model
    condition's answer keyed by task, falls back to the first complete object. Objects
    containing NaN or Infinity never decode. Validation after parsing is unchanged.
    """
    def reject_constant(value):
        raise ValueError(f"non-finite JSON constant {value}")

    decoder = json.JSONDecoder(parse_constant=reject_constant)
    first = None
    start = text.find("{")
    if start < 0:
        return {"decided": False, "reason": "reply was not a JSON object"}
    while start >= 0:
        try:
            result, _ = decoder.raw_decode(text, start)
        except ValueError:
            result = None
        if isinstance(result, dict):
            if "decided" in result:
                return result
            if first is None:
                first = result
        start = text.find("{", start + 1)
    return first if first is not None else {"decided": False, "reason": "reply was not valid JSON"}


class AnthropicBackend:
    """Untested here: no API key in this environment."""

    def __init__(self, model: str, max_tokens: int = 1024):
        import anthropic  # imported lazily so tests don't need it

        self.client = anthropic.Anthropic()
        self.model = model
        self.max_tokens = max_tokens
        self.label = f"anthropic:{model}"

    def respond(self, executor_name: str, context: dict[str, Any]) -> tuple[dict[str, Any], int, int]:
        msg = self.client.messages.create(
            model=self.model,
            max_tokens=self.max_tokens,
            system=MODEL_SYSTEM_PROMPT,
            messages=[{"role": "user", "content": json.dumps(context, default=str)}],
        )
        text = "".join(getattr(b, "text", "") for b in msg.content)
        return _parse_json_object(text), msg.usage.input_tokens, msg.usage.output_tokens

    def has(self, executor_name: str) -> bool:
        return True


class PiBackend:
    """The pi harness in RPC mode (JSON lines over stdin and stdout). Untested here.

    Starts `pi --mode rpc --no-session --provider P --model M`, sends one
    prompt command and collects streamed text until the run ends.
    """

    def __init__(self, provider: str, model: str, pi_cmd: str = "pi", timeout_s: float = 120):
        self.cmd = [pi_cmd, "--mode", "rpc", "--no-session", "--provider", provider, "--model", model]
        self.timeout_s = timeout_s
        self.label = f"pi:{provider}/{model}"

    @staticmethod
    def _delta_text(event: dict[str, Any]) -> str:
        for key in ("delta", "text", "content"):
            value = event.get(key)
            if isinstance(value, str):
                return value
            if isinstance(value, dict):
                inner = value.get("text") or value.get("delta")
                if isinstance(inner, str):
                    return inner
        return ""

    def respond(self, executor_name: str, context: dict[str, Any]) -> tuple[dict[str, Any], int, int]:
        prompt = MODEL_SYSTEM_PROMPT + "\n\n" + json.dumps(context, default=str)
        proc = subprocess.Popen(self.cmd, stdin=subprocess.PIPE, stdout=subprocess.PIPE, text=True)
        assert proc.stdin and proc.stdout
        proc.stdin.write(json.dumps({"id": executor_name, "type": "prompt", "message": prompt}) + "\n")
        proc.stdin.flush()
        parts: list[str] = []
        for line in proc.stdout:
            try:
                event = json.loads(line)
            except json.JSONDecodeError:
                continue
            if event.get("type") == "message_update":
                parts.append(self._delta_text(event))
            if event.get("type") in ("agent_end", "agent_settled"):
                break
        proc.terminate()
        text = "".join(parts)
        return _parse_json_object(text), estimate_tokens(prompt), estimate_tokens(text)

    def has(self, executor_name: str) -> bool:
        return True


class ModelExecutor:
    def __init__(self, tier: Tier, backend: Any):
        assert tier in (Tier.T1, Tier.T2)
        self.tier = tier
        self.backend = backend

    def can_decide(self, task: TaskModel) -> bool:
        name = task.executors.get(self.tier)
        return name is not None and self.backend.has(name)

    def decide(self, ctx: DecisionContext) -> Proposal:
        name = ctx.task.executors[self.tier]
        response, tokens_in, tokens_out = self.backend.respond(name, model_context(ctx))
        valid = isinstance(response, dict) and type(response.get("decided")) is bool
        confidence = response.get("confidence") if isinstance(response, dict) else None
        if confidence is not None and (
            isinstance(confidence, bool) or not isinstance(confidence, (int, float))
            or not math.isfinite(confidence) or not 0 <= confidence <= 1
        ):
            valid = False
        if isinstance(response, dict) and (not isinstance(response.get("reason", ""), str)
                or not isinstance(response.get("framings", []), list)
                or any(not isinstance(item, dict) for item in response.get("framings", []))):
            valid = False
        if not valid:
            return Proposal(decided=False, reason="invalid model response or confidence",
                            decided_by=f"{self.backend.label}:{name}",
                            tokens_in=tokens_in, tokens_out=tokens_out)
        return Proposal(
            decided=response["decided"],
            payload=response.get("payload"),
            confidence=response.get("confidence"),
            reason=response.get("reason", ""),
            framings=response.get("framings", []) or [],
            decided_by=f"{self.backend.label}:{name}",
            tokens_in=tokens_in,
            tokens_out=tokens_out,
        )


# ----------------------------------------------------------------------------
# T3: people


def judgement_request(ctx: DecisionContext, policy_features: dict[str, Any]) -> dict[str, Any]:
    """What a person is shown when a decision reaches them."""
    tm = ctx.task
    request: dict[str, Any] = {
        "task_id": tm.id,
        "task": tm.name,
        "question": tm.decision.rule,
        "why_you": ctx.reasons_so_far,
        "situation": {"observations": ctx.observations, "subtask_results": ctx.subtask_results},
        "answer_fields": tm.hand_off.fields,
    }
    if ctx.lower_proposals:
        last = ctx.lower_proposals[-1]
        request["proposal"] = {"payload": last.payload, "reason": last.reason}
        framings = [f for p in ctx.lower_proposals for f in p.framings]
        if framings:
            request["options"] = framings
    if policy_features:
        request["can_become_policy_for"] = policy_features
    return request


class Responder(Protocol):
    def decide(self, request: dict[str, Any]) -> Proposal: ...

    def approve(self, request: dict[str, Any]) -> bool: ...

    def veto(self, notice: dict[str, Any]) -> bool: ...


class PersonExecutor:
    tier = Tier.T3

    def __init__(self, responder: Responder, name: str = "person"):
        self.responder = responder
        self.name = name

    def can_decide(self, task: TaskModel) -> bool:
        return True

    def decide(self, ctx: DecisionContext, request: Optional[dict[str, Any]] = None) -> Proposal:
        p = self.responder.decide(request or judgement_request(ctx, {}))
        p.decided = True
        p.decided_by = f"person:{self.name}"
        return p


class ConsoleResponder:
    """Asks at the terminal. For trying the runtime by hand."""

    def decide(self, request: dict[str, Any]) -> Proposal:
        print("\n--- decision needed ---")
        print(json.dumps(request, indent=2, default=str))
        raw = input("Answer as JSON with the answer fields: ")
        payload = json.loads(raw)
        keep = input("Store this as a policy for the same situation? [y/N] ").strip().lower() == "y"
        return Proposal(decided=True, payload=payload, make_policy=keep, reason="entered at console")

    def approve(self, request: dict[str, Any]) -> bool:
        print("\n--- approval needed ---")
        print(json.dumps(request, indent=2, default=str))
        return input("Approve? [y/N] ").strip().lower() == "y"

    def veto(self, notice: dict[str, Any]) -> bool:
        print(f"\n[notice] {notice.get('task')}: {notice.get('action')} {notice.get('payload')}")
        return False
