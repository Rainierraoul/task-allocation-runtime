"""M0: the record of one run.

Every stage of every task model writes an event. The log is the instance
model in the thesis's metamodel stack, and the data source for the
evaluation studies (tokens, calls per tier, escalations, what people saw).
"""

from __future__ import annotations

import json
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


class EventLog:
    def __init__(self, run_id: str):
        self.run_id = run_id
        self.events: list[dict[str, Any]] = []
        self._t0 = time.perf_counter()

    def emit(self, kind: str, **fields: Any) -> dict[str, Any]:
        event = {
            "run": self.run_id,
            "seq": len(self.events),
            "ms": round((time.perf_counter() - self._t0) * 1000, 3),
            "kind": kind,
            **fields,
        }
        self.events.append(event)
        return event

    def of(self, kind: str) -> list[dict[str, Any]]:
        return [e for e in self.events if e["kind"] == kind]

    def write(self, path: Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w") as f:
            for e in self.events:
                f.write(json.dumps(e, default=str) + "\n")

    def metrics(self) -> dict[str, Any]:
        decided = Counter(e["tier"] for e in self.of("decided"))
        calls = Counter(e["tier"] for e in self.of("call"))
        tokens_in: dict[str, int] = defaultdict(int)
        tokens_out: dict[str, int] = defaultdict(int)
        max_context = 0
        for e in self.of("call"):
            tokens_in[e["tier"]] += e.get("tokens_in", 0)
            tokens_out[e["tier"]] += e.get("tokens_out", 0)
            max_context = max(max_context, e.get("tokens_in", 0))
        shown = sum(e.get("tokens_shown", 0) for e in self.events if e["kind"] in ("person_request", "notice", "approval_request"))
        reports = sum(e.get("tokens", 0) for e in self.of("hand_off") if e.get("to") == "people")
        return {
            "decisions_by_tier": dict(sorted(decided.items())),
            "calls_by_tier": dict(sorted(calls.items())),
            "abstentions": len(self.of("abstain")),
            "escalations": len(self.of("escalate")),
            "boundary_blocks": len(self.of("boundary")),
            "person_requests": len(self.of("person_request")),
            "approval_requests": len(self.of("approval_request")),
            "notices": len(self.of("notice")),
            "policies_applied": len(self.of("policy_applied")),
            "policies_set": len(self.of("policy_set")),
            "tokens_in_by_tier": dict(sorted(tokens_in.items())),
            "tokens_out_by_tier": dict(sorted(tokens_out.items())),
            "max_context_tokens": max_context,
            "tokens_shown_to_people": shown,
            "estimated_tokens_in_people_reports": reports,
            "estimated_total_text_to_people": shown + reports,
        }
