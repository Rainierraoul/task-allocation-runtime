"""Stored policies: people's past decisions, keyed by task and situation.

A policy applies only when every feature the task model names in
decision.policy_features has the same value as when the person decided.
There is no fuzzy matching: a new situation goes back to people.
"""

from __future__ import annotations

import json
from datetime import date
from pathlib import Path
from typing import Any, Optional

from .boundary import NormativePolicy


def feature_key(features: dict[str, Any], names: list[str]) -> tuple[tuple[str, Any], ...]:
    return tuple((n, features.get(n)) for n in sorted(names))


class PolicyStore:
    def __init__(self, path: Optional[Path] = None):
        self.path = Path(path) if path else None
        self._policies: dict[tuple, NormativePolicy] = {}
        if self.path and self.path.exists():
            for row in json.loads(self.path.read_text()):
                p = NormativePolicy(
                    task_id=row["task_id"],
                    features=tuple((k, v) for k, v in row["features"]),
                    decision=row["decision"],
                    set_by=row["set_by"],
                    set_on=row["set_on"],
                    rationale=row.get("rationale", ""),
                    scope=row.get("scope", ""),
                )
                self._policies[p.key()] = p

    def __len__(self) -> int:
        return len(self._policies)

    def match(self, task_id: str, features: tuple[tuple[str, Any], ...]) -> Optional[NormativePolicy]:
        return self._policies.get((task_id, features))

    def add(
        self,
        task_id: str,
        features: tuple[tuple[str, Any], ...],
        decision: dict[str, Any],
        set_by: str,
        rationale: str = "",
        scope: str = "",
        set_on: Optional[str] = None,
    ) -> NormativePolicy:
        p = NormativePolicy(
            task_id=task_id,
            features=features,
            decision=dict(decision),
            set_by=set_by,
            set_on=set_on or date.today().isoformat(),
            rationale=rationale,
            scope=scope,
        )
        self._policies[p.key()] = p
        self.save()
        return p

    def all(self) -> list[NormativePolicy]:
        return list(self._policies.values())

    def save(self) -> None:
        if not self.path:
            return
        rows = [
            {
                "task_id": p.task_id,
                "features": [list(f) for f in p.features],
                "decision": p.decision,
                "set_by": p.set_by,
                "set_on": p.set_on,
                "rationale": p.rationale,
                "scope": p.scope,
            }
            for p in self._policies.values()
        ]
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text(json.dumps(rows, indent=2))
