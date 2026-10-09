"""The rule that keeps normative decisions with people.

Machine tiers may make epistemic decisions: establishing what is the case.
A normative decision (a choice about what ought to happen) is made by people,
or by applying a policy that people set earlier for the same situation.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional


@dataclass(frozen=True)
class NormativePolicy:
    """A choice people made, stored so the same situation doesn't come back to them."""

    task_id: str
    features: tuple[tuple[str, Any], ...]
    decision: dict[str, Any]
    set_by: str
    set_on: str
    rationale: str = ""
    scope: str = ""

    def key(self) -> tuple[str, tuple[tuple[str, Any], ...]]:
        return (self.task_id, self.features)


def blocks_machine_decision(is_normative: bool, has_policy: bool) -> Optional[str]:
    """Return the reason a machine tier may not decide, or None if it may."""
    if is_normative and not has_policy:
        return "normative decision with no covering policy: people decide"
    return None
