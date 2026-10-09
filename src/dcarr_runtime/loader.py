"""Load and check an M1 task model set written in YAML."""

from __future__ import annotations

from pathlib import Path

import yaml

from .allocation import allocate
from .metamodel import TaskModelSet


def load_task_models(path: str | Path) -> TaskModelSet:
    data = yaml.safe_load(Path(path).read_text())
    return TaskModelSet.model_validate(data)


def allocation_table(model_set: TaskModelSet) -> list[dict[str, str]]:
    """One row per task model: the design-time allocation and why."""
    rows = []
    for tm in model_set.task_models:
        a = allocate(tm)
        rows.append(
            {
                "id": tm.id,
                "task model": tm.name,
                "decision": tm.decision.kind.value,
                "describability": tm.decision.describability.value,
                "cynefin": tm.cynefin.value,
                "control": tm.control_level.value,
                "clause": a.clause,
                "tier": a.tier.value,
                "action level": a.action_level.value,
            }
        )
    return rows
