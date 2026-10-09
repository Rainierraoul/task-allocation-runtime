"""Hand-offs: what a task model passes to its parent or to people.

A hand-off must contain exactly the declared fields, with the declared types,
and fit the declared token budget. A model tier whose output fails these
checks is treated as abstaining; a deterministic tool that fails them has a
design fault, and the run stops.

Token counts are estimates (characters / 4, rounded up) so the runtime needs
no tokenizer. Real backends report their own counts, which the event log
records alongside.
"""

from __future__ import annotations

import json
import math
from typing import Any

from .metamodel import HandOff

_TYPE_CHECKS = {
    "bool": lambda v: isinstance(v, bool),
    "number": lambda v: isinstance(v, (int, float)) and not isinstance(v, bool) and math.isfinite(v),
    "text": lambda v: isinstance(v, str),
    "enum": lambda v: isinstance(v, str),
    "list": lambda v: isinstance(v, list),
}


def estimate_tokens(obj: Any) -> int:
    text = obj if isinstance(obj, str) else json.dumps(obj, ensure_ascii=False, default=str)
    return math.ceil(len(text) / 4)


def validate(payload: Any, spec: HandOff) -> list[str]:
    problems: list[str] = []
    if not isinstance(payload, dict):
        return ["hand-off is not a set of named fields"]
    missing = [f for f in spec.fields if f not in payload]
    extra = [f for f in payload if f not in spec.fields]
    if missing:
        problems.append(f"missing fields {missing}")
    if extra:
        problems.append(f"undeclared fields {extra}")
    for name, kind in spec.fields.items():
        if name in payload and not _TYPE_CHECKS[kind](payload[name]):
            problems.append(f"field {name!r} is not {kind}")
    for name, allowed in spec.enum_values.items():
        if name in payload and isinstance(payload[name], str) and payload[name] not in allowed:
            problems.append(f"field {name!r} is not one of {allowed}")
    size = estimate_tokens(payload)
    if size > spec.max_tokens:
        problems.append(f"{size} tokens exceeds the budget of {spec.max_tokens}")
    return problems
