"""DCARR runtime: task models from DCARR's Describe phase, run across
deterministic tools, small models, reasoning models and people."""

from .allocation import Allocation, allocate, default_threshold
from .engine import DesignFault, Engine
from .eventlog import EventLog
from .executors import (
    AnthropicBackend,
    ConsoleResponder,
    DecisionContext,
    ModelExecutor,
    PersonExecutor,
    PiBackend,
    Proposal,
    ScriptedBackend,
    ToolExecutor,
)
from .loader import allocation_table, load_task_models
from .metamodel import TaskModel, TaskModelSet, Tier
from .policy import PolicyStore

__all__ = [
    "Allocation",
    "AnthropicBackend",
    "ConsoleResponder",
    "DecisionContext",
    "DesignFault",
    "Engine",
    "EventLog",
    "ModelExecutor",
    "PersonExecutor",
    "PiBackend",
    "PolicyStore",
    "Proposal",
    "ScriptedBackend",
    "TaskModel",
    "TaskModelSet",
    "Tier",
    "ToolExecutor",
    "allocate",
    "allocation_table",
    "default_threshold",
    "load_task_models",
]
