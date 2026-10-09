"""Build and run Scenario 5.

    python -m scenarios.simpilot.run --variant clean --policies out/policies.json

Variants:
  clean      the instruction as given in the thesis
  ambiguous  'Aircraft two or three, ...': the small model is unsure who is addressed
  jammed     the primary channel is jammed during a scheduled exercise inject
"""

from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path
from typing import Optional

from dcarr_runtime import (
    Engine,
    EventLog,
    ModelExecutor,
    PersonExecutor,
    PolicyStore,
    ScriptedBackend,
    Tier,
    ToolExecutor,
    allocation_table,
    load_task_models,
)

from .stand_ins import HANDLERS, ScriptedPerson
from .tools import ACTUATORS, ORIENTATION_STORE, SENSORS, TOOLS
from .world import World

HERE = Path(__file__).parent
CLEAN = "Aircraft 2, make your way to the fuel tanker, contact me when you are 20 nautical miles out."
AMBIGUOUS = "Aircraft two or three, make your way to the fuel tanker, contact me when you are 20 nautical miles out."


def make_world(variant: str) -> tuple[World, dict]:
    orientation = copy.deepcopy(ORIENTATION_STORE)
    if variant == "clean":
        world = World.sample(CLEAN)
    elif variant == "ambiguous":
        world = World.sample(AMBIGUOUS)
    elif variant == "jammed":
        world = World.sample(CLEAN, channel_state="jammed", jamming_inject_scheduled=True)
        orientation["exercise_injects"] = [{"type": "jamming", "channel": "Channel 5", "window": "sample"}]
    else:
        raise ValueError(f"unknown variant {variant!r}")
    return world, orientation


def build(variant: str, policies: PolicyStore, backend=None, person=None, run_id: Optional[str] = None):
    model_set = load_task_models(HERE / "task_models.yaml")
    world, orientation = make_world(variant)
    backend = backend or ScriptedBackend(HANDLERS)
    person = person or ScriptedPerson()
    log = EventLog(run_id or f"simpilot-{variant}")
    engine = Engine(
        model_set=model_set,
        executors={
            Tier.T0: ToolExecutor(TOOLS),
            Tier.T1: ModelExecutor(Tier.T1, backend),
            Tier.T2: ModelExecutor(Tier.T2, backend),
        },
        person=PersonExecutor(person, name="White Force"),
        policies=policies,
        log=log,
        sensors=SENSORS,
        orientation_store=orientation,
        actuators=ACTUATORS,
        world=world,
    )
    return engine, world, person, log


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--variant", default="clean", choices=["clean", "ambiguous", "jammed"])
    parser.add_argument("--policies", default=None, help="policy store file; kept between runs")
    parser.add_argument("--log", default=None, help="write the event log (JSON lines) here")
    parser.add_argument("--allocation", action="store_true", help="print the design-time allocation and stop")
    args = parser.parse_args()

    if args.allocation:
        for row in allocation_table(load_task_models(HERE / "task_models.yaml")):
            print(" | ".join(f"{v}" for v in row.values()))
        return

    policies = PolicyStore(Path(args.policies) if args.policies else None)
    engine, world, person, log = build(args.variant, policies)
    result = engine.run()
    print(json.dumps(result, indent=2))
    print(json.dumps(log.metrics(), indent=2))
    if args.log:
        log.write(Path(args.log))


if __name__ == "__main__":
    main()
