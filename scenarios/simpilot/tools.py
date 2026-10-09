"""Scenario 5 sensors, orientation entities, deterministic tools (T0) and actuators."""

from __future__ import annotations

import math
import copy
from typing import Any

from dcarr_runtime.executors import DecisionContext, Proposal

from .world import World

# ---------------------------------------------------------------- sensors
# Each returns the current value of one declared observation.

SENSORS = {
    "transmission": lambda w, _: w.transmission,
    "trainee_level": lambda w, _: w.trainee_level,
    "deviation_inject_scheduled": lambda w, _: w.deviation_inject_scheduled,
    "entity_list": lambda w, _: w.entities(),
    "aircraft2_state": lambda w, _: {
        "x": round(w.aircraft["aircraft 2"].x, 2),
        "y": round(w.aircraft["aircraft 2"].y, 2),
        "speed_kts": w.aircraft["aircraft 2"].speed_kts,
    },
    "tanker_state": lambda w, _: {"callsign": w.tanker.callsign, "x": round(w.tanker.x, 2), "y": round(w.tanker.y, 2)},
    "distance_nm": lambda w, _: round(w.distance_to_tanker(), 2),
    "channel_state": lambda w, _: w.channel_state,
    "jamming_inject_scheduled": lambda w, _: w.jamming_inject_scheduled,
}

# ---------------------------------------------------- orientation entities
# Queryable data structures (DCARR Step 17). Sample values.

ORIENTATION_STORE: dict[str, Any] = {
    "own_callsigns": ["aircraft 1", "aircraft 2"],
    "command_lexicon": {
        "join_tanker": ["make your way to the fuel tanker", "join the tanker", "proceed to the tanker"],
    },
    "training_objectives": "Trainee practises directing an aircraft to a tanker and handling range reports.",
    "exercise_plan": {"sortie": "sample sortie 3", "trainee_level": "initial_employment"},
    "briefing_material": {"tanker_callsign": "SHELL 1", "orbit": "8 NM racetrack, 300 kts"},
    "intercept_method": "Lead pursuit to the predicted orbit position",
    "radio_procedure": "Callsign of station called, own callsign, message",
    "comms_plan": {"primary": "Channel 5", "backup": "Channel 7"},
    "exercise_injects": [],
}

# -------------------------------------------------------------- T0 tools


def _reject(reason: str) -> Proposal:
    return Proposal(decided=False, reason=reason)


def locate_tanker(ctx: DecisionContext) -> Proposal:
    callsign = ctx.orientations["briefing_material"]["tanker_callsign"]
    if callsign not in ctx.observations["entity_list"]:
        return _reject(f"{callsign} is not in the entity list: input outside the state machine")
    orbit = ctx.orientations["briefing_material"]["orbit"]
    return Proposal(decided=True, payload={"tanker": callsign, "tanker_found": True, "orbit": orbit})


def select_entity(ctx: DecisionContext) -> Proposal:
    target = ctx.observations.get("addressee")
    if target not in (ctx.observations.get("entity_list") or []):
        return _reject(f"{target!r} is not in the entity list")
    return Proposal(decided=True, payload={"selected": target})


def intercept_heading(ctx: DecisionContext) -> Proposal:
    w: World = ctx.world
    a = w.aircraft["aircraft 2"]
    assert w.tanker
    speed_nm_s = a.speed_kts / 3600.0
    t = math.hypot(w.tanker.x - a.x, w.tanker.y - a.y) / speed_nm_s
    px, py = w.tanker.x, w.tanker.y
    for _ in range(25):
        px, py = w.tanker.position_after(t)
        t = math.hypot(px - a.x, py - a.y) / speed_nm_s
    heading = math.degrees(math.atan2(px - a.x, py - a.y)) % 360
    return Proposal(
        decided=True,
        payload={"heading_deg": round(heading, 1), "time_to_intercept_min": round(t / 60, 1)},
    )


def configure_tool(ctx: DecisionContext) -> Proposal:
    a, b = ctx.observations.get("selected"), ctx.observations.get("tanker")
    if not a or not b:
        return _reject("an end point for the distance line is missing")
    return Proposal(decided=True, payload={"tool_ready": True, "line": f"{a} to {b}"})


def wait_for_range(ctx: DecisionContext) -> Proposal:
    """Observes the distance every 5 simulated seconds until it reaches the report distance."""
    # Predict against a copy; the Action stage owns the simulated time advance.
    # A veto must not leave the aircraft moved by an unperformed action.
    w: World = copy.deepcopy(ctx.world)
    report_at = ctx.observations.get("report_at_nm")
    if not isinstance(report_at, (int, float)):
        return _reject("no reporting distance")
    start = w.time_s
    last_minute_distance = w.distance_to_tanker()
    not_closing = 0
    while w.distance_to_tanker() > report_at:
        w.step(5.0)
        if int(w.time_s - start) % 60 == 0:
            d = w.distance_to_tanker()
            not_closing = not_closing + 1 if d >= last_minute_distance else 0
            last_minute_distance = d
            if not_closing >= 3:
                return _reject("the distance has not closed for 3 minutes")
        if w.time_s - start > 45 * 60:
            return _reject("the reporting distance was not reached within 45 minutes")
    return Proposal(
        decided=True,
        payload={
            "at_range": True,
            "distance_nm": round(w.distance_to_tanker(), 1),
            "elapsed_min": round((w.time_s - start) / 60, 6),
        },
    )


def goal_check(ctx: DecisionContext) -> Proposal:
    w: World = ctx.world
    obs = ctx.observations
    d = obs["distance_nm"]
    met = d is not None and obs["report_at_nm"] is not None and d <= obs["report_at_nm"] and obs["call_status"] == "made"
    tanker = ctx.orientations["briefing_material"]["tanker_callsign"]
    summary = f"Aircraft 2 is {d:.1f} NM from {tanker} at {w.clock()}. Call {obs['call_status']}: {obs['message']}"
    return Proposal(
        decided=True,
        payload={
            "summary": summary,
            "goal_met": bool(met),
            "distance_nm": round(d, 1),
            "plan_choice": obs.get("choice") or "not_applicable",
            "next": "Await control's next instruction" if met else "Goal not met: check the call",
        },
    )


TOOLS = {
    "locate_tanker": locate_tanker,
    "select_entity": select_entity,
    "intercept_heading": intercept_heading,
    "configure_tool": configure_tool,
    "wait_for_range": wait_for_range,
    "goal_check": goal_check,
}

# -------------------------------------------------------------- actuators


def _radio(w: World, text: str, frequency: str = "") -> None:
    w.radio_log.append((w.time_s, frequency or "Channel 5", text))


def transmit_call(w: World, payload: dict, ctx: DecisionContext) -> None:
    _radio(w, payload["message"], payload.get("frequency", ""))
    w.control_acknowledged = payload.get("call_status") == "made"


ACTUATORS = {
    "transmit_acknowledgement": lambda w, p, c: _radio(w, f"{p['addressee']}, wilco"),
    "record_deviation_plan": lambda w, p, c: w.plan.update({"deviation": p["choice"]}),
    "centre_map_on_tanker": lambda w, p, c: setattr(w, "map_centre", p["tanker"]),
    "select_entity_on_console": lambda w, p, c: setattr(w, "selected", p["selected"]),
    "set_heading": lambda w, p, c: setattr(w.aircraft["aircraft 2"], "heading_deg", p["heading_deg"]),
    "configure_distance_tool": lambda w, p, c: setattr(w, "distance_line", tuple(p["line"].split(" to "))),
    "hold_until_range": lambda w, p, c: w.step(p["elapsed_min"] * 60),
    "transmit_call": transmit_call,
    "report_situation": lambda w, p, c: None,
}
