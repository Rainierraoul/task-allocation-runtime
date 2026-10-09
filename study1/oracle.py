"""Independent numerical reference and literal scoring; no model grading."""
from __future__ import annotations

import copy
import math
import re

from .instances import make_world, orientation

EPISTEMIC = ["ACK", "LOC", "SEL", "HDG", "CFG", "WAIT", "CALL", "FLY"]
ANGLE_TOL = 1.0
DISTANCE_TOL = 1.25
INTERCEPT_MIN_TOL = 0.2
WAIT_MIN_TOL = 1.0
# Orbit radius and speed are given to two decimals in the briefing material.
ORBIT_NUMBER_TOL = 0.01
# Control agency callsign. It is part of every CALL input (protocol v1.1).
CONTROL_CALLSIGN = "Wedgetail"


def heading_reference(w):
    """Earliest intercept via bracketed bisection, not runtime fixed-point iteration."""
    a, t = w.aircraft["aircraft 2"], w.tanker
    speed = a.speed_kts / 3600

    def residual(seconds):
        x, y = t.position_after(seconds)
        return math.hypot(x - a.x, y - a.y) - speed * seconds

    lo, hi = 0.0, 1.0
    while residual(hi) > 0:
        hi *= 2
        if hi > 7200:
            raise ValueError("No bracketed intercept in the material range")
    for _ in range(80):
        mid = (lo + hi) / 2
        if residual(mid) > 0:
            lo = mid
        else:
            hi = mid
    x, y = t.position_after(hi)
    return {"heading_deg": math.degrees(math.atan2(x - a.x, y - a.y)) % 360,
            "time_to_intercept_min": hi / 60}


def range_reference(w, report_at):
    """Closed-form positions sampled every five seconds from the initial state."""
    a = w.aircraft["aircraft 2"]
    heading = math.radians(a.heading_deg)
    for seconds in range(0, 2701, 5):
        travel = a.speed_kts * seconds / 3600
        tx, ty = w.tanker.position_after(seconds)
        d = math.hypot(tx - a.x - travel * math.sin(heading), ty - a.y - travel * math.cos(heading))
        if d <= report_at:
            return {"at_range": True, "distance_nm": d, "elapsed_min": seconds / 60}
    raise ValueError("Reporting range not reached by reference trajectory")


def call_reference(instance, report_at):
    w = instance["world"]
    comms = instance["orientation"]["comms_plan"]
    if w["channel_state"] == "in_use":
        return {"call_status": "deferred", "message": "Retry in 30 seconds", "frequency": comms["primary"]}
    if w["jamming_inject_scheduled"] and instance["white_force_plan"]["jammed_call"] == "deferred":
        return {"call_status": "deferred", "message": "Hold the call for the exercise inject", "frequency": comms["primary"]}
    frequency = comms["backup"] if w["channel_state"] == "jammed" else comms["primary"]
    return {"call_status": "made", "message": f"{CONTROL_CALLSIGN}, aircraft 2, {report_at:g} NM from the tanker.", "frequency": frequency}


def gold(instance):
    w = make_world(instance)
    ack = copy.deepcopy(instance["answer_key"]["ACK"])
    out = {"ACK": ack}
    if not ack["acknowledged"]:
        out["FLY"] = {"goal_met": False, "distance_nm": w.distance_to_tanker(), "plan_choice": "not_applicable", "summary": "Instruction addressed elsewhere", "next": "Wait"}
        return out
    ori = orientation(instance)
    out["DEV"] = {"choice": instance["white_force_plan"]["deviation"], "rationale": "Scripted exercise plan"}
    out["LOC"] = {"tanker": w.tanker.callsign, "tanker_found": True, "orbit": ori["briefing_material"]["orbit"]}
    out["SEL"] = {"selected": "aircraft 2"}
    out["HDG"] = heading_reference(w)
    w.aircraft["aircraft 2"].heading_deg = out["HDG"]["heading_deg"]
    out["CFG"] = {"tool_ready": True, "line": f"aircraft 2 to {w.tanker.callsign}"}
    out["WAIT"] = range_reference(w, ack["report_at_nm"])
    w.step(out["WAIT"]["elapsed_min"] * 60)
    out["CALL"] = call_reference(instance, ack["report_at_nm"])
    out["FLY"] = {"goal_met": out["CALL"]["call_status"] == "made", "distance_nm": w.distance_to_tanker(), "plan_choice": out["DEV"]["choice"], "summary": "Synthetic join status", "next": "Wait"}
    return out


def finite_number(v):
    return isinstance(v, (int, float)) and not isinstance(v, bool) and math.isfinite(v)


def near(a, b, tol):
    return finite_number(a) and abs(a - b) <= tol


def score(task, payload, expected, instance):
    if not isinstance(payload, dict):
        return False
    if task == "ACK":
        return (str(payload.get("addressee", "")).lower().strip() == expected["addressee"]
                and payload.get("instruction") == "join_tanker"
                and near(payload.get("report_at_nm"), expected["report_at_nm"], 0.01)
                and type(payload.get("acknowledged")) is bool and payload["acknowledged"] == expected["acknowledged"])
    if task == "HDG":
        angle = payload.get("heading_deg")
        return (finite_number(angle) and 0 <= angle < 360
                and abs((angle - expected["heading_deg"] + 180) % 360 - 180) <= ANGLE_TOL
                and near(payload.get("time_to_intercept_min"), expected["time_to_intercept_min"], INTERCEPT_MIN_TOL))
    if task == "WAIT":
        return (payload.get("at_range") is True
                and near(payload.get("distance_nm"), expected["distance_nm"], DISTANCE_TOL)
                and near(payload.get("elapsed_min"), expected["elapsed_min"], WAIT_MIN_TOL))
    if task == "CALL":
        if payload.get("call_status") != expected["call_status"] or payload.get("frequency") != expected["frequency"]:
            return False
        message = str(payload.get("message", "")).lower()
        if expected["call_status"] == "deferred":
            return any(word in message for word in ("retry", "wait", "hold", "defer"))
        distances = re.findall(r"(\d+(?:\.\d+)?)\s*(?:nm|nautical\s+miles|miles)", message)
        return (CONTROL_CALLSIGN.lower() in message and bool(re.search(r"aircraft\s+(?:2|two)\b", message))
                and "tanker" in message and len(distances) == 1
                and near(float(distances[0]), instance["answer_key"]["ACK"]["report_at_nm"], 0.01))
    if task == "LOC":
        # Content, not wording: the tanker's name, and exactly one radius (a number in NM) and one
        # speed (a number in kts or knots), each matching the briefing.
        orbit = str(payload.get("orbit", "")).lower()
        radii = [float(n) for n in re.findall(r"(\d+(?:\.\d+)?)\s*(?:nm\b|nautical\s+miles?)", orbit)]
        speeds = [float(n) for n in re.findall(r"(\d+(?:\.\d+)?)\s*(?:kts?\b|knots?\b)", orbit)]
        want_radius, want_speed = (float(n) for n in re.findall(r"\d+(?:\.\d+)?", expected["orbit"]))
        return (str(payload.get("tanker", "")).strip().lower() == expected["tanker"].lower()
                and payload.get("tanker_found") is True
                and len(radii) == 1 and abs(radii[0] - want_radius) <= ORBIT_NUMBER_TOL
                and len(speeds) == 1 and abs(speeds[0] - want_speed) <= ORBIT_NUMBER_TOL)
    if task == "CFG":
        # The distance tool must be ready and its line must run from aircraft 2 to the tanker,
        # named in full: "SHELL 20" is not "SHELL 2".
        line = str(payload.get("line", "")).lower()
        tanker = re.escape(expected["line"].split(" to ", 1)[1].lower())
        return (payload.get("tool_ready") is True
                and bool(re.search(rf"\baircraft\s+(?:2|two)\b.*\bto\b.*(?<![\w]){tanker}(?![\w])", line)))
    if task == "FLY":
        return (type(payload.get("goal_met")) is bool and payload["goal_met"] == expected["goal_met"]
                and near(payload.get("distance_nm"), expected["distance_nm"], DISTANCE_TOL))
    return all(payload.get(k) == v and type(payload.get(k)) is type(v) for k, v in expected.items())


def eligible_tasks(instance, expected):
    # A scheduled inject's channel choice is normative; do not mark a model's
    # different training-design choice as an epistemic error.
    return [t for t in EPISTEMIC if t in expected and not
            (t in ("CALL", "FLY") and instance["world"]["jamming_inject_scheduled"] and expected["ACK"]["acknowledged"])]


def score_run(instance, results):
    expected = gold(instance)
    scored = {t: score(t, results.get(t), expected[t], instance) for t in eligible_tasks(instance, expected)}
    if not expected["ACK"]["acknowledged"] and any(t in results for t in ("SEL", "HDG", "WAIT", "CALL")):
        scored["FLY"] = False
    return scored
