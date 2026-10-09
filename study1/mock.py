"""Deterministic plumbing checks. These outputs are not experiment results."""
from __future__ import annotations

import copy
import re
from dataclasses import asdict

from dcarr_runtime.handoff import estimate_tokens
from scenarios.simpilot.world import Aircraft, Tanker, World

from .oracle import call_reference, gold, heading_reference, range_reference


def parse_ack(context):
    transmission = context["observations"]["transmission"].lower()
    address = transmission.split(",")[0].replace("two", "2").replace("three", "3").replace("one", "1")
    numbers = re.findall(r"\d+", address)
    candidates = [f"aircraft {n}" for n in numbers]
    ours = context["orientation"]["own_callsigns"]
    target = next((a for a in candidates if a in ours), candidates[0])
    distance = float(re.search(r"(\d+)\s*(?:nautical miles|nm)", transmission).group(1))
    return {"addressee": target, "instruction": "join_tanker", "report_at_nm": distance, "acknowledged": target in ours}


def world_from_context(context):
    obs = context["observations"]
    w = World.sample("")
    w.aircraft["aircraft 2"] = Aircraft(**obs["aircraft2_state"])
    fields = {k: v for k, v in obs["tanker_state"].items() if k not in ("x", "y")}
    w.tanker = Tanker(**fields)
    return w


class MockBackend:
    config = {"provider": "scripted-smoke-test", "models": {"T1": "mock-small", "T2": "mock-strong"}}

    def __init__(self):
        self.last_rows, self.contexts = [], []
        self.run_id = ""

    def start(self, run_id):
        self.run_id, self.last_rows, self.contexts = run_id, [], []

    def call(self, tier, context, mode="allocated"):
        if mode == "monolithic":
            ack_ctx = {"observations": {"transmission": context["transmission"]}, "orientation": context["orientation"]}
            ack = parse_ack(ack_ctx)
            plan = context["orientation"]["exercise_plan"]["plan_id"]
            w = context["world"]
            instance = {"transmission": context["transmission"], "world": w,
                        "orientation": context["orientation"], "answer_key": {"ACK": ack},
                        "white_force_plan": {"deviation": "misinterpret" if w["deviation_inject_scheduled"] and w["trainee_level"] == "lfe_proficient" else "comply",
                                             "jammed_call": "deferred" if plan == "recover_communications" else "made"}}
            response = gold(instance)
            if "FLY" in response:
                response["FLY"]["distance_nm"] = round(response["FLY"]["distance_nm"], 2)
            if "HDG" in response:
                response["HDG"] = {k: round(v, 3) for k, v in response["HDG"].items()}
            if "WAIT" in response:
                response["WAIT"] = {k: round(v, 3) if type(v) is float else v for k, v in response["WAIT"].items()}
        else:
            task, obs, ori = context["task_id"], context["observations"], context["orientation"]
            self.contexts.append(copy.deepcopy(context))
            decided, confidence = True, 0.99
            if task == "ACK":
                payload = parse_ack(context)
            elif task == "LOC":
                payload = {"tanker": ori["briefing_material"]["tanker_callsign"], "tanker_found": True, "orbit": ori["briefing_material"]["orbit"]}
            elif task == "SEL":
                payload = {"selected": obs["addressee"]}
            elif task == "HDG":
                payload = {k: round(v, 3) for k, v in heading_reference(world_from_context(context)).items()}
            elif task == "CFG":
                payload = {"tool_ready": True, "line": f"{obs['selected']} to {obs['tanker']}"}
            elif task == "WAIT":
                payload = range_reference(world_from_context(context), obs["report_at_nm"])
                payload = {k: round(v, 6) if type(v) is float else v for k, v in payload.items()}
            elif task == "CALL":
                instance = {"world": obs, "orientation": ori,
                            "white_force_plan": {"jammed_call": "deferred" if ori["exercise_plan"]["plan_id"] == "recover_communications" else "made"}}
                payload = call_reference(instance, obs["report_at_nm"])
                if obs["jamming_inject_scheduled"]:
                    decided, confidence = False, 0.5
            elif task == "FLY":
                payload = {"summary": "Synthetic join status", "goal_met": obs["call_status"] == "made" and obs["distance_nm"] <= obs["report_at_nm"],
                           "distance_nm": obs["distance_nm"], "plan_choice": obs.get("choice") or "not_applicable", "next": "Await control"}
            else:
                raise ValueError(task)
            response = {"decided": True if mode == "forced" else decided, "payload": payload,
                        "confidence": confidence, "reason": "Deterministic smoke-test candidate"}
        tin, tout = estimate_tokens(context), estimate_tokens(response)
        self.last_rows.append({"request_id": f"{self.run_id}/{len(self.last_rows)+1}", "tier": tier,
                               "empirical": False, "input_tokens": tin, "output_tokens": tout,
                               "cost_usd": 0.0, "latency_s": 0.0, "parsed": response, "replayed": False})
        return response, tin, tout
