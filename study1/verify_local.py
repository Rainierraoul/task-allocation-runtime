"""Held-out verification of the local model pair (protocol v1.1, coordination issue 4).

Sends real held-out prompts, built by the study's own code, to both local
servers and checks the mechanics: the answer parses as JSON, hand-offs
validate, a thinking model's reasoning stays out of the answer, responses are
not cut off at the output limit, and the server's token counts add up.

It uses held-out instances only, writes outside any frozen study directory,
and changes no setting. Correctness counts are reported for information; they
must not be used to tune the configuration.

    .venv/bin/python -m study1.verify_local --config study1/config.open-weight.json --out runs/heldout-verification
"""
from __future__ import annotations

import argparse
import json
import math
import time
import urllib.request
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

from dcarr_runtime.executors import _parse_json_object
from dcarr_runtime.handoff import validate

from .backend import SYSTEMS
from .calibrate import clean_contexts
from .harness import run_instance, task_models
from .instances import generate
from .oracle import gold, score

# Verification only. The study sets its gates on held-out calibration.
PROVISIONAL_THRESHOLDS = {"ACK": {"T1": 0.7, "T2": 0.7}, "CALL": {"T1": 0.7, "T2": 0.7}}


def post(url, body, timeout):
    req = urllib.request.Request(url, data=json.dumps(body).encode(), method="POST",
                                 headers={"Content-Type": "application/json", "Authorization": "Bearer local"})
    with urllib.request.urlopen(req, timeout=timeout) as response:
        return json.loads(response.read())


class VerificationClient:
    """The call interface the harness expects, sending requests exactly as LocalBackend builds them."""

    def __init__(self, config, log):
        self.config, self.log = config, log
        self.run_id, self.call_index, self.last_rows = "", 0, []

    def start(self, run_id):
        self.run_id, self.call_index, self.last_rows = run_id, 0, []

    def count_tokens(self, tier, text):
        if not text:
            return 0
        base = self.config["base_urls"][tier].rstrip("/").removesuffix("/v1")
        return post(f"{base}/tokenize", {"model": self.config["models"][tier], "prompt": text,
                                         "add_special_tokens": False}, 60)["count"]

    def call(self, tier, context, mode="allocated"):
        self.call_index += 1
        max_tokens = self.config.get("max_output_tokens_by_tier", {}).get(tier, self.config["max_output_tokens"])
        request = {"model": self.config["models"][tier], "max_tokens": max_tokens,
                   "messages": [{"role": "system", "content": SYSTEMS[mode]},
                                {"role": "user", "content": json.dumps(context, sort_keys=True)}]}
        request.update(self.config.get("model_settings", {}).get(tier, {}))
        url = self.config["base_urls"][tier].rstrip("/") + "/chat/completions"
        start = time.perf_counter()
        raw = post(url, request, self.config.get("timeout_s", 900))
        latency = time.perf_counter() - start
        message = raw["choices"][0]["message"]
        content = message.get("content") or ""
        reasoning = message.get("reasoning_content") or message.get("reasoning") or ""
        parsed = _parse_json_object(content)
        usage = raw["usage"]
        content_tokens, reasoning_tokens = self.count_tokens(tier, content), self.count_tokens(tier, reasoning)
        row = {"run_id": self.run_id, "call": self.call_index, "tier": tier, "mode": mode,
               "task": context.get("task_id") if isinstance(context, dict) else None,
               "finish_reason": raw["choices"][0].get("finish_reason"), "parsed_ok": isinstance(parsed, dict),
               "think_tag_in_answer": "<think" in content, "content": content, "parsed": parsed,
               "reasoning_chars": len(reasoning), "content_tokens": content_tokens, "reasoning_tokens": reasoning_tokens,
               "input_tokens": usage["prompt_tokens"], "output_tokens": usage["completion_tokens"],
               "latency_s": latency, "cost_usd": 0.0, "empirical": False, "verification": True,
               "returned_model": raw["model"], "utc": datetime.now(timezone.utc).isoformat()}
        with self.log.open("a") as f:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
        self.last_rows.append(row)
        return parsed, row["input_tokens"], row["output_tokens"]


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--instances", type=int, default=8, help="held-out instances for the diagnostic check")
    parser.add_argument("--condition-instances", type=int, default=2, help="held-out instances run through full conditions")
    args = parser.parse_args()
    config = json.loads(args.config.read_text())
    args.out.mkdir(parents=True, exist_ok=True)
    log = args.out / "calls.jsonl"
    if log.exists():
        raise SystemExit(f"{log} exists; use a new --out directory so runs are not mixed")
    client = VerificationClient(config, log)
    heldout = [i for i in generate()["instances"] if i["split"] == "heldout"]
    models = task_models()

    diagnostics = []
    for instance in heldout[: args.instances]:
        expected = gold(instance)
        for task, context in clean_contexts(instance).items():
            for tier in ("T1", "T2"):
                client.start(f"verify/diagnostic/{instance['id']}/{task}/{tier}")
                response, _, _ = client.call(tier, context, "diagnostic")
                payload = response.get("payload") if isinstance(response, dict) else None
                conf = response.get("confidence") if isinstance(response, dict) else None
                handoff_ok = not validate(payload, models.get(task).hand_off)
                diagnostics.append({
                    "instance": instance["id"], "task": task, "tier": tier,
                    "decided_flag_valid": isinstance(response, dict) and type(response.get("decided")) is bool,
                    "confidence_valid": type(conf) in (int, float) and math.isfinite(conf) and 0 <= conf <= 1,
                    "handoff_valid": handoff_ok,
                    "candidate_correct": bool(handoff_ok and score(task, payload, expected[task], instance)),
                })
                print(f"diagnostic {instance['id']} {task} {tier}: {client.last_rows[-1]['finish_reason']}, "
                      f"{client.last_rows[-1]['output_tokens']} out, {client.last_rows[-1]['latency_s']:.0f}s", flush=True)

    runs = []
    for instance in heldout[: args.condition_instances]:
        for condition in ("single", "decomposed", "allocated"):
            row = run_instance(instance, condition, client, f"verify/{condition}/{instance['id']}",
                               thresholds=PROVISIONAL_THRESHOLDS)
            runs.append({k: row[k] for k in ("run_id", "condition", "status", "accuracy", "correct_by_task",
                                             "input_tokens", "output_tokens", "max_context_tokens", "time_s")})
            print(f"{condition} {instance['id']}: {row['status']}, {row['time_s']:.0f}s", flush=True)

    calls = [json.loads(line) for line in log.read_text().splitlines()]
    by_tier = defaultdict(list)
    for c in calls:
        by_tier[c["tier"]].append(c)
    summary = {"utc": datetime.now(timezone.utc).isoformat(), "config": config, "heldout_only": True,
               "instances": [i["id"] for i in heldout[: args.instances]], "tiers": {}, "diagnostics": diagnostics, "runs": runs}
    for tier, rows in sorted(by_tier.items()):
        with_reasoning = [r for r in rows if r["reasoning_tokens"]]
        summary["tiers"][tier] = {
            "calls": len(rows),
            "parsed_json": sum(r["parsed_ok"] for r in rows),
            "cut_off_at_limit": sum(r["finish_reason"] == "length" for r in rows),
            "think_tag_in_answer": sum(r["think_tag_in_answer"] for r in rows),
            "calls_with_separated_reasoning": len(with_reasoning),
            "max_output_tokens_seen": max(r["output_tokens"] for r in rows),
            "max_input_tokens_seen": max(r["input_tokens"] for r in rows),
            "token_accounting_gap_max": max(abs(r["output_tokens"] - r["content_tokens"] - r["reasoning_tokens"]) for r in rows),
            "mean_latency_s": sum(r["latency_s"] for r in rows) / len(rows),
            "max_latency_s": max(r["latency_s"] for r in rows),
            "returned_models": sorted({r["returned_model"] for r in rows}),
        }
        d = [x for x in diagnostics if x["tier"] == tier]
        summary["tiers"][tier]["diagnostic_valid_responses"] = sum(
            x["decided_flag_valid"] and x["confidence_valid"] and x["handoff_valid"] for x in d)
        summary["tiers"][tier]["diagnostic_calls"] = len(d)
    (args.out / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary["tiers"], indent=2))


if __name__ == "__main__":
    main()
