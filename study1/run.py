"""Run frozen stages, resume safely, and keep mock evidence separate."""
from __future__ import annotations

import argparse
import hashlib
import json
import random
import sys
import threading
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from dcarr_runtime import PolicyStore
from .backend import LocalBackend, PaidBackend
from .calibrate import calibrate, diagnostic_rows
from .harness import CONDITIONS, run_instance
from .instances import digest, generate
from .mock import MockBackend
from .router import BertRouter, MockRouter

REPETITIONS = 5
ORDER_SEEDS = [20261101, 20261102, 20261103, 20261104, 20261105]


def source_digest():
    root = Path(__file__).parents[1]
    paths = sorted([*root.glob("src/**/*.py"), *root.glob("study1/*.py"), *root.glob("scenarios/**/*.py"), *root.glob("scenarios/**/*.yaml")])
    return digest({str(p.relative_to(root)): hashlib.sha256(p.read_bytes()).hexdigest() for p in paths})


class ThreadSafeRouter:
    """Serialises scoring, and keeps router time per thread so each run's router_s is its own."""

    def __init__(self, router):
        self._router, self._lock, self._local = router, threading.Lock(), threading.local()

    @property
    def seconds(self):
        return getattr(self._local, "seconds", 0.0)

    def score(self, context):
        with self._lock:
            before = self._router.seconds
            value = self._router.score(context)
            spent = self._router.seconds - before
        self._local.seconds = self.seconds + spent
        return value


def existing_rows(path):
    if not path.exists():
        return {}
    rows = [json.loads(line) for line in path.read_text().splitlines()]
    result = {r["run_id"]: r for r in rows}
    if len(result) != len(rows):
        raise ValueError("Duplicate run IDs")
    return result


_APPEND_LOCK = threading.Lock()


def append(path, row):
    import os
    with _APPEND_LOCK, path.open("a") as f:
        f.write(json.dumps(row, allow_nan=False) + "\n")
        f.flush()
        os.fsync(f.fileno())


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=["smoke", "calibrate", "conditions", "diagnostics", "policies", "all"])
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--materials", type=Path)
    parser.add_argument("--config", type=Path, default=Path(__file__).with_name("config.example.json"))
    parser.add_argument("--budget-usd", type=float)
    parser.add_argument("--api-key-file", type=Path, help="Locally configured secret file; never logged or included in the manifest")
    parser.add_argument("--workers", type=int, default=1,
                        help="Independent runs in parallel (protocol v1.1). Needs a backend that declares thread_safe; "
                             "recorded in the manifest because run times depend on it")
    args = parser.parse_args()
    if args.workers < 1:
        parser.error("--workers must be at least 1")
    mock = args.stage == "smoke"
    config = json.loads(args.config.read_text())
    local = config.get("provider") == "local-openai"
    if not mock and not local and args.budget_usd is None:
        parser.error("Paid runs require an explicit --budget-usd")
    materials = json.loads(args.materials.read_text()) if args.materials else generate()
    args.out.mkdir(parents=True, exist_ok=True)
    frozen = {"materials_sha256": digest(materials), "config_sha256": digest(config), "source_sha256": source_digest(),
              "protocol_sha256": hashlib.sha256(Path(__file__).with_name("PROTOCOL.md").read_bytes()).hexdigest(),
              "empirical": not mock, "repetitions": REPETITIONS, "test_n": materials["test_n"], "heldout_n": materials["heldout_n"],
              "order_seeds": ORDER_SEEDS, "config": config, "workers": args.workers}
    manifest = args.out / "manifest.json"
    if manifest.exists() and json.loads(manifest.read_text()) != frozen:
        raise SystemExit("Frozen source/config/materials changed. Use a new output directory and record the deviation.")
    manifest.write_text(json.dumps(frozen, indent=2) + "\n")
    frozen_materials = args.out / "materials.json"
    if not frozen_materials.exists():
        frozen_materials.write_text(json.dumps(materials, indent=2) + "\n")
    if mock:
        paid = MockBackend()
    elif local:
        paid = LocalBackend(config, args.out / "provider-calls.jsonl")
    else:
        paid = PaidBackend(config, args.out / "provider-calls.jsonl", args.budget_usd, args.api_key_file)
    router = MockRouter() if mock else BertRouter(**config["router"])
    if args.workers > 1:
        if not getattr(paid, "thread_safe", False):
            raise SystemExit(f"{type(paid).__name__} does not declare thread_safe; run with --workers 1")
        router = ThreadSafeRouter(router)
    pool = ThreadPoolExecutor(args.workers) if args.workers > 1 else None
    pool_map = pool.map if pool else map
    try:
        heldout = [i for i in materials["instances"] if i["split"] == "heldout"]
        test = [i for i in materials["instances"] if i["split"] == "test"]
        calibration_file = args.out / "calibration.json"
        if args.stage in ("smoke", "calibrate", "all") and not calibration_file.exists():
            calibration = calibrate(heldout, paid, router, digest(materials), digest(config), pool_map=pool_map)
            calibration_file.write_text(json.dumps(calibration, indent=2) + "\n")
            print("Held-out calibration saved", flush=True)
        if not calibration_file.exists():
            raise SystemExit("Calibrate on held-out material first")
        calibration = json.loads(calibration_file.read_text())
        if calibration["materials_sha256"] != digest(materials) or calibration["config_sha256"] != digest(config) or calibration["empirical"] == mock:
            raise SystemExit("Calibration does not match this run")
        if args.stage in ("smoke", "conditions", "all"):
            path = args.out / "conditions.jsonl"
            seen = existing_rows(path)
            schedule = [(i, c, r) for i in test for c in CONDITIONS for r in range(REPETITIONS)]
            random.Random(20261009).shuffle(schedule)
            pending = [(i, c, r) for i, c, r in schedule if f"condition/{i['id']}/{c}/{r}" not in seen]

            def condition_run(unit):
                instance, condition, repetition = unit
                row = run_instance(instance, condition, paid, f"condition/{instance['id']}/{condition}/{repetition}",
                                   thresholds=calibration["thresholds"], router=router,
                                   router_threshold=calibration["router_threshold"])
                row["repetition"] = repetition
                append(path, row)
                return row

            for done, _ in enumerate(pool_map(condition_run, pending), 1):
                if done % 25 == 0 or done == len(pending):
                    print(f"Conditions: {len(schedule) - len(pending) + done}/{len(schedule)}", flush=True)
        if args.stage in ("smoke", "diagnostics", "all"):
            path = args.out / "diagnostics.jsonl"
            seen = existing_rows(path)
            # Cached paid responses make partial-stage resume reproducible.
            for row in diagnostic_rows(test, paid, pool_map=pool_map):
                if row["run_id"] not in seen:
                    row["runtime_abstained"] = (row["model_abstained"] or not row["valid_confidence"] or not row["handoff_valid"]
                                                or row["confidence"] < calibration["thresholds"][row["task"]][row["tier"]])
                    append(path, row)
            print("Diagnostic candidate and abstention records saved", flush=True)
        if args.stage in ("smoke", "policies", "all"):
            path = args.out / "policy-sequences.jsonl"
            seen = existing_rows(path)
            def policy_sequence(sequence, seed):
                order = list(test)
                random.Random(seed).shuffle(order)
                # Rebuild from the beginning: loading the terminal policy store
                # would change early cases. Cached calls retain their original data.
                store = PolicyStore()
                for position, instance in enumerate(order):
                    run_id = f"policy/{sequence}/{position}/{instance['id']}"
                    row = run_instance(instance, "allocated", paid, run_id, policies=store, thresholds=calibration["thresholds"])
                    row.update(sequence=sequence, order_seed=seed, position=position)
                    if run_id not in seen:
                        append(path, row)
                    elif {k: v for k, v in row.items() if k not in ("time_s", "events")} != {k: v for k, v in seen[run_id].items() if k not in ("time_s", "events")}:
                        raise RuntimeError("Policy replay changed the saved decisions")
                store.path = args.out / f"policy-store-{sequence}.json"
                store.save()
                print(f"Policy sequence {sequence+1}/5 saved", flush=True)

            # Each order is sequential within itself, since its policy store carries over; orders are independent.
            list(pool_map(policy_sequence, range(len(ORDER_SEEDS)), ORDER_SEEDS))
    finally:
        # Cancel queued work if a stage fails, so the process exits promptly.
        if pool:
            pool.shutdown(cancel_futures=True)
    if mock:
        print("SMOKE TEST ONLY — no model study has been run", flush=True)
    elif local:
        print(f"Stage complete; local models, {paid.failed_attempts} failed attempts logged", flush=True)
    else:
        print(f"Stage complete; logged spend ${paid.spent:.4f}", flush=True)


if __name__ == "__main__":
    main()
