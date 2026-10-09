"""Concurrency and approved retry timing, using fixtures rather than study data."""
import copy
import io
import json
import threading
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor

import pytest

from study1.analyze import check_ledger
from study1.backend import LocalBackend


CONFIG = {"provider": "local-openai", "models": {"T1": "fixture-model"},
          "base_urls": {"T1": "http://127.0.0.1:8000/v1"}, "max_output_tokens": 100}


class ParallelServer:
    def __init__(self, workers, fail_first=False):
        self.barrier = threading.Barrier(workers)
        self.lock = threading.Lock()
        self.posts = []
        self.gets = []
        self.fail_first = fail_first
        self.seen = set()

    def __call__(self, req, **kwargs):
        if req.full_url.endswith("/models"):
            self.gets.append(req.full_url)
            body = {"data": [{"id": "fixture-model", "root": "fixture-model"}]}
        elif req.full_url.endswith("/version"):
            self.gets.append(req.full_url)
            body = {"version": "fixture-version"}
        else:
            request = json.loads(req.data)
            context = json.loads(request["messages"][1]["content"])
            worker, call = context["worker"], context["call"]
            with self.lock:
                first = worker not in self.seen
                self.seen.add(worker)
                self.posts.append(context)
                message_id = f"fixture-{len(self.posts)}"
            if first:
                self.barrier.wait(timeout=5)  # Proves inference is not held under the ledger lock.
                if self.fail_first:
                    raise urllib.error.URLError("fixture interruption")
            body = {"id": message_id, "model": "fixture-model",
                    "usage": {"prompt_tokens": worker + 10, "completion_tokens": call + 20},
                    "choices": [{"finish_reason": "stop", "message": {"content": json.dumps({
                        "decided": True, "payload": {"echo": [worker, call], "padding": "x" * 4096},
                        "confidence": 0.9})}}]}
        return io.BytesIO(json.dumps(body).encode())


def ledger(path):
    return [json.loads(line) for line in path.read_text().splitlines()]


def test_parallel_units_keep_run_state_and_ledger_isolated_and_share_resume_cache(tmp_path, monkeypatch):
    n = 6
    server = ParallelServer(n)
    monkeypatch.setattr(urllib.request, "urlopen", server)
    path = tmp_path / "calls.jsonl"
    backend = LocalBackend(CONFIG, path)
    assert backend.thread_safe is True

    def unit(worker):
        run_id = f"unit/{worker}"
        backend.start(run_id)
        outputs = []
        for call in range(3):
            parsed, tin, tout = backend.call("T1", {"worker": worker, "call": call})
            assert parsed["payload"]["echo"] == [worker, call]
            assert (tin, tout) == (worker + 10, call + 20)
            parsed["payload"]["echo"] = "caller mutation must not change the cache"
            outputs.append(copy.deepcopy(backend.last_rows[-1]))
        assert backend.run_id == run_id and backend.call_index == 3
        assert len(backend.last_rows) == 3
        assert all(r["request_id"].startswith(run_id + "/") for r in outputs)
        return outputs

    with ThreadPoolExecutor(n) as pool:
        list(pool.map(unit, range(n)))
    calls = ledger(path)
    check_ledger(calls)
    ends = [r for r in calls if r["kind"] == "provider_call"]
    assert len(ends) == n * 3 and len({r["request_id"] for r in ends}) == n * 3
    assert len(server.gets) == 2  # Server/version checks only at construction.
    assert backend.run_id == "" and backend.last_rows == []  # No worker state leaks to the main thread.
    with ThreadPoolExecutor(n) as pool:
        replayed = list(pool.map(unit, range(n)))
    assert len(server.posts) == n * 3
    assert all(r["replayed"] for rows in replayed for r in rows)
    assert ledger(path) == calls


def test_parallel_failures_and_retries_are_recorded_without_lost_counts(tmp_path, monkeypatch):
    n = 5
    server = ParallelServer(n, fail_first=True)
    monkeypatch.setattr(urllib.request, "urlopen", server)
    path = tmp_path / "calls.jsonl"
    backend = LocalBackend(CONFIG, path)
    def unit(worker):
        run_id = f"retry/{worker}"
        context = {"worker": worker, "call": 0}
        backend.start(run_id)
        with pytest.raises(urllib.error.URLError):
            backend.call("T1", context)
        backend.start(run_id)
        backend.call("T1", context)
        timing = backend.run_timing()
        assert timing["failed_attempts"] == timing["retry_count"] == 1
        assert timing["unknown_timing_attempts"] == 0
        assert timing["prior_known_failed_latency_s"] == timing["known_failed_latency_s"]
        return timing
    with ThreadPoolExecutor(n) as pool:
        list(pool.map(unit, range(n)))
    check_ledger(ledger(path))
    assert backend.failed_attempts == n and len(server.posts) == n * 2


def test_a_concurrent_duplicate_request_cannot_send_a_second_inference(tmp_path, monkeypatch):
    entered, release = threading.Event(), threading.Event()
    server = ParallelServer(1)
    def blocked(req, **kwargs):
        if req.full_url.endswith("/chat/completions"):
            entered.set()
            assert release.wait(timeout=5)
        return server(req, **kwargs)
    monkeypatch.setattr(urllib.request, "urlopen", blocked)
    path = tmp_path / "calls.jsonl"
    backend = LocalBackend(CONFIG, path)
    def first():
        backend.start("same-unit")
        return backend.call("T1", {"worker": 0, "call": 0})
    with ThreadPoolExecutor(2) as pool:
        pending = pool.submit(first)
        assert entered.wait(timeout=5)
        backend.start("same-unit")
        try:
            with pytest.raises(RuntimeError, match="already in flight"):
                backend.call("T1", {"worker": 0, "call": 0})
        finally:
            release.set()
        pending.result(timeout=5)
    check_ledger(ledger(path))
    assert len(server.posts) == 1


@pytest.mark.parametrize("unknown", [0, 1])
def test_harness_counts_prior_failures_once_excludes_offline_gaps_and_keeps_other_measures(monkeypatch, unknown):
    from types import SimpleNamespace
    from study1 import harness
    from study1.instances import generate
    from study1.mock import MockBackend

    class Resumed(MockBackend):
        def call(self, *args, **kwargs):
            result = super().call(*args, **kwargs)
            self.last_rows[-1].update(replayed=True, latency_s=3.0)
            return result

        def run_timing(self):
            return {"failed_attempts": 2, "retry_count": 2,
                    "known_failed_latency_s": 9.0, "unknown_timing_attempts": unknown,
                    "prior_known_failed_latency_s": 7.0}

    instance = next(i for i in generate()["instances"] if i["split"] == "heldout")
    baseline = harness.run_instance(instance, "single", MockBackend(), "fixture")
    # Large clock origin stands for a later resumed process; only the active
    # interval matters. Its ten seconds already include two seconds of failure.
    clock = iter([1_000_000.0, 1_000_010.0])
    monkeypatch.setattr(harness, "time", SimpleNamespace(perf_counter=lambda: next(clock)))
    row = harness.run_instance(instance, "single", Resumed(), "fixture")
    assert row["time_s"] == (None if unknown else 20.0)  # 10 active + 7 prior failed + 3 replayed.
    for key in ("accuracy", "correct_by_task", "input_tokens", "output_tokens", "results"):
        assert row[key] == baseline[key]
    assert row["failed_attempts"] == row["retry_count"] == 2


def test_analysis_timing_requires_matching_failure_evidence():
    from study1.analyze import validate_timing
    rid = "fixture/run/1/T1/allocated"
    calls = [{"kind": "request_started", "request_id": rid},
             {"kind": "failed_call", "request_id": rid, "latency_s": None},
             {"kind": "request_started", "request_id": rid},
             {"kind": "failed_call", "request_id": rid, "latency_s": 4.0},
             {"kind": "request_started", "request_id": rid}]
    row = {"run_id": "fixture/run", "time_s": None, "failed_attempts": 2,
           "retry_count": 2, "unknown_timing_attempts": 1, "known_failed_latency_s": 4.0}
    validate_timing(row, calls)
    with pytest.raises(ValueError, match="unavailable run time"):
        validate_timing({**row, "time_s": 10.0}, calls)
    with pytest.raises(ValueError, match="disagrees with provider ledger"):
        validate_timing({**row, "retry_count": 1}, calls)
    with pytest.raises(ValueError, match="without unknown attempt evidence"):
        validate_timing({"time_s": None})
    for bad in (True, -1, float("nan"), float("inf")):
        with pytest.raises(ValueError, match="Invalid"):
            validate_timing({"time_s": bad})
    validate_timing({"time_s": 10.0, "known_failed_latency_s": 4.0})
    with pytest.raises(ValueError, match="Invalid"):
        validate_timing({"time_s": 3.0, "known_failed_latency_s": 4.0})


def test_missing_timing_bootstrap_pairs_repetitions_and_preserves_accuracy_tokens(tmp_path, monkeypatch):
    import importlib
    analysis = importlib.import_module("study1.analyze")
    ids = ["a", "b"]
    durations = {"single": [None, 100.0, None, None],
                 "decomposed": [20.0] * 4, "routed": [None] * 4,
                 "allocated": [12.0, 105.0, 40.0, 60.0]}
    rows = []
    for condition, times in durations.items():
        for n, duration in enumerate(times):
            rows.append({"instance_id": ids[n // 2], "condition": condition, "repetition": n % 2,
                         "accuracy": 0.75, "input_tokens": 10, "output_tokens": 20,
                         "max_context_tokens": 10, "cost_usd": 0.0, "time_s": duration,
                         "status": "completed", "failed_attempts": int(duration is None),
                         "retry_count": int(duration is None), "unknown_timing_attempts": int(duration is None),
                         "known_failed_latency_s": 0.0,
                         "metrics": {"tokens_in_by_tier": {"T1": 10}, "tokens_out_by_tier": {"T1": 20}}})
    policies = [{"sequence": s, "position": p, "instance_id": i, "order_seed": s,
                 "metrics": {"person_requests": p}} for s in range(5) for p, i in enumerate(ids)]
    manifest = {"empirical": False, "repetitions": 2}
    monkeypatch.setattr(analysis, "validate_dataset", lambda *args: (manifest, rows, [], policies, ids))
    out = analysis.analyze(tmp_path, allow_mock=True)
    result = json.loads((out / "results.json").read_text())
    single = result["condition_measures"]["single"]["time_s"]
    assert single["mean"] == 100 and single["available_runs"] == single["available_instances"] == 1
    assert single["unavailable_runs"] == 3 and single["ci95"] == [100, 100]
    paired = result["paired_differences"]["allocated minus single"]["time_s"]
    assert paired["mean"] == 5 and paired["ci95"] == [5, 5] and paired["available_runs"] == 1
    unavailable = result["condition_measures"]["routed"]["time_s"]
    assert unavailable["mean"] is None and unavailable["ci95"] == [None, None]
    assert unavailable["usable_bootstrap_resamples"] == 0
    assert result["condition_measures"]["routed"]["accuracy"]["mean"] == 0.75
    assert result["condition_measures"]["routed"]["output_tokens"]["mean"] == 20
    assert (out / "retry-summary.csv").exists()
    hashes = {p.name: p.read_bytes() for p in out.iterdir()}
    analysis.analyze(tmp_path, allow_mock=True)
    assert hashes == {p.name: p.read_bytes() for p in out.iterdir()}


def test_parallel_runner_uses_local_backend_for_every_stage_and_resumes_without_inference(tmp_path, monkeypatch):
    """Exercise --workers end to end with intercepted HTTP, never live servers."""
    import sys
    from study1 import run
    from study1.instances import generate
    from study1.mock import MockBackend
    from study1.router import MockRouter

    workers = 3
    barrier, lock = threading.Barrier(workers), threading.Lock()
    seen_threads, posts = set(), []
    config = {"provider": "local-openai", "models": {"T1": "fixture-small", "T2": "fixture-strong"},
              "base_urls": {"T1": "http://fixture/t1/v1", "T2": "http://fixture/t2/v1"},
              "max_output_tokens": 100, "router": {}}

    def transport(req, **kwargs):
        if req.full_url.endswith("/models"):
            tier = "T1" if "/t1/" in req.full_url else "T2"
            body = {"data": [{"id": config["models"][tier]}]}
        elif req.full_url.endswith("/version"):
            body = {"version": "fixture-version"}
        else:
            request = json.loads(req.data)
            context = json.loads(request["messages"][1]["content"])
            thread = threading.get_ident()
            with lock:
                first = thread not in seen_threads
                seen_threads.add(thread)
                posts.append(request)
                message_id = f"fixture-{len(posts)}"
            if first:
                barrier.wait(timeout=5)
            fixture = MockBackend()
            fixture.start("intercepted-HTTP-fixture")
            # The monolithic prompt alone requests a task-keyed object. Other
            # scripted replies already supply a best candidate at every tier.
            mode = "monolithic" if "task_models" in context else "forced"
            parsed, tin, tout = fixture.call("T1", context, mode)
            body = {"id": message_id, "model": request["model"],
                    "usage": {"prompt_tokens": tin, "completion_tokens": tout},
                    "choices": [{"finish_reason": "stop", "message": {"content": json.dumps(parsed)}}]}
        return io.BytesIO(json.dumps(body).encode())

    monkeypatch.setattr(urllib.request, "urlopen", transport)
    monkeypatch.setattr(run, "BertRouter", lambda **kwargs: MockRouter())
    material = generate()
    material["instances"] = [i for split in ("heldout", "test") for i in
                             [i for i in material["instances"] if i["split"] == split and i["address_kind"] == "direct"][:2]]
    material.update(test_n=2, heldout_n=2)
    inputs = tmp_path / "inputs"
    inputs.mkdir()
    config_path, materials_path = inputs / "config.json", inputs / "materials.json"
    config_path.write_text(json.dumps(config))
    materials_path.write_text(json.dumps(material))
    out = tmp_path / "intercepted-http-software-fixture"
    monkeypatch.setattr(sys, "argv", ["run", "all", "--out", str(out), "--config", str(config_path),
                                     "--materials", str(materials_path), "--workers", str(workers)])
    run.main()
    calls = ledger(out / "provider-calls.jsonl")
    check_ledger(calls)
    assert len(seen_threads) == workers
    assert len(ledger(out / "conditions.jsonl")) == 2 * 4 * 5
    assert len(ledger(out / "policy-sequences.jsonl")) == 2 * 5
    assert all(r["time_s"] is not None and r["retry_count"] == 0 for r in ledger(out / "conditions.jsonl"))
    first_count = len(posts)
    run.main()
    assert len(posts) == first_count
    assert ledger(out / "provider-calls.jsonl") == calls
