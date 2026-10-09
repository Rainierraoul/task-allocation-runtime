import copy
import json

import pytest

from dcarr_runtime.executors import DecisionContext
from dcarr_runtime.handoff import validate
from study1.calibrate import choose_threshold, clean_contexts
from study1.harness import run_instance, task_models
from study1.instances import digest, generate, make_world
from study1.mock import MockBackend
from study1.oracle import gold, heading_reference, score, score_run
from study1.router import MockRouter
from scenarios.simpilot.tools import intercept_heading, wait_for_range


def test_materials_are_reproducible_balanced_and_separate():
    first, second = generate(), generate()
    assert digest(first) == digest(second)
    assert len(first["instances"]) == 96
    assert len({i["id"] for i in first["instances"]}) == 96
    for split, n in [("test", 64), ("heldout", 32)]:
        selected = [i for i in first["instances"] if i["split"] == split]
        assert len(selected) == n
        assert sum(i["address_kind"] == "misaddressed" for i in selected) == n // 4
        assert sum(i["world"]["channel_state"] == "jammed" for i in selected) == n // 2


def test_independent_heading_reference_agrees_with_runtime_over_material_range():
    tm = task_models().get("HDG")
    for instance in generate()["instances"]:
        w = make_world(instance)
        reference = heading_reference(w)
        actual = intercept_heading(DecisionContext(tm, {}, {}, {}, world=w)).payload
        assert abs((actual["heading_deg"] - reference["heading_deg"] + 180) % 360 - 180) <= 1
        assert abs(actual["time_to_intercept_min"] - reference["time_to_intercept_min"]) <= 0.2


def test_wait_decision_leaves_world_unchanged_until_action():
    instance = next(i for i in generate()["instances"] if i["answer_key"]["ACK"]["acknowledged"])
    w = make_world(instance)
    w.aircraft["aircraft 2"].heading_deg = heading_reference(w)["heading_deg"]
    before = copy.deepcopy(w)
    ctx = DecisionContext(task_models().get("WAIT"), {"report_at_nm": 20.0}, {}, {}, world=w)
    p = wait_for_range(ctx)
    assert p.decided
    assert p.payload["elapsed_min"] > 0
    assert w == before


@pytest.mark.parametrize("condition", ["single", "decomposed", "routed", "allocated"])
def test_all_conditions_ignore_a_misaddressed_instruction(condition):
    instance = next(i for i in generate()["instances"] if i["address_kind"] == "misaddressed")
    row = run_instance(instance, condition, MockBackend(), "test", router=MockRouter(), router_threshold=0.5)
    assert row["accuracy"] == 1
    assert row["results"]["ACK"]["acknowledged"] is False
    assert "HDG" not in row["results"]
    assert row["empirical"] is False


def test_answer_keys_and_private_plans_never_enter_model_contexts():
    class Recording(MockBackend):
        def __init__(self):
            super().__init__()
            self.inputs = []

        def call(self, tier, context, mode="allocated"):
            self.inputs.append(copy.deepcopy(context))
            return super().call(tier, context, mode)

    instance = next(i for i in generate()["instances"] if i["address_kind"] == "direct")
    backend = Recording()
    for condition in ("single", "decomposed", "routed", "allocated"):
        run_instance(instance, condition, backend, condition, router=MockRouter(), router_threshold=0.5)
    for context in backend.inputs:
        text = json.dumps(context)
        assert "answer_key" not in text and "white_force_plan" not in text
    for context in clean_contexts(instance).values():
        text = json.dumps(context)
        assert "answer_key" not in text and "white_force_plan" not in text


def test_literal_scoring_rejects_wrong_channel_and_distance_and_nan():
    instance = next(i for i in generate()["instances"] if i["address_kind"] == "direct" and i["world"]["channel_state"] == "clear")
    expected = gold(instance)["CALL"]
    assert score("CALL", expected, expected, instance)
    assert not score("CALL", {**expected, "frequency": "Channel 99"}, expected, instance)
    assert not score("CALL", {**expected, "message": "Wedgetail, aircraft 2, 999 NM from the tanker"}, expected, instance)
    expected_hdg = gold(instance)["HDG"]
    assert not score("HDG", {**expected_hdg, "heading_deg": float("nan")}, expected_hdg, instance)


def test_failed_forced_candidate_is_not_rescued_by_the_scripted_person():
    class Broken(MockBackend):
        def call(self, tier, context, mode="allocated"):
            response, tin, tout = super().call(tier, context, mode)
            if context.get("task_id") == "ACK":
                response["payload"] = {"invented": True}
            return response, tin, tout

    instance = next(i for i in generate()["instances"] if i["address_kind"] == "direct")
    row = run_instance(instance, "decomposed", Broken(), "bad-candidate")
    assert row["status"] == "failed_candidate_or_action"
    assert row["metrics"]["person_requests"] == 0
    assert row["accuracy"] == 0


def test_threshold_failure_disables_autonomous_acceptance():
    rows = [{"model_abstained": False, "valid_confidence": True, "handoff_valid": True,
             "confidence": 1.0, "candidate_correct": False} for _ in range(20)]
    result = choose_threshold(rows)
    assert result["threshold"] > 1 and result["target_met"] is False


def test_research_analysis_rejects_mock_evidence(tmp_path):
    from study1.analyze import validate_dataset
    material = {"instances": []}
    (tmp_path / "materials.json").write_text(json.dumps(material))
    (tmp_path / "manifest.json").write_text(json.dumps({"materials_sha256": digest(material), "empirical": False}))
    with pytest.raises(ValueError, match="not empirical research"):
        validate_dataset(tmp_path)


def test_analysis_recomputes_accuracy_instead_of_trusting_saved_scores():
    from study1.analyze import validate_run_score
    instance = generate()["instances"][0]
    row = run_instance(instance, "allocated", MockBackend(), "score-audit")
    validate_run_score(row, instance)
    row["accuracy"] = 0.5
    with pytest.raises(ValueError, match="independent oracle"):
        validate_run_score(row, instance)


def test_analysis_requires_usage_to_match_provider_records():
    from study1.analyze import validate_usage
    row = {"run_id": "condition/test/single/0", "input_tokens": 10, "output_tokens": 20, "cost_usd": 0.00011}
    calls = [{"request_id": row["run_id"] + "/1/T2/monolithic", "input_tokens": 10, "output_tokens": 20, "cost_usd": 0.00011}]
    validate_usage(row, calls)
    with pytest.raises(ValueError, match="provider ledger"):
        validate_usage(row, [])


def test_paid_backend_requires_a_locally_configured_key(tmp_path, monkeypatch):
    from study1.backend import PaidBackend
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    with pytest.raises(RuntimeError, match="configured locally"):
        PaidBackend({}, tmp_path / "calls.jsonl", 1.0)
    assert not (tmp_path / "calls.jsonl").exists()


def test_paid_backend_cap_and_cache_preserve_usage_without_credentials(tmp_path, monkeypatch):
    import urllib.request
    from study1.backend import PaidBackend, BudgetExceeded
    monkeypatch.setenv("ANTHROPIC_API_KEY", "TEST-CREDENTIAL-DO-NOT-LOG")
    config = {"models": {"T1": "test-model"}, "max_output_tokens": 100,
              "prices_usd_per_million": {"T1": {"input": 1.0, "output": 5.0}}}
    calls = []

    class Response:
        def __enter__(self):
            return self

        def __exit__(self, *args):
            pass

        def read(self):
            return json.dumps({"model": "test-model", "id": "fake-provider-id", "usage": {"input_tokens": 10, "output_tokens": 20},
                               "content": [{"type": "text", "text": '{"decided":true,"payload":{},"confidence":1}'}]}).encode()

    def transport(req, **kwargs):
        calls.append(req)
        return Response()

    monkeypatch.setattr(urllib.request, "urlopen", transport)
    capped = PaidBackend(config, tmp_path / "capped.jsonl", 0.000001)
    capped.start("capped")
    with pytest.raises(BudgetExceeded):
        capped.call("T1", {})
    assert not calls
    backend = PaidBackend(config, tmp_path / "calls.jsonl", 1.0)
    backend.start("repeat-0")
    assert backend.call("T1", {})[1:] == (10, 20)
    assert backend.spent == pytest.approx(0.00011)
    backend.start("repeat-0")
    backend.call("T1", {})
    assert len(calls) == 1
    backend.start("repeat-1")
    backend.call("T1", {})
    assert len(calls) == 2
    assert "TEST-CREDENTIAL" not in (tmp_path / "calls.jsonl").read_text()


def test_uncertain_billed_call_cannot_be_silently_repeated(tmp_path, monkeypatch):
    from study1.backend import PaidBackend
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-only")
    path = tmp_path / "calls.jsonl"
    path.write_text(json.dumps({"kind": "request_started", "request_id": "unfinished"}) + "\n")
    with pytest.raises(RuntimeError, match="billing reconciliation"):
        PaidBackend({}, path, 1.0)


class _FakeLocalServer:
    """Stands in for a vLLM server: /v1/models, /version and /v1/chat/completions."""

    def __init__(self, model="local-small", version="0.30.0", fail_next=False):
        self.model, self.version, self.fail_next, self.posts = model, version, fail_next, []

    def __call__(self, req, **kwargs):
        import io
        import urllib.error
        url = req.full_url
        if url.endswith("/v1/models"):
            body = {"data": [{"id": self.model, "root": self.model, "max_model_len": 32768}]}
        elif url.endswith("/version"):
            body = {"version": self.version}
        else:
            self.posts.append(json.loads(req.data))
            if self.fail_next:
                self.fail_next = False
                raise urllib.error.URLError("connection reset")
            body = {"id": f"chatcmpl-{len(self.posts)}", "model": self.model,
                    "usage": {"prompt_tokens": 12, "completion_tokens": 30},
                    "choices": [{"finish_reason": "stop", "message": {
                        "content": '{"decided":true,"payload":{},"confidence":0.9}'}}]}

        class Response(io.BytesIO):
            def __enter__(self):
                return self

            def __exit__(self, *args):
                pass

        return Response(json.dumps(body).encode())


LOCAL_CONFIG = {"provider": "local-openai", "models": {"T1": "local-small"},
                "base_urls": {"T1": "http://127.0.0.1:8000/v1"}, "max_output_tokens": 100,
                "model_settings": {"T1": {"temperature": 0.7}}}


def test_local_backend_logs_usage_replays_and_costs_nothing(tmp_path, monkeypatch):
    import urllib.request
    from study1.backend import LocalBackend
    server = _FakeLocalServer()
    monkeypatch.setattr(urllib.request, "urlopen", server)
    backend = LocalBackend(LOCAL_CONFIG, tmp_path / "calls.jsonl")
    backend.start("repeat-0")
    parsed, tokens_in, tokens_out = backend.call("T1", {"x": 1})
    assert parsed["decided"] and (tokens_in, tokens_out) == (12, 30)
    assert server.posts[0]["temperature"] == 0.7 and server.posts[0]["messages"][0]["role"] == "system"
    backend.start("repeat-0")
    backend.call("T1", {"x": 1})
    assert len(server.posts) == 1
    rows = [json.loads(line) for line in (tmp_path / "calls.jsonl").read_text().splitlines()]
    assert [r["kind"] for r in rows] == ["server_info", "request_started", "response_received", "provider_call"]
    assert rows[0]["server"]["server_version"] == "0.30.0" and rows[-1]["cost_usd"] == 0.0


def test_local_backend_refuses_a_missing_model_or_a_changed_server(tmp_path, monkeypatch):
    import urllib.request
    from study1.backend import LocalBackend
    monkeypatch.setattr(urllib.request, "urlopen", _FakeLocalServer(model="something-else"))
    with pytest.raises(RuntimeError, match="does not serve local-small"):
        LocalBackend(LOCAL_CONFIG, tmp_path / "a.jsonl")
    monkeypatch.setattr(urllib.request, "urlopen", _FakeLocalServer(version="0.30.0"))
    LocalBackend(LOCAL_CONFIG, tmp_path / "b.jsonl")
    monkeypatch.setattr(urllib.request, "urlopen", _FakeLocalServer(version="0.31.0"))
    with pytest.raises(RuntimeError, match="server changed"):
        LocalBackend(LOCAL_CONFIG, tmp_path / "b.jsonl")


def test_a_failed_local_attempt_can_be_retried_and_still_passes_the_ledger_check(tmp_path, monkeypatch):
    import urllib.error
    import urllib.request
    from study1.analyze import check_ledger
    from study1.backend import LocalBackend
    server = _FakeLocalServer(fail_next=True)
    monkeypatch.setattr(urllib.request, "urlopen", server)
    backend = LocalBackend(LOCAL_CONFIG, tmp_path / "calls.jsonl")
    backend.start("run")
    with pytest.raises(urllib.error.URLError):
        backend.call("T1", {})
    resumed = LocalBackend(LOCAL_CONFIG, tmp_path / "calls.jsonl")
    resumed.start("run")
    resumed.call("T1", {})
    rows = [json.loads(line) for line in (tmp_path / "calls.jsonl").read_text().splitlines()]
    check_ledger(rows)
    assert resumed.failed_attempts == 1
    with pytest.raises(ValueError, match="unresolved"):
        check_ledger([r for r in rows if r["kind"] != "failed_call"])


@pytest.mark.parametrize("received", [False, True])
def test_process_interruption_preserves_journal_and_recovers_or_retries(tmp_path, monkeypatch, received):
    import urllib.request
    from study1.analyze import check_ledger
    from study1.backend import LocalBackend
    server = _FakeLocalServer()
    monkeypatch.setattr(urllib.request, "urlopen", server)
    path = tmp_path / "calls.jsonl"
    backend = LocalBackend(LOCAL_CONFIG, path)
    backend.start("run")
    expected = backend.call("T1", {"x": 1})
    # Simulate a killed process at either durable journal boundary, using only
    # the complete prefix it could have written before interruption.
    lines = path.read_text().splitlines(keepends=True)
    prefix = "".join(lines[:3 if received else 2])
    path.write_text(prefix)
    resumed = LocalBackend(LOCAL_CONFIG, path)
    resumed.start("run")
    assert resumed.call("T1", {"x": 1}) == expected
    assert path.read_text().startswith(prefix)
    rows = [json.loads(line) for line in path.read_text().splitlines()]
    check_ledger(rows)
    assert len(server.posts) == (1 if received else 2)
    if received:
        assert rows[-1]["recovered_response"] is True
    else:
        failures = [r for r in rows if r["kind"] == "failed_call"]
        assert failures[0]["error_type"] == "ProcessInterrupted"
        assert failures[0]["latency_s"] is None and failures[0]["timing_unknown"]
        starts = [r for r in rows if r["kind"] == "request_started"]
        assert starts[0]["attempt_id"] != starts[1]["attempt_id"]


def test_keyboard_interrupt_is_closed_and_records_measured_failed_time(tmp_path, monkeypatch):
    import urllib.request
    from study1.analyze import check_ledger
    from study1.backend import LocalBackend
    server = _FakeLocalServer()
    def interrupted(req, **kwargs):
        if req.full_url.endswith("/chat/completions"):
            raise KeyboardInterrupt()
        return server(req, **kwargs)
    monkeypatch.setattr(urllib.request, "urlopen", interrupted)
    path = tmp_path / "calls.jsonl"
    backend = LocalBackend(LOCAL_CONFIG, path)
    backend.start("run")
    clock = iter([10.0, 17.5])
    monkeypatch.setattr("study1.backend.time.perf_counter", lambda: next(clock))
    with pytest.raises(KeyboardInterrupt):
        backend.call("T1", {})
    failed = json.loads(path.read_text().splitlines()[-1])
    assert failed["latency_s"] == 7.5 and failed["timing_unknown"] is False
    monkeypatch.undo()
    monkeypatch.setattr(urllib.request, "urlopen", server)
    resumed = LocalBackend(LOCAL_CONFIG, path)
    resumed.start("run")
    resumed.call("T1", {})
    check_ledger([json.loads(line) for line in path.read_text().splitlines()])


@pytest.mark.parametrize("bad_rows", [
    [{"kind": "failed_call", "request_id": "r"}],
    [{"kind": "request_started", "request_id": "r", "attempt_id": "a"},
     {"kind": "provider_call", "request_id": "r", "attempt_id": "b"}],
    [{"kind": "request_started", "request_id": "r", "attempt_id": "a"},
     {"kind": "provider_call", "request_id": "r", "attempt_id": "a"},
     {"kind": "provider_call", "request_id": "r", "attempt_id": "a"}],
    [{"kind": "request_started", "request_id": "r", "attempt_id": "a", "request_sha256": "old"},
     {"kind": "failed_call", "request_id": "r", "attempt_id": "a"},
     {"kind": "request_started", "request_id": "r", "attempt_id": "b", "request_sha256": "new"}],
])
def test_ledger_rejects_unmatched_duplicate_and_changed_attempts(bad_rows):
    from study1.analyze import check_ledger
    with pytest.raises(ValueError):
        check_ledger(bad_rows)


def test_legacy_request_journal_remains_analysable():
    from study1.analyze import check_ledger
    check_ledger([{"kind": "request_started", "request_id": "r"},
                  {"kind": "failed_call", "request_id": "r"},
                  {"kind": "request_started", "request_id": "r"},
                  {"kind": "provider_call", "request_id": "r"}])


def test_interrupted_request_cannot_be_replaced_with_a_changed_request(tmp_path, monkeypatch):
    import urllib.request
    from study1.backend import LocalBackend
    server = _FakeLocalServer()
    monkeypatch.setattr(urllib.request, "urlopen", server)
    path = tmp_path / "calls.jsonl"
    backend = LocalBackend(LOCAL_CONFIG, path)
    backend.start("run")
    backend.call("T1", {"x": 1})
    path.write_text("\n".join(path.read_text().splitlines()[:2]) + "\n")
    resumed = LocalBackend(LOCAL_CONFIG, path)
    resumed.start("run")
    with pytest.raises(RuntimeError, match="Frozen request changed"):
        resumed.call("T1", {"x": 2})
    assert len(server.posts) == 1


def _tiny_snapshot(path):
    path.mkdir(parents=True)
    for name in ("config.json", "tokenizer_config.json", "tokenizer.json"):
        (path / name).write_text("{}")
    (path / "chat_template.jinja").write_text("{{ messages }}")
    (path / "model.safetensors").write_bytes(b"test weights only")


def test_snapshot_inventory_records_weights_tokenizer_and_template_and_detects_changes(tmp_path):
    from study1.serving import snapshot_inventory
    p = tmp_path / "snapshot"
    _tiny_snapshot(p)
    before = snapshot_inventory(p)
    assert set(before) == {"config.json", "tokenizer_config.json", "tokenizer.json",
                           "chat_template.jinja", "model.safetensors"}
    (p / "model.safetensors").write_bytes(b"changed weights")
    after = snapshot_inventory(p)
    assert before["model.safetensors"]["sha256"] != after["model.safetensors"]["sha256"]
    assert before["chat_template.jinja"] == after["chat_template.jinja"]
    (p / "model.safetensors.index.json").write_text(json.dumps({"weight_map": {"layer": "missing.safetensors"}}))
    with pytest.raises(RuntimeError, match="missing indexed weight shards"):
        snapshot_inventory(p)


def test_provenance_is_captured_from_live_launch_and_serving_interpreter(tmp_path, monkeypatch):
    from study1 import serving
    # This fixture models macOS/Metal regardless of the host running pytest.
    monkeypatch.setattr(serving, "_is_linux", lambda: False)
    revision = "a" * 40
    _tiny_snapshot(tmp_path / "models--local-small" / "snapshots" / revision)
    identity = {"pid": 123, "started_at": "fixed start"}
    monkeypatch.setattr(serving, "listener_identity", lambda url: identity)
    launch = ["local-small", {"--revision": revision}, "/serving/python"]
    monkeypatch.setattr(serving, "_launch", lambda pid: tuple(launch))
    monkeypatch.setattr(serving, "_environment", lambda python: {
        "cache_root": str(tmp_path), "packages": {"vllm": "0.30.0", "vllm-metal": "0.30.0", "mlx": "test"}})
    evidence = serving.capture_serving_evidence(LOCAL_CONFIG["base_urls"]["T1"], "local-small", revision)
    assert evidence["fingerprint"]["revision"] == revision
    assert evidence["observation"] == identity
    assert "model.safetensors" in evidence["fingerprint"]["snapshot_files"]
    launch[1] = {"--revision": "b" * 40}
    with pytest.raises(RuntimeError, match="Live server launch"):
        serving.capture_serving_evidence(LOCAL_CONFIG["base_urls"]["T1"], "local-small", revision)
    launch[1] = {"--revision": revision, "--tokenizer-revision": "b" * 40}
    with pytest.raises(RuntimeError, match="Live tokenizer"):
        serving.capture_serving_evidence(LOCAL_CONFIG["base_urls"]["T1"], "local-small", revision)


def test_local_launch_capture_omits_credentials(tmp_path, monkeypatch):
    from study1 import serving
    # The mocked ps command belongs to the macOS launch path.
    monkeypatch.setattr(serving, "_is_linux", lambda: False)
    interpreter = tmp_path / "python"
    interpreter.write_text("test interpreter")
    entry = tmp_path / "vllm"
    entry.write_text(f"#!{interpreter}\n")
    monkeypatch.setattr(serving, "_command", lambda args:
                        f"{interpreter} {entry} serve local-small --revision={'a'*40} --api-key TEST-SECRET --port 8000")
    model, flags, python = serving._launch(123)
    assert model == "local-small" and flags["--port"] == "8000"
    assert "TEST-SECRET" not in json.dumps(flags)


def _fake_serving_evidence():
    return {"fingerprint": {"model": "local-small", "revision": "a" * 40,
             "launch_settings": {"--revision": "a" * 40},
             "environment": {"packages": {"vllm": "0.30.0", "vllm-metal": "0.30.0", "mlx": "test"}},
             "snapshot_files": {n: {"bytes": 1, "sha256": "b" * 64} for n in
                                ("config.json", "tokenizer_config.json", "tokenizer.json", "model.safetensors")}},
            "observation": {"pid": 123, "started_at": "fixed start"}}


def test_pinned_backend_checks_changed_weights_and_dependencies_at_construction(tmp_path, monkeypatch):
    import urllib.request
    from study1 import serving
    from study1.backend import LocalBackend
    server = _FakeLocalServer()
    monkeypatch.setattr(urllib.request, "urlopen", server)
    evidence = _fake_serving_evidence()
    monkeypatch.setattr(serving, "capture_serving_evidence", lambda *args: copy.deepcopy(evidence))
    monkeypatch.setattr(serving, "listener_identity", lambda url: copy.deepcopy(evidence["observation"]))
    config = {**LOCAL_CONFIG, "revisions": {"T1": "a" * 40}}
    path = tmp_path / "calls.jsonl"
    backend = LocalBackend(config, path)
    backend.start("run")
    backend.call("T1", {})
    evidence["fingerprint"]["snapshot_files"]["model.safetensors"]["sha256"] = "c" * 64
    with pytest.raises(RuntimeError, match="server changed"):
        LocalBackend(config, path)
    evidence = _fake_serving_evidence()
    evidence["fingerprint"]["environment"]["packages"]["mlx"] = "changed"
    with pytest.raises(RuntimeError, match="server changed"):
        LocalBackend(config, path)
    evidence = _fake_serving_evidence()
    evidence["observation"]["pid"] = 456
    resumed = LocalBackend(config, path)
    resumed.start("run")
    resumed.call("T1", {})  # A new construction verifies an unchanged snapshot after restart.
    assert len(server.posts) == 1


def test_analysis_requires_actual_local_serving_evidence():
    from study1.analyze import validate_local_provenance
    config = {**LOCAL_CONFIG, "revisions": {"T1": "a" * 40}}
    row = {"kind": "server_info", "tier": "T1", "revision": "a" * 40,
           "server": {"server_version": "0.30.0"}}
    with pytest.raises(ValueError, match="unverified"):
        validate_local_provenance(config, [row])


def test_cpu_wheel_build_suffix_is_recordable_without_accepting_another_version():
    from study1.serving import vllm_version_matches
    assert vllm_version_matches("0.30.0", "0.30.0+cpu")
    assert not vllm_version_matches("0.30.0", "0.31.0+cpu")
    assert not vllm_version_matches(None, "0.30.0+cpu")


def test_analysis_accepts_verified_evidence_and_rejects_another_revision():
    from study1.analyze import validate_local_provenance
    config = {**LOCAL_CONFIG, "revisions": {"T1": "a" * 40}}
    row = {"kind": "server_info", "tier": "T1", "revision": "a" * 40,
           "server": {"server_version": "0.30.0"}}
    row["server"]["serving_evidence"] = _fake_serving_evidence()["fingerprint"]
    validate_local_provenance(config, [row])
    row["server"]["serving_evidence"]["revision"] = "c" * 40
    with pytest.raises(ValueError, match="unverified"):
        validate_local_provenance(config, [row])


def test_failed_attempt_timing_preserves_known_durations_and_marks_unknowns():
    from study1.backend import failed_attempt_timing
    rows = [{"kind": "failed_call", "request_id": "run/1/T1/allocated", "latency_s": 7.5},
            {"kind": "failed_call", "request_id": "run/1/T1/allocated", "latency_s": 2.0},
            {"kind": "failed_call", "request_id": "other/1/T1/allocated", "latency_s": None},
            {"kind": "provider_call", "request_id": "run/1/T1/allocated", "latency_s": 100.0}]
    measured = failed_attempt_timing(rows, "run")
    assert measured == {"failed_attempts": 2, "known_failed_latency_s": 9.5,
                        "unknown_timing_attempts": 0, "total_failed_latency_s": 9.5}
    total = failed_attempt_timing(rows)
    assert total["known_failed_latency_s"] == 9.5
    assert total["unknown_timing_attempts"] == 1 and total["total_failed_latency_s"] is None


@pytest.mark.parametrize("duration", [-1, float("nan"), float("inf"), True])
def test_invalid_failed_attempt_timing_is_rejected(duration):
    from study1.backend import failed_attempt_timing
    with pytest.raises(ValueError, match="duration"):
        failed_attempt_timing([{"kind": "failed_call", "request_id": "r", "latency_s": duration}])


def test_interrupted_legacy_local_journal_can_resume_without_deleting_evidence(tmp_path, monkeypatch):
    import urllib.request
    from study1.backend import LocalBackend
    from study1.analyze import check_ledger
    server = _FakeLocalServer()
    monkeypatch.setattr(urllib.request, "urlopen", server)
    path = tmp_path / "calls.jsonl"
    b = LocalBackend(LOCAL_CONFIG, path)
    b.start("run")
    b.call("T1", {})
    rows = [json.loads(line) for line in path.read_text().splitlines()[:2]]
    rows[1].pop("attempt_id")
    prefix = "".join(json.dumps(row) + "\n" for row in rows)
    path.write_text(prefix)
    resumed = LocalBackend(LOCAL_CONFIG, path)
    resumed.start("run")
    resumed.call("T1", {})
    assert path.read_text().startswith(prefix)
    check_ledger([json.loads(line) for line in path.read_text().splitlines()])
