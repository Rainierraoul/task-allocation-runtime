"""Linux/CUDA provenance using synthetic /proc, XML and intercepted HTTP.

These fixtures never launch a server, access a GPU or report model findings.
"""
import copy
import io
import json
import urllib.request

import pytest

from study1 import serving
from study1.analyze import validate_local_provenance
from study1.backend import LocalBackend


REVISION = "a" * 40
MODEL = "fixture/model"
URL = "http://127.0.0.1:8000/v1"
CONFIG = {"provider": "local-openai", "models": {"T1": MODEL},
          "revisions": {"T1": REVISION}, "base_urls": {"T1": URL}, "max_output_tokens": 100}


def gpu_xml():
    devices = "".join(f"""<gpu><minor_number>{n}</minor_number><uuid>GPU-fixture-{n}</uuid>
        <product_name>Fixture GPU</product_name><pci><pci_bus_id>0000:0{n}:00.0</pci_bus_id></pci>
        <fb_memory_usage><total>48000 MiB</total></fb_memory_usage></gpu>""" for n in range(3))
    return f"<nvidia_smi_log><driver_version>570.1</driver_version><cuda_version>12.8</cuda_version>{devices}</nvidia_smi_log>"


@pytest.fixture
def linux_server(tmp_path, monkeypatch):
    proc = tmp_path / "proc"
    (proc / "net").mkdir(parents=True)
    (proc / "net/tcp").write_text("header\n0: 0100007F:1F40 00000000:0000 0A 0:0 00:0 0 1000 0 99\n")
    (proc / "net/tcp6").write_text("header\n")
    boot = proc / "sys/kernel/random/boot_id"
    boot.parent.mkdir(parents=True)
    boot.write_text("fixture-boot-id")
    process = proc / "123"
    (process / "fd").mkdir(parents=True)
    (process / "fd/10").symlink_to("socket:[99]")
    (process / "stat").write_text("123 (python with spaces) " + " ".join(["S"] + ["0"] * 18 + ["456"]))
    python, entry = tmp_path / "python", tmp_path / "vllm"
    python.write_text("fixture interpreter")
    entry.write_text(f"#!{python}\n")
    args = [str(python), str(entry), "serve", MODEL, f"--revision={REVISION}",
            "--host", "127.0.0.1", "--port", "8000", "--tensor-parallel-size", "2",
            "--api-key", "DO-NOT-EMIT"]
    (process / "cmdline").write_bytes("\0".join(args).encode() + b"\0")
    environment = {"HF_HOME": str(tmp_path / "hub-home"), "CUDA_VISIBLE_DEVICES": "0,1",
                   "CUDA_DEVICE_ORDER": "PCI_BUS_ID", "SLURM_JOB_ID": "job-1",
                   "SLURMD_NODENAME": "fixture-node", "SLURM_JOB_NODELIST": "fixture-node",
                   "API_KEY": "DO-NOT-EMIT", "PRIVATE_VALUE": "DO-NOT-EMIT"}
    def save_env():
        (process / "environ").write_bytes("\0".join(f"{k}={v}" for k, v in environment.items()).encode() + b"\0")
    save_env()
    snapshot = tmp_path / "cache/models--fixture--model/snapshots" / REVISION
    snapshot.mkdir(parents=True)
    for name in ("config.json", "tokenizer_config.json", "tokenizer.json", "model.safetensors"):
        (snapshot / name).write_text("{}" if name.endswith(".json") else "fixture weights")
    probes = []
    def probe(interpreter, selected):
        assert interpreter == str(python)
        assert selected["HF_HOME"] == environment["HF_HOME"]
        probes.append(selected)
        return {"python": "fixture Python", "platform": "fixture Linux", "machine": "x86_64",
                "packages": {"vllm": "0.30.0", "torch": "fixture", "transformers": "fixture"},
                "torch_cuda_version": "12.8", "cache_root": str(tmp_path / "cache")}
    commands = []
    def command(args):
        commands.append(args)
        assert args == ["nvidia-smi", "-q", "-x"]
        return gpu_xml()
    monkeypatch.setattr(serving, "PROC_ROOT", proc)
    monkeypatch.setattr(serving, "_is_linux", lambda: True)
    monkeypatch.setattr(serving, "_environment", probe)
    monkeypatch.setattr(serving, "_command", command)
    return {"proc": proc, "environment": environment, "save_env": save_env,
            "snapshot": snapshot, "probes": probes, "commands": commands}


def test_linux_proc_listener_launch_and_environment_exclude_secrets(linux_server):
    identity = serving.listener_identity(URL)
    assert identity == {"pid": 123, "started_at": "fixture-boot-id:456"}
    model, flags, interpreter = serving._launch(identity["pid"])
    assert model == MODEL and flags["--revision"] == REVISION
    assert flags["--tensor-parallel-size"] == "2"
    selected = serving._server_environment(123)
    assert selected["SLURM_JOB_ID"] == "job-1"
    assert "DO-NOT-EMIT" not in json.dumps([flags, selected])
    assert interpreter.endswith("/python")
    with pytest.raises(RuntimeError, match="loopback"):
        serving.listener_identity("http://remote.example:8000/v1")
    with pytest.raises(RuntimeError, match="one identifiable Linux listener"):
        serving.listener_identity("http://localhost:8001/v1")


def test_linux_listener_handles_ipv6_pid_reuse_and_ambiguous_ownership(linux_server):
    proc = linux_server["proc"]
    original = serving.listener_identity(URL)
    (proc / "net/tcp").write_text("header\n")
    (proc / "net/tcp6").write_text("header\n0: 00000000000000000000000001000000:1F40 0:0 0A 0:0 00:0 0 1000 0 99\n")
    assert serving.listener_identity("http://[::1]:8000/v1") == original
    stat = proc / "123/stat"
    stat.write_text(stat.read_text().replace("456", "789"))
    assert serving.listener_identity(URL) != original
    (proc / "456/fd").mkdir(parents=True)
    (proc / "456/fd/10").symlink_to("socket:[99]")
    with pytest.raises(RuntimeError, match="one identifiable Linux listener"):
        serving.listener_identity(URL)


def test_linux_capture_links_snapshot_packages_cuda_devices_and_slurm(linux_server):
    result = serving.capture_serving_evidence(URL, MODEL, REVISION)
    fingerprint, observation = result["fingerprint"], result["observation"]
    assert fingerprint["backend"] == "linux-cuda"
    assert serving.dependencies_verified(fingerprint)
    assert serving.cuda_observation_verified(fingerprint, observation)
    assert fingerprint["hardware"]["cuda_driver_supported_version"] == "12.8"
    assert fingerprint["environment"]["torch_cuda_version"] == "12.8"
    assert len(fingerprint["hardware"]["gpu_models"]) == 2
    assert [d["uuid"] for d in observation["gpu_devices"]] == ["GPU-fixture-0", "GPU-fixture-1"]
    assert observation["slurm"]["job_id"] == "job-1" and observation["slurm"]["node"] == "fixture-node"
    assert "slurm" not in fingerprint and "gpu_devices" not in fingerprint
    assert "model.safetensors" in fingerprint["snapshot_files"]
    assert len(linux_server["probes"]) == 1 and len(linux_server["commands"]) == 1
    assert "DO-NOT-EMIT" not in json.dumps(result)


def test_cuda_runtime_and_driver_capability_are_recorded_separately(linux_server, monkeypatch):
    original = serving._environment
    def probe(*args):
        return {**original(*args), "torch_cuda_version": "12.9"}
    monkeypatch.setattr(serving, "_environment", probe)
    result = serving.capture_serving_evidence(URL, MODEL, REVISION)
    assert result["fingerprint"]["environment"]["torch_cuda_version"] == "12.9"
    assert result["fingerprint"]["hardware"]["cuda_driver_supported_version"] == "12.8"
    assert serving.dependencies_verified(result["fingerprint"])


@pytest.mark.parametrize("selector", ["", "0,0", "7", "GPU-fixture-"])
def test_cuda_capture_refuses_missing_duplicate_or_unidentifiable_devices(linux_server, selector):
    with pytest.raises(RuntimeError, match="CUDA"):
        serving._cuda_hardware(selector)


def test_cuda_uuid_selection_and_unavailable_driver_evidence(linux_server, monkeypatch):
    hardware, devices = serving._cuda_hardware("GPU-fixture-2")
    assert devices[0]["index"] == "2" and len(hardware["gpu_models"]) == 1
    monkeypatch.setattr(serving, "_command", lambda args: gpu_xml().replace("<cuda_version>12.8", "<cuda_version>N/A"))
    with pytest.raises(RuntimeError, match="Incomplete NVIDIA"):
        serving._cuda_hardware("0")


@pytest.mark.parametrize("key", ["SLURM_JOB_ID", "SLURMD_NODENAME"])
def test_linux_capture_refuses_missing_job_or_node(linux_server, key):
    linux_server["environment"].pop(key)
    linux_server["save_env"]()
    with pytest.raises(RuntimeError, match="SLURM"):
        serving.capture_serving_evidence(URL, MODEL, REVISION)


def test_linux_backend_records_resubmission_and_guards_weights(linux_server, tmp_path, monkeypatch):
    def http(req, **kwargs):
        assert req.data is None  # No inference is needed to test construction evidence.
        return io.BytesIO(json.dumps({"data": [{"id": MODEL}]} if req.full_url.endswith("/models")
                                    else {"version": "0.30.0"}).encode())
    monkeypatch.setattr(urllib.request, "urlopen", http)
    path = tmp_path / "calls.jsonl"
    LocalBackend(CONFIG, path)
    linux_server["environment"]["SLURM_JOB_ID"] = "job-2"
    linux_server["environment"]["SLURMD_NODENAME"] = "second-fixture-node"
    linux_server["save_env"]()
    LocalBackend(CONFIG, path)
    calls = [json.loads(line) for line in path.read_text().splitlines()]
    assert [r["kind"] for r in calls] == ["server_info", "server_observation"]
    assert calls[0]["observed_listener"]["slurm"]["job_id"] == "job-1"
    assert calls[1]["observation"]["slurm"]["job_id"] == "job-2"
    validate_local_provenance(CONFIG, calls)
    before = path.read_bytes()
    (linux_server["snapshot"] / "model.safetensors").write_text("changed fixture weights")
    with pytest.raises(RuntimeError, match="server changed"):
        LocalBackend(CONFIG, path)
    assert path.read_bytes() == before


@pytest.mark.parametrize("field", ["torch", "cuda_runtime", "driver", "gpu_uuid", "job", "node", "backend"])
def test_analysis_refuses_incomplete_cuda_provenance(linux_server, field):
    captured = serving.capture_serving_evidence(URL, MODEL, REVISION)
    row = {"kind": "server_info", "tier": "T1", "server": {
        "server_version": "0.30.0", "serving_evidence": captured["fingerprint"]},
        "observed_listener": captured["observation"]}
    validate_local_provenance(CONFIG, [row])
    row = copy.deepcopy(row)
    fp = row["server"]["serving_evidence"]
    if field == "torch": fp["environment"]["packages"]["torch"] = None
    elif field == "cuda_runtime": fp["environment"]["torch_cuda_version"] = None
    elif field == "driver": fp["hardware"]["driver_version"] = None
    elif field == "gpu_uuid": row["observed_listener"]["gpu_devices"][0]["uuid"] = None
    elif field == "job": row["observed_listener"]["slurm"]["job_id"] = None
    elif field == "node": row["observed_listener"]["slurm"]["node"] = None
    elif field == "backend": fp["backend"] = "unknown"
    with pytest.raises(ValueError, match="unverified|missing"):
        validate_local_provenance(CONFIG, [row])


def test_environment_probe_uses_the_server_cache_and_only_selected_facts(tmp_path, monkeypatch):
    import ast
    monkeypatch.setattr(serving, "_is_linux", lambda: True)
    monkeypatch.setenv("HF_HOME", "wrong-runner-cache")
    monkeypatch.setenv("HF_HUB_CACHE", "wrong-runner-hub")
    seen = []
    def output(args, **kwargs):
        seen.append((args, kwargs))
        return json.dumps({"cache_root": "/server/hub", "packages": {"vllm": "fixture"}, "torch_cuda_version": "fixture"})
    monkeypatch.setattr(serving.subprocess, "check_output", output)
    serving._environment("/server/python", {"HF_HOME": "/server/home", "API_KEY": "DO-NOT-EMIT"})
    args, kwargs = seen[0]
    ast.parse(args[2])
    assert args[:2] == ["/server/python", "-c"]
    assert "torch.version.cuda" in args[2] and "vllm-metal" not in args[2]
    assert "DO-NOT-EMIT" not in args[2]
    assert kwargs["env"]["HF_HOME"] == "/server/home" and "HF_HUB_CACHE" not in kwargs["env"]
