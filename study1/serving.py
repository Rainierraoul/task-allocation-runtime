"""Passive provenance for macOS/Metal and Linux/CUDA vLLM servers.

Evidence links a live launch command to a pinned on-disk snapshot and its
serving interpreter. It is not a direct inspection of tensors in GPU memory.
No model inference, downloads, environment dumps or server changes occur.
"""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import re
import shlex
import subprocess
import sys
import xml.etree.ElementTree as ET
from urllib.parse import urlsplit


PACKAGES = ("vllm", "vllm-metal", "mlx", "mlx-lm", "mlx-vlm", "transformers",
            "huggingface-hub", "torch", "numpy")
LINUX_PACKAGES = ("vllm", "torch", "transformers", "huggingface-hub", "numpy")
PROC_ROOT = Path("/proc")
SERVER_ENV_KEYS = ("HF_HOME", "HF_HUB_CACHE", "CUDA_VISIBLE_DEVICES", "CUDA_DEVICE_ORDER",
                   "SLURM_JOB_ID", "SLURMD_NODENAME", "SLURM_JOB_NODELIST", "SLURM_STEP_ID")
FLAGS = ("--host", "--port", "--revision", "--tokenizer", "--tokenizer-revision",
         "--code-revision", "--max-model-len", "--gpu-memory-utilization",
         "--reasoning-parser", "--tool-call-parser", "--download-dir",
         "--dtype", "--quantization", "--generation-config",
         "--enable-auto-tool-choice", "--enable-prefix-caching", "--no-enable-prefix-caching",
         "--tensor-parallel-size", "--pipeline-parallel-size", "--kv-cache-dtype")


def _is_linux():
    return sys.platform.startswith("linux")


def _fact(value):
    return isinstance(value, str) and bool(value.strip()) and value.strip().upper() not in ("N/A", "UNKNOWN")


def _linux_listener(port):
    """Match LISTEN socket inodes to readable process descriptors in /proc.

    Fail closed when permissions prevent identifying the listener. No root,
    process mutation or additional system package is required.
    """
    inodes = set()
    for table in (PROC_ROOT / "net/tcp", PROC_ROOT / "net/tcp6"):
        if not table.exists():
            continue
        for line in table.read_text().splitlines()[1:]:
            fields = line.split()
            if len(fields) >= 10 and fields[3] == "0A" and int(fields[1].rsplit(":", 1)[1], 16) == port:
                inodes.add(fields[9])
    pids = set()
    for process in PROC_ROOT.iterdir():
        if not process.name.isdigit():
            continue
        try:
            for fd in (process / "fd").iterdir():
                try:
                    target = os.readlink(fd)
                except OSError:
                    continue
                if target.startswith("socket:[") and target[8:-1] in inodes:
                    pids.add(int(process.name))
        except OSError:
            continue
    if len(pids) != 1:
        raise RuntimeError(f"Expected one identifiable Linux listener on port {port}")
    pid = pids.pop()
    # Field 22 is the process start tick; parsing after ')' allows spaces in comm.
    fields = (PROC_ROOT / str(pid) / "stat").read_text().rsplit(")", 1)[1].split()
    start_tick = int(fields[19])
    boot_id = (PROC_ROOT / "sys/kernel/random/boot_id").read_text().strip()
    return {"pid": pid, "started_at": f"{boot_id}:{start_tick}"}


def _server_environment(pid):
    """Select nonsecret serving/job facts; never return a full process environment."""
    selected = {}
    for entry in (PROC_ROOT / str(pid) / "environ").read_bytes().split(b"\0"):
        key, equal, value = entry.partition(b"=")
        name = key.decode("ascii", errors="ignore")
        if equal and name in SERVER_ENV_KEYS:
            selected[name] = value.decode("utf-8")
    return selected


def _cuda_hardware(visible_devices):
    """Selected facts from the read-only nvidia-smi XML query, not live utilization.

    cuda_driver_supported_version is driver capability, distinct from PyTorch's
    linked CUDA runtime captured in the interpreter environment.
    """
    root = ET.fromstring(_command(["nvidia-smi", "-q", "-x"]))
    driver = root.findtext("driver_version")
    cuda = root.findtext("cuda_version")
    inventory = []
    for gpu in root.findall("gpu"):
        inventory.append({"index": gpu.findtext("minor_number"), "uuid": gpu.findtext("uuid"),
                          "model": gpu.findtext("product_name"), "pci_bus_id": gpu.findtext("pci/pci_bus_id"),
                          "memory_total": gpu.findtext("fb_memory_usage/total")})
    selectors = [v.strip() for v in visible_devices.split(",") if v.strip()]
    if not selectors or len(selectors) != len(set(selectors)):
        raise RuntimeError("Explicit unique CUDA_VISIBLE_DEVICES are required")
    selected = []
    for selector in selectors:
        matches = [gpu for gpu in inventory if gpu["index"] == selector or
                   (selector.startswith("GPU-") and (gpu["uuid"] or "").startswith(selector))]
        if len(matches) != 1:
            raise RuntimeError("CUDA device selector cannot be verified against nvidia-smi")
        selected.append(matches[0])
    if (not _fact(driver) or not _fact(cuda)
            or any(not all(_fact(gpu[k]) for k in ("model", "uuid", "memory_total")) for gpu in selected)):
        raise RuntimeError("Incomplete NVIDIA GPU, driver or CUDA evidence")
    fingerprint = {"driver_version": driver, "cuda_driver_supported_version": cuda,
                   "gpu_models": [{"model": g["model"], "memory_total": g["memory_total"]} for g in selected],
                   "cuda_visible_devices": visible_devices}
    return fingerprint, selected


def dependencies_verified(evidence):
    """Accept historical macOS records; CUDA records require their own evidence."""
    environment = evidence.get("environment", {})
    packages = environment.get("packages", {})
    backend = evidence.get("backend")
    if backend in (None, "macos-metal"):
        return all(packages.get(name) for name in ("vllm", "vllm-metal", "mlx"))
    if backend != "linux-cuda":
        return False
    hardware = evidence.get("hardware", {})
    return bool(all(_fact(packages.get(name)) for name in ("vllm", "torch", "transformers"))
                and _fact(environment.get("torch_cuda_version"))
                and _fact(hardware.get("driver_version")) and _fact(hardware.get("cuda_driver_supported_version"))
                and _fact(hardware.get("cuda_visible_devices")) and hardware.get("gpu_models")
                and all(all(_fact(g.get(k)) for k in ("model", "memory_total")) for g in hardware["gpu_models"]))


def cuda_observation_verified(evidence, observation):
    if not isinstance(observation, dict):
        return False
    slurm = observation.get("slurm", {})
    devices = observation.get("gpu_devices", [])
    models = evidence.get("hardware", {}).get("gpu_models", [])
    return bool(_fact(slurm.get("job_id")) and _fact(slurm.get("node"))
                and devices and len(devices) == len(models)
                and all(_fact(d.get("uuid")) and d.get("model") == m.get("model")
                        and d.get("memory_total") == m.get("memory_total") for d, m in zip(devices, models)))


def vllm_version_matches(api_version, package_version):
    # The CPU wheel has a local build suffix that /version omits. Preserve the
    # complete package version in evidence, while comparing its public version.
    return (isinstance(api_version, str) and isinstance(package_version, str)
            and api_version.split("+", 1)[0] == package_version.split("+", 1)[0])


def _command(args):
    return subprocess.check_output(args, text=True, timeout=30).strip()


def listener_identity(base_url):
    url = urlsplit(base_url)
    if url.scheme != "http" or url.hostname not in ("127.0.0.1", "localhost", "::1"):
        raise RuntimeError("Loaded-weight inspection requires a local loopback HTTP server")
    port = url.port or 80
    if _is_linux():
        return _linux_listener(port)
    result = _command(["/usr/sbin/lsof", "-nP", f"-iTCP:{port}", "-sTCP:LISTEN", "-Fp"])
    pids = {line[1:] for line in result.splitlines() if line.startswith("p")}
    if len(pids) != 1:
        raise RuntimeError(f"Expected one identifiable listener on port {port}")
    pid = pids.pop()
    return {"pid": int(pid), "started_at": _command(["/bin/ps", "-p", pid, "-o", "lstart="])}


def _launch(pid):
    tokens = ((PROC_ROOT / str(pid) / "cmdline").read_bytes().decode("utf-8").rstrip("\0").split("\0")
              if _is_linux() else shlex.split(_command(["/bin/ps", "-p", str(pid), "-o", "args="])))
    try:
        serve = tokens.index("serve")
        model, entrypoint = tokens[serve + 1], Path(tokens[serve - 1]).resolve()
    except (ValueError, IndexError) as exc:
        raise RuntimeError("Listener does not have an identifiable vllm serve command") from exc
    if entrypoint.name != "vllm":
        raise RuntimeError("Unrecognised serving entry point")
    selected = {}
    for i in range(serve + 2, len(tokens)):
        name, equal, value = tokens[i].partition("=")
        if name in FLAGS:
            if equal:
                selected[name] = value
            elif i + 1 < len(tokens) and not tokens[i + 1].startswith("--"):
                selected[name] = tokens[i + 1]
            else:
                selected[name] = True
    with entrypoint.open() as stream:
        first_line = stream.readline().strip()
    interpreter = first_line[2:] if first_line.startswith("#!/") else ""
    if not interpreter or " " in interpreter or not Path(interpreter).is_file():
        raise RuntimeError("Cannot identify the serving interpreter from the vllm entry point")
    return model, selected, interpreter


def _environment(interpreter, server_environment=None):
    # Only explicitly selected package/platform facts are emitted. Never emit
    # process environments, credentials or the complete server configuration.
    linux = _is_linux()
    code = """import importlib.metadata as m, json, platform, sys
from huggingface_hub.constants import HF_HUB_CACHE
packages = {}
for name in %r:
    try: packages[name] = m.version(name)
    except m.PackageNotFoundError: packages[name] = None
evidence = {'python': sys.version, 'platform': platform.platform(),
            'machine': platform.machine(), 'packages': packages,
            'cache_root': HF_HUB_CACHE}
if %r:
    import torch
    evidence['torch_cuda_version'] = torch.version.cuda
print(json.dumps(evidence))
""" % (LINUX_PACKAGES if linux else PACKAGES, linux)
    environment = {**os.environ, "PYTHONDONTWRITEBYTECODE": "1"}
    if server_environment is not None:
        for key in ("HF_HOME", "HF_HUB_CACHE"):
            environment.pop(key, None)
            if key in server_environment:
                environment[key] = server_environment[key]
    return json.loads(subprocess.check_output([interpreter, "-c", code], text=True,
                                              timeout=30, env=environment))


def snapshot_inventory(snapshot):
    snapshot = Path(snapshot)
    paths = sorted(p for p in snapshot.rglob("*") if p.is_file() and
                   p.suffix in (".json", ".safetensors", ".bin", ".pt", ".model", ".jinja", ".txt", ".tiktoken"))
    names = {str(p.relative_to(snapshot)) for p in paths}
    if "config.json" not in names or "tokenizer_config.json" not in names:
        raise RuntimeError("Snapshot lacks model or tokenizer configuration")
    if not any(name.endswith((".safetensors", ".bin", ".pt")) for name in names):
        raise RuntimeError("Snapshot lacks model weights")
    if not any(name in names for name in ("tokenizer.json", "tokenizer.model", "vocab.json")):
        raise RuntimeError("Snapshot lacks tokenizer assets")
    for index in paths:
        if index.name.endswith(".index.json"):
            weight_map = json.loads(index.read_text()).get("weight_map", {})
            if not set(weight_map.values()).issubset(names):
                raise RuntimeError("Snapshot is missing indexed weight shards")
    inventory = {}
    for path in paths:
        h = hashlib.sha256()
        with path.open("rb") as stream:
            for block in iter(lambda: stream.read(1024 * 1024), b""):
                h.update(block)
        inventory[str(path.relative_to(snapshot))] = {"bytes": path.stat().st_size, "sha256": h.hexdigest()}
    return inventory


def capture_serving_evidence(base_url, model, revision):
    if not re.fullmatch(r"[0-9a-f]{40}", revision):
        raise RuntimeError("A full pinned model revision is required")
    identity = listener_identity(base_url)
    launched_model, flags, interpreter = _launch(identity["pid"])
    if launched_model != model or flags.get("--revision") != revision:
        raise RuntimeError("Live server launch does not match the pinned model and revision")
    if flags.get("--tokenizer", model) != model or flags.get("--tokenizer-revision", revision) != revision:
        raise RuntimeError("Live tokenizer does not match the pinned model snapshot")
    linux = _is_linux()
    server_environment = _server_environment(identity["pid"]) if linux else None
    environment = (_environment(interpreter, server_environment) if linux else _environment(interpreter))
    fingerprint = {"model": model, "revision": revision, "launch_settings": flags,
                   "environment": environment}
    observation = dict(identity)
    if linux:
        hardware, devices = _cuda_hardware(server_environment.get("CUDA_VISIBLE_DEVICES", ""))
        fingerprint.update(backend="linux-cuda", hardware=hardware)
        slurm = {"job_id": server_environment.get("SLURM_JOB_ID"),
                 "node": server_environment.get("SLURMD_NODENAME"),
                 "node_list": server_environment.get("SLURM_JOB_NODELIST"),
                 "step_id": server_environment.get("SLURM_STEP_ID")}
        if not slurm["job_id"] or not slurm["node"]:
            raise RuntimeError("Serving SLURM job and node evidence is required")
        observation.update(slurm=slurm, gpu_devices=devices,
                           cuda_device_order=server_environment.get("CUDA_DEVICE_ORDER"))
    if not dependencies_verified(fingerprint):
        raise RuntimeError("Serving dependency evidence is incomplete")
    cache_root = Path(flags.get("--download-dir") or environment.pop("cache_root"))
    environment.pop("cache_root", None)
    snapshot = cache_root / ("models--" + model.replace("/", "--")) / "snapshots" / revision
    inventory = snapshot_inventory(snapshot)
    if (listener_identity(base_url) != identity or _launch(identity["pid"]) != (launched_model, flags, interpreter)
            or (linux and _server_environment(identity["pid"]) != server_environment)):
        raise RuntimeError("Server changed during provenance capture")
    fingerprint["snapshot_files"] = inventory
    return {"fingerprint": fingerprint, "observation": observation}
