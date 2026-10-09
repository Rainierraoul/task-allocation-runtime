"""Model calls with durable evidence and a resume cache: a paid API with a spending cap, or local servers.

No credentials are written to a log. Cached calls preserve repetitions because
each request ID includes the condition, instance, repetition and task/call index.
"""
from __future__ import annotations

import copy
import json
import math
import os
import time
import threading
import urllib.error
import urllib.request
import uuid
from datetime import datetime, timezone
from pathlib import Path

from dcarr_runtime.executors import MODEL_SYSTEM_PROMPT, _parse_json_object

FORCED_SYSTEM = MODEL_SYSTEM_PROMPT.replace(
    '"payload": an object with exactly the hand-off fields listed, or null if you do not decide',
    '"payload": an object with exactly the hand-off fields listed; ALWAYS give your best candidate even if unsure'
) + "\nFor this comparison, always supply your best candidate and set decided=true. Do not abstain."
DIAGNOSTIC_SYSTEM = MODEL_SYSTEM_PROMPT.replace(
    '"payload": an object with exactly the hand-off fields listed, or null if you do not decide',
    '"payload": ALWAYS give your best candidate with exactly the hand-off fields, including when decided=false'
) + "\nYour candidate is scored separately from your willingness to decide. Do not omit it when abstaining."
MONOLITHIC_SYSTEM = """Make all nine decisions in this synthetic tanker-join scenario.
Return one JSON object with a key for each task ID. Each value is a best-candidate
payload with exactly that task's fields. Give null for tasks skipped by the plan.
Do not abstain. Work through the plan in order. Decide DEV yourself from the
training objectives and exercise-plan intent; no person's answer is available.
If ACK is not acknowledged, skip DEV through CALL and report goal_met=false in FLY.
Units: x,y nautical miles, speed knots, time seconds, heading 0 north (+y), 90 east (+x).
The tanker moves clockwise on a circular orbit: x=cx+r*sin(angle+speed*t/3600/r),
y=cy+r*cos(angle+speed*t/3600/r). Aircraft move at fixed heading and speed.
HDG uses the earliest point of interception with the moving tanker. WAIT reports
the first five-second sample at or below the ordered reporting distance.
If a channel is busy, defer. If unexpectedly jammed, use backup. A scheduled
jamming inject is a training-design choice; choose and explain the response.
Keep each task's payload within its stated budget. No prose outside JSON."""


SYSTEMS = {"allocated": MODEL_SYSTEM_PROMPT, "forced": FORCED_SYSTEM,
           "diagnostic": DIAGNOSTIC_SYSTEM, "monolithic": MONOLITHIC_SYSTEM}


class BudgetExceeded(RuntimeError):
    pass


def ledger_state(rows):
    """Validate attempt order, retaining support for pre-attempt-ID ledgers."""
    attempts, pending, completed, hashes = {}, {}, {}, {}
    for number, row in enumerate(rows):
        kind = row.get("kind")
        if kind == "unresolved_call":
            raise ValueError("Provider ledger contains unresolved requests")
        if kind not in ("request_started", "response_received", "failed_call", "provider_call"):
            continue
        request_id = row["request_id"]
        if kind == "request_started":
            attempt_id = row.get("attempt_id", f"legacy:{number}:{request_id}")
            if attempt_id in attempts or request_id in pending or request_id in completed:
                raise ValueError("Duplicate or overlapping request attempt")
            request_hash = row.get("request_sha256")
            if request_id in hashes and hashes[request_id] != request_hash:
                raise ValueError("Frozen request changed between attempts")
            hashes[request_id] = request_hash
            attempts[attempt_id] = {"start": row, "terminal": None, "response": None}
            pending[request_id] = attempt_id
        else:
            attempt_id = row.get("attempt_id", pending.get(request_id))
            if pending.get(request_id) != attempt_id or attempt_id not in attempts:
                raise ValueError("Response or failure has no matching open attempt")
            attempt = attempts[attempt_id]
            if row.get("request_sha256", hashes[request_id]) != hashes[request_id]:
                raise ValueError("Response request hash disagrees with its attempt")
            if kind == "response_received":
                if attempt["response"] is not None:
                    raise ValueError("Duplicate received response")
                attempt["response"] = row
            else:
                attempt["terminal"] = row
                del pending[request_id]
                if kind == "provider_call":
                    completed[request_id] = row
    return attempts, pending, completed


def failed_attempt_timing(rows, run_id=None):
    """Summarise measured failure durations without imputing missing times.

    This is raw evidence, not a decision about the published timing measure.
    """
    failures = [r for r in rows if r.get("kind") == "failed_call" and
                (run_id is None or r["request_id"].startswith(run_id + "/"))]
    known, unknown = [], 0
    for row in failures:
        duration = row.get("latency_s")
        if duration is None:
            unknown += 1
        elif (isinstance(duration, bool) or not isinstance(duration, (int, float))
              or not math.isfinite(duration) or duration < 0):
            raise ValueError("Invalid failed-attempt duration")
        else:
            known.append(duration)
    return {"failed_attempts": len(failures), "known_failed_latency_s": sum(known),
            "unknown_timing_attempts": unknown,
            "total_failed_latency_s": None if unknown else sum(known)}


class PaidBackend:
    """One provider and two explicitly configured model IDs; no silent fallback."""

    def __init__(self, config, log_path: Path, budget_usd: float, api_key_file: Path | None = None):
        if budget_usd <= 0:
            raise ValueError("A positive spending limit is required")
        self.key = os.environ.get("ANTHROPIC_API_KEY")
        if not self.key and api_key_file is not None:
            self.key = Path(api_key_file).read_text().strip()
        if not self.key:
            raise RuntimeError("ANTHROPIC_API_KEY must be configured locally; never paste it into chat")
        self.config = config
        self.path = Path(log_path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.budget_usd = budget_usd
        self.cache = {}
        self.spent = 0.0
        self.returned_models = {}
        if self.path.exists():
            pending = set()
            for line in self.path.read_text().splitlines():
                row = json.loads(line)
                if row.get("kind") == "request_started":
                    pending.add(row["request_id"])
                if row.get("kind") == "provider_call":
                    pending.discard(row["request_id"])
                    self.cache[row["request_id"]] = row
                    self.spent += row["cost_usd"]
                    previous = self.returned_models.setdefault(row["tier"], row["returned_model"])
                    if previous != row["returned_model"]:
                        raise RuntimeError("Provider ledger mixes model versions")
                elif row.get("kind") == "unresolved_call":
                    raise RuntimeError("An interrupted API call requires reconciliation before resuming")
            if pending:
                raise RuntimeError("Unfinished request journal requires billing reconciliation before resuming")
        self.run_id = ""
        self.call_index = 0
        self.last_rows = []

    def start(self, run_id):
        self.run_id, self.call_index, self.last_rows = run_id, 0, []

    def append(self, row):
        with self.path.open("a") as f:
            f.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n")
            f.flush()
            os.fsync(f.fileno())

    def call(self, tier, context, mode="allocated"):
        self.call_index += 1
        request_id = f"{self.run_id}/{self.call_index}/{tier}/{mode}"
        model = self.config["models"][tier]
        system = SYSTEMS[mode]
        request = {"model": model, "max_tokens": self.config["max_output_tokens"],
                   "system": system, "messages": [{"role": "user", "content": json.dumps(context, sort_keys=True)}]}
        # Configure thinking/effort explicitly in the saved model settings when used.
        request.update(self.config.get("model_settings", {}).get(tier, {}))
        encoded = json.dumps(request).encode()
        from .instances import digest
        request_hash = digest(request)
        if request_id in self.cache:
            row = self.cache[request_id]
            if row["request_sha256"] != request_hash:
                raise RuntimeError(f"Frozen request changed: {request_id}")
            self.last_rows.append({**row, "replayed": True})
            return row["parsed"], row["input_tokens"], row["output_tokens"]
        price = self.config["prices_usd_per_million"][tier]
        # Conservative byte bound plus ample provider wrapper allowance. All study
        # prompts are text. Reserve max output, including any reasoning tokens.
        reserve = ((len(encoded) + 4096) * price["input"] + request["max_tokens"] * price["output"]) / 1e6
        if self.spent + reserve > self.budget_usd:
            raise BudgetExceeded(f"Cap reached: spent ${self.spent:.4f}; next-call reserve ${reserve:.4f}")
        self.append({"kind": "request_started", "request_id": request_id, "request_sha256": request_hash,
                     "request": request, "reserved_usd": reserve, "utc": datetime.now(timezone.utc).isoformat()})
        req = urllib.request.Request("https://api.anthropic.com/v1/messages", data=encoded,
                                     headers={"Content-Type": "application/json", "x-api-key": self.key,
                                              "anthropic-version": "2023-06-01"}, method="POST")
        start = time.perf_counter()
        try:
            with urllib.request.urlopen(req, timeout=180) as response:
                raw = json.loads(response.read())
        except Exception as exc:
            # Do not retry an uncertain billed call or silently replace a model.
            self.append({"kind": "unresolved_call", "request_id": request_id, "error_type": type(exc).__name__})
            raise
        usage = raw["usage"]
        if usage.get("cache_creation_input_tokens", 0) or usage.get("cache_read_input_tokens", 0):
            self.append({"kind": "unresolved_call", "request_id": request_id, "error_type": "UnexpectedCacheUsage"})
            raise RuntimeError("Unexpected prompt-cache usage; reconcile prices before continuing")
        tokens_in, tokens_out = usage["input_tokens"], usage["output_tokens"]
        cost = (tokens_in * price["input"] + tokens_out * price["output"]) / 1e6
        text = "".join(block.get("text", "") for block in raw["content"] if block.get("type") == "text")
        parsed = _parse_json_object(text)
        row = {"kind": "provider_call", "request_id": request_id, "request_sha256": request_hash,
               "tier": tier, "requested_model": model, "returned_model": raw["model"],
               "provider_message_id": raw["id"], "request": request, "raw_response": raw,
               "parsed": parsed, "input_tokens": tokens_in, "output_tokens": tokens_out,
               "cost_usd": cost, "latency_s": time.perf_counter() - start,
               "utc": datetime.now(timezone.utc).isoformat(), "empirical": True}
        self.append(row)
        self.cache[request_id] = row
        self.spent += cost
        self.last_rows.append({**row, "replayed": False})
        previous = self.returned_models.setdefault(tier, raw["model"])
        if previous != raw["model"]:
            raise RuntimeError("Provider changed a model version; do not mix these records")
        return parsed, tokens_in, tokens_out


class LocalBackend:
    """Two open-weight models on local OpenAI-compatible servers, such as vLLM.

    The ledger has the same rows as PaidBackend's, so analysis is unchanged, and
    cost is zero. A local call is not billed, so a request that ends without a
    response may be sent again on resume; the journal keeps the failed attempt.
    Each tier's server version and served model are recorded once, and a change
    stops the study instead of mixing records.

    One instance may be shared by threads, each with its own active run. The
    ledger and response cache share one lock; inference runs outside that lock.
    Separate processes or backend instances must not share a ledger writer.
    """

    thread_safe = True

    def __init__(self, config, log_path: Path):
        self._lock = threading.RLock()
        self._local = threading.local()
        self._inflight = set()
        self._failed_rows = []
        self._attempt_counts = {}
        missing = [tier for tier in config["models"] if tier not in config.get("base_urls", {})]
        if missing:
            raise ValueError(f"No base_urls entry for {', '.join(missing)}")
        self.config = copy.deepcopy(config)
        self.path = Path(log_path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.budget_usd = None
        self.cache = {}
        self.spent = 0.0
        self.returned_models = {}
        self.servers = {}
        self.server_observations = {}
        self.failed_attempts = 0
        history = []
        if self.path.exists():
            for line in self.path.read_text().splitlines():
                row = json.loads(line)
                history.append(row)
                if row.get("kind") == "provider_call":
                    self.cache[row["request_id"]] = row
                    previous = self.returned_models.setdefault(row["tier"], row["returned_model"])
                    if previous != row["returned_model"]:
                        raise RuntimeError("Ledger mixes model versions")
                elif row.get("kind") == "server_info":
                    self.servers[row["tier"]] = row["server"]
                elif row.get("kind") == "failed_call":
                    self.failed_attempts += 1
                    self._failed_rows.append(row)
                if row.get("kind") == "request_started":
                    rid = row["request_id"]
                    self._attempt_counts[rid] = self._attempt_counts.get(rid, 0) + 1
        attempts, pending, _ = ledger_state(history)
        self.request_hashes = {a["start"]["request_id"]: a["start"].get("request_sha256")
                               for a in attempts.values()}
        for tier in config["models"]:
            server = self._server_info(tier)
            if tier in self.servers and self.servers[tier] != server:
                raise RuntimeError(f"The {tier} server changed since this ledger began: "
                                   f"{self.servers[tier]} became {server}. Use a new output directory.")
            if tier not in self.servers:
                self.servers[tier] = server
                self.append({"kind": "server_info", "tier": tier, "server": server,
                             "revision": config.get("revisions", {}).get(tier),
                             "observed_listener": self.server_observations.get(tier),
                             "utc": datetime.now(timezone.utc).isoformat()})
            elif server.get("serving_evidence", {}).get("backend") == "linux-cuda":
                # Job/PID/node observations change on resubmission. Preserve
                # them separately from the guarded snapshot/software fingerprint.
                self.append({"kind": "server_observation", "tier": tier,
                             "observation": self.server_observations[tier],
                             "utc": datetime.now(timezone.utc).isoformat()})
        # Close interrupted attempts explicitly. A durably received response is
        # recovered without another inference; an unobserved response stays unknown.
        for request_id, attempt_id in pending.items():
            attempt = attempts[attempt_id]
            started = attempt["start"]
            if attempt["response"] is not None:
                received = attempt["response"]
                row = self._provider_row(started, received["raw_response"], received["latency_s"])
                row["recovered_response"] = True
                row["utc"] = received["utc"]
                row["recovered_utc"] = datetime.now(timezone.utc).isoformat()
                self.append(row)
                self.cache[request_id] = row
            else:
                failed = {"kind": "failed_call", "request_id": request_id,
                             **({"attempt_id": started["attempt_id"]} if "attempt_id" in started else {}),
                             "request_sha256": started.get("request_sha256"),
                             "error_type": "ProcessInterrupted", "latency_s": None,
                             "timing_unknown": True, "utc": datetime.now(timezone.utc).isoformat()}
                self.append(failed)
                self._failed_rows.append(failed)
                self.failed_attempts += 1
        self.run_id = ""
        self.call_index = 0
        self.last_rows = []

    def _get(self, url):
        with urllib.request.urlopen(urllib.request.Request(url, headers={"Authorization": "Bearer local"}), timeout=30) as r:
            return json.loads(r.read())

    def _server_info(self, tier):
        base = self.config["base_urls"][tier].rstrip("/")
        model = self.config["models"][tier]
        served = {m["id"]: m for m in self._get(f"{base}/models")["data"]}
        if model not in served:
            raise RuntimeError(f"{base} does not serve {model}; it serves {sorted(served)}")
        root = base[: -len("/v1")] if base.endswith("/v1") else base
        try:
            version = self._get(f"{root}/version").get("version")
        except (urllib.error.URLError, json.JSONDecodeError):
            version = None
        entry = served[model]
        result = {"model": model, "root": entry.get("root"), "max_model_len": entry.get("max_model_len"),
                  "server_version": version}
        revision = self.config.get("revisions", {}).get(tier)
        if revision:
            from .serving import capture_serving_evidence, vllm_version_matches
            evidence = capture_serving_evidence(base, model, revision)
            if not vllm_version_matches(version, evidence["fingerprint"]["environment"]["packages"]["vllm"]):
                raise RuntimeError("Live API version disagrees with the serving interpreter")
            result["serving_evidence"] = evidence["fingerprint"]
            self.server_observations[tier] = evidence["observation"]
        return result

    @property
    def run_id(self):
        return getattr(self._local, "run_id", "")

    @run_id.setter
    def run_id(self, value):
        self._local.run_id = value

    @property
    def call_index(self):
        return getattr(self._local, "call_index", 0)

    @call_index.setter
    def call_index(self, value):
        self._local.call_index = value

    @property
    def last_rows(self):
        if not hasattr(self._local, "last_rows"):
            self._local.last_rows = []
        return self._local.last_rows

    @last_rows.setter
    def last_rows(self, value):
        self._local.last_rows = value

    def start(self, run_id):
        if not run_id:
            raise ValueError("A nonempty run ID is required")
        self.run_id, self.call_index, self.last_rows = run_id, 0, []
        with self._lock:
            self._local.prior_failed_timing = failed_attempt_timing(self._failed_rows, run_id)

    def run_timing(self):
        with self._lock:
            failed = failed_attempt_timing(self._failed_rows, self.run_id)
            retries = sum(max(0, n - 1) for rid, n in self._attempt_counts.items()
                          if rid.startswith(self.run_id + "/"))
        return {**failed, "retry_count": retries,
                "prior_known_failed_latency_s": self._local.prior_failed_timing["known_failed_latency_s"]}

    def append(self, row):
        # This same reentrant lock guards cache/reservations and each complete
        # durable journal write. It is never held during model inference.
        with self._lock:
            PaidBackend.append(self, row)

    def _provider_row(self, started, raw, latency_s):
        request = started["request"]
        model = request["model"]
        tier = started["tier"]
        if raw["model"] != model:
            raise RuntimeError("Server returned a model other than the requested model")
        return {"kind": "provider_call", "request_id": started["request_id"],
                "attempt_id": started["attempt_id"], "request_sha256": started["request_sha256"],
                "tier": tier, "requested_model": model, "returned_model": raw["model"],
                "provider_message_id": raw["id"], "request": request, "raw_response": raw,
                "finish_reason": raw["choices"][0].get("finish_reason"),
                "parsed": _parse_json_object(raw["choices"][0]["message"].get("content") or ""),
                "input_tokens": raw["usage"]["prompt_tokens"],
                "output_tokens": raw["usage"]["completion_tokens"],
                "cost_usd": 0.0, "latency_s": latency_s,
                "utc": datetime.now(timezone.utc).isoformat(), "empirical": True}

    def call(self, tier, context, mode="allocated"):
        if not self.run_id:
            raise RuntimeError("Call start(run_id) in this thread before inference")
        self.call_index += 1
        request_id = f"{self.run_id}/{self.call_index}/{tier}/{mode}"
        model = self.config["models"][tier]
        max_tokens = self.config.get("max_output_tokens_by_tier", {}).get(tier, self.config["max_output_tokens"])
        request = {"model": model, "max_tokens": max_tokens,
                   "messages": [{"role": "system", "content": SYSTEMS[mode]},
                                {"role": "user", "content": json.dumps(context, sort_keys=True)}]}
        request.update(self.config.get("model_settings", {}).get(tier, {}))
        from .instances import digest
        request_hash = digest(request)
        with self._lock:
            if request_id in self.request_hashes and self.request_hashes[request_id] != request_hash:
                raise RuntimeError(f"Frozen request changed: {request_id}")
            if request_id in self.cache:
                row = self.cache[request_id]
                if row["request_sha256"] != request_hash:
                    raise RuntimeError(f"Frozen request changed: {request_id}")
                self.last_rows.append({**copy.deepcopy(row), "replayed": True})
                return copy.deepcopy(row["parsed"]), row["input_tokens"], row["output_tokens"]
            if request_id in self._inflight:
                raise RuntimeError(f"Request already in flight: {request_id}")
            self._inflight.add(request_id)
        started = {"kind": "request_started", "request_id": request_id,
                   "attempt_id": str(uuid.uuid4()), "tier": tier, "request_sha256": request_hash,
                   "request": request, "utc": datetime.now(timezone.utc).isoformat()}
        with self._lock:
            try:
                self.append(started)
            except BaseException:
                self._inflight.discard(request_id)
                raise
            self.request_hashes[request_id] = request_hash
            self._attempt_counts[request_id] = self._attempt_counts.get(request_id, 0) + 1
        url = self.config["base_urls"][tier].rstrip("/") + "/chat/completions"
        req = urllib.request.Request(url, data=json.dumps(request).encode(), method="POST",
                                     headers={"Content-Type": "application/json", "Authorization": "Bearer local"})
        start = time.perf_counter()
        try:
            with urllib.request.urlopen(req, timeout=self.config.get("timeout_s", 600)) as response:
                raw = json.loads(response.read())
            latency_s = time.perf_counter() - start
            self.append({"kind": "response_received", "request_id": request_id,
                         "attempt_id": started["attempt_id"], "request_sha256": request_hash,
                         "raw_response": raw, "latency_s": latency_s,
                         "utc": datetime.now(timezone.utc).isoformat()})
            row = self._provider_row(started, raw, latency_s)
        except BaseException as exc:
            failed = {"kind": "failed_call", "request_id": request_id,
                         "attempt_id": started["attempt_id"], "request_sha256": request_hash,
                         "error_type": type(exc).__name__, "latency_s": time.perf_counter() - start,
                         "timing_unknown": False, "utc": datetime.now(timezone.utc).isoformat()}
            with self._lock:
                self.append(failed)
                self._failed_rows.append(failed)
                self.failed_attempts += 1
                self._inflight.discard(request_id)
            raise
        with self._lock:
            try:
                self.append(row)
                self.cache[request_id] = row
                previous = self.returned_models.setdefault(tier, raw["model"])
                if previous != raw["model"]:
                    raise RuntimeError("A server changed its model; do not mix these records")
            finally:
                self._inflight.discard(request_id)
        self.last_rows.append({**copy.deepcopy(row), "replayed": False})
        return copy.deepcopy(row["parsed"]), row["input_tokens"], row["output_tokens"]


class ExecutorBackend:
    def __init__(self, paid, tier, mode="allocated"):
        self.paid, self.tier, self.mode = paid, tier, mode
        self.label = f"{paid.config.get('provider', 'mock')}:{paid.config['models'][tier]}"

    def has(self, name):
        return True

    def respond(self, name, context):
        return self.paid.call(self.tier, context, self.mode)
