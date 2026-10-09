"""Paired instance bootstrap. Refuses incomplete or mock research evidence."""
from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path

import numpy as np

from .harness import CONDITIONS, task_models
from .instances import digest
from .oracle import eligible_tasks, gold, score_run, score


def validate_run_score(row, instance):
    scored = score_run(instance, row["results"])
    accuracy = sum(scored.values()) / len(scored)
    if row["correct_by_task"] != scored or not math.isclose(row["accuracy"], accuracy, abs_tol=1e-12):
        raise ValueError("Stored condition score disagrees with the independent oracle")


def validate_usage(row, calls):
    selected = [c for c in calls if c["request_id"].startswith(row["run_id"] + "/")]
    for measure in ("input_tokens", "output_tokens"):
        if row[measure] != sum(c[measure] for c in selected):
            raise ValueError("Run usage disagrees with provider ledger")
    if "cost_usd" in row and not math.isclose(row["cost_usd"], sum(c["cost_usd"] for c in selected), abs_tol=1e-12):
        raise ValueError("Run cost disagrees with provider ledger")
    if "response" in row and (len(selected) != 1 or row["response"] != selected[0]["parsed"]):
        raise ValueError("Diagnostic response disagrees with provider ledger")


def validate_timing(row, calls=None):
    """Unavailable completion time is explicit and supported by attempt evidence."""
    keys = ("failed_attempts", "retry_count", "unknown_timing_attempts")
    for key in keys:
        count = row.get(key, 0)
        if isinstance(count, bool) or not isinstance(count, int) or count < 0:
            raise ValueError("Invalid attempt count")
    if row.get("unknown_timing_attempts", 0) > row.get("failed_attempts", 0):
        raise ValueError("Unknown attempt count exceeds failed attempts")
    known = row.get("known_failed_latency_s", 0.0)
    duration = row["time_s"]
    if (isinstance(known, bool) or not isinstance(known, (int, float))
            or not math.isfinite(known) or known < 0):
        raise ValueError("Invalid known failed-attempt time")
    if row.get("unknown_timing_attempts", 0):
        if duration is not None:
            raise ValueError("Unknown attempt duration requires unavailable run time")
    elif (isinstance(duration, bool) or not isinstance(duration, (int, float))
          or not math.isfinite(duration) or duration < known):
        raise ValueError("Invalid or unavailable run time without unknown attempt evidence")
    if calls is not None:
        from .backend import failed_attempt_timing
        selected = [c for c in calls if c.get("request_id", "").startswith(row["run_id"] + "/")]
        expected = failed_attempt_timing(selected)
        counts = {}
        for call in selected:
            if call["kind"] == "request_started":
                rid = call["request_id"]
                counts[rid] = counts.get(rid, 0) + 1
        expected["retry_count"] = sum(max(n - 1, 0) for n in counts.values())
        if any(row.get(k, 0) != expected[k] for k in keys) or not math.isclose(
                known, expected["known_failed_latency_s"], abs_tol=1e-12):
            raise ValueError("Run attempt timing disagrees with provider ledger")

BOOTSTRAP_N = 10000
BOOTSTRAP_SEED = 20261010
MEASURES = ["accuracy", "input_tokens", "output_tokens", "max_context_tokens", "cost_usd", "time_s"]


def read_rows(path):
    return [json.loads(line) for line in path.read_text().splitlines()]


def interval(values):
    finite = np.asarray(values)[np.isfinite(values)]
    return [float(v) for v in np.percentile(finite, [2.5, 97.5])] if len(finite) else [None, None]


def available_mean(values, axis=None):
    """Mean over available observations, without zero imputation or warnings."""
    array = np.asarray(values, dtype=float)
    finite = np.isfinite(array)
    count = finite.sum(axis=axis)
    total = np.where(finite, array, 0.0).sum(axis=axis)
    return np.divide(total, count, out=np.full(np.shape(total), np.nan), where=count != 0)


def timing_estimate(rows, ids, draws, other=None):
    """Cluster by instance; paired timing uses jointly available repetitions."""
    own = {(r["instance_id"], r["repetition"]): r["time_s"] for r in rows}
    paired = None if other is None else {
        (r["instance_id"], r["repetition"]): r["time_s"] for r in other}
    instance_values, available_runs = [], 0
    rep_counts = []
    for instance in ids:
        values = []
        for key, value in own.items():
            if key[0] != instance or value is None:
                continue
            if paired is not None:
                if paired[key] is None:
                    continue
                value -= paired[key]
            values.append(value)
        available_runs += len(values)
        rep_counts.append(len(values))
        instance_values.append(float(available_mean(values)))
    per_instance = np.asarray(instance_values)
    boot = available_mean(per_instance[draws], axis=1)
    mean = float(available_mean(per_instance))
    return {"mean": mean if math.isfinite(mean) else None, "ci95": interval(boot),
            "available_runs": available_runs, "unavailable_runs": len(own) - available_runs,
            "available_instances": sum(n > 0 for n in rep_counts),
            "unavailable_instances": sum(n == 0 for n in rep_counts),
            "timed_repetitions_by_instance": dict(zip(ids, rep_counts)),
            "usable_bootstrap_resamples": int(np.isfinite(boot).sum())}


def write_csv(path, rows):
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def check_ledger(calls):
    """Every request must end in exactly one provider call, with nothing left unresolved."""
    from .backend import failed_attempt_timing, ledger_state
    try:
        attempts, pending, completed = ledger_state(calls)
    except ValueError as exc:
        raise ValueError(f"Provider ledger contains unresolved or invalid requests: {exc}") from exc
    requests = {a["start"]["request_id"] for a in attempts.values()}
    if pending or requests != set(completed):
        raise ValueError("Provider ledger contains unresolved requests")
    failed_attempt_timing(calls)


def validate_local_provenance(config, calls):
    """Local empirical evidence must establish both pinned serving snapshots."""
    if config.get("provider") != "local-openai":
        return
    import re
    from .serving import cuda_observation_verified, dependencies_verified, vllm_version_matches
    servers = [r for r in calls if r.get("kind") == "server_info"]
    models = config["models"]
    if len(servers) != len(models) or {r["tier"] for r in servers} != set(models):
        raise ValueError("Missing or duplicate local serving provenance")
    for row in servers:
        tier = row["tier"]
        revision = config.get("revisions", {}).get(tier)
        evidence = row["server"].get("serving_evidence", {})
        packages = evidence.get("environment", {}).get("packages", {})
        files = evidence.get("snapshot_files", {})
        if (not isinstance(revision, str) or not re.fullmatch(r"[0-9a-f]{40}", revision)
                or evidence.get("model") != models[tier] or evidence.get("revision") != revision
                or evidence.get("launch_settings", {}).get("--revision") != revision
                or not dependencies_verified(evidence)
                or not vllm_version_matches(row["server"].get("server_version"), packages.get("vllm"))
                or not {"config.json", "tokenizer_config.json"}.issubset(files)
                or not {"tokenizer.json", "tokenizer.model", "vocab.json"}.intersection(files)
                or not any(name.endswith((".safetensors", ".bin", ".pt")) for name in files)):
            raise ValueError("Local model revision, tokenizer or serving environment is unverified")
        if evidence.get("backend") == "linux-cuda":
            observations = [row.get("observed_listener")] + [r.get("observation") for r in calls
                            if r.get("kind") == "server_observation" and r.get("tier") == tier]
            if any(not cuda_observation_verified(evidence, o) for o in observations):
                raise ValueError("Linux SLURM job, node or GPU observation is missing")
        if any(not re.fullmatch(r"[0-9a-f]{64}", f.get("sha256", ""))
               or not isinstance(f.get("bytes"), int) or f["bytes"] < 0 for f in files.values()):
            raise ValueError("Invalid local snapshot file evidence")
    for row in calls:
        if row.get("kind") == "provider_call" and (
                row.get("requested_model") != models.get(row.get("tier"))
                or row.get("returned_model") != models.get(row.get("tier"))):
            raise ValueError("Call model disagrees with verified local serving provenance")


def validate_dataset(folder, allow_mock=False):
    manifest = json.loads((folder / "manifest.json").read_text())
    material = json.loads((folder / "materials.json").read_text())
    if digest(material) != manifest["materials_sha256"]:
        raise ValueError("Material hash mismatch")
    expected_empirical = manifest["empirical"]
    if not expected_empirical and not allow_mock:
        raise ValueError("Mock outputs are plumbing checks, not empirical research; analysis refused")
    conditions = read_rows(folder / "conditions.jsonl")
    diagnostics = read_rows(folder / "diagnostics.jsonl")
    policies = read_rows(folder / "policy-sequences.jsonl")
    test = [i for i in material["instances"] if i["split"] == "test"]
    by_id = {i["id"]: i for i in test}
    ids = {i["id"] for i in test}
    repetitions = manifest["repetitions"]
    wanted = {(i, c, r) for i in ids for c in CONDITIONS for r in range(repetitions)}
    got = {(r["instance_id"], r["condition"], r["repetition"]) for r in conditions}
    if got != wanted or len(got) != len(conditions):
        raise ValueError("Missing, extra or duplicate condition runs")
    wanted_d = {(i["id"], task, tier, r) for i in test for task in eligible_tasks(i, gold(i)) if task in ("ACK", "CALL")
                for tier in ("T1", "T2") for r in range(repetitions)}
    got_d = {(r["instance_id"], r["task"], r["tier"], r["repetition"]) for r in diagnostics}
    if got_d != wanted_d or len(got_d) != len(diagnostics):
        raise ValueError("Incomplete diagnostic evidence")
    wanted_p = {(seq, position) for seq in range(5) for position in range(len(test))}
    got_p = {(r["sequence"], r["position"]) for r in policies}
    if got_p != wanted_p or len(got_p) != len(policies):
        raise ValueError("Incomplete policy sequences")
    for seq in range(5):
        if {r["instance_id"] for r in policies if r["sequence"] == seq} != ids:
            raise ValueError("Policy order does not cover the frozen instances")
    if any(r["empirical"] != expected_empirical for r in conditions + diagnostics + policies):
        raise ValueError("Mixed mock and empirical records")
    for row in conditions + policies:
        validate_run_score(row, by_id[row["instance_id"]])
        validate_timing(row)
    ms = task_models()
    from dcarr_runtime.handoff import validate
    for row in diagnostics:
        payload = row["response"].get("payload") if isinstance(row["response"], dict) else None
        valid = not validate(payload, ms.get(row["task"]).hand_off)
        correct = valid and score(row["task"], payload, gold(by_id[row["instance_id"]])[row["task"]], by_id[row["instance_id"]])
        if row["candidate_correct"] != correct or row["handoff_valid"] != valid:
            raise ValueError("Diagnostic score disagrees with the independent oracle")
    if expected_empirical:
        if not (folder / "provider-calls.jsonl").exists():
            raise ValueError("Provider evidence is missing")
        calls = read_rows(folder / "provider-calls.jsonl")
        check_ledger(calls)
        validate_local_provenance(manifest["config"], calls)
        completed = [r for r in calls if r["kind"] == "provider_call"]
        for row in conditions + diagnostics + policies:
            validate_usage(row, completed)
        for row in conditions + policies:
            validate_timing(row, calls)
    return manifest, conditions, diagnostics, policies, sorted(ids)


def analyze(folder, allow_mock=False):
    manifest, rows, diag, policies, ids = validate_dataset(folder, allow_mock)
    index = {name: i for i, name in enumerate(ids)}
    arrays = {}
    for condition in CONDITIONS:
        array = np.zeros((len(ids), len(MEASURES)))
        for instance in ids:
            reps = [r for r in rows if r["condition"] == condition and r["instance_id"] == instance]
            array[index[instance]] = [float(available_mean([r[m] for r in reps])) if m == "time_s"
                                     else np.mean([r[m] for r in reps]) for m in MEASURES]
        arrays[condition] = array
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    draws = rng.integers(0, len(ids), size=(BOOTSTRAP_N, len(ids)))
    table7, differences = [], []
    summary = {"empirical": manifest["empirical"], "bootstrap_resamples": BOOTSTRAP_N, "bootstrap_seed": BOOTSTRAP_SEED,
               "condition_measures": {}, "paired_differences": {}, "abstention": {}, "policy_order_sensitivity": []}
    if not manifest["empirical"]:
        summary["warning"] = "SCRIPTED SMOKE TEST ONLY. These values are not language-model experimental results."
    for condition, array in arrays.items():
        boot = available_mean(array[draws], axis=1)
        summary["condition_measures"][condition] = {}
        for j, measure in enumerate(MEASURES):
            estimate = (timing_estimate([r for r in rows if r["condition"] == condition], ids, draws)
                        if measure == "time_s" else {"mean": float(array[:, j].mean()), "ci95": interval(boot[:, j])})
            value, bounds = estimate["mean"], estimate["ci95"]
            summary["condition_measures"][condition][measure] = estimate
            table7.append({"condition": condition, "measure": measure, "mean": value, "ci95_low": bounds[0], "ci95_high": bounds[1],
                           "available_runs": estimate.get("available_runs", len(ids) * manifest["repetitions"]),
                           "unavailable_runs": estimate.get("unavailable_runs", 0),
                           "available_instances": estimate.get("available_instances", len(ids))})
    for a, b in [("allocated", "single"), ("decomposed", "single"), ("routed", "decomposed"), ("allocated", "routed"), ("allocated", "decomposed")]:
        diff = arrays[a] - arrays[b]
        boot = available_mean(diff[draws], axis=1)
        label = f"{a} minus {b}"
        summary["paired_differences"][label] = {}
        for j, measure in enumerate(MEASURES):
            estimate = (timing_estimate([r for r in rows if r["condition"] == a], ids, draws,
                                       [r for r in rows if r["condition"] == b])
                        if measure == "time_s" else {"mean": float(diff[:, j].mean()), "ci95": interval(boot[:, j])})
            value, bounds = estimate["mean"], estimate["ci95"]
            summary["paired_differences"][label][measure] = estimate
            differences.append({"comparison": label, "measure": measure, "difference": value, "ci95_low": bounds[0], "ci95_high": bounds[1],
                                "available_pairs": estimate.get("available_runs", len(ids) * manifest["repetitions"]),
                                "available_instances": estimate.get("available_instances", len(ids))})
    table8 = []
    for tier in ("T1", "T2"):
        selected = [r for r in diag if r["tier"] == tier]
        counts = np.zeros((len(ids), 4))
        for row in selected:
            wrong = not row["candidate_correct"]
            j = 0 if wrong else 2
            counts[index[row["instance_id"]], j] += 1
            counts[index[row["instance_id"]], j + 1] += row["runtime_abstained"]
        totals = counts.sum(axis=0)
        bs = counts[draws].sum(axis=1)
        summary["abstention"][tier] = {"decisions": len(selected), "incorrect_candidates": int(totals[0]), "correct_candidates": int(totals[2])}
        for label, j in [("on_incorrect", 0), ("on_correct", 2)]:
            rate = float(totals[j + 1] / totals[j]) if totals[j] else None
            ratios = np.divide(bs[:, j+1], bs[:, j], out=np.full(BOOTSTRAP_N, np.nan), where=bs[:, j] != 0)
            bounds = interval(ratios)
            summary["abstention"][tier][label] = {"rate": rate, "ci95": bounds}
            table8.append({"tier": tier, "rate": label, "denominator": int(totals[j]), "abstentions": int(totals[j+1]),
                           "proportion": rate, "ci95_low": bounds[0], "ci95_high": bounds[1]})
    for sequence in range(5):
        seq = sorted([r for r in policies if r["sequence"] == sequence], key=lambda r: r["position"])
        early = np.mean([r["metrics"]["person_requests"] for r in seq[:10]])
        late = np.mean([r["metrics"]["person_requests"] for r in seq[-10:]])
        summary["policy_order_sensitivity"].append({"sequence": sequence, "order_seed": seq[0]["order_seed"],
                                                   "first_ten_mean_requests": float(early), "last_ten_mean_requests": float(late), "difference": float(late - early)})
    summary["policy_note"] = "Five orders reuse the same instances. Report their mean and range as order sensitivity, without an independent-instance confidence interval."
    summary["policy_mean_first_ten"] = float(np.mean([r["first_ten_mean_requests"] for r in summary["policy_order_sensitivity"]]))
    summary["policy_mean_last_ten"] = float(np.mean([r["last_ten_mean_requests"] for r in summary["policy_order_sensitivity"]]))
    tiers = []
    for c in CONDITIONS:
        for tier in ("T1", "T2"):
            selected = [r for r in rows if r["condition"] == c]
            tiers.append({"condition": c, "tier": tier,
                          "mean_input_tokens": float(np.mean([r["metrics"]["tokens_in_by_tier"].get(tier, 0) for r in selected])),
                          "mean_output_tokens": float(np.mean([r["metrics"]["tokens_out_by_tier"].get(tier, 0) for r in selected]))})
    failures = [{"condition": c, "completed_runs": sum(r["status"] == "completed" for r in rows if r["condition"] == c),
                 "failed_runs": sum(r["status"] != "completed" for r in rows if r["condition"] == c)} for c in CONDITIONS]
    retries = []
    for phase, records in (("conditions", rows), ("policy-sequences", policies)):
        groups = CONDITIONS if phase == "conditions" else range(5)
        for group in groups:
            selected = [r for r in records if r["condition" if phase == "conditions" else "sequence"] == group]
            retries.append({"phase": phase, "group": group, "runs": len(selected),
                            **{k: sum(r.get(k, 0) for r in selected) for k in (
                                "failed_attempts", "retry_count", "unknown_timing_attempts", "known_failed_latency_s")}})
    summary["retry_counts"] = retries
    if manifest["empirical"]:
        from .backend import failed_attempt_timing
        calls = read_rows(folder / "provider-calls.jsonl")
        starts = {}
        for call in calls:
            if call.get("kind") == "request_started":
                rid = call["request_id"]
                starts[rid] = starts.get(rid, 0) + 1
        summary["provider_attempts"] = {
            **failed_attempt_timing(calls), "request_attempts": sum(starts.values()),
            "retry_count": sum(max(n - 1, 0) for n in starts.values()),
            "scope": "Entire provider ledger, including calibration and test diagnostics"}
    summary["timing_note"] = ("Shared-load completion time includes each measured failed attempt once and excludes offline gaps. "
                              "Unknown attempt duration makes run time unavailable, without excluding accuracy or observed tokens. "
                              "Timing averages available repetitions within instances; paired timing uses jointly timed repetitions. "
                              "Bootstrap samples with no timed instances are unavailable. Token totals cover observed responses only.")
    out = folder / ("mock-analysis" if not manifest["empirical"] else "analysis")
    out.mkdir(exist_ok=True)
    write_csv(out / "table7.csv", table7)
    write_csv(out / "paired-differences.csv", differences)
    write_csv(out / "table8.csv", table8)
    write_csv(out / "tokens-by-tier.csv", tiers)
    write_csv(out / "policy-sequences.csv", summary["policy_order_sensitivity"])
    write_csv(out / "run-completion.csv", failures)
    write_csv(out / "retry-summary.csv", retries)
    (out / "results.json").write_text(json.dumps(summary, indent=2, allow_nan=False) + "\n")
    return out


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("folder", type=Path)
    parser.add_argument("--allow-mock", action="store_true", help="Explicitly produce mock-analysis only")
    args = parser.parse_args()
    print(analyze(args.folder, args.allow_mock))


if __name__ == "__main__":
    main()
