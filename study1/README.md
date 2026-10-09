# Running the tanker allocation study

Study 1 was collected on 8 and 9 October 2026 under protocol version 1.2 (cluster job 3910969). Its records and analysis outputs are archived on Zenodo (https://doi.org/10.5281/zenodo.23225358), with a copy of this source. This page explains how to check that analysis from the archived records, and how to run the study again. A new run is a replication, with its own output folder; it does not add to the archived records.

Read `PROTOCOL.md` and `../DEVIATIONS.md` first. `DEVIATIONS.md` records three differences between the protocol and the frozen implementation that were found after collection. Scripted smoke outputs are verification fixtures, not findings. Do not use commercial-in-confidence material, operational exercise data, private documents or manuscripts as experimental inputs.

## Setting up

From this repository root, use Python 3.12 and install the package and study/test dependencies in an isolated environment:

```sh
python -m venv .venv
.venv/bin/python -m pip install -r study1/requirements-tested.txt
.venv/bin/python -m pip install -e . --no-deps
.venv/bin/python -m pytest -q
```

The tested dependency snapshot is from macOS arm64, Python 3.12.14. Binary availability can differ on other platforms. The BERT checkpoint is fetched from Hugging Face at its pinned revision and runs locally, without an embedding-provider credential. Model loading and dependency versions must be reported separately from paid inference.

## Checking the archived analysis

The analysis reads only the saved records and makes no model call. It writes its outputs into an `analysis/` folder inside the folder it reads, so it runs on a copy.

Download both ZIP files from the Zenodo record and unzip them in the same folder. This gives `task-allocation-runtime-0.2.0/` (this source) and `study1-records/` side by side. Set up the environment as described above, then run from the source folder:

```sh
TASK_RECORDS="$(cd ../study1-records/collect-3910969 && pwd)"
mkdir ../check
cp -R "$TASK_RECORDS/study" ../check/study
rm -r ../check/study/analysis
.venv/bin/python -m study1.analyze ../check/study
(cd ../check && grep "./study/analysis/" "$TASK_RECORDS/SHA256SUMS" | shasum -a 256 -c)
```

The last command checks the eight new outputs against the archived checksums and prints `OK` for each (on Linux, use `sha256sum -c`). The descriptive counts in the article's Online Resource 2 can be checked the same way:

```sh
.venv/bin/python ../study1-records/publication/supplementary_counts.py "$TASK_RECORDS/study" ../check/supplementary-counts.json
cmp ../check/supplementary-counts.json ../study1-records/publication/supplementary-counts.json && echo identical
```

Give the script both paths: without arguments it looks for the records in the author's working layout. Both checks reproduced byte for byte in a separate environment built from `requirements-tested.txt`.

## Checking the software without a model

Verify the study's plumbing without a model server, credential or paid API:

```sh
.venv/bin/python -m study1.run smoke --out runs/smoke
.venv/bin/python -m study1.analyze runs/smoke --allow-mock
```

This produces `mock-analysis/` and prominently labels its output. Omitting `--allow-mock` rejects those records as research evidence. The full smoke schedule includes 1,280 condition runs, 1,000 paired test diagnostics and 320 policy-sequence runs, plus held-out checks.

## Collecting data on a GPU cluster (protocol v1.2)

This is how the archived collection was run. The study runs on two open-weight models, `Qwen/Qwen3-4B-Instruct-2507` and `Qwen/Qwen3.8-27B`, in their published bf16 form at the revisions pinned in `config.cluster.json`. Both are served by vLLM on one node with three NVIDIA L40S GPUs (48 GB each); the larger model is split across two of them. No API key or spending cap is needed. The scripts in `../cluster/` were written for Deakin University's A2I2 SLURM cluster and need small changes for another site (partition, quality of service and paths).

One-time setup on the cluster, under `/scratch/$USER/dcarr/`:

1. A vLLM environment (`envs/vllm`) with vLLM 0.30.0. Use the CUDA 12.9 build from vLLM's GitHub release if the GPU driver is older than 580; the default PyPI build targets CUDA 13.0. Remove `torchcodec` from it, since its prebuilt library targets CUDA 13 and the study does not decode audio.
2. A study environment (`envs/study`, Python 3.12) with `requirements-tested.txt` and this package installed without dependencies.
3. The two models and the router checkpoint downloaded into `hf/` at the pinned revisions.
4. A copy of this repository at a recorded commit in `code/<commit>/`, with the full commit hash in a `COMMIT` file.

Then submit, from the code copy:

```sh
sbatch --export=ALL,MODE=verify,CODE=$PWD cluster/study1.sbatch
sbatch --export=ALL,MODE=collect,CODE=$PWD,WORKERS=16 cluster/study1.sbatch
```

`verify` records the serving evidence once and runs `verify_local.py` on held-out instances only. `collect` generates the materials, runs all four stages with parallel workers and runs the analysis. A collect job resubmitted with the same `OUT` resumes from the saved ledger. The ledger begins with one `server_info` row per tier. If a server's software, model files or GPUs change, the backend stops; use a new output directory.

Version 1.1 ran the same models as MLX conversions on an Apple M4 Pro (`config.open-weight.json`). That setup was used for held-out verification only and is not used for collection.

## The paid configuration (protocol v1.0, not used)

The rest of this section describes the Anthropic configuration that version 1.0 proposed. It is kept for a possible replication and is not part of the current study.

Before real collection, confirm the funded Anthropic account and an authorised USD spending cap. Configure `ANTHROPIC_API_KEY` in the local process environment, or save only the key in an owner-readable `.secrets/anthropic-api-key` file and use `--api-key-file`. Do not put a credential in JSON configuration, a command argument, chat, a repository or provider logs. `.secrets/` is excluded from Git. A candidate model pair, explicit reasoning settings, prices checked on 7 October 2026 and the pinned router are in `config.example.json`. These are proposed settings, not evidence of successful calls. Check availability and prices again if collection is delayed, especially for Haiku 4.5.

The following commands are examples. Replace the limit with the amount authorised by the account holder. Stage calls share a single ledger, so the cap applies to its cumulative spend, including held-out calls. Execute the paid command only after credential/account and cap are established:

```sh
.venv/bin/python -m study1.instances --out runs/materials.json
.venv/bin/python -m study1.run calibrate --materials runs/materials.json --config study1/config.example.json --out runs/empirical --budget-usd 100 --api-key-file .secrets/anthropic-api-key
.venv/bin/python -m study1.run conditions --materials runs/materials.json --config study1/config.example.json --out runs/empirical --budget-usd 100 --api-key-file .secrets/anthropic-api-key
.venv/bin/python -m study1.run diagnostics --materials runs/materials.json --config study1/config.example.json --out runs/empirical --budget-usd 100 --api-key-file .secrets/anthropic-api-key
.venv/bin/python -m study1.run policies --materials runs/materials.json --config study1/config.example.json --out runs/empirical --budget-usd 100 --api-key-file .secrets/anthropic-api-key
.venv/bin/python -m study1.analyze runs/empirical
```

Alternatively, `run all` runs the four paid stages sequentially. Each call records exact request, provider response and message ID, returned model, usage, latency and cost. No provider-level retries or model substitutions occur. A conservative next-call reservation guards the cap. If access fails, a request is interrupted or its billing is uncertain, the process stops for reconciliation. Do not edit that record away or repeat a possibly charged request without accounting for it.

Hashes freeze materials, config, source and protocol. Changed inputs require a new output directory and a documented deviation. Resumption uses unique request IDs, including repetition, and rejects a changed request. Each policy order is reconstructed from its beginning, avoiding the bias of loading the terminal policy store into its first cases. An incomplete dataset cannot pass the analysis gate. Failed candidates remain failures; they are not silently omitted or rescued in the forced-candidate conditions.

Expected files:

| File | Meaning |
|---|---|
| `manifest.json`, `materials.json` | Frozen design, source/config/material hashes and synthetic cases |
| `provider-calls.jsonl` | Request journal, raw model-server responses, token use, serving evidence and (for the paid configuration) costs; never credentials |
| `calibration.json` | Held-out diagnostics, selected confidence gates and router threshold |
| `conditions.jsonl` | Five repetitions per instance/condition, task outputs, scores and event logs |
| `diagnostics.jsonl` | Same-response candidate, confidence and abstention data at both tiers |
| `policy-sequences.jsonl`, `policy-store-*.json` | Five sequential orders and terminal scoped policies |
| `analysis/table7.csv`, `table8.csv`, `results.json` | Empirical condition and diagnostic estimates from a complete dataset |
| `analysis/paired-differences.csv` | Paired instance bootstrap comparisons |
| `analysis/tokens-by-tier.csv`, `run-completion.csv`, `policy-sequences.csv` | Supporting counts, tier usage, failures and order sensitivity |

For Study 1, the analysis was rerun from the unchanged saved records in a separate environment and its output hashes compared before the results were reported. The code and data are archived with a persistent identifier.
