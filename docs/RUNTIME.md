# DCARR runtime

*Version 0.2.0 · October 2026*

This runs task models written in DCARR's Describe phase across four tiers: deterministic tools, small models, reasoning models and people. An ordered design rule selects the starting tier for each decision; it is not a measured guarantee of the cheapest capable executor. A tier that can't decide abstains, and the decision escalates to the next tier. People receive judgment requests, approval requests, notices and a final hand-off as specified by the task models.

It is the prototype for the paper *Allocating Cognitive Work Across People, Language Models, and Deterministic Tools: A Task-Analytic Method*, and builds outward from Vajda (2023), *Model Driven White Force Automation*.

For the project boundaries and a guided demo, read the [root README](../README.md),
[architecture](ARCHITECTURE.md), [demo](DEMO.md) and
[paper map](PAPER.md). The separate OODA agent harness is not required to
run this Python runtime or its scripted demonstration. `study1/harness.py` is the
DCARR experiment harness; it is not the `ooda-harness` agent project.

The [future-direction notes](directions/README.md) describe a bounded
agent executor and externally judged abstention as separate follow-ups. Both
are proposed and have no executable implementation or results in this release.
The completed Study 1 is their baseline.

## What maps to what

| Thesis | Here |
|---|---|
| M2 OODA metamodel | `src/dcarr_runtime/metamodel.py`, exported as `schema/task_model_set.schema.json` |
| M1 task models | YAML files, one per scenario (`scenarios/simpilot/task_models.yaml`) |
| M0 instance models | the event log of each run (`EventLog`, written as JSON lines) |
| Describe Steps 4–12, Analyse Steps 17–23 | fields of `TaskModel`; the field docstring lists which step each comes from |
| Cynefin as classifier, levels of automation as ranking | `allocation.py`: clauses A–G set the tier, and the action level sets how freely a decision is acted on |

## The allocation rule

The first clause that applies sets the tier.

| Clause | Condition | Tier |
|---|---|---|
| A | The decision can't be described or decomposed (DCARR Step 5a exit) | people |
| B | The decision is normative: a choice about what ought to happen | people, unless a stored policy covers the situation |
| C | The Cynefin domain is chaotic | people |
| D | The decision reaches a state machine or an algorithm (Steps 9, 18) | deterministic tools |
| E | Plain-language decision in a complex domain | reasoning models, proposing only |
| F | Plain-language functional task at the skill or rule level | small models |
| G | Any other plain-language decision | reasoning models |

Clauses A–D restate the thesis. E–G cover the decisions the thesis ruled out of scope, which language models can now make.

## Running it

Needs Python 3.10 or later, `pydantic` and `pyyaml`.
Run these commands from the repository root.

```
pip install -e ".[test]"
pytest
python -m scenarios.simpilot.run --allocation
python -m scenarios.simpilot.run --variant clean --policies out/policies.json --log out/clean.jsonl
python -m scenarios.simpilot.run --variant clean --policies out/policies.json   # second run: the stored policy answers
python -m scenarios.simpilot.run --variant ambiguous
python -m scenarios.simpilot.run --variant jammed
```

## Backends

- `ScriptedBackend` runs deterministic stand-ins for the model tiers. They are not language models. The tests and the worked example use them so the allocation, abstention, escalation and policy logic can be checked without a model API.
- `AnthropicBackend` calls the Anthropic Messages API. It is untested here because this environment has no API key.
- The study's `LocalBackend` (`study1/backend.py`) calls an OpenAI-compatible server. Protocol v1.2 uses it with two published bf16 checkpoints served by vLLM on the cluster; see `study1/README.md`.
- `PiBackend` drives the pi agent harness in RPC mode (`pi --mode rpc --no-session`). It is untested, and it reads the field names for streamed text defensively because the RPC documentation doesn't pin them down.

## The allocation study

The study code is in [`study1/`](../study1/README.md): a frozen synthetic 64-instance
test set, 32 held-out instances, four comparison conditions, learned routing,
held-out calibration, a request journal with raw responses, and a paired
bootstrap analysis. The protocol is `study1/PROTOCOL.md`, and every change from
the original study brief is documented in `DEVIATIONS.md`.

Study 1 was collected on 8 and 9 October 2026 under protocol version 1.2, using
the published bf16 Qwen checkpoints on a Linux/CUDA cluster
(`study1/config.cluster.json`, `cluster/`). The frozen records and analysis
outputs are archived on Zenodo (https://doi.org/10.5281/zenodo.23225358).
`DEVIATIONS.md` also records three differences between the protocol and the
frozen implementation found after collection. The records and analysis were kept
as collected. Earlier Mac and paid-model instructions are kept as history.

Software verification is covered by `tests/`. These checks and any scripted
output are verification data, not language-model results.

The curated public release uses MIT for code and CC BY 4.0 for documentation,
protocol and synthetic materials, as selected by the author. License and citation
files are at the repository root. The separately attributed learned router's
Apache 2.0 notice and license are retained in `study1/`.

## Limits of this version

- Token counts from the scripted backend are estimates (characters divided by 4), not model counts.
- People answer through a scripted responder or the console. The situation board, the interface people will actually use, isn't built yet. It needs its own design work, starting from the supervisor information requirements.
- Scenario 5 doesn't model aircraft 1, following the thesis. If a person chooses 'misinterpret' in DEV, the run records the plan but doesn't act it out.
- Scenario positions, callsigns and channels are sample values, not exercise data.
