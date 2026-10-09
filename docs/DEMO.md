# Scripted DCARR demo

This demo shows the allocation mechanics, information hand-offs and stored
policies. All model responses and person answers are scripted. It demonstrates
software behavior, not language-model performance or human workload.

## Setup

From the repository root, install the demo dependencies:

```sh
python -m venv .venv
.venv/bin/python -m pip install -e '.[test]'
```

Run every command below from this directory. No model server or GPU is
needed. The scripts write only their specified local output files.

## Inspect the starting allocation

```sh
.venv/bin/python -m scenarios.simpilot.run --allocation
```

The nine task models include six tool decisions, ACK and CALL starting at the
small-model tier, and DEV at the person tier. This is the design-time starting
allocation; a completed run can escalate or use a covering policy.

## Follow three situations

```sh
.venv/bin/python -m scenarios.simpilot.run --variant clean --log out/clean.jsonl
.venv/bin/python -m scenarios.simpilot.run --variant ambiguous --log out/ambiguous.jsonl
.venv/bin/python -m scenarios.simpilot.run --variant jammed --log out/jammed.jsonl
```

Each command prints the final report and run metrics. The JSONL log exposes the
allocation clause, calls, abstentions, escalations, judgment requests, actions
and validated hand-offs. In the ambiguous variant the small-model stand-in
abstains and the reasoning stand-in resolves the instruction. Under scheduled
jamming, both scripted model stand-ins abstain because the response is a
training-design choice, so the choice escalates to the scripted person. The
task model still classes this radio decision as epistemic. The choice reaches
the person here only because the stand-ins abstain, and in the computational
study real models sometimes made it themselves (see the
[paper map](PAPER.md#implementation-findings)).

## Reuse an explicitly scoped policy

Choose a new filename for an initially empty store. Running these twice with the
same filename preserves the choices authorized on the first run:

```sh
.venv/bin/python -m scenarios.simpilot.run --variant jammed --policies out/demo-policies.json --log out/policy-first.jsonl
.venv/bin/python -m scenarios.simpilot.run --variant jammed --policies out/demo-policies.json --log out/policy-second.jsonl
```

Compare `person_request`, `policy_set` and `policy_applied` events. This exact
repeat shows lookup of a matching policy. It does not establish generalization
or a favorable trend across new cases. Policies include task design/version,
declared orientation and selected observations; different situations can return
the choice to a person.

## Check the implementation

```sh
.venv/bin/python -m pytest -q
```

The test extra installs NumPy, which the study-analysis tests need. The scripted
demonstration itself needs neither NumPy nor any model-serving software.

The tests include intercepted model calls and synthetic process/GPU facts. A
passing fixture suite does not establish that a particular live server, model
pair or cluster configuration works. See [the paper map](PAPER.md) for the
separate real-model verification and frozen-data requirements.
