# task-allocation-runtime

This repository is the implementation of *Allocating cognitive work across people, language models and deterministic tools: A task-analytic method* (Simon Vajda, Applied Artificial Intelligence Initiative, Deakin University). The [Research Square preprint](https://doi.org/10.21203/rs.3.rs-11312211/v1), version 1, was posted on 26 October 2026. It is under review at *Cognition, Technology & Work* and has not been peer reviewed by a journal.

The method decides, before a system runs, which kind of decision maker should make each decision in a cognitive task: a deterministic tool, a small language model, a larger reasoning model, or a person. It extends DCARR, the task-analysis framework from the author's doctoral thesis (Vajda, 2023). Each task is written as a task model with four stages (observe, orient, decide and act), and an ordered rule assigns each decision using four properties: how formally it can be specified, the level of cognitive control it needs, its Cynefin domain, and whether it establishes what is the case or decides what ought to be done. A machine tier that cannot decide abstains and the decision moves up. Choices about what ought to be done stay with people, who can store them as narrowly scoped policies. What each task passes on, including what supervisors receive, is typed and limited in size.

## Status

The method, the runtime and its computational study (Study 1) are complete. Study 1 ran on 8 and 9 October 2026 on one node of Deakin University's A2I2 GPU cluster. It produced 1,280 condition runs, 1,000 paired diagnostic replies and 320 policy-sequence runs from 6,363 model requests, none of which failed or was repeated. A second analysis in a separate environment reproduced every analysis output byte for byte.

This repository holds the code, protocol, task models, synthetic materials and model configuration. The collected records and the analysis outputs are in the [Zenodo archive](https://doi.org/10.5281/zenodo.23225358), together with a copy of this source. The scripted demonstrations and the software tests below check the software only. They are not study results.

## Try it without a model

The demo replaces the language models and the person with scripted stand-ins, so it needs no GPU, model download or API key. Python 3.10 or later is enough.

```sh
python -m venv .venv
.venv/bin/python -m pip install -e '.[test]'
.venv/bin/python -m scenarios.simpilot.run --allocation
.venv/bin/python -m scenarios.simpilot.run --variant ambiguous --log out/ambiguous.jsonl
.venv/bin/python -m pytest -q
```

The first command prints which tier each of the nine tanker-join task models starts at, and why. The second runs a case where the radio call is addressed to "aircraft two or three": the small-model stand-in abstains and the reasoning-model stand-in resolves it. The [demo guide](docs/DEMO.md) walks through more cases, including how a stored policy answers a repeated question.

## What is in this repository

```text
src/dcarr_runtime/   task-model types, the allocation rule, the engine, hand-off checks, policies
scenarios/simpilot/  the synthetic tanker-join example: task models, simulation, scripted stand-ins
schema/              the task-model schema in JSON Schema form
study1/              the computational study: protocol, materials, conditions, scoring, analysis
cluster/             scripts that run the study on a SLURM GPU cluster
tests/               software tests
docs/                guides to the architecture, demo, paper and future directions
```

| To | Read |
|---|---|
| See how the method maps to the code | [Paper and implementation map](docs/PAPER.md) |
| Understand the components and the allocation rule | [Architecture](docs/ARCHITECTURE.md), [runtime guide](docs/RUNTIME.md) |
| Run the scripted demo | [Demo guide](docs/DEMO.md) |
| Inspect the study design | [Study README](study1/README.md), [protocol](study1/PROTOCOL.md), [deviations from the original brief](DEVIATIONS.md) |
| Read about planned extensions | [Future directions](docs/directions/README.md) |

## The study

Study 1 compared four ways of handling the same synthetic tanker-join scenario: one reasoning model given the whole task, the task decomposed with the reasoning model making every decision, decisions routed by a learned router, and the full allocation method with abstention, escalation and policies. It used 64 generated test instances and 32 held-out instances, five repetitions per condition, and two open-weight models (Qwen3-4B-Instruct-2507 and Qwen3.8-27B) served by vLLM. The questions were how allocation affects accuracy, tokens and the largest input any model processes, and whether abstention sends wrong decisions upward. Protocol [version 1.2](study1/PROTOCOL.md) was fixed before the archived collection. Earlier held-out verification runs led to amendments before collection, and [DEVIATIONS.md](DEVIATIONS.md) records those amendments and the implementation findings below.

The allocated condition was correct on 89.6% of required epistemic decisions (95% CI 82.8 to 95.5), against 67.2% for one reasoning model given the whole task. It read about a third of the tokens and wrote about an eighth. A scripted person, a program standing in for the exercise supervisor, supplied 13% of its scored decisions, so this accuracy describes the system together with its simulated supervisor. The paper reports these results with their intervals and the remaining measures.

Three differences between the protocol and the frozen implementation were found while reporting the results. None changed a record or the analysis, and the paper reports all three as limitations:

1. Reference recovery by the scripted person also operated in the routed condition, although the protocol described it only for the allocated condition.
2. The aircraft-selection tool required an exact name, while scoring ignored letter case, so a correct acknowledgement addressed to "Aircraft 2" was then rejected at selection.
3. The response to a scheduled jamming inject was meant to stay with people as a training choice. The frozen task model classed that radio decision as epistemic, so models made some of these responses without asking the person. The study therefore does not show the method's normative boundary holding for that decision.

[DEVIATIONS.md](DEVIATIONS.md) gives the counts. The scenario's positions, callsigns and channels are sample values. No exercise records and no human participants were used. A computational study cannot show effects on supervisors' workload or situation awareness, which need a separate study with practitioners.

## Future directions

Two follow-up studies are planned, each to be compared with Study 1 as the baseline: a bounded agent loop that may consult declared, read-only sources before deciding, and abstention gated by a separate decision model instead of the deciding model's own confidence. See the [direction notes](docs/directions/README.md). Neither is implemented here.

## Related project

[ooda-harness](https://github.com/Rainierraoul/ooda-harness), by the same author, is a separate, exploratory TypeScript agent harness. It runs a coding agent through repeated observe–orient–decide–act cycles, uses a decision model to judge progress, and can tune its own settings. It shares vocabulary with this project but is a different system. It is not used in Study 1 and is not evidence for the method. The [architecture guide](docs/ARCHITECTURE.md) compares the two.

## How to cite

GitHub's "Cite this repository" button reads [CITATION.cff](CITATION.cff). In text:

> Vajda, S. R. L. (2026). *task-allocation-runtime* [Computer software]. Zenodo. https://doi.org/10.5281/zenodo.23225358

The software DOI points to the release archived on Zenodo, which holds this source and the study records. The associated paper is available as a preprint:

> Vajda, S. (2026). *Allocating cognitive work across people, language models and deterministic tools: A task-analytic method* (Version 1) [Preprint]. Research Square. https://doi.org/10.21203/rs.3.rs-11312211/v1

The preprint and software have separate DOIs. The preprint is under review and has not been peer reviewed by a journal.

## Licence

Code is released under the [MIT licence](LICENSE). Documentation, the study protocol, the task models, the synthetic materials and the archived study records are released under [CC BY 4.0](LICENSE-docs). The learned router used in Study 1 keeps its Apache 2.0 [licence](study1/ROUTER-LICENSE) and [notice](study1/NOTICE). Model weights are not distributed here; the study downloads them from their publishers at pinned revisions.
