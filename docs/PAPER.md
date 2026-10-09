# Paper and implementation map

The allocation paper extends DCARR, the framework from the author's doctoral
thesis (Vajda, 2023), from identifying automation candidates to assigning
cognitive decisions across tools, language models and people.

## Method contribution

- Describe work as a hierarchy of Observe, Orient, Decide and Act task models.
- Derive the supervisor's information requirements and specify the hand-offs
  that supply them.
- Allocate each decision using formal specifiability, cognitive control,
  Cynefin domain and epistemic versus normative character.
- Allow machine abstention and escalation while retaining human authority for
  normative choices and specified action permissions.
- Record a person's decision as a policy only for its explicitly declared scope.

The prototype implements these mechanisms. It does not automate the entire
analyst's task-analysis process or provide a validated supervisor interface.

## Computational evaluation

**RQ1:** Relative to a reasoning model given the whole task, how do allocation
and decomposition affect epistemic accuracy, model tokens and largest input?
Completion time and requests to the scripted person are additional measures.
The scripted person is a program that stands in for the exercise supervisor.

**RQ2:** For acknowledging the command (ACK) and contacting the control agency
(CALL), does tier abstention separate correct from incorrect candidates? Across
repeated cases, how do requests to the scripted person change as policies
accumulate?

| Condition | Implementation |
|---|---|
| Single | One reasoning-model call returns the whole scenario's task payloads |
| Decomposed | The reasoning model makes the eight machine decisions in plan order |
| Routed | Tools retain six formally specified decisions; a learned router chooses the model for ACK and CALL |
| Allocated | DCARR allocation, gates, validation, escalation, policies and scripted recovery run together |

The design used 64 synthetic test cases, 32 separate held-out cases, five
repetitions per condition, paired ACK/CALL diagnostics and five policy orders.
Confidence gates and the router threshold were chosen on held-out cases before
test collection. No test outcome selected a setting.

Collection ran on 8 and 9 October 2026 under protocol version 1.2, with the
published bf16 Qwen checkpoints served by vLLM on one node of Deakin
University's A2I2 cluster (job 3910969). `study1/config.cluster.json` records
the model revisions and settings, and `cluster/` holds the launch scripts. All
6,363 model requests returned a response; none failed or was repeated, and
every run has a recorded time. A second analysis in a separate environment
reproduced all eight analysis outputs byte for byte. Version 1.1's Mac
configuration and version 1.0's paid-model configuration are kept as history
and were not used for collection.

The study measured accuracy, observed model-token use and completion time under
shared load. It did not measure monetary inference cost, or real supervisors'
workload or situation awareness. Scripted recovery contributes to the
allocated and routed systems' accuracy and is reported with the requests to
the scripted person.

### Implementation findings

Three differences between the protocol and the frozen implementation were found
while reporting the results. The records and analysis were kept as collected,
and the paper reports each one as a limitation; `DEVIATIONS.md` gives the
counts.

| Finding | Effect |
|---|---|
| Reference recovery also operated in the routed condition | The scripted person supplied 96 aircraft-selection decisions there, which count toward that condition's accuracy |
| The selection tool matched names exactly, while scoring ignored letter case | An addressee of "Aircraft 2" was scored correct and then rejected at selection, 29 times in the allocated and 47 in the routed condition |
| The scheduled-jamming response, meant as a training choice for people, was classed as an epistemic CALL decision | In the allocated condition the reasoning model made 25 of the 60 responses without asking the person; the study does not show the normative boundary holding for this decision |

## Evidence to manuscript

| Paper element | Source or output |
|---|---|
| Allocation rationale | [allocation.py](../src/dcarr_runtime/allocation.py) |
| Task-model metamodel and contracts | [metamodel.py](../src/dcarr_runtime/metamodel.py), [handoff.py](../src/dcarr_runtime/handoff.py), [schema](../schema/task_model_set.schema.json) |
| Worked tanker example | [task models](../scenarios/simpilot/task_models.yaml), [demo runner](../scenarios/simpilot/run.py) |
| Conditions, prompts and scripted person | [harness.py](../study1/harness.py), [backend.py](../study1/backend.py) |
| Materials and independent reference scoring | [instances.py](../study1/instances.py), [oracle.py](../study1/oracle.py), [materials.json](../study1/materials.json) |
| Held-out gates and router | [calibrate.py](../study1/calibrate.py), [router.py](../study1/router.py) |
| Raw attempts, responses and serving provenance | Frozen `provider-calls.jsonl`, `study1/serving.py` |
| Table 7 and paired condition comparisons | Saved `analysis/table7.csv`, `paired-differences.csv`, `results.json` |
| Table 8 diagnostics | Saved `analysis/table8.csv`, `results.json` |
| Policy trends | Saved `analysis/policy-sequences.csv` |
| Timing, retries and completion | Saved `analysis/retry-summary.csv`, `run-completion.csv`, `results.json` |

Source paths are relative to the repository root. Analysis paths are relative
to the frozen output directory `collect-3910969/study/`, which is in the
[Zenodo archive](https://doi.org/10.5281/zenodo.23225358) rather than this
repository.

## Current manuscript status

The paper is in preparation for Cognition, Technology & Work. It reports the
method, implementation, worked example and the completed computational
comparison, including the three implementation findings above. Its
supplementary results give per-task scores and the descriptive counts behind
these findings.

Study 1 uses the cluster launcher and the study's direct model-server adapter.
ooda-harness's agent loop, Jev judgements and setting evolution are not Study 1
conditions. The runtime's optional `PiBackend` is a generic interface to the pi
agent harness, not an evaluated integration with ooda-harness.

## Follow-ups and the broader architecture

The [direction notes](directions/README.md) describe two separate follow-ups:

| Direction | Change to the baseline | Status |
|---|---|---|
| [Bounded agent executor](directions/bounded-agent-executor.md) | How a tier reaches its decision; declared read-only investigation and explicit resource limits | Proposed; not implemented |
| [Calibrated abstention](directions/calibrated-abstention.md) | How a candidate is accepted or escalated; external judging against self-confidence | Proposed; not implemented |

Study 1 is complete and unchanged. A further-work paragraph names both directions,
without assigning a priority, adding a study condition or claiming a benefit.
The baseline's results can inform the author's choice of the next question.
The inner executor must preserve the Decide/Act distinction; its declared
sources require an explicit extension of the current input contract. Neither
direction automatically adopts dynamic Cynefin classification or self-tuning.
