# Direction 1: a bounded agent executor

*Proposed follow-up study. Not implemented and not part of Study 1.*

## The idea

In the current runtime, a model tier makes its decision in one call. It sees exactly the observations and orientation entities its task model declares, and it returns a typed hand-off or abstains.

The proposal adds an optional executor for that decision: a short agent loop. The model observes, orients, and may call a small set of declared tools to gather more evidence before it decides. When it finishes, it returns the same typed hand-off or abstention as a single call would. Everything around the decision stays with the runtime: allocation, hand-off validation, confidence gates, escalation, policies and the action levels.

This extends a feature the runtime already has. The paper's Implementation section describes a backend that drives an existing agent harness (pi) so that the same task models can run inside it. ooda-harness is built on pi's durable execution layer and already runs an observe–orient–decide–act loop with crash recovery.

## What it could add

- **Variable reasoning effort.** A routine decision uses one call; a harder one can use several observe–decide cycles, up to a declared limit.
- **Investigative tasks.** A task model could state the outcome it needs without prescribing every intermediate lookup.
- **Durable execution.** The current study backend already recovers individual model calls: its request ledger resumes an interrupted run without repeating completed calls. It cannot save the working state of a multi-step decision, such as which sources an agent has already read and what it concluded from them. A durable agent loop would checkpoint that state after each step, so a long decision could resume partway through. Section 4.5 of the paper lists the prototype's lack of concurrency control and time limits as limitations.

## Constraints the method requires

The paper makes two design claims that an inner agent could break without anyone noticing. Any implementation has to keep them.

1. **Deciding and acting stay separate.** The method attaches approval, veto and reversibility to the Action stage of each task model. The agent may only observe and orient, using read-only tools, and return a decision. Anything that changes the world is done by the runtime's Action stage under the task model's action level. Checking the agent's final answer cannot undo an action it has already taken, so the agent must not be able to take one.
2. **Inputs stay inspectable before a run.** Section 3.3 says a model receives only what its task model declares, so an analyst can inspect, before any run, everything a model will see. An agent that gathers evidence breaks this unless its tools and sources are declared in the task model. The inspectable boundary then moves from "these inputs" to "these sources". That is a change to the method, and it should be stated as one.

The loop also needs explicit limits: iterations, tokens, context size, wall time and tool calls. Exceeding any limit counts as abstention, like an over-budget hand-off does now.

## How it would be evaluated

Use Study 1's tanker materials, scoring, held-out calibration and analysis unchanged. Compare the allocated condition with direct calls (Study 1) against the allocated condition with the agent executor, for the plain-language decisions (ACK and CALL), with the same models, settings and limits. Pre-specify:

- accuracy of epistemic decisions, tokens read and written, largest context and time per run (Study 1's RQ1 measures);
- abstention among incorrect and correct decisions, and escalations (RQ2);
- how many loop iterations and tool calls each decision used;
- a durability check: interrupt runs at random points and confirm they resume without repeating completed tool calls or decisions.

The tanker decisions are narrow and fully specified, so the agent may have little to investigate. A second scenario with genuinely incomplete information would show the executor's value more clearly. That would be a new scenario with its own task analysis, not a change to Study 1.

## Risks

- More tokens and more time for the same decision, which works against RQ1's aim.
- A larger and less predictable input set, which works against the supervisor's ability to know in advance what a model saw.
- Several mechanisms interacting at once. Evaluating this direction separately from [calibrated abstention](calibrated-abstention.md) keeps the effect of each identifiable.

## Sources

- The paper, Sections 3.1 (terms: agent and harness), 3.3 (declared inputs), 3.6 (action levels) and 4.2 (backends).
- [ooda-harness](https://github.com/Rainierraoul/ooda-harness): a durable OODA agent loop on pi-durable.
- Anthropic, [Building effective agents](https://www.anthropic.com/engineering/building-effective-agents), on the difference between predefined workflows and agents that choose their own intermediate steps.
- Proposed by Codex on 8 October 2026; constraints 1 and 2 added by Claude.
