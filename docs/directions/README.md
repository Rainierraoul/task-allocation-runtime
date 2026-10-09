# Future directions

These notes describe two proposed follow-up studies. Neither is part of the paper's evaluation (Study 1), and no code for them is in this repository yet.

Study 1 is the baseline for both. It allocates each decision to a tier at design time, and a model tier answers with a single call whose own stated confidence decides whether it abstains. Each direction changes one of those two things and is meant to be compared against Study 1 on the same tanker materials and protocol, one change at a time, so that any difference can be attributed to it.

| Note | What changes | Question it answers |
|---|---|---|
| [Bounded agent executor](bounded-agent-executor.md) | How a model tier reaches its decision: a short observe–orient–decide loop with declared, read-only tools, instead of one call | Does letting a model investigate within strict limits improve decisions enough to justify its cost and the wider set of inputs? |
| [Calibrated abstention](calibrated-abstention.md) | What decides whether a model tier abstains: a separate decision model's probability that the answer is correct, instead of the model's own confidence | Does an external, calibrated judge send more of the wrong decisions upward and fewer of the right ones? |

Both build on [ooda-harness](https://github.com/Rainierraoul/ooda-harness), an exploratory agent harness by the same author. It has not been evaluated, and the paper does not rely on it.
