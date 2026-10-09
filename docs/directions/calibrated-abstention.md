# Direction 2: calibrated abstention

*Proposed follow-up study. Not implemented and not part of Study 1.*

## The problem it addresses

A model tier abstains when its stated confidence falls below a threshold set on held-out cases. The paper names the weakness of this itself (Section 4.5): a model's stated confidence may not track its accuracy, so the threshold gives only coarse control over abstention. RQ2 measures how well abstention catches wrong decisions, but every gate in Study 1 relies on the model judging itself.

## The idea

Add a second kind of gate: a separate decision model that is asked whether a candidate answer is correct and complete, given only the inputs the deciding model received. It returns a probability. The tier abstains when that probability falls below a threshold set on held-out cases, exactly as the confidence gate is set now.

Whether that probability is calibrated for this task is a question the study would answer, not an assumption. Choosing a threshold that meets an error target on held-out cases does not by itself show calibration. The study would also report, on held-out candidates, how well the judge's probabilities match observed correctness, for example with a reliability curve and a Brier score.

Decision models of this kind answer typed questions (yes/no, a choice among options, or a score) with a probability for each answer, in one forward pass, without generating text. Their developers calibrate them on their own training tasks; that calibration may not carry over to judging hand-offs. Open-Jev is an open re-implementation of one (TypeSafe's Jev) and runs locally. ooda-harness already uses it to screen tool calls and judge whether acceptance criteria are met.

## How it would be evaluated

This study can reuse Study 1's data. Study 1's diagnostics already save, for every eligible ACK and CALL context on held-out and test instances, both models' best candidate together with its stated confidence and its abstention flag. A judge can score those saved candidates afterwards, with no new calls to the deciding models. The comparison is then paired: the same candidates, gated three ways.

1. Self-reported confidence (Study 1).
2. The judge's probability.
3. Both together, abstaining when either falls below its threshold.

Set each gate's threshold on held-out candidates only, with Study 1's rule: the lowest threshold giving at least ten accepted responses and at most 5% observed error. Then report, on test candidates, abstention among incorrect candidates and among correct candidates, with denominators and bootstrap intervals over instances, as Study 1 does. The judge must see only the deciding model's declared inputs and its candidate, never the answer key or the White Force plan.

Because candidates are reused, this study costs only the judge's calls and can run as soon as Study 1's data exist.

## Two ideas from ooda-harness that do not fit as they stand

- **Re-classifying the Cynefin domain while running.** The paper fixes each task model's domain at design time, as it presents to a competent practitioner. It argues that fixed tiers let supervisors know in advance which decisions can reach them. A judge that re-classifies the domain during a run would trade that predictability for responsiveness. At most it should flag a mismatch to a person. In ooda-harness tests, Open-Jev's 2B model was confident only about the chaotic case and put five of six tasks in "confused".
- **Settings that tune themselves.** The paper treats thresholds as normative and leaves them to people. Automatic tuning, as ooda-harness's evolution loop does, could only propose changes for a person to approve.

## Risks

- The judge may share the deciding model's blind spots, especially if both come from the same model family.
- Open-Jev was trained on general decision tasks, not on judging candidate hand-offs. Its calibration may not carry over, which the held-out step will show.
- A second model adds latency and its own failure modes. The judge's failures must count as abstention, never as acceptance.

## Sources

- The paper, Sections 3.6 (confidence thresholds), 3.8 (abstention conditions), 4.5 (limitations) and 5.1 (RQ2 diagnostics).
- [ooda-harness](https://github.com/Rainierraoul/ooda-harness): Jev-based judgements and fusion of evidence over time.
- [Open-Jev](https://github.com/Zefan-Cai/Open-Jev): an open decision model returning calibrated probabilities.
- Proposed by Claude on 8 October 2026.
