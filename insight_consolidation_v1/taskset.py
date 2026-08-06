"""insight-consolidation-v1 - the task, the rubric and the taskset.

Rubric
------
Trained rewards
  grounded_insight   judged semantic match against the latent driver, GATED by citation
                     grounding. This is the headline number.
  evidence_grounding F1 of cited document ids against the minimal entailing set.
  confabulation      abstained correctly on a null episode, or correctly did not abstain.

Diagnostics (weight 0, recorded not trained)
  insight_ungated    the judged match without the citation gate, so the gate's effect is
                     visible rather than assumed.
  anti_prior         judged match restricted to the anti-prior split.
  revision           judged match restricted to revision episodes.
  volume             this episode's document count, so a sparsity curve can be built by
                     bucketing runs rather than by running three separate evals.
  docs_read          investigation cost. Accuracy without a budget is half a score.
  citation_precision / citation_recall / answer_parsed

The gate is the design's load-bearing idea: a correct-sounding driver with the wrong
evidence earns almost nothing, so an agent cannot be paid for guessing the stereotype and
dressing it up. Everything except the semantic match is set arithmetic.
"""

from __future__ import annotations

import verifiers.v1 as vf

from insight_consolidation_v1.generator import build_episode, plan_episodes
from insight_consolidation_v1.judge import GroundedInsightJudge, GroundedInsightJudgeConfig
from insight_consolidation_v1.scoring import ANSWER_CONTRACT, citation_f1, parse_answer
from insight_consolidation_v1.servers.tool import (
    EvidenceToolset,
    EvidenceToolsetConfig,
    InsightState,
)

SYSTEM_PROMPT = """\
You are an analyst on a vendor's account team, reviewing everything that team has recorded \
about one customer: emails, meeting notes, support tickets, CRM entries and call summaries.

Your job is to work out what is actually driving this account's behaviour. Customers \
routinely misreport their own motives, so the stated reason is usually not the real one, \
and the real one is never written down in any single document. It is entailed only by \
several ordinary-looking traces taken together, scattered across months and channels.

Most of what you can see is irrelevant. Some of it will point convincingly at the obvious \
wrong answer. Investigate with the tools, then commit.
"""

PROMPT = """\
You work at {vendor}. You sell {product}.

Account: {company}
{count} documents are available covering roughly eight months.

Work out what is really driving this account's behaviour, and identify the specific \
documents that entail it.

{contract}"""


class InsightData(vf.TaskData):
    """One episode. The corpus is rebuilt from the spec rather than carried on the wire,
    which is what keeps a 5,000-document episode a few hundred bytes here."""

    episode_seed: int
    schema_key: str
    volume: int
    variant: str
    answer_key: str
    answer_description: str
    minimal_evidence: list[str]
    prior_label: str
    should_abstain: bool
    vendor: str = ""
    product: str = ""


class InsightTaskConfig(vf.TaskConfig):
    tools: EvidenceToolsetConfig = EvidenceToolsetConfig()
    judge: GroundedInsightJudgeConfig = GroundedInsightJudgeConfig()

    gate_full: float = 0.5
    """Citation F1 at which the judged insight counts in full. The gate ramps linearly
    from `gate_floor` at F1=0 to 1.0 here, rather than switching at a threshold - a cliff
    would itself be a reward-hacking surface, and would make the score jump on a single
    lucky citation."""

    gate_floor: float = 0.1
    """What a correct-sounding but wholly ungrounded answer keeps. Not zero, because a
    lucky guess is worth marginally more than a wrong one - but close to it."""

    max_turns: int = 60


class InsightTask(vf.Task[InsightData, InsightState, InsightTaskConfig]):
    tools = (EvidenceToolset,)

    def _judge(self) -> GroundedInsightJudge:
        return GroundedInsightJudge(self.config.judge)

    @vf.stop
    async def turn_budget(self, trace: vf.Trace) -> bool:
        return trace.num_turns >= self.config.max_turns

    # --- trained rewards ---------------------------------------------------

    @vf.reward(weight=1.0)
    async def grounded_insight(self, trace: vf.Trace) -> float:
        answer = parse_answer(trace.last_reply)
        judged = await self._judged_score(trace, answer)
        if self.data.should_abstain:
            return judged  # nothing to cite; abstention is the whole task
        _, _, f1 = citation_f1(answer.evidence, self.data.minimal_evidence)
        floor, full = self.config.gate_floor, max(1e-6, self.config.gate_full)
        gate = floor + (1.0 - floor) * min(1.0, f1 / full)
        return judged * gate

    @vf.reward(weight=0.5)
    async def evidence_grounding(self, trace: vf.Trace) -> float:
        if self.data.should_abstain:
            # Citing evidence for a driver that does not exist is the failure here.
            answer = parse_answer(trace.last_reply)
            return 1.0 if not answer.evidence else 0.0
        answer = parse_answer(trace.last_reply)
        return citation_f1(answer.evidence, self.data.minimal_evidence)[2]

    @vf.reward(weight=0.25)
    async def confabulation(self, trace: vf.Trace) -> float:
        answer = parse_answer(trace.last_reply)
        return float(answer.abstain == self.data.should_abstain)

    # --- diagnostics -------------------------------------------------------

    @vf.metric
    async def insight_ungated(self, trace: vf.Trace) -> float:
        return await self._judged_score(trace, parse_answer(trace.last_reply))

    @vf.metric
    async def anti_prior(self, trace: vf.Trace) -> float:
        if self.data.variant != "anti_prior":
            return float("nan")
        return await self.grounded_insight(trace)

    @vf.metric
    async def revision(self, trace: vf.Trace) -> float:
        if self.data.variant != "revision":
            return float("nan")
        return await self.grounded_insight(trace)

    @vf.metric
    async def volume(self, trace: vf.Trace) -> float:
        return float(self.data.volume)

    @vf.metric
    async def docs_read(self, trace: vf.Trace) -> float:
        return float(len(trace.state.read_ids)) if trace.state else 0.0

    @vf.metric
    async def citation_precision(self, trace: vf.Trace) -> float:
        return citation_f1(parse_answer(trace.last_reply).evidence, self.data.minimal_evidence)[0]

    @vf.metric
    async def citation_recall(self, trace: vf.Trace) -> float:
        return citation_f1(parse_answer(trace.last_reply).evidence, self.data.minimal_evidence)[1]

    @vf.metric
    async def answer_parsed(self, trace: vf.Trace) -> float:
        return float(parse_answer(trace.last_reply).parsed)

    # --- internals ---------------------------------------------------------

    async def _judged_score(self, trace: vf.Trace, answer) -> float:
        text = answer.driver if answer.parsed else trace.last_reply
        if self.data.should_abstain and answer.abstain and not answer.driver:
            text = "The analyst declined to name a driver and stated the evidence is insufficient."
        return await self._judge().grade(
            response=text,
            reference=self.data.answer_description,
            distractor=self.data.prior_label,
            abstain_expected=self.data.should_abstain,
            trace=trace,
        )


class InsightConfig(vf.TasksetConfig):
    num_tasks: int = 60
    """Size of the fixed evaluation set. Ignored when `procedural` is true."""

    seed: int = 0
    procedural: bool = False
    """True yields episodes forever for training. Use `-n` to bound a run."""

    task: InsightTaskConfig = InsightTaskConfig()


class InsightTaskset(vf.Taskset[InsightTask, InsightConfig]):
    def load(self):
        count = 10_000 if self.config.procedural else self.config.num_tasks
        for i, spec in enumerate(plan_episodes(count, seed=self.config.seed)):
            episode = build_episode(spec)
            yield InsightTask(
                InsightData(
                    idx=i,
                    name=f"{spec.schema_key}/{spec.variant}/n{spec.volume}",
                    system_prompt=SYSTEM_PROMPT,
                    prompt=PROMPT.format(
                        vendor=episode.vendor,
                        product=episode.product,
                        company=episode.company,
                        count=spec.volume,
                        contract=ANSWER_CONTRACT,
                    ),
                    episode_seed=spec.seed,
                    schema_key=spec.schema_key,
                    volume=spec.volume,
                    variant=spec.variant,
                    answer_key=episode.answer_key,
                    answer_description=episode.answer_description,
                    minimal_evidence=episode.minimal_evidence,
                    prior_label=episode.prior_label,
                    should_abstain=episode.should_abstain,
                    vendor=episode.vendor,
                    product=episode.product,
                ),
                self.config.task,
            )


class InsightProceduralTaskset(InsightTaskset):
    """Infinite variant for training runs. Bound it with `-n`."""

    INFINITE = True

    def load(self):
        self.config.procedural = True
        return super().load()
