"""The grounded insight judge.

This is the one place a model enters the scoring path, and it enters holding the answer key.
That distinction is load-bearing. An open-ended "how insightful is this?" judge is the
failure mode the memory-benchmark literature is currently drowning in - GPT-4o-mini accepts
roughly 63% of intentionally wrong-but-topical answers when asked to grade quality blind.
A judge asked "does this claim mean the same thing as this reference claim?" is a semantic
equivalence check with a fixed target, which is a far narrower and far more reliable ask.

It is still a judge. The environment therefore reports it alongside deterministic
grounding, and gates it: an insight judged correct without the evidence that entails it
earns almost nothing.
"""

from __future__ import annotations

from typing import cast

from pydantic import BaseModel, Field

import verifiers.v1 as vf

GROUNDED_PROMPT = """\
You are grading whether an analyst identified the real driver behind a customer account's \
behaviour. You have the reference answer. Grade meaning, not wording.

REFERENCE (the true driver, established by construction):
{reference}

A COMMON WRONG ANSWER for this account (the surface reading an analyst lands on if they \
read stated positions instead of behaviour):
{distractor}

THE ANALYST'S ANSWER:
{response}

Score 0-4:
4 - Same driver as the reference. Wording differs, meaning matches, including the part \
that distinguishes it from the common wrong answer.
3 - Substantially the reference driver, but hedged, partial, or missing the distinguishing \
element.
2 - Touches the right area but does not commit to the reference driver, or blends it with \
the wrong answer.
1 - Mostly the common wrong answer, or a generic observation that would fit any account.
0 - Wrong, empty, or unrelated.

Grade only the analyst's claim about the driver. Ignore style, length, confidence and any \
document ids. Do not reward a claim for being well written.

Respond with the score and a one-sentence reason.
"""

ABSTAIN_PROMPT = """\
An analyst was shown an account whose documents do NOT entail any coherent underlying \
driver. The correct behaviour was to say so.

THE ANALYST'S ANSWER:
{response}

Score 4 if the analyst clearly declined to name a driver and said the evidence is \
insufficient. Score 0 if the analyst asserted a driver. Score 2 if it is genuinely \
ambiguous which they did.

Respond with the score and a one-sentence reason.
"""


class Verdict(BaseModel):
    score: int = Field(ge=0, le=4)
    reason: str = ""


class GroundedInsightJudgeConfig(vf.JudgeConfig):
    id: vf.ID = "grounded_insight"
    model: str = "openai/gpt-5.4-nano"


class GroundedInsightJudge(vf.Judge[float, GroundedInsightJudgeConfig]):
    """Semantic equivalence against a known latent driver, normalized to [0, 1]."""

    prompt = GROUNDED_PROMPT
    schema = Verdict

    def parse(self, response: vf.JudgeResponse[float]) -> float:
        parsed = response.parsed
        if isinstance(parsed, Verdict):
            return parsed.score / 4.0
        return 0.0

    async def grade(
        self,
        *,
        response: str,
        reference: str,
        distractor: str,
        abstain_expected: bool,
        trace: vf.Trace | None = None,
    ) -> float:
        if not response.strip():
            return 0.0
        template = ABSTAIN_PROMPT if abstain_expected else GROUNDED_PROMPT
        messages = template.format(
            response=response[:6000], reference=reference, distractor=distractor
        )
        result = await self.complete(
            messages, trace=trace, schema=self.schema, parse=self.parse
        )
        return cast(float, result.parsed)


__all__ = ["GroundedInsightJudge", "GroundedInsightJudgeConfig", "Verdict"]
