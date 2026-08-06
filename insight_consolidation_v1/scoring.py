"""Answer parsing and the deterministic half of the rubric.

Nothing here calls a model. Citation grounding, abstention and budget are set arithmetic,
which is the point: the only judged dimension is whether the agent's free-text driver
matches the latent one, and even that judge holds the answer key.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass

_FENCE = re.compile(r"```(?:json)?\s*(\{.*?\})\s*```", re.DOTALL)
_BARE = re.compile(r"(\{[^{}]*\"driver\"[^{}]*\})", re.DOTALL)
_DOC_ID = re.compile(r"doc_\d{5}")

# Reasoning models emit their scratchpad inline. It is not the answer, and leaving it in
# means the judge grades deliberation - including any wrong hypothesis the model considered
# and discarded - instead of the claim the agent actually committed to. Handles an unclosed
# trailing <think> too, which is what a truncated response looks like.
_THINK = re.compile(r"<think>.*?</think>|<think>.*\Z", re.DOTALL | re.IGNORECASE)


def strip_reasoning(reply: str) -> str:
    """Remove reasoning-model scratchpad blocks. Document ids mentioned only inside the
    scratchpad do not count as citations - the agent has to commit to them."""
    return _THINK.sub(" ", reply or "").strip()


@dataclass(frozen=True)
class Answer:
    driver: str = ""
    evidence: tuple[str, ...] = ()
    abstain: bool = False
    parsed: bool = False
    """False when no structured answer could be recovered at all."""


def parse_answer(reply: str) -> Answer:
    """Recover the structured claim from the final reply.

    Deliberately forgiving about wrapping - fenced block, bare object, or prose with ids -
    and strict about the fields. A model that cannot emit the contract at all scores zero
    rather than erroring, but we record that it failed to parse so the metric is visible.
    """
    if not reply or not reply.strip():
        return Answer()

    reply = strip_reasoning(reply)
    if not reply:
        return Answer()

    blob = None
    for pattern in (_FENCE, _BARE):
        matches = pattern.findall(reply)
        if matches:
            blob = matches[-1]
            break

    if blob is not None:
        try:
            data = json.loads(blob)
        except json.JSONDecodeError:
            data = None
        if isinstance(data, dict):
            raw_ev = data.get("evidence") or []
            if isinstance(raw_ev, str):
                raw_ev = _DOC_ID.findall(raw_ev)
            evidence = tuple(
                dict.fromkeys(str(e).strip() for e in raw_ev if _DOC_ID.fullmatch(str(e).strip()))
            )
            return Answer(
                driver=str(data.get("driver") or "").strip(),
                evidence=evidence,
                abstain=bool(data.get("abstain", False)),
                parsed=True,
            )

    # Last resort: unstructured prose. Harvest ids so a near-miss still gets grounding
    # credit, but treat the whole reply as the driver claim.
    ids = tuple(dict.fromkeys(_DOC_ID.findall(reply)))
    abstain = bool(re.search(r"\b(abstain|insufficient evidence|cannot determine)\b", reply, re.I))
    return Answer(driver=reply.strip()[:2000], evidence=ids, abstain=abstain, parsed=False)


def citation_f1(cited: tuple[str, ...], truth: list[str]) -> tuple[float, float, float]:
    """Precision, recall, F1 of cited document ids against the minimal entailing set."""
    if not truth:
        return (0.0, 0.0, 0.0)
    cited_set, truth_set = set(cited), set(truth)
    if not cited_set:
        return (0.0, 0.0, 0.0)
    hits = len(cited_set & truth_set)
    precision = hits / len(cited_set)
    recall = hits / len(truth_set)
    f1 = 0.0 if precision + recall == 0 else 2 * precision * recall / (precision + recall)
    return (precision, recall, f1)


def tokenize(text: str) -> list[str]:
    return re.findall(r"[a-z0-9]+", text.lower())


ANSWER_CONTRACT = """\
When you have finished investigating, reply with a single fenced JSON block and nothing else:

```json
{
  "driver": "one or two sentences naming what is actually driving this account's behaviour",
  "evidence": ["doc_00123", "doc_00456"],
  "abstain": false
}
```

Rules:
- `driver` is free text. Say what is really going on, not what they say is going on.
- `evidence` must list the specific document ids that jointly support your conclusion.
  A correct conclusion with the wrong evidence scores close to zero, so do not pad this
  list and do not guess.
- Set `abstain` to true, with an empty `evidence` list, if the documents do not actually
  entail any coherent driver. Some accounts genuinely contain nothing. Guessing confidently
  on those is penalised.
"""
