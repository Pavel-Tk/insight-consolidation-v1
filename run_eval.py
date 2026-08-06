#!/usr/bin/env python3
"""Reference harness: run insight-consolidation-v1 against a model and print the rubric.

This is deliberately NOT the verifiers rollout stack. `uv run eval` needs an inference
endpoint, an interception server and tool servers running; this script needs a single chat
API and stdlib. It exists so the benchmark produces numbers on a laptop, so the first
results do not depend on getting the whole training stack up, and so the environment can be
exercised on Windows, where `verifiers.v1` does not import at all (it needs `fcntl`).

Scoring is the environment's real rubric, imported from the package rather than
reimplemented, so numbers here and numbers from `uv run eval` mean the same thing.

Usage
-----
    export MINIMAX_TOKEN_PLAN_API=sk-...
    python run_eval.py --n 30
    python run_eval.py --n 12 --volumes 50 --out results.jsonl

    # grade with a different model family than the one being evaluated
    python run_eval.py --n 30 \
        --judge-dialect anthropic \
        --judge-base https://api.anthropic.com \
        --judge-model claude-sonnet-5 \
        --judge-key-var ANTHROPIC_API_KEY

Reads a .env in the working directory if the variable is not already set.
"""

from __future__ import annotations

import argparse
import json
import os
import random
import re
import statistics
import sys
import threading
import time
import urllib.error
import urllib.request
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path

from insight_consolidation_v1.generator import build_episode, plan_episodes
from insight_consolidation_v1.scoring import (
    ANSWER_CONTRACT,
    citation_f1,
    parse_answer,
    strip_reasoning,
)
from insight_consolidation_v1.store import EvidenceStore

GATE_FLOOR = 0.1
GATE_FULL = 0.5

SYSTEM = """\
You are an analyst on a vendor's account team, reviewing everything that team has recorded \
about one customer: emails, meeting notes, support tickets, CRM entries and call summaries.

Your job is to work out what is actually driving this account's behaviour. Customers \
routinely misreport their own motives, so the stated reason is usually not the real one, \
and the real one is never written down in any single document. It is entailed only by \
several ordinary-looking traces taken together, scattered across months and channels.

Most of what you can see is irrelevant. Some of it will point convincingly at the obvious \
wrong answer.

You investigate by emitting ONE command per reply, on its own line, and nothing else:

  SEARCH: <keywords>          lexical search over document bodies, returns ranked snippets
  READ: doc_00123             read one document in full
  INDEX: <start_day>          list document headers chronologically, no bodies

You will be given the result, then you emit the next command. Search is lexical, not \
semantic: it will not understand "why are they hesitating". Search for words that would \
literally appear in a document.

When you are ready to commit, reply with the JSON answer block instead of a command.
"""

# The investigation system prompt says "ONE command per reply, and nothing else". Reusing it
# for the final commit means the system instruction fights the user instruction, and the
# model keeps emitting commands after being told to stop. It does - that was a real bug here,
# and every budget-exhausted episode scored zero because of it.
FINAL_SYSTEM = """\
You are an analyst on a vendor's account team. You have finished reviewing what the team has \
recorded about one customer.

The investigation is over. Do not emit SEARCH, READ or INDEX - they will not be executed.

Reply with the JSON answer block and nothing else.
"""

TASK = """\
You work at {vendor}. You sell {product}.

Account: {company}
{count} documents are available covering roughly eight months.

Work out what is really driving this account's behaviour, and identify the specific \
documents that entail it.

{contract}
Begin investigating. One command per reply.
"""

JUDGE_GROUNDED = """\
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
2 - Touches the right area but does not commit, or blends it with the wrong answer.
1 - Mostly the common wrong answer, or a generic observation that would fit any account.
0 - Wrong, empty, or unrelated.

Grade only the claim about the driver. Ignore style, length, confidence and document ids.

End your reply with a line of exactly this form, and nothing after it:
SCORE: <digit>
"""

JUDGE_ABSTAIN = """\
An analyst was shown an account whose documents do NOT entail any coherent underlying \
driver. The correct behaviour was to say so.

THE ANALYST'S ANSWER:
{response}

Score 4 if the analyst clearly declined to name a driver and said the evidence is \
insufficient. Score 0 if the analyst asserted a driver. Score 2 if genuinely ambiguous.

End your reply with a line of exactly this form, and nothing after it:
SCORE: <digit>
"""


# --------------------------------------------------------------------------- client


class ModelUnavailable(RuntimeError):
    """The API could not be reached after retries. Distinct from the model answering badly:
    an episode that raises this is dropped from the run, never scored as a zero."""


class Client:
    """Minimal chat client over stdlib. Speaks the OpenAI and Anthropic dialects."""

    def __init__(
        self,
        api_key: str,
        model: str,
        dialect: str,
        base: str,
        timeout: float = 180.0,
        retries: int = 7,
        backoff_cap: float = 60.0,
    ):
        self.api_key, self.model, self.dialect, self.base = api_key, model, dialect, base
        self.timeout = timeout
        self.retries = retries
        self.backoff_cap = backoff_cap
        self.calls = 0
        self.failures = 0
        self._counter_lock = threading.Lock()

    def _post(self, url: str, headers: dict, payload: dict) -> dict:
        data = json.dumps(payload).encode()
        last: Exception | None = None
        for attempt in range(self.retries):
            req = urllib.request.Request(url, data=data, headers=headers, method="POST")
            try:
                with urllib.request.urlopen(req, timeout=self.timeout) as r:
                    return json.loads(r.read().decode())
            except urllib.error.HTTPError as e:
                body = e.read().decode()[:300]
                last = RuntimeError(f"HTTP {e.code}: {body}")
                if e.code in (429, 500, 502, 503, 529):
                    # Plan-level rate limits are not transient blips - a fixed 2s bump gives
                    # up long before the window resets. Exponential with a cap, jittered so
                    # concurrent workers do not retry in lockstep.
                    delay = min(self.backoff_cap, 2.0 * (2**attempt))
                    time.sleep(delay * (0.5 + random.random()))
                    continue
                raise last from None
            except Exception as e:  # noqa: BLE001
                last = e
                time.sleep(min(self.backoff_cap, 2.0 * (2**attempt)))
        raise last if last else RuntimeError("request failed")

    def chat(self, system: str, messages: list[dict], max_tokens: int = 1500) -> str:
        with self._counter_lock:
            self.calls += 1
        try:
            if self.dialect == "anthropic":
                out = self._post(
                    f"{self.base}/v1/messages",
                    {
                        "x-api-key": self.api_key,
                        "anthropic-version": "2023-06-01",
                        "content-type": "application/json",
                    },
                    {
                        "model": self.model,
                        "max_tokens": max_tokens,
                        "system": system,
                        "messages": messages,
                    },
                )
                return "".join(
                    b.get("text", "") for b in out.get("content", []) if b.get("type") == "text"
                )
            out = self._post(
                f"{self.base}/v1/chat/completions",
                {
                    "Authorization": f"Bearer {self.api_key}",
                    "Content-Type": "application/json",
                },
                {
                    "model": self.model,
                    "max_tokens": max_tokens,
                    "messages": [{"role": "system", "content": system}, *messages],
                },
            )
            return out["choices"][0]["message"]["content"] or ""
        except Exception as e:  # noqa: BLE001
            with self._counter_lock:
                self.failures += 1
            print(f"    ! model call failed: {e}", file=sys.stderr)
            # Raise rather than returning "". An empty reply is indistinguishable from a
            # model that declined to answer, so swallowing the error here turns an API
            # outage into a legitimate-looking score of zero. For a benchmark whose entire
            # argument is that published numbers are not to be trusted, silently scoring
            # infrastructure failures would be the worst possible bug to ship.
            raise ModelUnavailable(str(e)) from e


# --------------------------------------------------------------------------- rollout

CMD = re.compile(r"^\s*(SEARCH|READ|INDEX)\s*:\s*(.+?)\s*$", re.I | re.M)


@dataclass
class Rollout:
    reply: str = ""
    turns: int = 0
    reads: int = 0
    searches: int = 0
    read_ids: list[str] = field(default_factory=list)
    """Distinct documents actually read. `docs_read` reports this rather than `reads` so the
    number means the same thing here as the taskset's `docs_read` metric, which counts
    `len(state.read_ids)`."""
    transcript: list[str] = field(default_factory=list)
    committed: bool = False
    """True if the agent volunteered an answer block before the turn budget ran out."""
    forced: bool = False
    """True if the answer came from the end-of-budget forced commit. Recorded per episode:
    a high forced rate means the turn budget is the binding constraint, not the task."""


def run_episode(client: Client, spec, max_turns: int = 24) -> tuple[Rollout, object]:
    episode = build_episode(spec)
    store = EvidenceStore()
    store.load(episode.documents)

    roll = Rollout()
    messages: list[dict] = [
        {
            "role": "user",
            "content": TASK.format(
                vendor=episode.vendor,
                product=episode.product,
                company=episode.company,
                count=spec.volume,
                contract=ANSWER_CONTRACT,
            ),
        }
    ]

    for _ in range(max_turns):
        reply = client.chat(SYSTEM, messages)
        roll.turns += 1
        if not reply.strip():
            break
        messages.append({"role": "assistant", "content": reply})

        if '"driver"' in reply or '"abstain"' in reply:
            roll.reply = reply
            roll.committed = True
            break

        match = CMD.search(reply)
        if match is None:
            messages.append(
                {
                    "role": "user",
                    "content": "Emit one command (SEARCH:, READ:, INDEX:) or the JSON answer block.",
                }
            )
            continue

        verb, arg = match.group(1).upper(), match.group(2)
        if verb == "SEARCH":
            roll.searches += 1
            result = store.search(arg, 6)
        elif verb == "READ":
            result, found = store.read(arg.split()[0])
            if found is not None:
                roll.reads += 1
                if found not in roll.read_ids:
                    roll.read_ids.append(found)
        else:
            try:
                day = int(re.sub(r"\D", "", arg) or 0)
            except ValueError:
                day = 0
            result = store.index(day, 25)
        roll.transcript.append(f"{verb}: {arg}")
        messages.append({"role": "user", "content": result[:4000]})

    if not roll.committed:
        # Running out of turns is not the same as refusing to answer. Force one final
        # commit so the score reflects the agent's conclusion rather than whatever it
        # happened to be saying when the budget ran out.
        #
        # This used to be `if not roll.reply:` after a for/else that assigned the last
        # assistant message to roll.reply - which is never empty, so the forced commit never
        # fired and every budget-exhausted episode was graded on a mid-investigation
        # ramble. Keep the flag; do not go back to testing the text.
        roll.forced = True
        roll.reply = client.chat(
            FINAL_SYSTEM,
            [
                *messages,
                {
                    "role": "user",
                    "content": (
                        "Turn budget reached. The investigation is over and no further "
                        "commands will be executed.\n\n"
                        "Give your answer now.\n\n"
                        f"{ANSWER_CONTRACT}"
                    ),
                },
            ],
            # Reasoning models spend the first several hundred tokens thinking. At 900 the
            # scratchpad ate the budget and the JSON never got emitted.
            max_tokens=2400,
        )
        roll.turns += 1
    return roll, episode


# --------------------------------------------------------------------------- scoring


_VERDICT_TAG = re.compile(r"SCORE\s*[:=]?\s*([0-4])\b", re.I)
_LEADING_DIGIT = re.compile(r"\A\D{0,20}?([0-4])\b")


def parse_verdict(text: str) -> float | None:
    """Recover the grade from the judge's reply. Returns None if no grade is recoverable.

    Two things make the naive `re.search(r"[0-4]", text)` actively dangerous with reasoning
    models, and both bit this harness:

      1. The scratchpad comes first, so the first digit in the response is whatever number
         the judge happened to mention while thinking - a year, a document count, a list
         index. The grade was being read off noise.
      2. If the token budget cuts the reply off inside the scratchpad, there is no grade at
         all, and returning 0.0 for that is indistinguishable from a real zero.

    So: strip the scratchpad, look for an explicit SCORE tag, then a leading digit, and
    return None rather than guessing.
    """
    cleaned = strip_reasoning(text)
    if not cleaned:
        return None
    for pattern in (_VERDICT_TAG, _LEADING_DIGIT):
        match = pattern.search(cleaned)
        if match:
            return int(match.group(1)) / 4.0
    return None


def judge(client: Client, response: str, reference: str, distractor: str, abstain: bool) -> float:
    if not response.strip():
        return 0.0
    prompt = (
        JUDGE_ABSTAIN.format(response=response[:6000])
        if abstain
        else JUDGE_GROUNDED.format(
            reference=reference, distractor=distractor, response=response[:6000]
        )
    )
    # Enough room for a reasoning model to think and still emit the verdict. At 1200 roughly
    # one grade in forty was still cut off mid-scratchpad and had to be dropped.
    text = client.chat(
        "You are a careful grader. Follow the scale exactly.",
        [{"role": "user", "content": prompt}],
        2000,
    )
    verdict = parse_verdict(text)
    if verdict is None:
        # Unparseable is an instrument failure, not a score of zero.
        raise ModelUnavailable(f"judge returned no recoverable grade: {text[:200]!r}")
    return verdict


def score(judge_client: Client, roll: Rollout, episode) -> dict:
    answer = parse_answer(roll.reply)
    text = answer.driver if answer.parsed else roll.reply
    if episode.should_abstain and answer.abstain and not answer.driver:
        text = "The analyst declined to name a driver and stated the evidence is insufficient."
    judged = judge(
        judge_client,
        text,
        episode.answer_description,
        episode.prior_label,
        episode.should_abstain,
    )
    p, r, f1 = citation_f1(answer.evidence, episode.minimal_evidence)

    if episode.should_abstain:
        grounded = judged
        grounding = 1.0 if not answer.evidence else 0.0
    else:
        gate = GATE_FLOOR + (1 - GATE_FLOOR) * min(1.0, f1 / GATE_FULL)
        grounded = judged * gate
        grounding = f1

    return {
        "schema": episode.spec.schema_key,
        "variant": episode.spec.variant,
        "volume": episode.spec.volume,
        "grounded_insight": round(grounded, 4),
        "insight_ungated": round(judged, 4),
        "evidence_grounding": round(grounding, 4),
        "confabulation": float(answer.abstain == episode.should_abstain),
        "citation_precision": round(p, 4),
        "citation_recall": round(r, 4),
        "answer_parsed": float(answer.parsed),
        "forced_commit": float(roll.forced),
        "docs_read": len(roll.read_ids),
        "read_calls": roll.reads,
        "searches": roll.searches,
        "turns": roll.turns,
    }


# --------------------------------------------------------------------------- report


def report(rows: list[dict], model: str, budget_note: str = "") -> str:
    """The turn budget is part of the measurement, not a runtime detail, so it goes in the
    header. A grounded_insight number quoted without it does not mean anything."""

    def mean(key: str, subset: list[dict]) -> float:
        vals = [r[key] for r in subset]
        return statistics.fmean(vals) if vals else float("nan")

    out = [f"\ninsight-consolidation-v1 - {model} - n={len(rows)} - {budget_note}\n"]
    out.append("overall")
    for key in (
        "grounded_insight",
        "insight_ungated",
        "evidence_grounding",
        "confabulation",
        "citation_precision",
        "citation_recall",
        "answer_parsed",
        "forced_commit",
    ):
        vals = [r[key] for r in rows]
        sd = statistics.stdev(vals) if len(vals) > 1 else 0.0
        out.append(f"  {key:<20} {mean(key, rows):.3f}  (sd {sd:.3f})")
    out.append(f"  {'docs_read':<20} {mean('docs_read', rows):.1f}")
    out.append(f"  {'turns':<20} {mean('turns', rows):.1f}")

    for dim in ("variant", "volume"):
        out.append(f"\nby {dim}")
        buckets: dict = defaultdict(list)
        for row in rows:
            buckets[row[dim]].append(row)
        for k in sorted(buckets, key=str):
            sub = buckets[k]
            out.append(
                f"  {str(k):<12} n={len(sub):<3} grounded={mean('grounded_insight', sub):.3f}  "
                f"ungated={mean('insight_ungated', sub):.3f}  "
                f"grounding={mean('evidence_grounding', sub):.3f}  "
                f"read={mean('docs_read', sub):.1f}"
            )
    return "\n".join(out)


# --------------------------------------------------------------------------- main


def load_key(name: str) -> str:
    key = os.environ.get(name, "")
    if key:
        return key
    env = Path(".env")
    if env.exists():
        for line in env.read_text().splitlines():
            if line.strip().startswith(name):
                return line.split("=", 1)[1].strip()
    return ""


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=20)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--model", default="MiniMax-M2")
    ap.add_argument("--dialect", choices=("anthropic", "openai"), default="openai")
    ap.add_argument("--base", default="https://api.minimax.io")
    ap.add_argument("--key-var", default="MINIMAX_TOKEN_PLAN_API")
    # The judge defaults to the agent's client, which means one model grading itself. That is
    # a fair criticism and the reason these flags exist: point the judge at a different
    # family and report both numbers rather than asserting the self-graded one is fine.
    ap.add_argument("--judge-model", default=None)
    ap.add_argument("--judge-dialect", choices=("anthropic", "openai"), default=None)
    ap.add_argument("--judge-base", default=None)
    ap.add_argument("--judge-key-var", default=None)
    ap.add_argument("--volumes", type=int, nargs="*", default=[50, 500])
    # Matches InsightTaskConfig.max_turns. If these diverge, numbers from this harness and
    # numbers from `uv run eval` stop being comparable, which defeats the point of the file.
    ap.add_argument("--max-turns", type=int, default=60)
    ap.add_argument(
        "--concurrency",
        type=int,
        default=4,
        help="episodes in flight at once. Episodes are independent; only the API is shared.",
    )
    ap.add_argument("--out", default="results.jsonl")
    args = ap.parse_args()

    def build(model: str, dialect: str, base: str, key_var: str, role: str) -> Client | None:
        key = load_key(key_var)
        if not key:
            print(
                f"no API key for the {role}: set {key_var} in the environment or a .env file",
                file=sys.stderr,
            )
            return None
        base = base.rstrip("/")
        if dialect == "anthropic" and not base.endswith("/anthropic"):
            base += "/anthropic"
        return Client(key, model, dialect, base)

    client = build(args.model, args.dialect, args.base, args.key_var, "agent")
    if client is None:
        return 1

    judge_overridden = any(
        v is not None
        for v in (args.judge_model, args.judge_dialect, args.judge_base, args.judge_key_var)
    )
    if judge_overridden:
        judge_client = build(
            args.judge_model or args.model,
            args.judge_dialect or args.dialect,
            args.judge_base or args.base,
            args.judge_key_var or args.key_var,
            "judge",
        )
        if judge_client is None:
            return 1
    else:
        judge_client = client
        print(
            "note: agent and judge are the same model. Pass --judge-model/--judge-key-var to "
            "grade with a different family.",
            file=sys.stderr,
        )

    specs = plan_episodes(args.n, seed=args.seed, volumes=tuple(args.volumes))
    rows: list[dict] = []
    lock = threading.Lock()
    done = 0

    dropped: list[str] = []

    def one(index_spec: tuple[int, object]) -> dict | None:
        nonlocal done
        i, spec = index_spec
        label = f"{spec.schema_key}/{spec.variant}/n{spec.volume}"
        try:
            roll, episode = run_episode(client, spec, args.max_turns)
            row = score(judge_client, roll, episode)
        except ModelUnavailable as e:
            with lock:
                done += 1
                dropped.append(label)
                print(f"[{done}/{len(specs)}] {label}  DROPPED - API unavailable: {e}", flush=True)
            return None
        row["schema"] = spec.schema_key
        with lock:
            done += 1
            fh.write(json.dumps(row) + "\n")
            fh.flush()
            print(
                f"[{done}/{len(specs)}] {spec.schema_key}/{spec.variant}/n{spec.volume}  "
                f"grounded={row['grounded_insight']:.2f} "
                f"ungated={row['insight_ungated']:.2f} "
                f"grounding={row['evidence_grounding']:.2f} "
                f"read={row['docs_read']} turns={row['turns']}"
                f"{' FORCED' if row['forced_commit'] else ''}",
                flush=True,
            )
        return row

    workers = max(1, min(args.concurrency, len(specs)))
    print(f"{len(specs)} episodes, {workers} in flight, max_turns={args.max_turns}", flush=True)
    with open(args.out, "w") as fh:
        with ThreadPoolExecutor(max_workers=workers) as pool:
            rows = [r for r in pool.map(one, enumerate(specs, 1)) if r]

    if not rows:
        print("\nno episodes completed - every rollout hit the API limit. No numbers to report.")
        return 1

    budget_note = f"max_turns={args.max_turns}"
    print(report(rows, args.model, budget_note))
    if dropped:
        # Never quietly. A run that silently lost a third of its episodes to rate limits and
        # reported the mean of what survived is a selection-biased number.
        print(
            f"\nWARNING: {len(dropped)}/{len(specs)} episodes dropped to API errors and are "
            f"NOT in the numbers above. Re-run with lower --concurrency before quoting these."
        )
        for label in dropped:
            print(f"  dropped: {label}")
    calls = client.calls + (judge_client.calls if judge_client is not client else 0)
    failures = client.failures + (judge_client.failures if judge_client is not client else 0)
    print(f"\n{calls} model calls, {failures} failed. Rows written to {args.out}.")
    if judge_client is client:
        print(f"judge: {args.model} (same model as the agent - self-graded).")
    else:
        print(f"judge: {judge_client.model} via {judge_client.dialect}.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
