#!/usr/bin/env python3
"""Judge calibration: does the grader actually separate a right answer from the trap?

The headline reward calls a model, and a judge nobody has measured is the failure mode this
whole environment argues against. KnowMe-Bench reports Cohen's kappa against human labels;
this environment does not have human labels yet, so it measures the next best thing -
construct validity. The judge holds the answer key, so we know in advance what it *should*
say about four kinds of answer:

    reference    the latent driver's own description, verbatim       expect 4
    label        the same driver compressed to one line              expect 3-4
    stereotype   the decoy the anti-prior split is built from        expect 0-1
    generic      a plausible-sounding claim that fits any account    expect 0-1

The number that matters is the separation between `label` and `stereotype`. If the judge
cannot tell a correct compressed answer from the wrong answer it was explicitly warned
about, the grounded_insight reward is noise and no model score computed from it means
anything.

Cheap: four calls per schema, forty in total, no rollouts.

    python judge_check.py
    python judge_check.py --model MiniMax-M2 --out judge_check.json
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
from collections import defaultdict

from insight_consolidation_v1.schemas import SCHEMAS
from run_eval import Client, ModelUnavailable, judge, load_key

GENERIC = (
    "The account is being cautious and wants to see more value demonstrated before "
    "committing. There are competing priorities internally and the timeline has slipped."
)

# What each probe should score, as a band. Used only to flag violations, never to grade.
EXPECTED = {
    "reference": (0.75, 1.0),
    "label": (0.5, 1.0),
    "stereotype": (0.0, 0.35),
    "generic": (0.0, 0.35),
}


def probes(schema) -> dict[str, str]:
    return {
        "reference": schema.description,
        "label": schema.label,
        "stereotype": schema.prior_label,
        "generic": GENERIC,
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="MiniMax-M2")
    ap.add_argument("--dialect", choices=("anthropic", "openai"), default="openai")
    ap.add_argument("--base", default="https://api.minimax.io")
    ap.add_argument("--key-var", default="MINIMAX_TOKEN_PLAN_API")
    ap.add_argument("--out", default="judge_check.json")
    args = ap.parse_args()

    key = load_key(args.key_var)
    if not key:
        print(f"no API key: set {args.key_var}", file=sys.stderr)
        return 1
    base = args.base.rstrip("/")
    if args.dialect == "anthropic" and not base.endswith("/anthropic"):
        base += "/anthropic"
    client = Client(key, args.model, args.dialect, base)

    scores: dict[str, list[float]] = defaultdict(list)
    rows = []
    for schema in SCHEMAS:
        for kind, answer in probes(schema).items():
            try:
                s = judge(client, answer, schema.description, schema.prior_label, False)
            except ModelUnavailable as e:
                print(f"  ! {schema.key}/{kind} unavailable: {e}", file=sys.stderr)
                continue
            scores[kind].append(s)
            rows.append({"schema": schema.key, "probe": kind, "score": s})
            print(f"  {schema.key:<32} {kind:<11} {s:.2f}", flush=True)

    print(f"\njudge calibration - {args.model} - {len(rows)} probes\n")
    summary = {}
    for kind in ("reference", "label", "stereotype", "generic"):
        vals = scores.get(kind, [])
        if not vals:
            continue
        mean = statistics.fmean(vals)
        lo, hi = EXPECTED[kind]
        flag = "" if lo <= mean <= hi else "   <-- OUTSIDE EXPECTED BAND"
        summary[kind] = mean
        print(f"  {kind:<11} mean={mean:.3f}  (n={len(vals)}, expected {lo:.2f}-{hi:.2f}){flag}")

    sep = summary.get("label", 0.0) - summary.get("stereotype", 1.0)
    print(f"\n  separation (label - stereotype) = {sep:+.3f}")
    if sep < 0.4:
        print(
            "  WARNING: the judge barely distinguishes the right answer from the decoy.\n"
            "  grounded_insight is not a trustworthy reward with this judge model."
        )
    else:
        print("  The judge separates the correct driver from the stereotype it was warned about.")

    with open(args.out, "w") as fh:
        json.dump({"model": args.model, "summary": summary, "rows": rows}, fh, indent=2)
    print(f"\nWritten to {args.out}.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
