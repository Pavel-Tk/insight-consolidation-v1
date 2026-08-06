"""Generator invariants. These are the tests that keep the benchmark honest."""

from __future__ import annotations

import pytest

from insight_consolidation_v1.corpus import Document
from insight_consolidation_v1.generator import (
    VARIANTS,
    VOLUMES,
    EpisodeSpec,
    build_episode,
    plan_episodes,
)
from insight_consolidation_v1.schemas import SCHEMAS
from insight_consolidation_v1.scoring import citation_f1, parse_answer, tokenize


def test_determinism():
    spec = EpisodeSpec(seed=7, schema_key="career_exposure", volume=50)
    a, b = build_episode(spec), build_episode(spec)
    assert [d.body for d in a.documents] == [d.body for d in b.documents]
    assert a.minimal_evidence == b.minimal_evidence


def test_determinism_across_processes():
    """The in-process check above is not enough. Python randomizes string hashing per
    process, so a PRNG seeded from builtin `hash()` reproduces perfectly within one run
    and differs between runs - which would quietly void every published score."""
    import os
    import subprocess
    import sys

    snippet = (
        "from insight_consolidation_v1.generator import EpisodeSpec, build_episode;"
        "e = build_episode(EpisodeSpec(seed=7, schema_key='career_exposure', volume=50));"
        "print(e.minimal_evidence, e.documents[3].body[:40])"
    )
    # Inherit the real environment and override only PYTHONHASHSEED. A hand-built env with a
    # POSIX PATH cannot launch the interpreter on Windows, and the point of this test is that
    # it runs everywhere the benchmark does.
    runs = {
        subprocess.run(
            [sys.executable, "-c", snippet],
            capture_output=True,
            text=True,
            check=True,
            env={**os.environ, "PYTHONHASHSEED": seed},
        ).stdout
        for seed in ("0", "1", "12345")
    }
    assert len(runs) == 1, f"episode is not reproducible across processes: {runs}"


def test_volumes_and_ids_unique():
    for volume in (50, 500):
        ep = build_episode(EpisodeSpec(seed=1, schema_key="budget_freeze", volume=volume))
        assert len(ep.documents) == volume
        assert len({d.doc_id for d in ep.documents}) == volume


def test_timeline_is_monotonic():
    ep = build_episode(EpisodeSpec(seed=3, schema_key="champion_is_leaving", volume=200))
    days = [d.day for d in ep.documents]
    assert days == sorted(days)


@pytest.mark.parametrize("schema", SCHEMAS, ids=lambda s: s.key)
def test_every_schema_builds_and_entails(schema):
    ep = build_episode(EpisodeSpec(seed=11, schema_key=schema.key, volume=80))
    assert ep.answer_key == schema.key
    assert len(ep.minimal_evidence) >= 2, "a driver entailed by one document is not sparse"
    assert ep.prior_label


@pytest.mark.parametrize("variant", VARIANTS)
def test_variants(variant):
    ep = build_episode(
        EpisodeSpec(seed=5, schema_key="career_exposure", volume=100, variant=variant)
    )
    if variant == "null":
        assert ep.should_abstain and ep.answer_key == "" and not ep.minimal_evidence
    else:
        assert not ep.should_abstain and ep.minimal_evidence


def test_revision_answer_is_the_late_driver():
    ep = build_episode(
        EpisodeSpec(seed=5, schema_key="career_exposure", volume=120, variant="revision")
    )
    assert ep.answer_key == "budget_freeze"
    index = ep.index
    days = [index[d].day for d in ep.minimal_evidence]
    # The evidence that settles a revision episode must sit late on the timeline.
    assert min(days) > 120, days


def test_no_single_document_names_the_driver():
    """Rule 1 from schemas.py, enforced. If one document gives the answer away, the task
    is retrieval rather than consolidation."""
    for schema in SCHEMAS:
        key_terms = {t for t in tokenize(schema.label) if len(t) > 4}
        for signal in schema.signals:
            overlap = key_terms & set(tokenize(signal.text))
            assert len(overlap) <= 1, (
                f"{schema.key}: signal leaks the driver via {overlap}: {signal.text!r}"
            )


def test_anti_prior_raises_decoy_density():
    normal = build_episode(EpisodeSpec(seed=2, schema_key="budget_freeze", volume=200))
    anti = build_episode(
        EpisodeSpec(seed=2, schema_key="budget_freeze", volume=200, variant="anti_prior")
    )
    from insight_consolidation_v1.corpus import PRIOR_TEMPLATES

    stems = [t.split("{")[0][:20] for t in PRIOR_TEMPLATES["slow_procurement"]]

    def decoys(ep) -> int:
        return sum(1 for d in ep.documents if any(s and s in d.body for s in stems))

    assert decoys(anti) > decoys(normal)


def test_plan_is_balanced_and_reproducible():
    plan = plan_episodes(60, seed=0)
    assert plan == plan_episodes(60, seed=0)
    assert {s.schema_key for s in plan} == {s.key for s in SCHEMAS}
    assert set(VOLUMES) & {s.volume for s in plan}


def test_large_volume_is_tractable():
    ep = build_episode(EpisodeSpec(seed=9, schema_key="trust_damaged_earlier", volume=5000))
    assert len(ep.documents) == 5000
    assert all(isinstance(d, Document) for d in ep.documents[:5])


def test_citation_f1():
    assert citation_f1(("doc_00001",), ["doc_00001"]) == (1.0, 1.0, 1.0)
    assert citation_f1((), ["doc_00001"]) == (0.0, 0.0, 0.0)
    p, r, f = citation_f1(("doc_00001", "doc_00002"), ["doc_00001", "doc_00003"])
    assert p == 0.5 and r == 0.5 and f == 0.5


def test_parse_answer_forms():
    fenced = parse_answer(
        'here you go\n```json\n{"driver": "blame risk", '
        '"evidence": ["doc_00001", "doc_00002"], "abstain": false}\n```'
    )
    assert fenced.parsed and fenced.driver == "blame risk" and len(fenced.evidence) == 2

    bare = parse_answer('{"driver": "x", "evidence": [], "abstain": true}')
    assert bare.parsed and bare.abstain

    prose = parse_answer("I think it is doc_00007 and doc_00009 that matter here.")
    assert not prose.parsed and prose.evidence == ("doc_00007", "doc_00009")

    assert not parse_answer("").parsed
    assert parse_answer("I must abstain, insufficient evidence.").abstain


def test_reasoning_scratchpad_is_not_the_answer():
    """Reasoning models emit <think> inline. Grading it means grading deliberation - every
    hypothesis the model raised and rejected - instead of what it committed to, and it lets
    document ids the model merely mused about count as citations."""
    from insight_consolidation_v1.scoring import strip_reasoning

    reply = (
        "<think>Maybe doc_00001 matters. Could be a budget freeze. No, discard that.</think>\n"
        '```json\n{"driver": "blame risk", "evidence": ["doc_00042"], "abstain": false}\n```'
    )
    answer = parse_answer(reply)
    assert answer.parsed
    assert answer.driver == "blame risk"
    assert answer.evidence == ("doc_00042",), "ids from the scratchpad must not be cited"

    # An unclosed <think> is what a truncated response looks like. It must not survive into
    # the graded text either.
    assert "discard" not in strip_reasoning("<think>rambling, discard this")
    prose = parse_answer("<think>doc_00003 maybe</think> It is doc_00007.")
    assert prose.evidence == ("doc_00007",)


def test_judge_verdict_is_not_read_off_the_scratchpad():
    """Regression: the harness used to grade with `re.search(r"[0-4]", text)`, which on a
    reasoning model picks up the first digit in the <think> block - a year, a document count
    - rather than the verdict. It made the judged reward pure noise, and the calibration
    probe caught it scoring the stereotype above the reference. Unparseable must be None,
    never 0.0, or an instrument failure is indistinguishable from a real zero."""
    from run_eval import parse_verdict

    assert parse_verdict("<think>2023 and 4 docs</think>\nSCORE: 0") == 0.0
    assert parse_verdict("<think>3 things, 2 points</think>\nSCORE: 4") == 1.0
    assert parse_verdict("SCORE: 2") == 0.5
    assert parse_verdict("2\nThe analyst hedged.") == 0.5
    assert parse_verdict("<think>cut off mid thought with 3 items") is None
    assert parse_verdict("prose with no grade in it") is None
    assert parse_verdict("") is None


def test_world_is_consistent_and_specific():
    """Every schema names its product, its teams and what is at stake, and the whole corpus
    lives in that one world. Signals must not be the only documents that name the real
    teams - that would make them separable by vocabulary alone."""
    from insight_consolidation_v1.schemas import SCHEMAS

    for schema in SCHEMAS:
        setting = schema.setting
        assert setting.vendor and setting.product and len(setting.parties) >= 3, schema.key
        ep = build_episode(EpisodeSpec(seed=4, schema_key=schema.key, volume=200))
        assert ep.vendor == setting.vendor

        signal_ids = set(ep.minimal_evidence)
        non_signal = [d for d in ep.documents if d.doc_id not in signal_ids]
        for party in setting.parties:
            in_signal = any(party in ep.index[i].body for i in signal_ids)
            if in_signal:
                assert any(party in d.body for d in non_signal), (
                    f"{schema.key}: team {party!r} appears only in signal documents"
                )


def test_uniqueness_is_not_a_shortlist():
    """Noise repeated verbatim while signals never do would let an agent shortlist the
    entailing documents by looking for text that appears exactly once."""
    from collections import Counter

    for volume in (500, 5000):
        ep = build_episode(
            EpisodeSpec(seed=1, schema_key="already_built_internally", volume=volume)
        )
        counts = Counter(d.body for d in ep.documents)
        appear_once = sum(1 for body, k in counts.items() if k == 1)
        assert appear_once > volume * 0.5, (
            f"volume={volume}: only {appear_once} unique bodies, signals stand out"
        )


def test_behavioural_signal_rule():
    """Rule 2: at least two signals per schema must be things the buyer DID."""
    from insight_consolidation_v1.schemas import SCHEMAS

    for schema in SCHEMAS:
        behavioural = sum(1 for s in schema.signals if s.behavioural)
        assert behavioural >= 2, f"{schema.key}: only {behavioural} behavioural signals"
