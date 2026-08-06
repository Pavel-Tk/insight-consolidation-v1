"""Episode construction: construct-then-hide.

The order matters and is the whole design. We sample the latent driver FIRST, then
manufacture evidence that entails it only in aggregate. Ground truth is therefore a
variable we chose, never a human annotation of prose, and never an artifact of how the
answer happens to be phrased.

An episode is fully determined by `EpisodeSpec`, so the corpus never travels on the wire:
the taskset puts the spec on `TaskData`, and the tool server rebuilds byte-identical
documents from it.

Variants
--------
`normal`     the driver is entailed by the minimal set. Answer it and cite it.
`null`       no coherent driver. Only noise and prior-decoys. Correct behaviour is to
             abstain. This is the confabulation control - without it a confident guesser
             looks competent.
`revision`   the driver holds for the first two thirds of the timeline, then late evidence
             flips it to `revises_to`. Correct answer is the NEW driver. Measures whether
             the agent updates or anchors.
`anti_prior` prior-supporting decoys are boosted so the stereotype answer is maximally
             attractive while the evidence still entails the true driver.
"""

from __future__ import annotations

import random
import zlib
from dataclasses import dataclass, field

from insight_consolidation_v1.corpus import (
    NOISE_TEMPLATES,
    PRIOR_TEMPLATES,
    Document,
    Personae,
    make_document,
)
from insight_consolidation_v1.schemas import SCHEMAS, LatentSchema, get_schema

VARIANTS = ("normal", "null", "revision", "anti_prior")
VOLUMES = (50, 500, 5000)


@dataclass(frozen=True)
class EpisodeSpec:
    """Everything needed to rebuild an episode deterministically."""

    seed: int
    schema_key: str
    volume: int = 50
    variant: str = "normal"

    def __post_init__(self) -> None:
        if self.variant not in VARIANTS:
            raise ValueError(f"unknown variant {self.variant!r}")
        if self.volume < 10:
            raise ValueError("volume must be at least 10")


@dataclass
class Episode:
    spec: EpisodeSpec
    documents: list[Document]
    answer_key: str
    """Latent key the agent must land on, or "" when the correct answer is abstain."""
    answer_description: str
    """What the judge compares the agent's free-text answer against."""
    minimal_evidence: list[str] = field(default_factory=list)
    """Document ids that jointly entail the driver. The citation answer key."""
    prior_key: str = ""
    prior_label: str = ""
    company: str = ""
    vendor: str = ""
    product: str = ""
    should_abstain: bool = False

    @property
    def index(self) -> dict[str, Document]:
        return {d.doc_id: d for d in self.documents}


def _prior_pool(prior_key: str) -> tuple[str, ...]:
    return PRIOR_TEMPLATES.get(prior_key, PRIOR_TEMPLATES["value_driven"])


def _place(slots: list[int], rng: random.Random, count: int) -> list[int]:
    """Take `count` positions from the remaining slot pool, without replacement."""
    picked = rng.sample(slots, min(count, len(slots)))
    for p in picked:
        slots.remove(p)
    return sorted(picked)


def _stable_hash(text: str) -> int:
    """CRC32, not builtin `hash`. Python randomizes string hashing per process, so seeding
    a PRNG from `hash()` would silently produce a different corpus on every run - and this
    benchmark's whole reproducibility claim is that an episode is recoverable from its
    spec. Caught by the cross-process determinism test; do not reintroduce."""
    return zlib.crc32(text.encode("utf-8"))


def build_episode(spec: EpisodeSpec) -> Episode:
    rng = random.Random(
        (spec.seed * 1_000_003)
        ^ (_stable_hash(spec.schema_key) << 8)
        ^ _stable_hash(spec.variant)
    )
    schema: LatentSchema = get_schema(spec.schema_key)
    personae = Personae(schema.setting, rng)

    n = spec.volume
    slots = list(range(n))
    bodies: dict[int, tuple[str, str]] = {}  # position -> (doc_type, body)
    minimal: list[int] = []

    signal_source: list = []
    answer_key = schema.key
    answer_desc = schema.description
    should_abstain = False

    if spec.variant == "null":
        # No driver is entailed. The episode contains only ambient activity and
        # decoys, so a well-calibrated agent abstains.
        answer_key = ""
        answer_desc = "No coherent driver is entailed by this evidence."
        should_abstain = True
    elif spec.variant == "revision":
        target_key = schema.revises_to or _fallback_revision(schema, rng)
        target = get_schema(target_key)
        # Early timeline entails the original driver; late timeline overturns it.
        signal_source = [(s, "early") for s in schema.minimal_set()]
        signal_source += [(s, "late") for s in target.minimal_set()]
        answer_key = target.key
        answer_desc = target.description
    else:
        signal_source = [(s, "any") for s in schema.signals]

    # --- entailing signals -------------------------------------------------
    if not should_abstain:
        positions = _place(slots, rng, len(signal_source))
        # Order positions so "early" signals land before "late" ones on the timeline.
        ordered = sorted(
            range(len(signal_source)),
            key=lambda i: {"early": 0, "any": 1, "late": 2}[signal_source[i][1]],
        )
        for slot_i, sig_i in enumerate(ordered):
            signal, phase = signal_source[sig_i]
            pos = positions[slot_i] if slot_i < len(positions) else slots.pop()
            bodies[pos] = (signal.doc_type, personae.body(signal.text, rng))
            if signal.weight >= 1.0 and (spec.variant != "revision" or phase == "late"):
                minimal.append(pos)

    # --- prior-supporting decoys ------------------------------------------
    decoy_count = max(2, n // 25)
    if spec.variant == "anti_prior":
        decoy_count *= 3
    elif spec.variant == "null":
        decoy_count = max(3, n // 20)
    pool = _prior_pool(schema.prior_key)
    for pos in _place(slots, rng, decoy_count):
        bodies[pos] = (rng.choice(("email", "crm_note", "meeting_note")),
                       personae.body(rng.choice(pool), rng))

    # --- ambient noise -----------------------------------------------------
    # Bodies are deduplicated within an episode. Verbatim repeats among noise while
    # signals are always unique would make "appears exactly once" a free shortlist of
    # the entailing documents.
    seen_bodies = {body for _, body in bodies.values()}
    for pos in slots:
        for _ in range(24):
            candidate = personae.body(rng.choice(NOISE_TEMPLATES), rng)
            if candidate not in seen_bodies:
                break
        seen_bodies.add(candidate)
        bodies[pos] = (
            rng.choice(("email", "meeting_note", "ticket", "crm_note", "call_summary")),
            candidate,
        )

    # Timeline ordering: documents are dated in index order so "late" evidence
    # really is late, which is what makes the revision variant meaningful.
    documents = [
        make_document(
            i,
            bodies[i][0],
            bodies[i][1],
            personae,
            rng,
            day=1 + int(i * (240 / max(1, n - 1))),
        )
        for i in range(n)
    ]

    return Episode(
        spec=spec,
        documents=documents,
        answer_key=answer_key,
        answer_description=answer_desc,
        minimal_evidence=[documents[p].doc_id for p in sorted(minimal)],
        prior_key=schema.prior_key,
        prior_label=schema.prior_label,
        company=personae.company,
        vendor=schema.setting.vendor,
        product=schema.setting.product,
        should_abstain=should_abstain,
    )


def _fallback_revision(schema: LatentSchema, rng: random.Random) -> str:
    others = [s.key for s in SCHEMAS if s.key != schema.key]
    return rng.choice(others)


def plan_episodes(
    num_tasks: int,
    seed: int = 0,
    volumes: tuple[int, ...] = VOLUMES,
    variant_mix: tuple[tuple[str, float], ...] = (
        ("normal", 0.45),
        ("anti_prior", 0.25),
        ("revision", 0.15),
        ("null", 0.15),
    ),
) -> list[EpisodeSpec]:
    """A balanced, reproducible plan: every schema, every volume, mixed variants."""
    rng = random.Random(seed)
    variants = [v for v, _ in variant_mix]
    weights = [w for _, w in variant_mix]
    specs: list[EpisodeSpec] = []
    for i in range(num_tasks):
        schema = SCHEMAS[i % len(SCHEMAS)]
        volume = volumes[(i // len(SCHEMAS)) % len(volumes)]
        variant = rng.choices(variants, weights=weights, k=1)[0]
        if variant == "revision" and schema.revises_to is None and rng.random() < 0.5:
            variant = "normal"
        specs.append(
            EpisodeSpec(seed=seed * 10_000 + i, schema_key=schema.key,
                        volume=volume, variant=variant)
        )
    return specs
