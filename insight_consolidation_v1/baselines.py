"""Model-free baselines. Run these before believing any model score.

Letta scored 74.0% on LoCoMo using nothing but a filesystem and `grep`, which is the single
most useful datapoint in the agent-memory literature: most published "memory architectures"
beat a trivial baseline by single digits or not at all. A benchmark that cannot separate
grep from reasoning is not measuring reasoning.

These baselines need no API key and no model. They answer one question: how much of the
minimal entailing set is recoverable by retrieval alone?

    python -m insight_consolidation_v1.baselines
"""

from __future__ import annotations

import random
import statistics
from dataclasses import dataclass

from insight_consolidation_v1.generator import EpisodeSpec, build_episode, plan_episodes
from insight_consolidation_v1.scoring import _DOC_ID, citation_f1
from insight_consolidation_v1.store import EvidenceStore

# The queries a competent analyst would actually try first. If these recover the entailing
# documents, the task is retrieval and the benchmark is void.
PROBE_QUERIES = (
    "why hesitating stalling delay",
    "concern risk worried",
    "budget price discount approval",
    "decision maker owner sponsor",
    "reschedule postponed moved",
    "internal politics team ownership",
    "reference blame failed previous",
    "renewal churn usage adoption",
)


@dataclass
class BaselineResult:
    name: str
    precision: float
    recall: float
    f1: float
    n: int

    def line(self) -> str:
        return (
            f"  {self.name:<22} P={self.precision:.3f}  R={self.recall:.3f}  "
            f"F1={self.f1:.3f}   (n={self.n})"
        )


def _store(spec: EpisodeSpec) -> tuple[EvidenceStore, list[str]]:
    episode = build_episode(spec)
    store = EvidenceStore()
    store.load(episode.documents)
    return store, episode.minimal_evidence


def _cited_from_search(store: EvidenceStore, budget: int) -> tuple[str, ...]:
    seen: list[str] = []
    for query in PROBE_QUERIES:
        for doc_id in _DOC_ID.findall(store.search(query, limit=5)):
            if doc_id not in seen:
                seen.append(doc_id)
    return tuple(seen[:budget])


def run(num_tasks: int = 60, seed: int = 0) -> list[BaselineResult]:
    specs = [s for s in plan_episodes(num_tasks, seed=seed) if s.variant != "null"]
    rng = random.Random(seed)
    rows: dict[str, list[tuple[float, float, float]]] = {
        "grep (top hits)": [],
        "grep (budget 4)": [],
        "random 4 docs": [],
        "cite everything": [],
    }
    for spec in specs:
        store, truth = _store(spec)
        ids = [d.doc_id for d in store.documents]
        rows["grep (top hits)"].append(citation_f1(_cited_from_search(store, 40), truth))
        rows["grep (budget 4)"].append(citation_f1(_cited_from_search(store, 4), truth))
        rows["random 4 docs"].append(
            citation_f1(tuple(rng.sample(ids, min(4, len(ids)))), truth)
        )
        rows["cite everything"].append(citation_f1(tuple(ids), truth))

    results = []
    for name, triples in rows.items():
        results.append(
            BaselineResult(
                name=name,
                precision=statistics.fmean(t[0] for t in triples),
                recall=statistics.fmean(t[1] for t in triples),
                f1=statistics.fmean(t[2] for t in triples),
                n=len(triples),
            )
        )
    return results


def main() -> None:
    print("insight-consolidation-v1 - model-free retrieval baselines")
    print("Citation quality against the minimal entailing set.\n")
    for result in run():
        print(result.line())
    print(
        "\nIf `grep (top hits)` recall approaches 1.0, retrieval alone solves the task and\n"
        "the episodes need regenerating. The gap between these numbers and a model's is the\n"
        "only part of a model score that means anything."
    )


if __name__ == "__main__":
    main()
