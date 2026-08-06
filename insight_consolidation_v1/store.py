"""The evidence store, as plain Python.

Deliberately a *weak* retrieval surface: lexical TF-IDF search over document bodies, a
read-one-document call, and a bounded chronological index. That weakness is the point. If
search were semantic and strong it would hand back the entailing documents for a query like
"why are they really stalling", and we would be measuring the retriever rather than the
agent's consolidation policy.

At volume 5000 the agent cannot read everything inside a sane turn budget, so it has to
decide what to look at. That decision is the skill under test and the reason this is an RL
environment rather than a QA set.

This module has no `verifiers` dependency, on purpose. The baselines and the reference
harness (`run_eval.py`) claim to run on stdlib alone, and routing them through the rollout
stack's toolset class would have made that claim false - `verifiers.v1` imports `fcntl` and
does not import at all on Windows. `servers/tool.py` is a thin adapter that exposes these
same three operations as verifiers tools.

Backends
--------
There is one backend: this one. It is fed documents rebuilt deterministically from the
episode spec, so the environment is self-contained and anyone can run it with no vendor
service and no key beyond the judge model. See `servers/tool.py` for the state of the
planned pluggable backend, which does not exist.
"""

from __future__ import annotations

import math
from collections import Counter

from insight_consolidation_v1.corpus import Document
from insight_consolidation_v1.scoring import tokenize

MAX_SEARCH_LIMIT = 10
MAX_INDEX_WINDOW = 40


class EvidenceStore:
    """Lexical TF-IDF search, single-document read, and a headers-only timeline index."""

    def __init__(self) -> None:
        self._docs: list[Document] = []
        self._by_id: dict[str, Document] = {}
        self._idf: dict[str, float] = {}
        self._postings: list[Counter] = []

    def load(self, documents: list[Document]) -> None:
        self._docs = documents
        self._by_id = {d.doc_id: d for d in documents}
        self._postings = [Counter(tokenize(f"{d.subject} {d.body}")) for d in documents]
        df: Counter = Counter()
        for counts in self._postings:
            df.update(counts.keys())
        n = max(1, len(documents))
        self._idf = {term: math.log(1 + n / (1 + count)) for term, count in df.items()}

    @property
    def documents(self) -> list[Document]:
        return self._docs

    def search(self, query: str, limit: int = 5) -> str:
        limit = max(1, min(int(limit or 5), MAX_SEARCH_LIMIT))
        terms = tokenize(query)
        if not terms:
            return "No query terms."
        scored: list[tuple[float, int]] = []
        for i, counts in enumerate(self._postings):
            total = sum(counts.values()) or 1
            score = sum(
                (counts[t] / total) * self._idf.get(t, 0.0) for t in terms if t in counts
            )
            if score > 0:
                scored.append((score, i))
        if not scored:
            return "No matches."
        scored.sort(key=lambda pair: (-pair[0], pair[1]))
        lines = [self._docs[i].snippet() for _, i in scored[:limit]]
        return f"{len(scored)} matches, showing {len(lines)}:\n" + "\n".join(lines)

    def read(self, doc_id: str) -> tuple[str, str | None]:
        """Render one document. Returns (text, doc_id_or_None) so the caller can record the
        read against rollout state without this module knowing what rollout state is."""
        doc = self._by_id.get((doc_id or "").strip())
        if doc is None:
            return (f"No document {doc_id!r}.", None)
        return (doc.render(), doc.doc_id)

    def index(self, start_day: int = 0, limit: int = 20) -> str:
        limit = max(1, min(int(limit or 20), MAX_INDEX_WINDOW))
        window = [d for d in self._docs if d.day >= int(start_day or 0)][:limit]
        if not window:
            return "No documents in that window."
        header = f"{len(self._docs)} documents total, days 1-{self._docs[-1].day}."
        return header + "\n" + "\n".join(d.render(full=False) for d in window)


__all__ = ["EvidenceStore", "MAX_SEARCH_LIMIT", "MAX_INDEX_WINDOW"]
