"""The verifiers toolset the agent investigates through.

A thin adapter. All the retrieval logic lives in `insight_consolidation_v1.store`, which has
no `verifiers` dependency so the baselines and the reference harness can run on stdlib alone.
This file exists to expose those three operations as verifiers tools and to carry rollout
state.

Why the search is weak is documented in `store.py`, and it is a design decision rather than
an omission.

Backends
--------
There is one backend. It rebuilds the corpus deterministically from the episode spec, so the
environment is self-contained and anyone can run it with no vendor service and no key beyond
the judge model.

`EvidenceToolsetConfig.substrate_base_url` is a placeholder. Nothing reads it. A pluggable
backend that would let the same episodes be served by a real memory system - and so let
memory architectures be compared on identical ground truth - is planned for v0.2 and does
not exist today. Until it does, this environment measures agents, not memory systems.
"""

from __future__ import annotations

import verifiers.v1 as vf

from insight_consolidation_v1.corpus import Document
from insight_consolidation_v1.generator import EpisodeSpec, build_episode
from insight_consolidation_v1.store import MAX_INDEX_WINDOW, MAX_SEARCH_LIMIT, EvidenceStore

__all__ = [
    "EvidenceToolset",
    "EvidenceToolsetConfig",
    "InsightState",
    "MAX_SEARCH_LIMIT",
    "MAX_INDEX_WINDOW",
]


class InsightState(vf.State):
    """Shared rollout state. Read counts feed the investigation-budget metric."""

    searches: int = 0
    reads: int = 0
    read_ids: list[str] = []


class EvidenceToolsetConfig(vf.ToolsetConfig):
    substrate_base_url: str = ""
    """Reserved for the v0.2 pluggable backend. NOT IMPLEMENTED - nothing reads this field,
    and setting it changes nothing. Kept so the config shape does not break when the backend
    lands."""


class EvidenceToolset(vf.Toolset[EvidenceToolsetConfig, InsightState]):
    TOOL_PREFIX = "evidence"

    def __init__(self, config: EvidenceToolsetConfig) -> None:
        super().__init__(config)
        self.store = EvidenceStore()

    async def setup_task(self, task) -> None:
        spec = EpisodeSpec(
            seed=task.episode_seed,
            schema_key=task.schema_key,
            volume=task.volume,
            variant=task.variant,
        )
        self.store.load(build_episode(spec).documents)

    def _load(self, documents: list[Document]) -> None:
        """Load documents directly. Used by tests and by anything driving the store outside
        a rollout."""
        self.store.load(documents)

    # --- tools -------------------------------------------------------------

    @vf.tool
    def search(self, query: str, limit: int = 5) -> str:
        """Search the account's documents by keyword. Returns ranked snippets with ids.

        Lexical matching only - it will not understand a conceptual question like "why are
        they hesitating". Search for the words that would actually appear in a document.
        """
        self.state.searches += 1
        return self.store.search(query, limit)

    @vf.tool
    def read(self, doc_id: str) -> str:
        """Read one document in full by its id, for example doc_00123."""
        text, found = self.store.read(doc_id)
        if found is None:
            return text
        self.state.reads += 1
        if found not in self.state.read_ids:
            self.state.read_ids = [*self.state.read_ids, found]
        return text

    @vf.tool
    def index(self, start_day: int = 0, limit: int = 20) -> str:
        """List documents chronologically from a starting day, headers only, no bodies.

        Use this to understand the shape of the timeline. It never returns document text,
        so it cannot substitute for reading.
        """
        return self.store.index(start_day, limit)


if __name__ == "__main__":
    EvidenceToolset.run()
