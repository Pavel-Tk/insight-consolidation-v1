"""insight-consolidation-v1.

The taskset and judge are imported lazily. They pull in `verifiers.v1`, which is
Linux/macOS-only (it imports `fcntl`), and the generator, store, baselines and reference
harness deliberately do not need it. Importing this package should not decide which platform
you can run the baselines on.
"""

from typing import TYPE_CHECKING, Any

from insight_consolidation_v1.store import EvidenceStore

if TYPE_CHECKING:  # pragma: no cover
    from insight_consolidation_v1.judge import GroundedInsightJudge
    from insight_consolidation_v1.taskset import (
        InsightProceduralTaskset,
        InsightTask,
        InsightTaskset,
    )

_LAZY = {
    "GroundedInsightJudge": "insight_consolidation_v1.judge",
    "InsightTask": "insight_consolidation_v1.taskset",
    "InsightTaskset": "insight_consolidation_v1.taskset",
    "InsightProceduralTaskset": "insight_consolidation_v1.taskset",
}


def __getattr__(name: str) -> Any:
    module = _LAZY.get(name)
    if module is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    import importlib

    return getattr(importlib.import_module(module), name)


def __dir__() -> list[str]:
    return sorted([*_LAZY, "EvidenceStore"])


__all__ = [
    "EvidenceStore",
    "InsightTaskset",
    "InsightProceduralTaskset",
    "InsightTask",
    "GroundedInsightJudge",
]
