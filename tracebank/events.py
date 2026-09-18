"""Core data types: Event and Trace (Section 3.2 of the paper)."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence


@dataclass(frozen=True)
class Event:
    """e = (act, ts, pos, attr)."""
    act: str
    pos: int
    ts: Optional[float] = None
    attr: Dict[str, Any] = field(default_factory=dict, compare=False)


@dataclass
class Trace:
    """sigma = <e_1, ..., e_n> plus run-level metadata."""
    run_id: str
    events: List[Event]
    reward: float = 0.0
    task_id: Optional[int] = None
    trial: Optional[int] = None
    domain: Optional[str] = None
    model: Optional[str] = None
    instruction: Optional[str] = None
    meta: Dict[str, Any] = field(default_factory=dict)

    @property
    def labels(self) -> List[str]:
        return [e.act for e in self.events]

    @property
    def success(self) -> bool:
        return self.reward >= 1.0 - 1e-6

    def __len__(self) -> int:
        return len(self.events)

    def prefix(self, i: int) -> "Trace":
        return Trace(
            run_id=self.run_id, events=self.events[:i], reward=self.reward,
            task_id=self.task_id, trial=self.trial, domain=self.domain,
            model=self.model, instruction=self.instruction, meta=dict(self.meta),
        )

    def to_json(self) -> Dict[str, Any]:
        return {
            "run_id": self.run_id, "task_id": self.task_id, "trial": self.trial,
            "reward": self.reward, "domain": self.domain, "model": self.model,
            "instruction": self.instruction, "meta": self.meta,
            "events": [
                {"act": e.act, "pos": e.pos, "ts": e.ts, "attr": e.attr}
                for e in self.events
            ],
        }

    @staticmethod
    def from_json(d: Dict[str, Any]) -> "Trace":
        return Trace(
            run_id=d["run_id"],
            events=[Event(act=e["act"], pos=e["pos"], ts=e.get("ts"), attr=e.get("attr", {}))
                    for e in d["events"]],
            reward=d.get("reward", 0.0), task_id=d.get("task_id"), trial=d.get("trial"),
            domain=d.get("domain"), model=d.get("model"),
            instruction=d.get("instruction"), meta=d.get("meta", {}),
        )


def alphabet(traces: Sequence[Trace]) -> List[str]:
    """The finite activity-label set A observed across traces."""
    return sorted({e.act for t in traces for e in t.events})
