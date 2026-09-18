"""Policies consume immutable decisions and return validated domain commands."""

import hashlib
from collections.abc import Sequence
from typing import Protocol

from edge_sim_models import DecisionCommand, DecisionRequest, Place, Reject


class Policy(Protocol):
    def decide(self, decision: DecisionRequest) -> Sequence[DecisionCommand]: ...


class FirstFit:
    def decide(self, decision: DecisionRequest) -> tuple[DecisionCommand, ...]:
        return tuple(
            Place(request_id=c.request_id, stage_id=c.stage_id, node_id=c.nodes[0])
            if c.nodes
            else Reject(request_id=c.request_id, reason="no_feasible_node")
            for c in decision.candidates
        )


def derive_seed(seed: int, run_id: str, stream: str) -> int:
    """Stable across process counts and Python hash randomization."""
    payload = f"{seed}\0{run_id}\0{stream}".encode()
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "big")
