"""Typed extension contracts. Plugins are trusted local Python factories."""

from typing import Protocol

from . import DecisionCommand, DecisionRequest, RequestSpec, StageState, StateView


class AdmissionPolicy(Protocol):
    def admit(self, request: RequestSpec, view: StateView) -> bool: ...


class PlacementPolicy(Protocol):
    def decide(self, decision: DecisionRequest) -> tuple[DecisionCommand, ...]: ...


class ReplicaPolicy(Protocol):
    def select(self, artifact_id: str, destination: str, candidates: tuple[str, ...]) -> str: ...


class SchedulingPolicy(Protocol):
    def order(self, stages: tuple[StageState, ...]) -> tuple[str, ...]:
        """Return a permutation of request_id/stage_id keys."""
        ...


class PreemptionPolicy(Protocol):
    def decide(self, view: StateView) -> tuple[DecisionCommand, ...]:
        """Return Suspend/Resume commands at a new domain event boundary."""
        ...


class CachePolicy(Protocol):
    def order(self, node_id: str, evictable_ids: tuple[str, ...]) -> tuple[str, ...]:
        """Reorder the already LRU-sorted, unpinned candidates."""
        ...
