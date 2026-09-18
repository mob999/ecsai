"""Capacity accounting. SimGrid alone owns computational and network progress."""

from pydantic import BaseModel, ConfigDict, Field


class MutableState(BaseModel):
    model_config = ConfigDict(validate_assignment=True, extra="forbid")


class ResourceLedger(MutableState):
    memory_capacity: int
    storage_capacity: int
    memory: dict[str, int] = Field(default_factory=dict)
    storage: dict[str, int] = Field(default_factory=dict)

    @property
    def memory_used(self) -> int:
        return sum(self.memory.values())

    @property
    def storage_used(self) -> int:
        return sum(self.storage.values())

    def reserve_memory(self, owner: str, amount: int) -> bool:
        if owner in self.memory:
            return True
        if self.memory_used + amount > self.memory_capacity:
            return False
        self.memory[owner] = amount
        return True

    def reserve_storage(self, owner: str, amount: int) -> bool:
        if owner in self.storage:
            return True
        if self.storage_used + amount > self.storage_capacity:
            return False
        self.storage[owner] = amount
        return True


class ReplicaState(MutableState):
    artifact_id: str
    node_id: str
    size_bytes: int
    cacheable: bool
    available: bool = True
    last_access: float = 0.0
    pinned: bool = False


class TransferState(MutableState):
    artifact_id: str
    src: str
    dst: str
    size_bytes: int
    started_s: float
    waiters: set[str] = Field(default_factory=set)


class StageState(MutableState):
    request_id: str
    stage_id: str
    status: str = "WAITING_DEPENDENCIES"
    node_id: str | None = None
    entered_s: float = 0.0
    started_s: float | None = None
    finished_s: float | None = None
    reason: str | None = None
    durations: dict[str, float] = Field(default_factory=dict)
    ready_after_s: float = 0.0

    @property
    def key(self) -> str:
        return f"{self.request_id}/{self.stage_id}"

    def transition(self, status: str, now: float) -> None:
        self.durations[self.status] = self.durations.get(self.status, 0) + now - self.entered_s
        self.status = status
        self.entered_s = now
        if status == "RUNNING" and self.started_s is None:
            self.started_s = now
        if status in {"SUCCEEDED", "CANCELLED", "FAILED"}:
            self.finished_s = now


class RequestState(MutableState):
    request_id: str
    status: str = "PENDING"
    finished_s: float | None = None
    reason: str | None = None


class CommandError(ValueError):
    """An invalid command batch; no changes have been applied."""

    def __init__(self, code: str, message: str):
        self.code = code
        super().__init__(message)
