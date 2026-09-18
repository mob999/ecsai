"""One spawned simulation process per session; no engine objects cross the pipe."""

from __future__ import annotations

import math
import multiprocessing as mp
from multiprocessing.connection import wait
from time import monotonic
from typing import TYPE_CHECKING, Any

from .settings import Settings

if TYPE_CHECKING:
    from edge_sim_models import (
        AdvanceResult,
        DecisionCommand,
        DomainEvent,
        RunResult,
        RunSpec,
        ScenarioSpec,
        StateView,
    )


class SDKError(RuntimeError):
    """A worker or transport failure with a stable machine-readable code."""

    def __init__(self, code: str, message: str):
        self.code = code
        self.message = message
        super().__init__(f"{code}: {message}")


class Session:
    """Synchronous RPC facade. A failed transport closes the session without retry."""

    def __init__(self, run: RunSpec, timeout_s: float | None = None):
        if timeout_s is None:
            timeout_s = Settings().timeout_s
        if not math.isfinite(timeout_s) or timeout_s <= 0:
            raise ValueError("timeout_s must be finite and positive")
        # The worker module itself must be engine-free; engine imports belong in its target.
        from edge_sim_simgrid.worker import worker_main

        self.run_id = run.run_id
        self.timeout_s = timeout_s
        self._closed = False
        self._pending: str | None = "boot"
        self._deadline = monotonic() + timeout_s
        context = mp.get_context("spawn")
        self._conn, child = context.Pipe()
        self._process = context.Process(target=worker_main, args=(child, run))
        try:
            self._process.start()
            child.close()
            self._receive()
        except BaseException:
            child.close()
            self.close()
            raise

    @property
    def pid(self) -> int | None:
        return self._process.pid

    @property
    def closed(self) -> bool:
        return self._closed

    def _fail(self, code: str, message: str) -> None:
        self.close()
        raise SDKError(code, message)

    def _send(self, op: str, payload: Any = None) -> None:
        if self._closed:
            raise SDKError("closed", "Session is closed")
        if self._pending is not None:
            raise SDKError("busy", "Only one outstanding RPC is allowed per session")
        self._pending = op
        self._deadline = monotonic() + self.timeout_s
        try:
            self._conn.send((op, payload))
        except (OSError, EOFError) as exc:
            self._fail("worker_exited", str(exc))
        except BaseException:
            # Serialization failures happen before a request is transmitted.
            self._pending = None
            raise

    def _receive(self) -> Any:
        if self._closed:
            raise SDKError("closed", "Session is closed")
        if self._pending is None:
            raise SDKError("no_pending_rpc", "No response is outstanding")
        try:
            if not wait([self._conn], max(0, self._deadline - monotonic())):
                self._fail("timeout", f"Worker timed out during {self._pending}")
            response = self._conn.recv()
        except (EOFError, OSError) as exc:
            self._fail("worker_exited", f"Worker disconnected: {exc}")
        op, self._pending = self._pending, None
        if not isinstance(response, tuple) or len(response) != 2:
            self._fail("protocol_error", "Expected a two-item worker response")
        status, value = response
        if status == "error":
            if not isinstance(value, dict) or not all(
                isinstance(value.get(key), str) for key in ("code", "message")
            ):
                self._fail("protocol_error", "Malformed worker error")
            raise SDKError(value["code"], value["message"])
        if status != ("ready" if op == "boot" else "ok"):
            self._fail("protocol_error", f"Unexpected response status: {status!r}")
        if op == "boot" and value is not None:
            self._fail("protocol_error", "Unexpected boot payload")
        return value

    def _rpc(self, op: str, payload: Any = None) -> Any:
        self._send(op, payload)
        return self._receive()

    def advance(self, until_time: float | None = None) -> AdvanceResult:
        return self._rpc("advance", until_time)

    def apply(self, decision_id: str, commands: tuple[DecisionCommand, ...]) -> None:
        self._rpc("apply", (decision_id, tuple(commands)))

    def inspect(self, selection: tuple[str, ...] | None = None) -> StateView:
        return self._rpc("inspect", None if selection is None else tuple(selection))

    def result(self) -> RunResult:
        return self._rpc("result")

    def events(self) -> tuple[DomainEvent, ...]:
        return self._rpc("events")

    def commands(self) -> tuple:
        return self._rpc("commands")

    def close(self) -> None:
        """Bounded, idempotent teardown, including when a worker has hung or died."""
        if self._closed:
            return
        self._closed = True
        try:
            if self._pending is None and self._process.pid is not None:
                try:
                    self._conn.send(("close", None))
                except (EOFError, OSError):
                    pass
        finally:
            self._conn.close()
            if self._process.pid is not None:
                self._process.join(0.2)
                if self._process.is_alive():
                    self._process.terminate()
                    self._process.join(0.5)
                if self._process.is_alive():
                    self._process.kill()
                    self._process.join(0.5)
            self._pending = None

    def __enter__(self) -> Session:
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()


def start(run: RunSpec, timeout_s: float | None = None) -> Session:
    return Session(run, timeout_s)


def validate(scenario: ScenarioSpec | dict) -> ScenarioSpec:
    """Validate domain data in the parent without importing the engine backend."""
    from edge_sim_models import ScenarioSpec

    return ScenarioSpec.model_validate(scenario)
