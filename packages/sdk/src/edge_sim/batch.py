"""Bounded process scheduling using pipe readiness, without response-waiting threads."""

from __future__ import annotations

import math
from collections import deque
from multiprocessing.connection import wait
from time import monotonic
from typing import TYPE_CHECKING, Literal

from .session import SDKError, Session
from .settings import Settings

if TYPE_CHECKING:
    from edge_sim_models import AdvanceResult, RunResult, RunSpec, WindowControl, WindowResult


class BatchRunner:
    """Queue runs behind at most ``workers`` live sessions.

    ``submit_advance`` accepts one request per run, including queued runs.
    ``recv_ready`` returns a mapping of run IDs to advances or SDKError objects.
    A finished advance is delivered only after its result has been cached.
    The class, like Session, is intended for use by a single calling thread.
    """

    def __init__(self, workers: int | None = None, timeout_s: float | None = None):
        if workers is None or timeout_s is None:
            overrides = {}
            if workers is not None:
                overrides["workers"] = workers
            if timeout_s is not None:
                overrides["timeout_s"] = timeout_s
            settings = Settings(**overrides)
            workers = settings.workers if workers is None else workers
            timeout_s = settings.timeout_s if timeout_s is None else timeout_s
        if isinstance(workers, bool) or not isinstance(workers, int) or workers < 1:
            raise ValueError("workers must be a positive integer")
        if not math.isfinite(timeout_s) or timeout_s <= 0:
            raise ValueError("timeout_s must be finite and positive")
        self.workers = workers
        self.timeout_s = timeout_s
        self._closed = False
        self._runs: dict[str, RunSpec] = {}
        self._queue: deque[str] = deque()
        self._active: dict[str, Session] = {}
        self._requested: dict[str, float | None] = {}
        self._results: dict[str, RunResult] = {}
        self._errors: dict[str, SDKError] = {}
        self._ready: dict[str, AdvanceResult | WindowResult | SDKError] = {}

    def _check_open(self) -> None:
        if self._closed:
            raise SDKError("closed", "Batch runner is closed")

    @property
    def active_ids(self) -> tuple[str, ...]:
        return tuple(self._active)

    @property
    def pending_ids(self) -> tuple[str, ...]:
        return tuple(self._queue)

    def submit(self, run: RunSpec) -> str:
        self._check_open()
        run_id = run.run_id
        if run_id in self._runs:
            raise SDKError("duplicate_run_id", f"Run ID already submitted: {run_id}")
        self._runs[run_id] = run
        self._queue.append(run_id)
        self._fill()
        return run_id

    def _record_error(self, run_id: str, error: SDKError) -> None:
        self._errors[run_id] = error
        self._ready[run_id] = error
        self._requested.pop(run_id, None)
        session = self._active.pop(run_id, None)
        if session is not None:
            session.close()

    def _fill(self) -> None:
        while self._queue and len(self._active) < self.workers:
            run_id = self._queue.popleft()
            try:
                session = Session(self._runs[run_id], self.timeout_s)
                self._active[run_id] = session
                if run_id in self._requested:
                    session._send("advance", self._requested[run_id])
            except SDKError as error:
                self._record_error(run_id, error)

    def session(self, run_id: str) -> Session:
        """Access an active run for inspect/apply; outstanding RPCs enforce backpressure."""
        self._check_open()
        if run_id in self._errors:
            raise self._errors[run_id]
        if run_id not in self._runs:
            raise KeyError(run_id)
        if run_id not in self._active:
            raise SDKError("inactive_run", f"Run is queued or finished: {run_id}")
        return self._active[run_id]

    def submit_advance(self, run_id: str, until_time: float | None = None) -> None:
        self._check_open()
        if run_id not in self._runs:
            raise KeyError(run_id)
        if run_id in self._requested or run_id in self._ready:
            raise SDKError("busy", f"Run has an outstanding or unread response: {run_id}")
        if run_id in self._errors:
            raise self._errors[run_id]
        if run_id in self._results:
            raise SDKError("finished", f"Run has finished: {run_id}")
        if run_id in self._active:
            try:
                self._active[run_id]._send("advance", until_time)
            except SDKError as error:
                if error.code == "busy":
                    raise
                self._record_error(run_id, error)
                self._fill()
                return
        self._requested[run_id] = until_time

    def submit_window(
        self,
        run_id: str,
        until_s: float,
        control: WindowControl,
        *,
        scope: Literal["full", "scheduling"] = "full",
    ) -> None:
        """Submit a window on an active session; collect with recv_ready()."""
        self._check_open()
        if run_id in self._requested or run_id in self._ready:
            raise SDKError("busy", "Run has an outstanding or unread response")
        session = self.session(run_id)
        session._send("advance_window", (until_s, control, scope))
        self._requested[run_id] = until_s

    def recv_ready(
        self, timeout: float | None = None
    ) -> dict[str, AdvanceResult | WindowResult | SDKError]:
        """Collect available replies, waiting at most timeout for pipe readiness.

        ``None`` waits until a reply or an RPC deadline; zero polls. Worker boot
        and bounded teardown when promoting queued runs may add lifecycle time.
        Transport errors terminate their run; worker errors also retire the run.
        """
        self._check_open()
        if timeout is not None and (not math.isfinite(timeout) or timeout < 0):
            raise ValueError("timeout must be finite and nonnegative, or None")
        deadline = math.inf if timeout is None else monotonic() + timeout
        while not self._ready:
            waiting = {
                session._conn: run_id
                for run_id, session in self._active.items()
                if run_id in self._requested
            }
            if not waiting:
                break
            now = monotonic()
            rpc_deadline = min(self._active[r]._deadline for r in waiting.values())
            ready = set(wait(list(waiting), max(0, min(deadline, rpc_deadline) - now)))
            for conn, run_id in waiting.items():
                session = self._active[run_id]
                if conn not in ready and monotonic() < session._deadline:
                    continue
                try:
                    response = session._receive()
                    if response.kind == "finished" and hasattr(response, "result"):
                        self._results[run_id] = response.result
                        self._ready[run_id] = response
                        self._requested.pop(run_id)
                        session.close()
                        del self._active[run_id]
                    else:
                        self._ready[run_id] = response
                        self._requested.pop(run_id)
                except SDKError as error:
                    self._record_error(run_id, error)
            self._fill()
            if monotonic() >= deadline:
                break
        responses, self._ready = self._ready, {}
        return responses

    def advance_all(
        self, until_time: float | None = None
    ) -> dict[str, AdvanceResult | WindowResult | SDKError]:
        """Barrier over the active snapshot, excluding runs queued for a worker slot.

        Existing asynchronous work must first be collected with recv_ready.
        Errors are values, so one failed run does not hide other barrier replies.
        """
        self._check_open()
        if self._requested or self._ready:
            raise SDKError("busy", "Collect asynchronous responses before starting a barrier")
        targets = self.active_ids
        for run_id in targets:
            self.submit_advance(run_id, until_time)
        responses: dict[str, AdvanceResult | WindowResult | SDKError] = {}
        while any(run_id not in responses for run_id in targets):
            responses.update(self.recv_ready())
        return responses

    def result(self, run_id: str) -> RunResult:
        """Read a captured terminal result, including after runner.close()."""
        if run_id not in self._runs:
            raise KeyError(run_id)
        if run_id in self._errors:
            raise self._errors[run_id]
        if run_id not in self._results:
            raise SDKError("not_finished", f"No terminal result captured for {run_id}")
        return self._results[run_id]

    def cancel(self, run_id: str) -> None:
        """Cancel an active or queued run; other runs continue using the released slot."""
        self._check_open()
        if run_id not in self._runs:
            raise KeyError(run_id)
        if run_id in self._results or run_id in self._errors:
            return
        if run_id in self._queue:
            self._queue.remove(run_id)
        self._record_error(run_id, SDKError("cancelled", f"Run cancelled: {run_id}"))
        self._fill()

    def discard(self, run_id: str) -> None:
        """Release a collected run and its metadata, for bounded episode recycling."""
        self._check_open()
        if run_id in self._requested:
            raise SDKError("busy", "Collect the outstanding response before discarding")
        session = self._active.pop(run_id, None)
        if session is not None:
            session.close()
        if run_id in self._queue:
            self._queue.remove(run_id)
        for mapping in (self._runs, self._results, self._errors, self._ready):
            mapping.pop(run_id, None)
        self._fill()

    def run(self) -> dict[str, RunResult]:
        """Drive all submitted built-in-policy runs to completion, draining the queue.

        External decisions require explicit submit_advance/recv_ready/apply calls.
        No decision is implicitly answered or mutation retried.
        """
        self._check_open()
        if self._requested or self._ready:
            raise SDKError("busy", "Collect asynchronous responses before calling run")
        while self._active:
            for response in self.advance_all().values():
                if isinstance(response, SDKError):
                    raise response
                if response.kind != "finished":
                    raise SDKError("decision_required", "Run paused; use the explicit stepping API")
        if self._errors:
            raise next(iter(self._errors.values()))
        return dict(self._results)

    def close(self) -> None:
        if not self._closed:
            self._closed = True
            for session in self._active.values():
                session.close()
            self._active.clear()
            self._queue.clear()
            self._requested.clear()

    def __enter__(self) -> BatchRunner:
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()
