"""Transport fault tests use actual spawned processes and the worker wire protocol."""

import importlib.util
import multiprocessing as mp
import os
import subprocess
import sys
import time

import pytest
from edge_sim.batch import BatchRunner
from edge_sim.session import SDKError, Session, start, validate
from edge_sim_models import (
    AdvanceResult,
    NodeSpec,
    RequestSpec,
    RunResult,
    RunSpec,
    ScenarioSpec,
    StageSpec,
    StateView,
    WorkflowSpec,
)


def _protocol_worker(connection, run):
    """A controllable worker for transport failures independent of engine scheduling."""
    mutations = 0
    try:
        if run.run_id == "bad-boot":
            connection.send(("error", {"code": "startup_failed", "message": "test failure"}))
            return
        connection.send(("ready", None))
        while True:
            op, payload = connection.recv()
            if op == "close":
                return
            if op == "advance":
                if run.run_id == "eof":
                    return
                if run.run_id == "hang":
                    time.sleep(60)
                if run.run_id == "slow":
                    time.sleep(0.3)
                if run.run_id == "malformed":
                    connection.send(("wrong", None))
                    continue
                view = StateView()
                result = RunResult(run_id=run.run_id)
                value = AdvanceResult(kind="finished", time_s=0, view=view, result=result)
            elif op == "result":
                if run.run_id == "capture":
                    time.sleep(0.3)
                value = RunResult(run_id=run.run_id)
            elif op == "apply":
                mutations += 1
                connection.send(("error", {"code": "rejected", "message": "do not retry"}))
                continue
            elif op == "inspect":
                value = (mp.get_start_method(), os.getpid(), mutations, payload)
            elif op == "events":
                value = ()
            elif op == "commands":
                value = (mutations,)
            else:
                raise AssertionError(op)
            connection.send(("ok", value))
    except (EOFError, BrokenPipeError):
        pass
    finally:
        connection.close()


@pytest.fixture
def protocol_backend(monkeypatch):
    import edge_sim_simgrid.worker

    monkeypatch.setattr(edge_sim_simgrid.worker, "worker_main", _protocol_worker)


def _run(run_id="test"):
    return RunSpec(
        run_id=run_id,
        scenario=ScenarioSpec(
            nodes=(NodeSpec(id="edge", speed_flops=100, memory_bytes=100, storage_bytes=100),)
        ),
    )


def test_parent_imports_and_validation_never_load_engine():
    code = """
import sys
from edge_sim.session import Session, validate
from edge_sim.batch import BatchRunner
validate({'nodes': [{'id': 'edge', 'speed_flops': 1, 'memory_bytes': 1, 'storage_bytes': 1}]})
assert 'simgrid' not in sys.modules
assert 'edge_sim_simgrid.runtime' not in sys.modules
"""
    subprocess.run([sys.executable, "-c", code], check=True, timeout=10)


def test_spawn_isolation_context_cleanup_and_no_mutation_retry(protocol_backend):
    with start(_run()) as session:
        method, pid, mutations, selection = session.inspect(("request",))
        assert method == "spawn" and pid != os.getpid()
        assert selection == ("request",)
        assert mutations == 0
        with pytest.raises(SDKError, match="rejected"):
            session.apply("decision", ())
        assert session.inspect()[2] == 1
        assert session.commands() == (1,)
        assert session.events() == ()
    assert session.closed
    assert all(child.pid != pid for child in mp.active_children())
    session.close()
    with pytest.raises(SDKError, match="closed"):
        session.advance()


@pytest.mark.parametrize("run_id,code", [("eof", "worker_exited"), ("malformed", "protocol_error")])
def test_transport_failures_are_typed_and_reaped(protocol_backend, run_id, code):
    with Session(_run(run_id)) as session:
        pid = session.pid
        with pytest.raises(SDKError) as caught:
            session.advance()
        assert caught.value.code == code
        assert session.closed
        assert all(child.pid != pid for child in mp.active_children())


def test_timeout_and_busy_do_not_retry_or_leak_process(protocol_backend):
    with BatchRunner() as batch:
        run_id = batch.submit(_run("hang"))
        session = batch.session(run_id)
        session.timeout_s = 0.05
        batch.submit_advance(run_id)
        with pytest.raises(SDKError, match="busy"):
            session.inspect()
        with pytest.raises(SDKError, match="busy"):
            batch.submit_advance(run_id)
        started = time.monotonic()
        reply = batch.recv_ready(timeout=2)[run_id]
        assert isinstance(reply, SDKError) and reply.code == "timeout"
        assert time.monotonic() - started < 2
        assert session.closed and not batch.active_ids


def test_boot_failure_reaps_child(protocol_backend):
    before = {child.pid for child in mp.active_children()}
    with pytest.raises(SDKError, match="startup_failed"):
        Session(_run("bad-boot"))
    assert {child.pid for child in mp.active_children()} == before


def test_settings_defaults_and_explicit_overrides(protocol_backend, monkeypatch):
    monkeypatch.setenv("EDGE_SIM_TIMEOUT_S", "7")
    monkeypatch.setenv("EDGE_SIM_WORKERS", "2")
    with start(_run()) as session:
        assert session.timeout_s == 7
    with BatchRunner() as batch:
        assert batch.timeout_s == 7 and batch.workers == 2
    with BatchRunner(workers=3) as batch:
        assert batch.timeout_s == 7 and batch.workers == 3
    monkeypatch.setenv("EDGE_SIM_TIMEOUT_S", "invalid")
    with Session(_run(), timeout_s=5) as session:
        assert session.timeout_s == 5
    with BatchRunner(timeout_s=5) as batch:
        assert batch.timeout_s == 5 and batch.workers == 2


def test_result_captured_before_queued_worker_starts(protocol_backend):
    with BatchRunner(workers=1) as batch:
        batch.submit(_run("capture"))
        batch.submit(_run("next"))
        assert batch.active_ids == ("capture",)
        assert batch.pending_ids == ("next",)
        batch.submit_advance("capture")
        batch.submit_advance("next")
        assert batch.recv_ready(timeout=0.05) == {}
        assert batch.active_ids == ("capture",)
        assert batch.pending_ids == ("next",)
        with pytest.raises(SDKError, match="not_finished"):
            batch.result("capture")
        replies = batch.recv_ready(timeout=5)
        assert replies["capture"].kind == "finished"
        assert batch.result("capture").run_id == "capture"
        assert batch.active_ids == ("next",)
        assert batch.recv_ready(timeout=5)["next"].kind == "finished"
        assert not batch.active_ids
    assert batch.result("next").run_id == "next"


def test_readiness_not_submission_order_and_duplicate_ids(protocol_backend):
    with BatchRunner(workers=2) as batch:
        batch.submit(_run("slow"))
        batch.submit(_run("fast"))
        with pytest.raises(SDKError, match="duplicate_run_id"):
            batch.submit(_run("slow"))
        batch.submit_advance("slow")
        batch.submit_advance("fast")
        replies = batch.recv_ready(timeout=5)
        assert tuple(replies) == ("fast",)
        assert batch.recv_ready(timeout=5)["slow"].kind == "finished"


def test_barrier_active_snapshot_and_queue_drain(protocol_backend):
    with BatchRunner(workers=2) as batch:
        for index in range(5):
            batch.submit(_run(str(index)))
        assert len(batch.active_ids) == 2
        assert set(batch.advance_all()) == {"0", "1"}
        assert set(batch.active_ids) == {"2", "3"}
        results = batch.run()
        assert set(results) == {str(index) for index in range(5)}
        assert all(key == result.run_id for key, result in results.items())
        assert not batch.active_ids and not batch.pending_ids


@pytest.mark.integration
@pytest.mark.skipif(importlib.util.find_spec("simgrid") is None, reason="SimGrid not installed")
def test_real_backend_sessions_are_repeatable_and_independent():
    scenario = validate(
        ScenarioSpec(
            nodes=_run().scenario.nodes,
            workflows=(
                WorkflowSpec(
                    id="cpu",
                    stages=(StageSpec(id="compute", flops=200, memory_bytes=10),),
                ),
            ),
            requests=(RequestSpec(id="request", workflow="cpu", receiver="edge"),),
        ).model_dump()
    )
    with Session(RunSpec(run_id="one", scenario=scenario)) as first:
        with Session(RunSpec(run_id="two", scenario=scenario)) as second:
            assert first.pid != second.pid
            assert first.advance().kind == second.advance().kind == "finished"
            assert first.result().run_id == "one"
            assert second.result().run_id == "two"
            assert first.result().now_s == pytest.approx(2)
            assert second.result().now_s == pytest.approx(2)
            assert first.result().state.requests[0].status == "SUCCEEDED"
            assert first.result().state == second.result().state
    assert "simgrid" not in sys.modules
