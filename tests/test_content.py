"""Analytical content-service contracts against native SimGrid."""

import pytest
from edge_sim import SDKError, start
from edge_sim_models import (
    ArtifactSpec,
    CacheNodeSpec,
    ContentRequest,
    ContentServiceSpec,
    LinkSpec,
    NodeSpec,
    RouteSpec,
    RunSpec,
    ScenarioSpec,
    SchedulerControl,
    SchedulerSpec,
    TransferPoolSpec,
    WindowControl,
)

pytestmark = pytest.mark.integration


def scenario(
    requests,
    *,
    caches=1,
    active=8,
    waiting=50,
    scheduler_waiting=100,
    size=100,
    bandwidth=100,
    shared=False,
    storage=10000,
):
    nodes = tuple(
        NodeSpec(
            id=n,
            speed_flops=1,
            memory_bytes=0,
            storage_bytes=max(storage, len({r[1] for r in requests}) * size)
            if n == "origin"
            else storage,
        )
        for n in ["origin", "user"] + [f"c{i}" for i in range(caches)]
    )
    links, routes, cache_specs = [], [], []
    for i in range(caches):
        for kind, src, dst in [("b", "origin", f"c{i}"), ("d", f"c{i}", "user")]:
            lid = kind if shared else f"{kind}{i}"
            if not any(link.id == lid for link in links):
                links.append(LinkSpec(id=lid, bandwidth_bytes_s=bandwidth, latency_s=0))
            routes.append(RouteSpec(src=src, dst=dst, links=(lid,)))
        cache_specs.append(
            CacheNodeSpec(
                node_id=f"c{i}",
                cluster_id=f"s{i}",
                backhaul_link="b" if shared else f"b{i}",
                delivery_link="d" if shared else f"d{i}",
                backhaul=TransferPoolSpec(max_active=active, max_waiting=waiting),
                delivery=TransferPoolSpec(max_active=active, max_waiting=waiting),
            )
        )
    artifacts = tuple(
        ArtifactSpec(id=aid, size_bytes=size, locations=("origin",))
        for aid in sorted({r[1] for r in requests})
    )
    return RunSpec(
        trace=True,
        control_mode="window",
        scenario=ScenarioSpec(
            nodes=nodes, links=tuple(links), routes=tuple(routes), artifacts=artifacts
        ),
        content=ContentServiceSpec(
            origin="origin",
            caches=tuple(cache_specs),
            schedulers=tuple(
                SchedulerSpec(id=f"s{i}", max_waiting=scheduler_waiting, service_s=0.001)
                for i in range(caches)
            ),
            requests=tuple(
                ContentRequest(
                    id=rid,
                    artifact_id=aid,
                    cluster_id=f"s{cluster}",
                    receiver="user",
                    arrival_s=arrival,
                    deadline_s=deadline,
                )
                for rid, aid, cluster, arrival, deadline in requests
            ),
        ),
    )


def control(run, bias=-5):
    return WindowControl(
        schedulers=tuple(
            SchedulerControl(cluster_id=s.id, weights=(0, 0, 0, bias))
            for s in run.content.schedulers
        )
    )


def test_independent_links_and_shared_competition():
    arrivals = [("r0", "a", 0, 0, 20), ("r1", "a", 1, 0, 20)]
    ends = []
    for shared in (False, True):
        run = scenario(arrivals, caches=2, shared=shared)
        with start(run) as session:
            out = session.advance_window(10, control(run))
            assert out.view.completed == 2
            ends.append(max(r.completed_s for r in out.view.requests))
            assert sum(link.bytes_sent for link in out.view.links) == pytest.approx(400)
    assert ends[0] == pytest.approx(2.001)
    assert ends[1] == pytest.approx(4.001)


def test_coalescing_and_unicast_and_cache_hit():
    run = scenario([("r0", "a", 0, 0, 20), ("r1", "a", 0, 0.01, 20), ("r2", "a", 0, 4, 20)])
    with start(run) as session:
        out = session.advance_window(8, control(run))
        assert out.view.completed == 3
        assert out.view.cache_hits == 1
        counters = {link.link_id: link.bytes_sent for link in out.view.links}
        assert counters == pytest.approx({"b0": 100, "d0": 300})


def test_fifo_concurrent_service_and_overflow():
    run = scenario([(f"r{i}", f"a{i}", 0, i * 0.01, 20) for i in range(5)], active=2, waiting=1)
    with start(run) as session:
        out = session.advance_window(0.1, control(run))
        backhaul = next(p for p in out.view.pools if p.kind == "backhaul")
        assert (backhaul.active, backhaul.waiting) == (2, 1)
        assert out.view.rejected == 2
        assert out.view.overflows == 2
        out = session.advance_window(10, control(run))
        assert out.view.completed == 3
        events = session.events()
        started = [e.entity_id for e in events if e.kind == "transfer_started"]
        assert (
            started.index("transfer-1") < started.index("transfer-2") < started.index("transfer-3")
        )


def test_timeout_cancels_only_last_coalesced_waiter():
    run = scenario([("r0", "a", 0, 0, 0.5), ("r1", "a", 0, 0.01, 10)])
    with start(run) as session:
        out = session.advance_window(0.6, control(run))
        assert out.view.timed_out == 1
        assert len(out.view.transfers) == 1
        assert out.view.transfers[0].request_ids == ("r1",)
        out = session.advance_window(4, control(run))
        assert out.view.completed == 1
        assert sum(link.bytes_sent for link in out.view.links) == pytest.approx(200)
    run = scenario([("r0", "a", 0, 0, 0.5)])
    with start(run) as session:
        out = session.advance_window(0.6, control(run))
        assert not out.view.transfers
        assert any(e.kind == "transfer_cancelled" for e in session.events())


def test_exact_deadline_and_arrival_boundary():
    run = scenario([("r0", "a", 0, 0, 2.001), ("r1", "a", 0, 3, 10)], caches=2)
    with start(run) as session:
        out = session.advance_window(3, control(run))
        assert out.view.completed == 1 and out.view.arrived == 1
        out = session.advance_window(6, control(run, 5))
        assert out.view.forwarded == 1
        assert out.view.completed == 2
        assert next(r for r in out.view.requests if r.request_id == "r1").origin_cluster == "s0"


def test_scheduler_waiting_capacity_excludes_service_and_forward_once():
    run = scenario([(f"r{i}", "a", 0, 0, 20) for i in range(3)], scheduler_waiting=1, caches=2)
    with start(run) as session:
        out = session.advance_window(0.0005, control(run, 5))
        assert out.view.rejected == 1
        assert len(out.view.schedulers[0].waiting) == 1
        assert out.view.schedulers[0].active_request is not None
        out = session.advance_window(5, control(run, 5))
        assert out.view.forwarded == 2
        assert out.view.completed == 2


def test_invalid_controls_recover_and_modes_do_not_mix():
    run = scenario([("r", "a", 0, 0, 10)])
    with start(run) as session:
        with pytest.raises(SDKError, match="wrong_mode"):
            session.advance()
        with pytest.raises(SDKError, match="invalid_control"):
            session.advance_window(1, {"schedulers": []})
        assert session.advance_window(3, control(run)).view.completed == 1


def test_lru_capacity_and_forward_target_overflow():
    run = scenario(
        [("r0", "a", 0, 0, 20), ("r1", "b", 0, 3, 20), ("r2", "a", 0, 6, 20)], storage=100
    )
    # Origin must retain the complete catalog independently of cache capacity.
    nodes = tuple(
        n.model_copy(update={"storage_bytes": 200}) if n.id == "origin" else n
        for n in run.scenario.nodes
    )
    run = run.model_copy(update={"scenario": run.scenario.model_copy(update={"nodes": nodes})})
    with start(run) as session:
        out = session.advance_window(10, control(run))
        assert out.view.completed == 3 and out.view.cache_hits == 0
        assert len([e for e in session.events() if e.kind == "cache_evict"]) == 2
    run = scenario(
        [("r0", "a", 0, 0, 10), ("r1", "a", 1, 0.0005, 10)], caches=2, scheduler_waiting=0
    )
    with start(run) as session:
        out = session.advance_window(5, control(run, 5))
        assert out.view.overflows == 1
        assert out.view.forwarded == 2
        assert out.view.completed == 1


def test_simultaneous_scheduler_completions_release_capacity_first():
    run = scenario([("r0", "a", 0, 0, 10), ("r1", "a", 1, 0, 10)], caches=2, scheduler_waiting=0)
    with start(run) as session:
        out = session.advance_window(5, control(run, 5))
        assert out.view.completed == 2
        assert out.view.overflows == 0


def test_partial_bytes_at_boundaries_and_cancellation_are_not_double_counted():
    run = scenario([("r0", "a", 0, 0, 0.5)])
    with start(run) as session:
        first = session.advance_window(0.25, control(run))
        assert first.view.links[0].bytes_sent == pytest.approx(24.9)
        assert session.inspect().links == first.view.links
        second = session.advance_window(0.6, control(run))
        assert second.view.timed_out == 1
        assert second.view.links[0].bytes_sent == pytest.approx(49.9)
        assert second.link_bytes[0].bytes_sent == pytest.approx(25)
        third = session.advance_window(1, control(run))
        assert sum(link.bytes_sent for link in third.link_bytes) == 0
        assert third.view.links[0].bytes_sent == pytest.approx(49.9)


def test_scheduling_snapshot_preserves_counters_and_full_inspection():
    run = scenario([(f"r{i}", f"a{i % 2}", 0, i * 0.0001, 0.5 + i) for i in range(6)])
    with start(run) as full, start(run) as compact:
        for boundary in (0.0005, 0.25, 0.6, 1.5, 3, 8):
            a = full.advance_window(boundary, control(run))
            b = compact.advance_window(boundary, control(run), scope="scheduling")
            assert b.view.scope == "scheduling" and not b.view.transfers
            assert {r.request_id for r in b.view.requests} == {
                rid for s in b.view.schedulers for rid in s.waiting
            }
            exclude = {"scope", "requests", "transfers"}
            assert a.view.model_dump(exclude=exclude) == b.view.model_dump(exclude=exclude)
            assert a.link_bytes == b.link_bytes
            assert compact.inspect() == full.inspect()
        assert compact.result().state == full.result().state
        with pytest.raises(SDKError, match="invalid_scope"):
            compact.advance_window(9, control(run), scope="invalid")
        assert compact.advance_window(9, control(run)).view.scope == "full"


def test_simultaneous_delivery_completions_win_exact_deadline():
    run = scenario([("r0", "a", 0, 0, 2.001), ("r1", "a", 1, 0, 2.001)], caches=2)
    with start(run) as session:
        out = session.advance_window(2.001, control(run))
        assert out.completed == 2 and out.timed_out == 0
        assert not out.view.transfers
        assert sum(link.bytes_sent for link in out.link_bytes) == pytest.approx(400)
