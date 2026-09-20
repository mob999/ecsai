"""Event-driven content service. SimGrid owns every positive-byte transfer.

Only scheduler service delays use timers. Queues count unique transfers (not
coalesced waiters); delivery is always per request. No training dependencies.
"""

import hashlib
import heapq
import math
import random
from collections import Counter, OrderedDict, deque
from dataclasses import dataclass, field
from time import perf_counter

import simgrid as sg
from edge_sim_models import (
    ContentRequestState,
    ContentTransferState,
    ContentView,
    Detail,
    DomainEvent,
    LinkCounter,
    PoolState,
    RequestState,
    RunManifest,
    RunMetrics,
    RunResult,
    SchedulerState,
    Serve,
    StateView,
    WindowControl,
    WindowResult,
)
from pydantic import ValidationError

from .resources import CommandError

TERMINAL = {"SUCCEEDED", "TIMED_OUT", "REJECTED"}


@dataclass
class Request:
    spec: object
    cluster: str
    status: str = "PENDING"
    cache: str | None = None
    forwarded: bool = False
    finished: float | None = None
    reason: str | None = None


@dataclass
class Transfer:
    id: str
    artifact: str
    cache: str
    kind: str
    src: str
    dst: str
    waiters: set = field(default_factory=set)
    activity: object = None


class ContentRuntime:
    def __init__(self, run):
        self.run, self.spec = run, run.content
        self.requests = {r.id: Request(r, r.cluster_id) for r in self.spec.requests}
        self.arrivals = sorted(self.spec.requests, key=lambda r: (r.arrival_s, r.id))
        self.arrival_index = 0
        self.deadlines = []
        self.live = set()
        self.changed = set()
        self.schedulers = {s.id: s for s in self.spec.schedulers}
        self.queues = {s.id: deque() for s in self.spec.schedulers}
        self.busy = {}
        self.caches = {c.node_id: c for c in self.spec.caches}
        self.nodes = {n.id: n for n in run.scenario.nodes}
        controlled_links = {
            link
            for cache in self.spec.caches
            for link in (cache.backhaul_link, cache.delivery_link)
        }
        self.links = {link.id: link for link in run.scenario.links if link.id in controlled_links}
        self.routes = {(r.src, r.dst): r.links for r in run.scenario.routes}
        self.artifacts = {a.id: a for a in run.scenario.artifacts}
        self.replicas = {c: OrderedDict() for c in self.caches}
        self.reserved = {c: {} for c in self.caches}
        self.active = {(c, k): {} for c in self.caches for k in ("backhaul", "delivery")}
        self.waiting = {key: deque() for key in self.active}
        self.fetches = {}
        self.transfers = {}
        self.link_bytes = Counter()
        self.counts = Counter()
        self.latencies = []
        self.events = []
        self.command_log = []
        self.sequence = 0
        self.transfer_sequence = 0
        self.rr = Counter()
        self.control = None
        self.broadcast = {}
        self.rng = random.Random(run.seed)
        self.closed = False

    @property
    def now(self):
        return sg.Engine.clock

    def emit(self, kind, entity, **details):
        self.sequence += 1
        if self.run.trace:
            self.events.append(
                DomainEvent(
                    sequence=self.sequence,
                    time_s=self.now,
                    kind=kind,
                    entity_id=entity,
                    details=tuple(Detail(name=k, value=v) for k, v in details.items()),
                )
            )

    def _pool_spec(self, key):
        return getattr(self.caches[key[0]], key[1])

    def _load(self, cache, kind):
        key = cache, kind
        cfg = self._pool_spec(key)
        denominator = (cfg.max_active or 1) + (cfg.max_waiting or 0)
        return min(1, (len(self.active[key]) + len(self.waiting[key])) / denominator)

    def _cluster_load(self, cluster):
        values = [
            (self._load(c, "backhaul") + self._load(c, "delivery")) / 2
            for c, cfg in self.caches.items()
            if cfg.cluster_id == cluster
        ]
        return sum(values) / len(values)

    def _enqueue(self, rid, cluster):
        request = self.requests[rid]
        request.cluster = cluster
        if cluster not in self.busy and self.queues[cluster]:
            waiting = self.queues[cluster].popleft()
            self.busy[cluster] = (waiting, self.now + self.schedulers[cluster].service_s)
            self.requests[waiting].status = "SCHEDULING"
        if cluster not in self.busy and not self.queues[cluster]:
            self.busy[cluster] = (rid, self.now + self.schedulers[cluster].service_s)
            request.status = "SCHEDULING"
        elif len(self.queues[cluster]) < self.schedulers[cluster].max_waiting:
            self.queues[cluster].append(rid)
            request.status = "WAITING_SCHEDULER"
        else:
            self._finish(rid, "REJECTED", "scheduler_overflow")

    def _schedule(self, rid):
        request = self.requests[rid]
        cluster = request.cluster
        w = next(s.weights for s in self.control.schedulers if s.cluster_id == cluster)
        size = self.artifacts[request.spec.artifact_id].size_bytes
        x = (
            self._cluster_load(cluster),
            min(1, size / self.spec.size_scale_bytes),
            min(1, max(0, request.spec.deadline_s - self.now) / self.spec.deadline_scale_s),
        )
        # sigmoid(logit) >= .5 is exactly logit >= 0; avoids overflow.
        forward = sum(a * b for a, b in zip(w[:3], x, strict=True)) + w[3] >= 0
        if self.control.policy != "threshold":
            forward = {"local": False, "forward": True, "random": self.rng.random() < 0.5}[
                self.control.policy
            ]
        others = sorted(s for s in self.schedulers if s != cluster)
        if forward and not request.forwarded and others:
            target = min(others, key=lambda s: (self.broadcast[s], s))
            request.forwarded = True
            self.counts["forwarded"] += 1
            self.emit("request_forwarded", rid, src=cluster, dst=target)
            self._enqueue(rid, target)
            return
        caches = sorted(c for c, cfg in self.caches.items() if cfg.cluster_id == cluster)
        # Smooth weighted round robin, weights use configured delivery capacities.
        weights = {c: self.links[self.caches[c].delivery_link].bandwidth_bytes_s for c in caches}
        for c in caches:
            self.rr[c] += weights[c]
        target = min(caches, key=lambda c: (-self.rr[c], c))
        self.rr[target] -= sum(weights.values())
        self.serve(Serve(request_id=rid, cache_node=target))

    def _pinned(self, cache, aid):
        return any(
            t.cache == cache and t.artifact == aid and t.kind == "delivery"
            for t in self.transfers.values()
        )

    def _reserve(self, cache, aid):
        size = self.artifacts[aid].size_bytes
        capacity = self.nodes[cache].storage_bytes
        used = sum(self.replicas[cache].values()) + sum(self.reserved[cache].values())
        for old in list(self.replicas[cache]):
            if used + size <= capacity:
                break
            if not self._pinned(cache, old):
                used -= self.replicas[cache].pop(old)
                self.emit("cache_evict", old, node_id=cache)
        if used + size > capacity:
            return False
        self.reserved[cache][aid] = size
        return True

    def serve(self, operation):
        rid, cache = operation.request_id, operation.cache_node
        request = self.requests[rid]
        request.cache = cache
        aid = request.spec.artifact_id
        self.counts["cache_lookups"] += 1
        self.emit("content_served", rid, node_id=cache, artifact_id=aid)
        if aid in self.replicas[cache]:
            self.counts["cache_hits"] += 1
            self.replicas[cache].move_to_end(aid)
            self._delivery(rid)
        elif (cache, aid) in self.fetches:
            transfer = self.fetches[cache, aid]
            transfer.waiters.add(rid)
            request.status = "WAITING_DATA"
        else:
            if not self._reserve(cache, aid):
                self._finish(rid, "REJECTED", "storage_capacity")
                return
            request.status = "WAITING_DATA"
            transfer = self._new_transfer(aid, cache, "backhaul", self.spec.origin, cache, {rid})
            self.fetches[cache, aid] = transfer
            self._admit(transfer)

    def _new_transfer(self, aid, cache, kind, src, dst, waiters):
        self.transfer_sequence += 1
        t = Transfer(f"transfer-{self.transfer_sequence}", aid, cache, kind, src, dst, waiters)
        self.transfers[t.id] = t
        return t

    def _delivery(self, rid):
        r = self.requests[rid]
        r.status = "DELIVERING"
        self.replicas[r.cache].move_to_end(r.spec.artifact_id)
        self._admit(
            self._new_transfer(
                r.spec.artifact_id,
                r.cache,
                "delivery",
                r.cache,
                r.spec.receiver,
                {rid},
            )
        )

    def _admit(self, t):
        key = t.cache, t.kind
        cfg = self._pool_spec(key)
        while self.waiting[key] and (
            cfg.max_active is None or len(self.active[key]) < cfg.max_active
        ):
            self._start(self.waiting[key].popleft())
        if cfg.max_active is None or len(self.active[key]) < cfg.max_active:
            self._start(t)
        elif cfg.max_waiting is None or len(self.waiting[key]) < cfg.max_waiting:
            self.waiting[key].append(t)
            self.emit("transfer_queued", t.id, kind_name=t.kind, node_id=t.cache)
        else:
            for rid in sorted(t.waiters):
                self._finish(rid, "REJECTED", f"{t.kind}_overflow")

    def _start(self, t):
        t.activity = sg.Comm.sendto_async(
            sg.Host.by_name(t.src),
            sg.Host.by_name(t.dst),
            self.artifacts[t.artifact].size_bytes,
        )
        self.active[t.cache, t.kind][t.id] = t
        self.emit(
            "transfer_started",
            t.id,
            artifact_id=t.artifact,
            src=t.src,
            dst=t.dst,
            bytes=self.artifacts[t.artifact].size_bytes,
        )

    def _remove_transfer(self, t, cancelled=False):
        key = t.cache, t.kind
        self.active[key].pop(t.id, None)
        if t in self.waiting[key]:
            self.waiting[key].remove(t)
        self.transfers.pop(t.id, None)
        if t.kind == "backhaul":
            self.fetches.pop((t.cache, t.artifact), None)
            self.reserved[t.cache].pop(t.artifact, None)
        if cancelled:
            self.counts["cancelled_transfers"] += 1
            if t.activity is not None:
                t.activity.cancel()
            self.emit("transfer_cancelled", t.id, src=t.src, dst=t.dst)

    def _finish(self, rid, status, reason=None):
        r = self.requests[rid]
        if r.status in TERMINAL:
            return
        r.status, r.finished, r.reason = status, self.now, reason
        self.live.discard(rid)
        self.changed.add(rid)
        self.counts[
            {"SUCCEEDED": "completed", "TIMED_OUT": "timed_out", "REJECTED": "rejected"}[status]
        ] += 1
        if reason and reason.endswith("overflow"):
            self.counts["overflows"] += 1
            self.emit("queue_overflow", rid, reason=reason)
        if status == "SUCCEEDED":
            self.latencies.append(self.now - r.spec.arrival_s)
        for queue in self.queues.values():
            if rid in queue:
                queue.remove(rid)
        for cluster, (active, _) in list(self.busy.items()):
            if active == rid:
                del self.busy[cluster]
        for t in list(self.transfers.values()):
            if rid in t.waiters:
                t.waiters.remove(rid)
                if not t.waiters:
                    self._remove_transfer(t, cancelled=True)
        self.emit("request_finished", rid, status=status, reason=reason)

    def _settle(self, arrivals=True):
        # Completed delivery wins an exact deadline tie. Start newly available
        # work after expiration, so dead requests never consume a service slot.
        completed = [
            t for t in self.transfers.values() if t.activity is not None and t.activity.test()
        ]
        # Release all simultaneous completions before admitting anything new.
        for t in completed:
            self._remove_transfer(t)
            self.emit("transfer_finished", t.id, src=t.src, dst=t.dst)
        for t in completed:
            if t.kind == "delivery":
                for rid in sorted(t.waiters):
                    self._finish(rid, "SUCCEEDED")
        while self.deadlines and self.deadlines[0][0] <= self.now:
            _, rid = heapq.heappop(self.deadlines)
            if rid in self.live:
                self._finish(rid, "TIMED_OUT", "deadline")
        for t in completed:
            if t.kind == "backhaul":
                self.replicas[t.cache][t.artifact] = self.artifacts[t.artifact].size_bytes
                for rid in sorted(t.waiters):
                    if rid in self.live:
                        self._delivery(rid)
        ready = [
            (cluster, rid) for cluster, (rid, finish) in self.busy.items() if finish <= self.now
        ]
        for cluster, _ in ready:
            del self.busy[cluster]
        for _, rid in ready:
            if rid in self.live:
                self._schedule(rid)
        if arrivals:
            while self.arrival_index < len(self.arrivals):
                spec = self.arrivals[self.arrival_index]
                if spec.arrival_s > self.now:
                    break
                self.arrival_index += 1
                self.counts["arrived"] += 1
                self.live.add(spec.id)
                heapq.heappush(self.deadlines, (spec.deadline_s, spec.id))
                self.emit("request_arrived", spec.id)
                if spec.deadline_s <= self.now:
                    self._finish(spec.id, "TIMED_OUT", "deadline")
                else:
                    self._enqueue(spec.id, spec.cluster_id)
        for cluster, queue in self.queues.items():
            if queue and cluster not in self.busy:
                rid = queue.popleft()
                self.busy[cluster] = rid, self.now + self.schedulers[cluster].service_s
                self.requests[rid].status = "SCHEDULING"
        for key, queue in self.waiting.items():
            cap = self._pool_spec(key).max_active
            while queue and (cap is None or len(self.active[key]) < cap):
                self._start(queue.popleft())

    def _wait(self, date):
        active = [t for t in self.transfers.values() if t.activity is not None]
        remaining = {t.id: t.activity.remaining for t in active}
        if active:
            try:
                sg.ActivitySet([t.activity for t in active]).wait_any_for(max(0, date - self.now))
            except sg.TimeoutException:
                pass
        else:
            sg.this_actor.sleep_for(max(0, date - self.now))
        for t in active:
            served = max(0, remaining[t.id] - t.activity.remaining)
            for link in self.routes[t.src, t.dst]:
                self.link_bytes[link] += served

    def advance_window(self, until_s, control):
        started = perf_counter()
        if not math.isfinite(until_s) or until_s <= self.now:
            raise CommandError("invalid_time", "window boundary must be finite and in the future")
        try:
            control = WindowControl.model_validate(control)
        except ValidationError as error:
            raise CommandError("invalid_control", str(error)) from error
        if {c.cluster_id for c in control.schedulers} != set(self.schedulers):
            raise CommandError("invalid_control", "provide exactly one control per scheduler")
        if self.run.until_s is not None:
            until_s = min(until_s, self.run.until_s)
            if until_s <= self.now:
                raise CommandError("finished", "run time limit reached")
        self.control = control
        self.broadcast = {
            s: 0.5 * self._cluster_load(s)
            + 0.5 * len(self.queues[s]) / max(1, self.schedulers[s].max_waiting)
            for s in self.schedulers
        }
        start, before, link_before = self.now, self.counts.copy(), self.link_bytes.copy()
        self.changed.clear()
        if self.run.trace:
            self.command_log.append((start, until_s, control))
        while self.now < until_s:
            self._settle()
            dates = [until_s] + [t for _, t in self.busy.values()]
            if self.arrival_index < len(self.arrivals):
                dates.append(self.arrivals[self.arrival_index].arrival_s)
            while self.deadlines and self.deadlines[0][1] not in self.live:
                heapq.heappop(self.deadlines)
            if self.deadlines:
                dates.append(self.deadlines[0][0])
            self._wait(min(t for t in dates if t > self.now))
        self._settle(arrivals=False)
        finished = self.arrival_index == len(self.arrivals) and not self.live
        return WindowResult(
            kind="finished" if finished else "time",
            start_s=start,
            view=self.inspect(),
            completed=self.counts["completed"] - before["completed"],
            timed_out=self.counts["timed_out"] - before["timed_out"],
            rejected=self.counts["rejected"] - before["rejected"],
            link_bytes=tuple(
                LinkCounter(
                    link_id=lid,
                    bytes_sent=self.link_bytes[lid] - link_before[lid],
                    capacity_byte_seconds=(self.now - start) * link.bandwidth_bytes_s,
                )
                for lid, link in sorted(self.links.items())
            ),
            simulation_wall_s=perf_counter() - started,
        )

    def inspect(self, selection=None):
        selected = self.live | self.changed if selection is None else set(selection)
        if selected - self.requests.keys():
            raise CommandError("unknown_request", "unknown content request")
        return ContentView(
            now_s=self.now,
            schedulers=tuple(
                SchedulerState(
                    cluster_id=s,
                    waiting=tuple(q),
                    capacity=self.schedulers[s].max_waiting,
                    active_request=self.busy[s][0] if s in self.busy else None,
                )
                for s, q in sorted(self.queues.items())
            ),
            pools=tuple(
                PoolState(
                    node_id=c,
                    kind=k,
                    active=len(self.active[c, k]),
                    waiting=len(self.waiting[c, k]),
                    max_active=self._pool_spec((c, k)).max_active,
                    max_waiting=self._pool_spec((c, k)).max_waiting,
                )
                for c, k in sorted(self.active)
            ),
            requests=tuple(
                ContentRequestState(
                    request_id=rid,
                    origin_cluster=r.spec.cluster_id,
                    cluster_id=r.cluster,
                    artifact_id=r.spec.artifact_id,
                    size_bytes=self.artifacts[r.spec.artifact_id].size_bytes,
                    arrival_s=r.spec.arrival_s,
                    deadline_s=r.spec.deadline_s,
                    status=r.status,
                    cache_node=r.cache,
                    forwarded=r.forwarded,
                    completed_s=r.finished,
                    reason=r.reason,
                )
                for rid in sorted(selected)
                for r in (self.requests[rid],)
            ),
            transfers=tuple(
                ContentTransferState(
                    transfer_id=t.id,
                    artifact_id=t.artifact,
                    node_id=t.cache,
                    kind=t.kind,
                    status="active" if t.activity is not None else "waiting",
                    request_ids=tuple(sorted(t.waiters)),
                    src=t.src,
                    dst=t.dst,
                    size_bytes=self.artifacts[t.artifact].size_bytes,
                    remaining_bytes=max(0, t.activity.remaining)
                    if t.activity is not None
                    else self.artifacts[t.artifact].size_bytes,
                )
                for t in self.transfers.values()
            ),
            links=tuple(
                LinkCounter(
                    link_id=lid,
                    bytes_sent=self.link_bytes[lid],
                    capacity_byte_seconds=self.now * link.bandwidth_bytes_s,
                )
                for lid, link in sorted(self.links.items())
            ),
            **{
                k: self.counts[k]
                for k in (
                    "arrived",
                    "completed",
                    "timed_out",
                    "rejected",
                    "cache_hits",
                    "cache_lookups",
                    "forwarded",
                    "overflows",
                    "cancelled_transfers",
                )
            },
            # Only interval latencies travel per window; result() retains the full record.
            latencies_s=tuple(
                self.requests[r].finished - self.requests[r].spec.arrival_s
                for r in sorted(self.changed)
                if self.requests[r].status == "SUCCEEDED"
            ),
        )

    def result(self):
        done = self.arrival_index == len(self.arrivals) and not self.live
        return RunResult(
            run_id=self.run.run_id,
            seed=self.run.seed,
            now_s=self.now,
            completed=done,
            end_reason="completed"
            if done
            else "truncated"
            if self.run.until_s is not None and self.now >= self.run.until_s
            else "in_progress",
            state=StateView(
                now_s=self.now,
                requests=tuple(
                    RequestState(
                        request_id=rid,
                        status=r.status if r.status in TERMINAL | {"PENDING"} else "ACTIVE",
                        arrival_s=r.spec.arrival_s,
                        deadline_s=r.spec.deadline_s,
                        completed_s=r.finished,
                        reason=r.reason,
                    )
                    for rid, r in self.requests.items()
                ),
            ),
            metrics=RunMetrics(
                arrived=self.counts["arrived"],
                completed=self.counts["completed"],
                timed_out=self.counts["timed_out"],
                rejected=self.counts["rejected"],
                unfinished=len(self.live),
                cache_hits=self.counts["cache_hits"],
                transfers=self.transfer_sequence,
                events=self.sequence,
            ),
            events=tuple(self.events),
            manifest=RunManifest(
                run_id=self.run.run_id,
                seed=self.run.seed,
                run_spec=self.run,
                scenario_hash=hashlib.sha256(self.run.model_dump_json().encode()).hexdigest(),
                simgrid_version="4.1",
            ),
        )

    def close(self):
        if not self.closed:
            for t in list(self.transfers.values()):
                self._remove_transfer(t, cancelled=True)
            self.closed = True
