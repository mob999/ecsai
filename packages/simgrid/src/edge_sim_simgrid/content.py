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
    original_id: str | None = None
    attempt_index: int = 0


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
    accounted_remaining: float = 0.0
    order: int = 0
    fetch_key: tuple | None = None


class ContentRuntime:
    def __init__(self, run):
        self.run, self.spec = run, run.content
        self.requests = {r.id: Request(r, r.cluster_id) for r in self.spec.requests}
        self.arrivals = sorted(self.spec.requests, key=lambda r: (r.arrival_s, r.id))
        self.arrival_index = 0
        self.retry_arrivals = []
        self.retry_sequence = 0
        self.originals = {r.id: r for r in self.spec.requests}
        self.deadlines = []
        self.live = set()
        self.changed = set()
        self.schedulers = {s.id: s for s in self.spec.schedulers}
        self.queues = {s.id: deque() for s in self.spec.schedulers}
        self.busy = {}
        self.caches = {c.node_id: c for c in self.spec.caches}
        self.cluster_load_caches = {
            s: tuple(c for c, cfg in self.caches.items() if cfg.cluster_id == s)
            for s in self.schedulers
        }
        self.cluster_caches = {s: tuple(sorted(c)) for s, c in self.cluster_load_caches.items()}
        self.nodes = {n.id: n for n in run.scenario.nodes}
        controlled_links = {
            link
            for cache in self.spec.caches
            for link in (cache.backhaul_link, cache.delivery_link)
        }
        self.links = {link.id: link for link in run.scenario.links if link.id in controlled_links}
        self.link_rates = {lid: link.bandwidth_bytes_s for lid, link in self.links.items()}
        self.link_capacity = Counter()
        self.capacity_clock = 0.0
        self.native_links = {lid: sg.Link.by_name(lid) for lid in self.links}
        self.routes = {(r.src, r.dst): r.links for r in run.scenario.routes}
        self.artifacts = {a.id: a for a in run.scenario.artifacts}
        self.hosts = {n: sg.Host.by_name(n) for n in self.nodes}
        self.delivery_weights = {
            c: (cfg.total_bandwidth_bytes_s or self.links[cfg.delivery_link].bandwidth_bytes_s)
            for c, cfg in self.caches.items()
        }
        self.cluster_weights = {
            s: sum(self.delivery_weights[c] for c in caches)
            for s, caches in self.cluster_caches.items()
        }
        self.replicas = {c: OrderedDict() for c in self.caches}
        self.reserved = {c: {} for c in self.caches}
        self.active = {(c, k): {} for c in self.caches for k in ("backhaul", "delivery")}
        self.waiting = {key: deque() for key in self.active}
        self.fetches = {}
        self.fetch_counts = Counter()
        self.transfers = {}
        self.activities = {}
        self.pending = sg.ActivitySet([])
        self.completed_ids = set()
        self.request_transfer = {}
        self.pins = Counter()
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
            for c in self.cluster_load_caches[cluster]
        ]
        return sum(values) / len(values)

    def _eligible(self, rid):
        return (
            self.spec.scheduler_release == "continuous"
            or self.requests[rid].forwarded
            or rid in self.window_requests
        )

    def _enqueue(self, rid, cluster):
        request = self.requests[rid]
        request.cluster = cluster
        if (
            cluster not in self.busy
            and self.queues[cluster]
            and self._eligible(self.queues[cluster][0])
        ):
            waiting = self.queues[cluster].popleft()
            self.busy[cluster] = (waiting, self.now + self.schedulers[cluster].service_s)
            self.requests[waiting].status = "SCHEDULING"
        if cluster not in self.busy and not self.queues[cluster] and self._eligible(rid):
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
        w = self.control_weights[cluster]
        size = self.artifacts[request.spec.artifact_id].size_bytes
        x = (
            self._cluster_load(cluster),
            min(1, size / self.spec.size_scale_bytes),
            min(1, max(0, request.spec.deadline_s - self.now) / self.spec.deadline_scale_s),
        )
        # sigmoid(logit) >= .5 is exactly logit >= 0; avoids overflow.
        forward = sum(a * b for a, b in zip(w[:3], x, strict=True)) + w[3] >= 0
        if self.control.policy == "direct":
            forward = False if request.forwarded else self.direct_decisions[rid]
        elif self.control.policy != "threshold":
            forward = {"local": False, "forward": True, "random": self.rng.random() < 0.5}[
                self.control.policy
            ]
        target = self.forward_targets[cluster]
        if forward and not request.forwarded and target is not None:
            request.forwarded = True
            self.counts["forwarded"] += 1
            self.emit("request_forwarded", rid, src=cluster, dst=target)
            self._enqueue(rid, target)
            return
        caches = self.cluster_caches[cluster]
        # Smooth weighted round robin, weights use configured delivery capacities.
        for c in caches:
            self.rr[c] += self.delivery_weights[c]
        target = min(caches, key=lambda c: (-self.rr[c], c))
        self.rr[target] -= self.cluster_weights[cluster]
        self.serve(Serve(request_id=rid, cache_node=target))

    def _pinned(self, cache, aid):
        return self.pins[cache, aid] > 0

    def _fetch_key(self, cache, aid, rid):
        return (cache, aid) if self.spec.coalesce_backhaul else (cache, aid, rid)

    def _account_capacity(self):
        elapsed = self.now - self.capacity_clock
        for lid, rate in self.link_rates.items():
            self.link_capacity[lid] += elapsed * rate
        self.capacity_clock = self.now

    def _set_bandwidth(self, control):
        self._account_capacity()
        if self.spec.bandwidth_mode != "shared":
            return
        ratios = {c.cluster_id: c.backhaul_ratio for c in control.schedulers}
        for cache in self.caches.values():
            r = ratios[cache.cluster_id]
            for lid, fraction in ((cache.backhaul_link, r), (cache.delivery_link, 1 - r)):
                rate = cache.total_bandwidth_bytes_s * fraction
                self.native_links[lid].set_bandwidth(rate)
                self.link_rates[lid] = rate
        self.emit("bandwidth_allocated", "network", ratios=str(ratios))

    def _reserve(self, cache, aid):
        if aid in self.reserved[cache] or aid in self.replicas[cache]:
            return True
        size = self.artifacts[aid].size_bytes
        capacity = self.nodes[cache].storage_bytes
        used = sum(self.replicas[cache].values()) + sum(
            v for k, v in self.reserved[cache].items() if k not in self.replicas[cache]
        )
        for old in list(self.replicas[cache]):
            if used + size <= capacity:
                break
            if not self._pinned(cache, old) and old not in self.reserved[cache]:
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
        elif self._fetch_key(cache, aid, rid) in self.fetches:
            transfer = self.fetches[self._fetch_key(cache, aid, rid)]
            transfer.waiters.add(rid)
            self.request_transfer[rid] = transfer
            request.status = "WAITING_DATA"
        else:
            if not self._reserve(cache, aid):
                self._finish(rid, "REJECTED", "storage_capacity")
                return
            request.status = "WAITING_DATA"
            transfer = self._new_transfer(aid, cache, "backhaul", self.spec.origin, cache, {rid})
            transfer.fetch_key = self._fetch_key(cache, aid, rid)
            self.fetches[transfer.fetch_key] = transfer
            self.fetch_counts[cache, aid] += 1
            self._admit(transfer)

    def _new_transfer(self, aid, cache, kind, src, dst, waiters):
        self.transfer_sequence += 1
        t = Transfer(f"transfer-{self.transfer_sequence}", aid, cache, kind, src, dst, waiters)
        t.order = self.transfer_sequence
        self.transfers[t.id] = t
        for rid in waiters:
            self.request_transfer[rid] = t
        if kind == "delivery":
            self.pins[cache, aid] += 1
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
            self.hosts[t.src],
            self.hosts[t.dst],
            self.artifacts[t.artifact].size_bytes,
        )
        self.active[t.cache, t.kind][t.id] = t
        t.accounted_remaining = float(self.artifacts[t.artifact].size_bytes)
        self.activities[t.activity] = t
        self.pending.push(t.activity)
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
        self._account_bytes(t)
        self.active[key].pop(t.id, None)
        if t.activity is None and t in self.waiting[key]:
            self.waiting[key].remove(t)
        self.transfers.pop(t.id, None)
        if t.activity is not None:
            if t.id not in self.completed_ids:
                self.pending.erase(t.activity)
            self.completed_ids.discard(t.id)
            self.activities.pop(t.activity)
        for rid in t.waiters:
            self.request_transfer.pop(rid, None)
        if t.kind == "delivery":
            self.pins[t.cache, t.artifact] -= 1
            if not self.pins[t.cache, t.artifact]:
                del self.pins[t.cache, t.artifact]
        if t.kind == "backhaul":
            self.fetches.pop(t.fetch_key, None)
            self.fetch_counts[t.cache, t.artifact] -= 1
            if not self.fetch_counts[t.cache, t.artifact]:
                del self.fetch_counts[t.cache, t.artifact]
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
        queue = self.queues[r.cluster]
        if rid in queue:
            queue.remove(rid)
        if r.cluster in self.busy and self.busy[r.cluster][0] == rid:
            del self.busy[r.cluster]
        t = self.request_transfer.pop(rid, None)
        if t is not None:
            t.waiters.remove(rid)
            if not t.waiters:
                self._remove_transfer(t, cancelled=True)
        self.emit("request_finished", rid, status=status, reason=reason)
        if status != "SUCCEEDED" and r.attempt_index < self.spec.max_retries:
            original_id = r.original_id or rid
            original = self.originals[original_id]
            self.retry_sequence += 1
            retry_id = f"__retry_{self.retry_sequence}"
            while retry_id in self.requests:
                self.retry_sequence += 1
                retry_id = f"__retry_{self.retry_sequence}"
            arrival = self.now + self.spec.retry_delay_s
            attempt = original.model_copy(
                update={
                    "id": retry_id,
                    "arrival_s": arrival,
                    "deadline_s": arrival + original.deadline_s - original.arrival_s,
                }
            )
            self.requests[retry_id] = Request(
                attempt,
                original.cluster_id,
                original_id=original_id,
                attempt_index=r.attempt_index + 1,
            )
            heapq.heappush(self.retry_arrivals, (arrival, self.retry_sequence, retry_id))
            self.emit(
                "retry_scheduled",
                retry_id,
                original_request_id=original_id,
                previous_request_id=rid,
                arrival_s=arrival,
                attempt_index=r.attempt_index + 1,
            )

    def _account_bytes(self, t):
        # Progress remains entirely native. Sampling at a boundary or just before
        # removal accounts partial/cancelled transfers exactly without polling
        # every unrelated event. No guessed rates or separate network solver.
        if t.activity is None:
            return
        remaining = max(0.0, t.activity.remaining)
        served = max(0.0, t.accounted_remaining - remaining)
        for link in self.routes[t.src, t.dst]:
            self.link_bytes[link] += served
        t.accounted_remaining = remaining

    def _collect_completed(self):
        # Native test_any removes the returned activity. Preserve transfer creation
        # order, not native completion notification order, for simultaneous events.
        while self.pending.size:
            activity = self.pending.test_any()
            if activity is None:
                break
            self.completed_ids.add(self.activities[activity].id)
        return sorted((self.transfers[tid] for tid in self.completed_ids), key=lambda t: t.order)

    def _settle(self, arrivals=True):
        # Completed delivery wins an exact deadline tie. Start newly available
        # work after expiration, so dead requests never consume a service slot.
        completed = self._collect_completed()
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
            # Original arrivals precede retries at an exact tie. Boundary arrivals
            # (including retries) are deferred when arrivals=False.
            while self.retry_arrivals and self.retry_arrivals[0][0] <= self.now:
                _, _, rid = heapq.heappop(self.retry_arrivals)
                spec = self.requests[rid].spec
                self.counts["arrived"] += 1
                self.live.add(rid)
                heapq.heappush(self.deadlines, (spec.deadline_s, rid))
                self.emit("request_arrived", rid)
                if spec.deadline_s <= self.now:
                    self._finish(rid, "TIMED_OUT", "deadline")
                else:
                    self._enqueue(rid, spec.cluster_id)
        for cluster, queue in self.queues.items():
            if queue and cluster not in self.busy and self._eligible(queue[0]):
                rid = queue.popleft()
                self.busy[cluster] = rid, self.now + self.schedulers[cluster].service_s
                self.requests[rid].status = "SCHEDULING"
        for key, queue in self.waiting.items():
            cap = self._pool_spec(key).max_active
            while queue and (cap is None or len(self.active[key]) < cap):
                self._start(queue.popleft())

    def _wait(self, date):
        if self.activities:
            try:
                activity = self.pending.wait_any_for(max(0, date - self.now))
                self.completed_ids.add(self.activities[activity].id)
            except sg.TimeoutException:
                pass
        else:
            sg.this_actor.sleep_for(max(0, date - self.now))

    def advance_window(self, until_s, control, scope="full"):
        started = perf_counter()
        if scope not in {"full", "scheduling"}:
            raise CommandError("invalid_scope", "expected full or scheduling snapshot")
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
        if control.policy == "direct":
            if self.spec.scheduler_release != "window":
                raise CommandError("invalid_control", "direct decisions require window release")
            for item in control.schedulers:
                expected = set(self.queues[item.cluster_id])
                if item.cluster_id in self.busy:
                    expected.add(self.busy[item.cluster_id][0])
                expected = {rid for rid in expected if not self.requests[rid].forwarded}
                if set(item.request_decisions) != expected:
                    raise CommandError(
                        "invalid_control", "direct decisions must cover visible requests"
                    )
        self.window_requests = set(self.live)
        self.direct_decisions = {
            rid: forward
            for item in control.schedulers
            for rid, forward in item.request_decisions.items()
        }
        self._set_bandwidth(control)
        self.control = control
        self.control_weights = {c.cluster_id: c.weights for c in control.schedulers}
        self.broadcast = {
            s: 0.5 * self._cluster_load(s)
            + 0.5 * len(self.queues[s]) / max(1, self.schedulers[s].max_waiting)
            for s in self.schedulers
        }
        self.forward_targets = {
            s: min(
                (other for other in self.schedulers if other != s),
                key=lambda other: (self.broadcast[other], other),
                default=None,
            )
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
            if self.retry_arrivals:
                dates.append(self.retry_arrivals[0][0])
            while self.deadlines and self.deadlines[0][1] not in self.live:
                heapq.heappop(self.deadlines)
            if self.deadlines:
                dates.append(self.deadlines[0][0])
            self._wait(min(t for t in dates if t > self.now))
        self._settle(arrivals=False)
        finished = (
            self.arrival_index == len(self.arrivals) and not self.live and not self.retry_arrivals
        )
        return WindowResult(
            kind="finished" if finished else "time",
            start_s=start,
            view=self.inspect(scope=scope),
            completed=self.counts["completed"] - before["completed"],
            timed_out=self.counts["timed_out"] - before["timed_out"],
            rejected=self.counts["rejected"] - before["rejected"],
            link_bytes=tuple(
                LinkCounter(
                    link_id=lid,
                    bytes_sent=self.link_bytes[lid] - link_before[lid],
                    capacity_byte_seconds=(self.now - start) * self.link_rates[lid],
                )
                for lid, link in sorted(self.links.items())
            ),
            simulation_wall_s=perf_counter() - started,
        )

    def inspect(self, selection=None, *, scope="full"):
        self._account_capacity()
        for t in self.activities.values():
            self._account_bytes(t)
        selected = self.live | self.changed if selection is None else set(selection)
        if scope == "scheduling":
            selected = {rid for queue in self.queues.values() for rid in queue}
            if self.spec.scheduler_release == "window":
                selected.update(rid for rid, _ in self.busy.values())
        if selected - self.requests.keys():
            raise CommandError("unknown_request", "unknown content request")
        return ContentView(
            scope=scope,
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
                    remaining_bytes=sum(
                        max(0, t.activity.remaining) for t in self.active[c, k].values()
                    )
                    + sum(self.artifacts[t.artifact].size_bytes for t in self.waiting[c, k]),
                )
                for c, k in sorted(self.active)
            ),
            requests=tuple(
                ContentRequestState(
                    request_id=rid,
                    original_request_id=r.original_id or rid,
                    attempt_index=r.attempt_index,
                    first_arrival_s=self.originals[r.original_id or rid].arrival_s,
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
                if scope == "full"
            ),
            links=tuple(
                LinkCounter(
                    link_id=lid,
                    bytes_sent=self.link_bytes[lid],
                    capacity_byte_seconds=self.link_capacity[lid],
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
            attempt_outcomes=tuple(
                ContentRequestState(
                    request_id=rid,
                    original_request_id=r.original_id or rid,
                    attempt_index=r.attempt_index,
                    first_arrival_s=self.originals[r.original_id or rid].arrival_s,
                    origin_cluster=r.spec.cluster_id,
                    cluster_id=r.cluster,
                    artifact_id=r.spec.artifact_id,
                    size_bytes=self.artifacts[r.spec.artifact_id].size_bytes,
                    arrival_s=r.spec.arrival_s,
                    deadline_s=r.spec.deadline_s,
                    completed_s=r.finished,
                    status=r.status,
                    reason=r.reason,
                    cache_node=r.cache,
                    forwarded=r.forwarded,
                )
                for rid in sorted(self.changed)
                for r in (self.requests[rid],)
                if self.spec.max_retries and r.status in TERMINAL
            ),
        )

    def result(self):
        done = (
            self.arrival_index == len(self.arrivals) and not self.live and not self.retry_arrivals
        )
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
