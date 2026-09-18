"""Domain orchestration over real SimGrid activities, confined to one actor."""

import hashlib
import importlib
import math
from collections import Counter

import simgrid as sg

from .resources import (
    CommandError,
    ReplicaState,
    RequestState,
    ResourceLedger,
    StageState,
    TransferState,
)


class Runtime:
    def __init__(self, run):
        self.run = run
        self.scenario = run.scenario
        self.nodes = {n.id: n for n in self.scenario.nodes}
        self.workflows = {w.id: w for w in self.scenario.workflows}
        self.requests = {r.id: r for r in self.scenario.requests}
        self.request_states = {r.id: RequestState(request_id=r.id) for r in self.scenario.requests}
        self.stages = {}
        self.stage_specs = {}
        self.inputs = {}
        self.outputs = {}
        self.dependencies = {}
        self.artifacts = {}
        self.replicas = {}
        self.origins = set()
        self.ledgers = {
            n.id: ResourceLedger(memory_capacity=n.memory_bytes, storage_capacity=n.storage_bytes)
            for n in self.scenario.nodes
        }
        self.transfers = {}
        self.comms = {}
        self.transfer_timers = {}
        self.execs = {}
        self.routes = {(r.src, r.dst): r.links for r in self.scenario.routes}
        self.events = []
        self.command_log = []
        self.event_count = 0
        self.revision = 0
        self.pending = None
        self.decision_dirty = False
        self.wake_s = None
        self.finished_reason = None
        self.round_robin = 0
        self.cpu_busy = Counter()
        self.link_bytes = Counter()
        self.cache_hits = 0
        self.transfer_count = 0
        self.data_lookups = set()
        self.plugins = {}
        for plugin in run.policy.plugins:
            module, factory = plugin.factory.split(":")
            self.plugins[plugin.role] = getattr(importlib.import_module(module), factory)()
        self.preemption_revision = -1
        self._build()

    @property
    def now(self):
        return sg.Engine.clock

    def _build(self):
        for a in self.scenario.artifacts:
            self.artifacts[a.id] = a
            for node_id in a.locations:
                self.ledgers[node_id].reserve_storage(a.id, a.size_bytes)
                self.replicas[a.id, node_id] = ReplicaState(
                    artifact_id=a.id,
                    node_id=node_id,
                    size_bytes=a.size_bytes,
                    cacheable=a.cacheable,
                    pinned=True,
                )
                self.origins.add((a.id, node_id))
        for request in self.scenario.requests:
            workflow = self.workflows[request.workflow]
            bindings = {b.artifact_id: b.global_artifact_id for b in request.input_bindings}
            mapping = {}
            for a in workflow.artifacts:
                aid = bindings.get(a.id, f"{request.id}/{a.id}")
                mapping[a.id] = aid
                if aid not in self.artifacts:
                    self.artifacts[aid] = a
            for stage in workflow.stages:
                key = f"{request.id}/{stage.id}"
                self.stages[key] = StageState(request_id=request.id, stage_id=stage.id)
                self.stage_specs[key] = stage
                self.inputs[key] = tuple(mapping[a] for a in stage.inputs)
                self.outputs[key] = tuple(mapping[a] for a in stage.outputs)
                deps = set(stage.depends_on)
                for aid in stage.inputs:
                    producer = next(a.producer_stage for a in workflow.artifacts if a.id == aid)
                    if producer is not None:
                        deps.add(producer)
                self.dependencies[key] = tuple(f"{request.id}/{d}" for d in sorted(deps))
            consumed = {a for s in workflow.stages for a in s.inputs}
            self.outputs[request.id] = tuple(
                mapping[a.id]
                for a in workflow.artifacts
                if a.producer_stage and a.id not in consumed
            )
            # A content-only workflow declares input artifacts as its deliverables.
            if not workflow.stages:
                self.outputs[request.id] = tuple(mapping.values())

    def emit(self, kind, entity_id, **details):
        from edge_sim_models import Detail, DomainEvent

        self.revision += 1
        self.event_count += 1
        if self.run.trace:
            self.events.append(
                DomainEvent(
                    time_s=self.now,
                    sequence=self.event_count,
                    kind=kind,
                    entity_id=entity_id,
                    details=tuple(Detail(name=k, value=v) for k, v in sorted(details.items())),
                )
            )

    def _transition(self, stage, status):
        stage.transition(status, self.now)
        self.emit("stage_state", stage.key, status=status)

    def _active(self, request_id):
        return self.request_states[request_id].status == "ACTIVE"

    def _pin(self, aid, node):
        if (aid, node) in self.origins:
            return True
        if any(t.artifact_id == aid and t.src == node for t in self.transfers.values()):
            return True
        alternatives = {n for a, n in self.replicas if a == aid and n != node}

        def needed_at(dst):
            return (node == dst or (node, dst) in self.routes) and not any(
                n == dst or (n, dst) in self.routes for n in alternatives
            )

        for key, stage in self.stages.items():
            if not self._active(stage.request_id):
                continue
            if stage.status not in {"SUCCEEDED", "FAILED", "CANCELLED"}:
                if aid in self.inputs[key]:
                    if stage.node_id is not None:
                        if stage.node_id == node or needed_at(stage.node_id):
                            return True
                    else:
                        spec = self.stage_specs[key]
                        needed = sum(
                            self.artifacts[a].size_bytes
                            for a in set(self.inputs[key] + self.outputs[key])
                        )
                        if any(
                            n.id == node or needed_at(n.id)
                            for n in self.nodes.values()
                            if (not spec.eligible_nodes or n.id in spec.eligible_nodes)
                            and spec.memory_bytes <= n.memory_bytes
                            and needed <= n.storage_bytes
                        ):
                            return True
        return any(
            self._active(rid)
            and aid in self.outputs[rid]
            and (req.receiver == node or needed_at(req.receiver))
            for rid, req in self.requests.items()
        )

    def _reserve_data(self, aid, node):
        ledger = self.ledgers[node]
        if aid in ledger.storage:
            return True
        size = self.artifacts[aid].size_bytes
        if size > ledger.storage_capacity:
            return False
        candidates = sorted(
            (r for (a, n), r in self.replicas.items() if n == node and not self._pin(a, n)),
            key=lambda r: (r.last_access, r.artifact_id),
        )
        if "cache" in self.plugins:
            ids = tuple(r.artifact_id for r in candidates)
            ordered = self.plugins["cache"].order(node, ids)
            self._check_permutation(ids, ordered)
            by_id = {r.artifact_id: r for r in candidates}
            candidates = [by_id[i] for i in ordered]
        for replica in candidates:
            if ledger.storage_used + size <= ledger.storage_capacity:
                break
            # Earlier evictions may have made this the last reachable copy.
            if self._pin(replica.artifact_id, node):
                continue
            self._remove_replica(replica.artifact_id, node)
            self.emit("cache_evict", replica.artifact_id, node_id=node)
        return ledger.reserve_storage(aid, size)

    def _remove_replica(self, aid, node):
        self.replicas.pop((aid, node), None)
        self.ledgers[node].storage.pop(aid, None)

    def _ensure_data(self, aid, node, waiter):
        lookup = (aid, node, waiter)
        if lookup not in self.data_lookups:
            self.data_lookups.add(lookup)
            if (aid, node) in self.replicas:
                self.cache_hits += 1
        if (aid, node) in self.replicas:
            self.replicas[aid, node].last_access = self.now
            return True
        key = (aid, node)
        if key in self.transfers:
            self.transfers[key].waiters.add(waiter)
            return False
        sources = sorted(n for (a, n) in self.replicas if a == aid and (n, node) in self.routes)
        if not sources or not self._reserve_data(aid, node):
            return False
        src = sources[0]
        if "replica" in self.plugins:
            src = self.plugins["replica"].select(aid, node, tuple(sources))
            if src not in sources:
                raise ValueError("Replica policy returned an unavailable source")
        artifact = self.artifacts[aid]
        self.transfers[key] = TransferState(
            artifact_id=aid,
            src=src,
            dst=node,
            size_bytes=artifact.size_bytes,
            started_s=self.now,
            waiters={waiter},
        )
        if artifact.size_bytes:
            self.comms[key] = sg.Comm.sendto_async(
                sg.Host.by_name(src), sg.Host.by_name(node), artifact.size_bytes
            )
        else:
            # SimGrid raw zero-byte comms never complete. Only propagation remains;
            # these timers consume no bandwidth and share normal transfer ownership.
            links = {link.id: link for link in self.scenario.links}
            self.transfer_timers[key] = self.now + sum(
                links[link].latency_s for link in self.routes[src, node]
            )
        self.transfer_count += 1
        self.emit("transfer_started", aid, src=src, dst=node, bytes=artifact.size_bytes)
        if key in self.transfer_timers and self.transfer_timers[key] <= self.now:
            self._complete_transfer(key)
            return True
        return False

    def _complete_transfer(self, key):
        t = self.transfers.pop(key)
        self.comms.pop(key, None)
        self.transfer_timers.pop(key, None)
        a = self.artifacts[t.artifact_id]
        self.replicas[key] = ReplicaState(
            artifact_id=t.artifact_id,
            node_id=t.dst,
            size_bytes=a.size_bytes,
            cacheable=a.cacheable,
            last_access=self.now,
        )
        self.emit("transfer_finished", t.artifact_id, src=t.src, dst=t.dst)

    def _finish_request(self, rid, status, reason=None):
        state = self.request_states[rid]
        if state.status not in {"ACTIVE", "PENDING"}:
            return
        state.status, state.finished_s, state.reason = status, self.now, reason
        for key, stage in self.stages.items():
            if stage.request_id != rid:
                continue
            if stage.status not in {"SUCCEEDED", "FAILED", "CANCELLED"}:
                if key in self.execs:
                    self.execs.pop(key).cancel()
                self._transition(stage, "CANCELLED")
                if stage.node_id:
                    ledger = self.ledgers[stage.node_id]
                    ledger.memory.pop(key, None)
                    for aid in self.outputs[key]:
                        if (aid, stage.node_id) not in self.replicas:
                            ledger.storage.pop(aid, None)
        for key, transfer in list(self.transfers.items()):
            transfer.waiters = {
                w for w in transfer.waiters if w != rid and not w.startswith(rid + "/")
            }
            if not transfer.waiters:
                activity = self.comms.pop(key, None)
                if activity is not None:
                    activity.cancel()
                self.transfer_timers.pop(key, None)
                self.ledgers[transfer.dst].storage.pop(transfer.artifact_id, None)
                del self.transfers[key]
        self.emit("request_finished", rid, status=status, reason=reason)
        self._collect_data()

    def _collect_data(self):
        for (aid, node), replica in list(self.replicas.items()):
            if not replica.cacheable and not self._pin(aid, node):
                self._remove_replica(aid, node)

    def _reap(self):
        changed = False
        for key in list(self.transfers):
            if key in self.transfer_timers:
                done = self.transfer_timers[key] <= self.now
            else:
                done = self.comms[key].test()
            if not done:
                continue
            self._complete_transfer(key)
            changed = True
        for key, activity in list(self.execs.items()):
            stage = self.stages[key]
            if stage.status == "SUSPENDED" or not activity.test():
                continue
            del self.execs[key]
            self.ledgers[stage.node_id].memory.pop(key, None)
            for aid in self.outputs[key]:
                a = self.artifacts[aid]
                self.replicas[aid, stage.node_id] = ReplicaState(
                    artifact_id=aid,
                    node_id=stage.node_id,
                    size_bytes=a.size_bytes,
                    cacheable=a.cacheable,
                    last_access=self.now,
                )
            self._transition(stage, "SUCCEEDED")
            changed = True
        return changed

    def _candidates(self, key):
        spec = self.stage_specs[key]
        needed = sum(
            self.artifacts[a].size_bytes for a in set(self.inputs[key] + self.outputs[key])
        )
        candidates = []
        for node in self.nodes.values():
            if spec.eligible_nodes and node.id not in spec.eligible_nodes:
                continue
            if spec.memory_bytes > node.memory_bytes or needed > node.storage_bytes:
                continue
            possible = True
            for aid in self.inputs[key]:
                existing = [n for a, n in self.replicas if a == aid]
                if existing and not any(
                    n == node.id or (n, node.id) in self.routes for n in existing
                ):
                    possible = False
            if possible:
                candidates.append(node.id)
        return tuple(sorted(candidates))

    def _place(self, key, node):
        stage = self.stages[key]
        stage.node_id = node
        self._transition(stage, "WAITING_DATA")

    def _dispatch(self):
        busy = Counter(s.node_id for s in self.stages.values() if s.status == "RUNNING")
        queued = [s for s in self.stages.values() if s.status == "QUEUED"]
        queued.sort(
            key=lambda s: (
                (
                    self.requests[s.request_id].deadline_s
                    if self.requests[s.request_id].deadline_s is not None
                    else math.inf
                )
                if self.run.policy.name == "edf"
                else s.entered_s,
                s.key,
            )
        )
        if "scheduling" in self.plugins and queued:
            ids = tuple(s.key for s in queued)
            views = {f"{s.request_id}/{s.stage_id}": s for s in self.inspect().stages}
            ordered = self.plugins["scheduling"].order(tuple(views[k] for k in ids))
            self._check_permutation(ids, ordered)
            queued = [self.stages[k] for k in ordered]
        changed = False
        for stage in queued:
            key, node = stage.key, self.nodes[stage.node_id]
            if self.now < stage.ready_after_s or busy[node.id] >= node.cores:
                continue
            spec, ledger = self.stage_specs[key], self.ledgers[node.id]
            if (
                key not in ledger.memory
                and ledger.memory_used + spec.memory_bytes > node.memory_bytes
            ):
                stage.reason = "memory"
                continue
            # Capacity checks and evictions precede an atomic reservation of outputs.
            missing = [a for a in self.outputs[key] if a not in ledger.storage]
            reserved = []
            for aid in missing:
                if not self._reserve_data(aid, node.id):
                    for allocated in reserved:
                        ledger.storage.pop(allocated, None)
                    stage.reason = "storage"
                    break
                reserved.append(aid)
            else:
                ledger.reserve_memory(key, spec.memory_bytes)
                if key in self.execs:
                    self.execs[key].resume()
                else:
                    activity = sg.this_actor.exec_init(spec.flops)
                    activity.host = sg.Host.by_name(node.id)
                    activity.start()
                    self.execs[key] = activity
                    if spec.flops == 0:
                        activity.wait()
                stage.reason = None
                self._transition(stage, "RUNNING")
                busy[node.id] += 1
                changed = True
        return changed

    def settle(self):
        # Drain completions before expiring deadlines, but let simultaneous
        # arrivals compete with queued work before making scheduling decisions.
        while True:
            changed = self._reap()
            changed = self._deliver_completed() or changed
            if not changed:
                break
        self._expire_deadlines()
        for rid, req in sorted(self.requests.items()):
            state = self.request_states[rid]
            if state.status == "PENDING" and req.arrival_s <= self.now:
                state.status = "ACTIVE"
                for stage in self.stages.values():
                    if stage.request_id == rid:
                        stage.entered_s = self.now
                self.emit("request_arrived", rid)
                if not self.workflows[req.workflow].stages:
                    deliverables = set(self.outputs[rid])
                    if sum(self.artifacts[a].size_bytes for a in deliverables) > self.nodes[
                        req.receiver
                    ].storage_bytes or any(
                        not any(
                            n == req.receiver or (n, req.receiver) in self.routes
                            for n in self.artifacts[a].locations
                        )
                        for a in deliverables
                    ):
                        self._finish_request(rid, "REJECTED", "no_feasible_receiver")
                        continue
                if "admission" in self.plugins and not self.plugins["admission"].admit(
                    req, self.inspect()
                ):
                    self._finish_request(rid, "REJECTED", "admission_policy")
        self._settle_active()
        if self._expire_deadlines():
            self._settle_active()
        self._collect_data()

    def _settle_active(self):
        # Zero-work activities can complete without advancing time; drain to a fixed point.
        for _ in range(len(self.stages) + len(self.artifacts) + 2):
            changed = False
            placements = []
            for key, stage in sorted(self.stages.items()):
                if not self._active(stage.request_id):
                    continue
                if stage.status == "WAITING_DEPENDENCIES" and all(
                    self.stages[d].status == "SUCCEEDED" for d in self.dependencies[key]
                ):
                    self._transition(stage, "READY")
                    changed = True
                if stage.status == "READY":
                    candidates = self._candidates(key)
                    if not candidates:
                        self._finish_request(stage.request_id, "REJECTED", "no_feasible_node")
                        changed = True
                    elif not self.run.external:
                        from edge_sim_models import Place

                        index = self.round_robin if self.run.policy.name == "round_robin" else 0
                        placements.append(
                            Place(
                                request_id=stage.request_id,
                                stage_id=stage.stage_id,
                                node_id=candidates[index % len(candidates)],
                            )
                        )
                        self.round_robin += 1
                        changed = True
            placements = [c for c in placements if self._active(c.request_id)]
            if placements:
                self.pending = self._decision()
                commands = tuple(placements)
                if "placement" in self.plugins:
                    commands = self.plugins["placement"].decide(self.pending)
                self.apply(self.pending.decision_id, commands)
            for key, stage in sorted(self.stages.items()):
                if not self._active(stage.request_id):
                    continue
                if stage.status == "WAITING_DATA":
                    ready = [self._ensure_data(a, stage.node_id, key) for a in self.inputs[key]]
                    if all(ready):
                        self._transition(stage, "QUEUED")
                        changed = True
            if "preemption" in self.plugins and self.preemption_revision != self.revision:
                commands = self.plugins["preemption"].decide(self.inspect())
                if commands:
                    from edge_sim_models import Resume, Suspend

                    if any(not isinstance(c, (Suspend, Resume)) for c in commands):
                        raise ValueError("Preemption policy must return Suspend/Resume commands")
                    self.pending = self._decision()
                    self.apply(self.pending.decision_id, commands)
                self.preemption_revision = self.revision
            changed = self._dispatch() or changed
            changed = self._reap() or changed
            changed = self._deliver_completed() or changed
            if not changed:
                break

    def _deliver_completed(self):
        changed = False
        for rid, req in sorted(self.requests.items()):
            if not self._active(rid):
                continue
            stages = [s for s in self.stages.values() if s.request_id == rid]
            if all(s.status == "SUCCEEDED" for s in stages):
                done = [self._ensure_data(a, req.receiver, rid) for a in self.outputs[rid]]
                if all(done):
                    self._finish_request(rid, "SUCCEEDED")
                    changed = True
        return changed

    def _expire_deadlines(self):
        changed = False
        for rid, req in sorted(self.requests.items()):
            if self._active(rid) and req.deadline_s is not None and self.now >= req.deadline_s:
                self._finish_request(rid, "TIMED_OUT", "deadline")
                changed = True
        return changed

    def _wait(self, date):
        date = min(date, min(self.transfer_timers.values(), default=math.inf))
        active = list(self.comms.values()) + [
            a for key, a in self.execs.items() if self.stages[key].status == "RUNNING"
        ]
        before = self.now
        transfer_remaining = {k: a.remaining for k, a in self.comms.items()}
        busy = Counter(s.node_id for s in self.stages.values() if s.status == "RUNNING")
        if active:
            try:
                activity_set = sg.ActivitySet(active)
                if math.isfinite(date):
                    activity_set.wait_any_for(max(0, date - self.now))
                else:
                    activity_set.wait_any()
            except sg.TimeoutException:
                pass
        elif math.isfinite(date):
            sg.this_actor.sleep_for(max(0, date - self.now))
        else:
            self.finished_reason = "deadlock"
            for rid in self.requests:
                if self._active(rid):
                    self._finish_request(rid, "FAILED", "deadlock")
        elapsed = self.now - before
        for node, count in busy.items():
            self.cpu_busy[node] += elapsed * count
        for key, amount in transfer_remaining.items():
            t = self.transfers[key]
            served = max(0, amount - self.comms[key].remaining)
            for link in self.routes[t.src, t.dst]:
                self.link_bytes[link] += served

    def close(self):
        for activity in [*self.execs.values(), *self.comms.values()]:
            activity.cancel()
        self.execs.clear()
        self.comms.clear()
        self.transfer_timers.clear()

    @staticmethod
    def _check_permutation(expected, actual):
        if len(expected) != len(actual) or set(expected) != set(actual):
            raise ValueError("Policy must return a permutation of supplied candidates")

    def inspect(self, selection=None):
        from edge_sim_models import (
            ArtifactState,
            Metric,
            NodeState,
            StateView,
        )
        from edge_sim_models import RequestState as RequestView
        from edge_sim_models import StageState as StageView

        selected = set(self.requests if selection is None else selection)
        unknown = selected - self.requests.keys()
        if unknown:
            raise CommandError("unknown_request", f"Unknown requests: {sorted(unknown)}")
        stages = []
        for key, state in sorted(self.stages.items()):
            if state.request_id not in selected:
                continue
            remaining = self.stage_specs[key].flops
            if key in self.execs:
                remaining = max(0, self.execs[key].remaining)
            elif state.status == "SUCCEEDED":
                remaining = 0
            durations = dict(state.durations)
            if state.finished_s is None and self._active(state.request_id):
                durations[state.status] = (
                    durations.get(state.status, 0) + self.now - state.entered_s
                )
            stages.append(
                StageView(
                    request_id=state.request_id,
                    stage_id=state.stage_id,
                    status=state.status,
                    node_id=state.node_id,
                    remaining_flops=remaining,
                    started_s=state.started_s,
                    completed_s=state.finished_s,
                    reason=state.reason,
                    durations=tuple(Metric(name=k, value=v) for k, v in sorted(durations.items())),
                )
            )
        return StateView(
            now_s=self.now,
            nodes=tuple(
                NodeState(
                    node_id=node,
                    memory_used_bytes=ledger.memory_used,
                    storage_used_bytes=ledger.storage_used,
                    cores_used=sum(
                        s.node_id == node and s.status == "RUNNING" for s in self.stages.values()
                    ),
                )
                for node, ledger in sorted(self.ledgers.items())
            ),
            artifacts=tuple(
                ArtifactState(
                    artifact_id=aid,
                    size_bytes=a.size_bytes,
                    cacheable=a.cacheable,
                    locations=tuple(sorted(n for x, n in self.replicas if x == aid)),
                )
                for aid, a in sorted(self.artifacts.items())
                if any(x == aid for x, _ in self.replicas)
            ),
            stages=tuple(stages),
            requests=tuple(
                RequestView(
                    request_id=rid,
                    status=s.status,
                    arrival_s=self.requests[rid].arrival_s,
                    deadline_s=self.requests[rid].deadline_s,
                    completed_s=s.finished_s,
                    reason=s.reason,
                )
                for rid, s in sorted(self.request_states.items())
                if rid in selected
            ),
        )

    def _decision(self):
        from edge_sim_models import Candidate, DecisionRequest

        return DecisionRequest(
            decision_id=f"{self.run.run_id}:{self.revision}",
            run_id=self.run.run_id,
            revision=self.revision,
            time_s=self.now,
            view=self.inspect(),
            candidates=tuple(
                Candidate(request_id=s.request_id, stage_id=s.stage_id, nodes=self._candidates(key))
                for key, s in sorted(self.stages.items())
                if s.status == "READY"
            ),
        )

    def advance(self, until_time=None):
        from edge_sim_models import AdvanceResult

        if until_time is not None and (not math.isfinite(until_time) or until_time < self.now):
            raise CommandError("invalid_time", "Time boundary must be finite and >= current time")
        if self.pending is not None:
            return AdvanceResult(
                kind="decision", time_s=self.now, view=self.inspect(), decision=self.pending
            )
        changed_since_decision = self.decision_dirty
        self.decision_dirty = False
        while True:
            revision = self.revision
            self.settle()
            changed_since_decision |= self.revision != revision
            if all(s.status not in {"PENDING", "ACTIVE"} for s in self.request_states.values()):
                self.finished_reason = self.finished_reason or "completed"
            if self.run.until_s is not None and self.now >= self.run.until_s:
                self.finished_reason = self.finished_reason or "truncated"
            if self.finished_reason:
                return AdvanceResult(
                    kind="finished", time_s=self.now, view=self.inspect(), result=self.result()
                )
            waking = self.wake_s is not None and self.now >= self.wake_s
            if waking:
                self.wake_s = None
            if (
                self.run.external
                and (waking or changed_since_decision)
                and (self.wake_s is None)
                and any(self._active(r) for r in self.requests)
            ):
                self.pending = self._decision()
                return AdvanceResult(
                    kind="decision", time_s=self.now, view=self.inspect(), decision=self.pending
                )
            if until_time is not None and self.now >= until_time:
                return AdvanceResult(kind="time", time_s=self.now, view=self.inspect())
            dates = [
                r.arrival_s
                for rid, r in self.requests.items()
                if self.request_states[rid].status == "PENDING"
            ]
            dates += [
                r.deadline_s
                for rid, r in self.requests.items()
                if self._active(rid) and r.deadline_s is not None
            ]
            dates += [
                s.ready_after_s
                for s in self.stages.values()
                if s.status == "QUEUED" and s.ready_after_s > self.now
            ]
            dates += [t for t in (until_time, self.run.until_s, self.wake_s) if t is not None]
            date = min((t for t in dates if t > self.now), default=math.inf)
            previous = self.now
            self._wait(date)
            changed_since_decision |= self.now != previous

    def apply(self, decision_id, commands):
        from edge_sim_models import CommandRecord, Defer, Place, Reject, Resume, Suspend

        if self.pending is None or decision_id != self.pending.decision_id:
            raise CommandError("stale_decision", "Decision is missing, stale, or already consumed")
        if not commands:
            raise CommandError("empty_commands", "Provide actions or an explicit Defer")
        touched = set()
        rejected = {c.request_id for c in commands if isinstance(c, Reject)}
        defer_count = 0
        # Validate the entire batch before any resource, activity or log mutation.
        for command in commands:
            if isinstance(command, Defer):
                defer_count += 1
                if defer_count > 1 or command.until_s <= self.now:
                    raise CommandError("invalid_defer", "Defer must specify one future time")
                continue
            if isinstance(command, Reject):
                key = command.request_id
                if key not in self.requests or not self._active(key):
                    raise CommandError("invalid_request", "Can only reject active requests")
            elif isinstance(command, (Place, Suspend, Resume)):
                key = f"{command.request_id}/{command.stage_id}"
                if key not in self.stages or command.request_id in rejected:
                    raise CommandError("invalid_stage", "Unknown stage or conflicting rejection")
                stage = self.stages[key]
                if isinstance(command, Place):
                    if stage.status != "READY" or command.node_id not in self._candidates(key):
                        raise CommandError(
                            "invalid_placement", "Stage or target node is not eligible"
                        )
                elif isinstance(command, Suspend) and stage.status != "RUNNING":
                    raise CommandError("invalid_suspend", "Only running stages can be suspended")
                elif isinstance(command, Resume) and stage.status != "SUSPENDED":
                    raise CommandError("invalid_resume", "Only suspended stages can be resumed")
            else:
                raise CommandError("unknown_command", "Use a typed domain command")
            if key in touched:
                raise CommandError(
                    "conflicting_commands", "A target may appear only once per batch"
                )
            touched.add(key)
        for command in commands:
            if isinstance(command, Place):
                self._place(f"{command.request_id}/{command.stage_id}", command.node_id)
            elif isinstance(command, Suspend):
                key = f"{command.request_id}/{command.stage_id}"
                self.execs[key].suspend()
                self._transition(self.stages[key], "SUSPENDED")
                self.stages[key].ready_after_s = self.now + self.run.pause_overhead_s
            elif isinstance(command, Resume):
                key = f"{command.request_id}/{command.stage_id}"
                stage = self.stages[key]
                stage.ready_after_s = (
                    max(self.now, stage.ready_after_s) + self.run.resume_overhead_s
                )
                self._transition(stage, "QUEUED")
            elif isinstance(command, Reject):
                self._finish_request(command.request_id, "REJECTED", command.reason)
            elif isinstance(command, Defer):
                self.wake_s = command.until_s
        self.command_log.append(
            CommandRecord(
                run_id=self.run.run_id,
                decision_id=decision_id,
                time_s=self.now,
                revision=self.pending.revision,
                commands=tuple(commands),
                view=self.pending.view if self.run.trace else None,
                policy_id="external" if self.run.external else self.run.policy.name,
            )
        )
        self.pending = None
        self.decision_dirty = True
        self.emit("decision_applied", decision_id)

    def result(self):
        import platform

        from edge_sim_models import Metric, RunManifest, RunMetrics, RunResult

        statuses = Counter(s.status for s in self.request_states.values())
        view = self.inspect()
        metrics = RunMetrics(
            arrived=sum(s.status != "PENDING" for s in self.request_states.values()),
            completed=statuses["SUCCEEDED"],
            rejected=statuses["REJECTED"],
            failed=statuses["FAILED"],
            timed_out=statuses["TIMED_OUT"],
            unfinished=statuses["ACTIVE"] + statuses["PENDING"],
            cache_hits=self.cache_hits,
            transfers=self.transfer_count,
            events=self.event_count,
            cpu_utilization=tuple(
                Metric(
                    name=n.id, value=(self.cpu_busy[n.id] / (self.now * n.cores) if self.now else 0)
                )
                for n in self.scenario.nodes
            ),
            link_utilization=tuple(
                Metric(
                    name=link.id,
                    value=(
                        self.link_bytes[link.id] / (self.now * link.bandwidth_bytes_s)
                        if self.now
                        else 0
                    ),
                )
                for link in self.scenario.links
            ),
            request_latency=tuple(
                Metric(name=rid, value=s.finished_s - self.requests[rid].arrival_s)
                for rid, s in self.request_states.items()
                if s.status == "SUCCEEDED"
            ),
            stage_durations=tuple(
                Metric(name=f"{s.request_id}/{s.stage_id}/{m.name}", value=m.value)
                for s in view.stages
                for m in s.durations
            ),
        )
        manifest = RunManifest(
            run_id=self.run.run_id,
            run_spec=self.run,
            seed=self.run.seed,
            scenario_hash=hashlib.sha256(self.scenario.model_dump_json().encode()).hexdigest(),
            python_version=platform.python_version(),
            simgrid_version="4.1",
            policy=self.run.policy.name,
            policy_plugins=self.run.policy.plugins,
            external_policy=self.run.external,
        )
        return RunResult(
            run_id=self.run.run_id,
            seed=self.run.seed,
            now_s=self.now,
            state=view,
            events=tuple(self.events),
            completed=self.finished_reason == "completed",
            end_reason=self.finished_reason or "in_progress",
            metrics=metrics,
            manifest=manifest,
        )
