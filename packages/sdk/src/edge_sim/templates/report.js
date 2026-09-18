/* No network, dependencies, or engine access. All labels use textContent. */
(() => {
  'use strict';
  const data = JSON.parse(document.getElementById('edge-sim-data').textContent);
  const {scenario: sc, result, run, commands} = data;
  const $ = id => document.getElementById(id);
  const el = (tag, text, cls) => {
    const node = document.createElement(tag);
    if (text !== undefined) node.textContent = text;
    if (cls) node.className = cls;
    return node;
  };
  const fmt = n => n == null ? '—' : Number(n).toLocaleString('en-US', {maximumSignificantDigits: 6});
  const seconds = n => n == null ? '—' : `${fmt(n)} s`;
  const bytes = n => {
    if (n == null) return '—';
    const units = ['B', 'KB', 'MB', 'GB', 'TB'];
    let i = 0;
    while (n >= 1000 && i < units.length - 1) { n /= 1000; i++; }
    return `${fmt(n)} ${units[i]}`;
  };
  const empty = (node, text) => node.replaceChildren(el('div', text, 'empty'));
  const detail = (title, value) => {
    $('detail-title').textContent = title;
    $('detail-json').textContent = JSON.stringify(value, null, 2);
    $('detail').showModal();
  };
  $('close-detail').onclick = () => $('detail').close();
  // Paginate every data table so large experiments do not create huge DOM trees.
  function table(target, headers, rows, open, message = 'No records available.') {
    target.replaceChildren();
    if (!rows.length) { empty(target, message); return; }
    let page = 0;
    const size = 50;
    const wrap = el('div', undefined, 'table-wrap');
    const grid = el('table');
    const head = el('thead');
    const hr = el('tr');
    headers.forEach(h => hr.append(el('th', h)));
    head.append(hr);
    const body = el('tbody');
    grid.append(head, body); wrap.append(grid); target.append(wrap);
    const pager = el('div', undefined, 'pager');
    const previous = el('button', '← Previous'), next = el('button', 'Next →'), label = el('span');
    pager.append(label, previous, next); target.append(pager);
    function render() {
      body.replaceChildren();
      rows.slice(page * size, (page + 1) * size).forEach((row, offset) => {
        const tr = el('tr');
        row.forEach(value => tr.append(el('td', value ?? '—')));
        if (open) {
          tr.className = 'clickable'; tr.tabIndex = 0;
          tr.setAttribute('role', 'button');
          const activate = () => open(page * size + offset);
          tr.onclick = activate;
          tr.onkeydown = e => { if (e.key === 'Enter' || e.key === ' ') { e.preventDefault(); activate(); } };
        }
        body.append(tr);
      });
      label.textContent = `${page * size + 1}–${Math.min((page + 1) * size, rows.length)} of ${rows.length}`;
      previous.disabled = page === 0; next.disabled = (page + 1) * size >= rows.length;
    }
    previous.onclick = () => { page--; render(); };
    next.onclick = () => { page++; render(); };
    render();
  }
  const specTable = (id, headings, items, cells) => table($(id), headings, items.map(cells), i => detail(items[i].id || 'Configuration', items[i]));
  function options(target, values) {
    values.forEach(value => { const option = el('option', value); option.value = value; target.append(option); });
  }
  function stats(items) {
    items.forEach(([label, value]) => {
      const card = el('div', undefined, 'stat');
      card.append(el('strong', value), el('span', label)); $('summary').append(card);
    });
  }
  $('title').textContent = result?.run_id || run?.run_id || 'Scenario configuration';
  $('subtitle').textContent = result
    ? `${result.end_reason} · ${seconds(result.now_s)} simulated · Seed ${result.seed} · ${run?.external ? 'External policy' : run?.policy.name || result.manifest?.policy || 'Policy unavailable'}`
    : 'Explore the infrastructure, data dependencies, and request schedule.';
  stats(result ? [
    ['Completed requests', fmt(result.metrics.completed)], ['Arrived', fmt(result.metrics.arrived)],
    ['Timed out', fmt(result.metrics.timed_out)], ['Rejected / failed', `${result.metrics.rejected} / ${result.metrics.failed}`],
    ['Unfinished', fmt(result.metrics.unfinished)], ['Cache hits', fmt(result.metrics.cache_hits)], ['Transfers', fmt(result.metrics.transfers)]]
    : [['Nodes', sc?.nodes.length || 0], ['Directed routes', sc?.routes.length || 0], ['Workflows', sc?.workflows.length || 0], ['Requests', sc?.requests.length || 0]]);
  document.querySelectorAll('[data-tab]').forEach(button => {
    button.onclick = () => {
      document.querySelectorAll('[data-tab]').forEach(b => {
        const active = b === button;
        b.classList.toggle('active', active); b.setAttribute('aria-pressed', String(active));
        $(b.dataset.tab).hidden = !active;
      });
    };
  });
  $('download').onclick = () => {
    const url = URL.createObjectURL(new Blob([JSON.stringify(result || run || sc, null, 2)], {type: 'application/json'}));
    const a = el('a'); a.href = url; a.download = result ? 'result.json' : run ? 'run.json' : 'scenario.json'; a.click();
    setTimeout(() => URL.revokeObjectURL(url), 1000);
  };
  const svgNS = 'http://www.w3.org/2000/svg';
  const svgEl = (tag, attrs, text) => {
    const node = document.createElementNS(svgNS, tag);
    Object.entries(attrs || {}).forEach(([key, value]) => node.setAttribute(key, value));
    if (text !== undefined) node.textContent = text;
    return node;
  };
  // Graph positions are derived from topology/DAG layers, never from user markup.
  function graph(target, nodes, edges, width, height) {
    target.replaceChildren();
    if (!nodes.length) { empty(target, 'No graph to display.'); return; }
    const svg = svgEl('svg', {width, height, viewBox: `0 0 ${width} ${height}`, role: 'img', 'aria-label': target.id === 'dag' ? 'Workflow dependency graph' : 'Directed network topology'});
    const markerId = `${target.id}-arrow`;
    const defs = svgEl('defs');
    const marker = svgEl('marker', {id: markerId, viewBox: '0 0 10 10', refX: 9, refY: 5, markerWidth: 6, markerHeight: 6, orient: 'auto-start-reverse'});
    marker.append(svgEl('path', {d: 'M 0 0 L 10 5 L 0 10 z', fill: '#91a592'})); defs.append(marker); svg.append(defs);
    const positions = new Map(nodes.map(n => [n.id, n]));
    for (const edge of edges) {
      const a = positions.get(edge.src), b = positions.get(edge.dst);
      if (!a || !b) continue;
      const dx = b.x - a.x, dy = b.y - a.y;
      const angle = Math.atan2(dy, dx), r = 50;
      const sx = a.x + Math.cos(angle) * r, sy = a.y + Math.sin(angle) * r;
      const ex = b.x - Math.cos(angle) * r, ey = b.y - Math.sin(angle) * r;
      const path = a === b ? `M ${a.x + 30} ${a.y - 26} C ${a.x + 110} ${a.y - 100},${a.x - 110} ${a.y - 100},${a.x - 30} ${a.y - 26}`
        : `M ${sx} ${sy} Q ${(sx + ex) / 2 - Math.sin(angle) * 22} ${(sy + ey) / 2 + Math.cos(angle) * 22} ${ex} ${ey}`;
      const line = svgEl('path', {d: path, stroke: '#b3c3b0', fill: 'none', 'stroke-width': 1.5, 'marker-end': `url(#${markerId})`});
      line.append(svgEl('title', {}, edge.label)); svg.append(line);
    }
    for (const n of nodes) {
      const g = svgEl('g', {transform: `translate(${n.x},${n.y})`, class: 'graph-node', tabindex: 0, role: 'button', 'aria-label': n.id});
      g.append(svgEl('rect', {x: -66, y: -27, width: 132, height: 54, rx: 9, fill: n.role === 'cloud' ? '#e6eddf' : n.role === 'client' ? '#f3eadb' : '#edf5ed', stroke: '#cbd9c8'}));
      g.append(svgEl('text', {'text-anchor': 'middle', y: -2}, n.id.length > 17 ? n.id.slice(0, 15) + '…' : n.id));
      g.append(svgEl('text', {'text-anchor': 'middle', y: 16, style: 'font-size:10px;fill:#6d7c76'}, n.caption));
      g.append(svgEl('title', {}, n.id));
      g.onclick = () => detail(n.id, n.raw);
      g.onkeydown = e => { if (e.key === 'Enter' || e.key === ' ') { e.preventDefault(); g.onclick(); } };
      svg.append(g);
    }
    target.append(svg);
  }
  if (sc) {
    const count = sc.nodes.length;
    const radius = Math.max(140, count * 25), width = Math.max(700, radius * 2 + 190), height = radius * 2 + 120;
    if (count <= 160) graph($('topology'), sc.nodes.map((n, i) => ({...n, raw: n, caption: `${n.role} · ${n.cores} cores`, x: width / 2 + radius * Math.cos(i / count * Math.PI * 2 - Math.PI / 2), y: height / 2 + radius * Math.sin(i / count * Math.PI * 2 - Math.PI / 2)})), sc.routes.map(r => ({...r, label: `${r.src} → ${r.dst}: ${r.links.join(' → ')}`})), width, height);
    else empty($('topology'), 'Topology exceeds 160 nodes. Use the complete node and route tables below.');
    $('topology-note').textContent = `${count} nodes · ${sc.links.length} links · Click nodes for details`;
    specTable('nodes', ['Node', 'Role', 'Cores', 'FLOP/s', 'Memory', 'Storage'], sc.nodes, n => [n.id, n.role, n.cores, fmt(n.speed_flops), bytes(n.memory_bytes), bytes(n.storage_bytes)]);
    specTable('links', ['Link', 'Bandwidth', 'Latency'], sc.links, l => [l.id, `${bytes(l.bandwidth_bytes_s)}/s`, seconds(l.latency_s)]);
    specTable('routes', ['From → To', 'Ordered links'], sc.routes, r => [`${r.src} → ${r.dst}`, r.links.join(' → ')]);
    specTable('artifacts', ['Artifact', 'Size', 'Replicas', 'Cacheable'], sc.artifacts, a => [a.id, bytes(a.size_bytes), a.locations.join(', '), a.cacheable ? 'Yes' : 'No']);
    specTable('requests', ['Request', 'Workflow', 'Arrival', 'Deadline', 'Receiver', 'Input bindings'], sc.requests, r => [r.id, r.workflow, seconds(r.arrival_s), seconds(r.deadline_s), r.receiver, r.input_bindings.map(b => `${b.artifact_id} ← ${b.global_artifact_id}`).join(', ')]);
    options($('workflow'), sc.workflows.map(w => w.id));
    function workflow() {
      const w = sc.workflows.find(w => w.id === $('workflow').value);
      if (!w) { ['dag', 'stages', 'workflow-artifacts'].forEach(id => empty($(id), 'No workflows configured.')); return; }
      const producers = new Map(w.artifacts.filter(a => a.producer_stage).map(a => [a.id, a.producer_stage]));
      const edges = [], depths = new Map(), pending = new Map(w.stages.map(s => [s.id, s]));
      for (const s of w.stages) {
        const parents = new Set([...s.depends_on, ...s.inputs.map(a => producers.get(a)).filter(Boolean)]);
        for (const p of parents) edges.push({src: p, dst: s.id, label: `${p} → ${s.id}`});
      }
      while (pending.size) {
        let progress = false;
        for (const [id] of pending) {
          const parents = edges.filter(e => e.dst === id).map(e => e.src);
          if (parents.every(p => depths.has(p))) { depths.set(id, Math.max(-1, ...parents.map(p => depths.get(p))) + 1); pending.delete(id); progress = true; }
        }
        if (!progress) break;
      }
      const lanes = new Map();
      const nodes = w.stages.map(s => { const depth = depths.get(s.id) || 0, lane = lanes.get(depth) || 0; lanes.set(depth, lane + 1); return {id: s.id, raw: s, x: 100 + depth * 210, y: 70 + lane * 100, caption: `${fmt(s.flops)} FLOPs`}; });
      if (nodes.length <= 200) graph($('dag'), nodes, edges, Math.max(700, 220 + Math.max(0, ...depths.values()) * 210), Math.max(170, 60 + Math.max(0, ...lanes.values()) * 100));
      else empty($('dag'), 'Workflow exceeds 200 stages. Use the stage table below.');
      if (!nodes.length) empty($('dag'), 'Content-only workflow: inputs are delivered directly to the receiver.');
      specTable('stages', ['Stage', 'FLOPs', 'Memory', 'Inputs → Outputs', 'Dependencies', 'Eligible nodes'], w.stages, s => [s.id, fmt(s.flops), bytes(s.memory_bytes), `${s.inputs.join(', ') || '—'} → ${s.outputs.join(', ') || '—'}`, edges.filter(e => e.dst === s.id).map(e => e.src).join(', ') || '—', s.eligible_nodes.join(', ') || 'All feasible nodes']);
      specTable('workflow-artifacts', ['Artifact', 'Size', 'Producer', 'Cacheable'], w.artifacts, a => [a.id, bytes(a.size_bytes), a.producer_stage || 'External input', a.cacheable ? 'Yes' : 'No']);
    }
    $('workflow').onchange = workflow; workflow();
  } else {
    ['topology', 'nodes', 'links', 'routes', 'dag', 'stages', 'workflow-artifacts', 'artifacts', 'requests'].forEach(id => empty($(id), 'Configuration unavailable: this result does not contain a full run manifest.'));
    $('workflow').disabled = true;
  }
  if (run) {
    const settings = el('dl');
    [['Run', run.run_id], ['Seed', run.seed], ['Policy', `${run.policy.name} v${run.policy.version}`], ['Control', run.external ? 'External' : 'Built-in'], ['Trace recording', run.trace ? 'Enabled' : 'Disabled'], ['Time limit', seconds(run.until_s)], ['Pause overhead', seconds(run.pause_overhead_s)], ['Resume overhead', seconds(run.resume_overhead_s)], ['Plugins', run.policy.plugins.map(p => `${p.role}: ${p.factory} (${p.version})`).join('\n') || 'None']].forEach(([k, v]) => settings.append(el('dt', k), el('dd', v)));
    $('settings').append(settings);
  } else empty($('settings'), 'No run settings supplied.');

  // Historical emit() stores fields in details; typed top-level fields are also supported.
  const events = (result?.events || []).map((e, index) => {
    const fields = Object.fromEntries(e.details.map(d => [d.name, d.value]));
    for (const [k, v] of Object.entries(e)) if (v != null && k !== 'details') fields[k] = v;
    const entity = e.entity_id || (e.request_id && e.stage_id ? `${e.request_id}/${e.stage_id}` : e.request_id || e.artifact_id || '');
    const stageKey = e.kind === 'stage_state' ? entity : null;
    const request = e.request_id || (stageKey ? stageKey.split('/')[0] : e.kind.startsWith('request_') ? entity : null);
    return {...fields, entity, stageKey, request, index, raw: e};
  }).sort((a, b) => a.time_s - b.time_s || a.sequence - b.sequence || a.index - b.index);
  const stageStates = result?.state.stages || [];
  const stageMap = new Map(stageStates.map(s => [`${s.request_id}/${s.stage_id}`, s]));
  const colors = {WAITING_DEPENDENCIES: '#cbd5df', READY: '#9cb6cc', WAITING_DATA: '#e3b65c', QUEUED: '#b4c784', RUNNING: '#23755c', SUSPENDED: '#a78cba', SUCCEEDED: '#55a985', CANCELLED: '#d67c6d', FAILED: '#b74343'};
  for (const [status, color] of Object.entries(colors)) {
    const item = el('span'), swatch = el('i', undefined, 'swatch'); swatch.style.background = color;
    item.append(swatch, document.createTextNode(status)); $('legend').append(item);
  }
  const stageEvents = new Map();
  for (const event of events) if (event.stageKey) {
    if (!stageEvents.has(event.stageKey)) stageEvents.set(event.stageKey, []);
    stageEvents.get(event.stageKey).push(event);
  }
  const transfers = [], activeTransfers = new Map();
  for (const event of events) {
    const key = JSON.stringify([event.entity, event.src, event.dst]);
    if (event.kind === 'transfer_started') {
      const transfer = {artifact: event.entity, src: event.src, dst: event.dst, size_bytes: event.size_bytes ?? event.bytes, start_s: event.time_s, end_s: null, start_event: event.raw};
      transfers.push(transfer); activeTransfers.set(key, transfer);
    } else if (event.kind === 'transfer_finished') {
      const transfer = activeTransfers.get(key);
      if (transfer) { transfer.end_s = event.time_s; transfer.finish_event = event.raw; activeTransfers.delete(key); }
      else transfers.push({artifact: event.entity, src: event.src, dst: event.dst, start_s: null, end_s: event.time_s, finish_event: event.raw});
    }
  }
  options($('request-filter'), [...new Set([...(sc?.requests || []).map(r => r.id), ...(result?.state.requests || []).map(r => r.request_id), ...stageStates.map(s => s.request_id), ...events.map(e => e.request).filter(Boolean)])].sort());
  options($('node-filter'), [...new Set([...(sc?.nodes || []).map(n => n.id), ...(result?.state.nodes || []).map(n => n.node_id), ...events.flatMap(e => [e.node_id, e.src, e.dst]).filter(Boolean)])].sort());
  options($('event-filter'), [...new Set(events.map(e => e.kind))].sort());
  $('trace-notice').textContent = !result ? 'Configuration only. Export a RunResult to explore its trace.'
    : !events.length ? 'No event trace is available. Aggregate metrics and the final snapshot are shown; execution intervals cannot be reconstructed.'
    : `${events.length} recorded events · ${commands.length} decision batches. Timeline uses recorded transitions only; metrics and final snapshot are not historical replay.`;
  function meter(target, metrics) {
    if (!metrics?.length) { empty(target, 'No utilization metrics.'); return; }
    for (const m of metrics) {
      const row = el('div', undefined, 'meter-row'), bar = el('div', undefined, 'meter'), fill = el('span');
      fill.style.width = `${m.value * 100}%`; bar.append(fill);
      row.append(el('span', m.name), bar, el('span', `${fmt(m.value * 100)}%`)); target.append(row);
    }
  }
  meter($('cpu'), result?.metrics.cpu_utilization); meter($('network'), result?.metrics.link_utilization);
  function renderTrace() {
    const req = $('request-filter').value, node = $('node-filter').value, search = $('search').value.toLowerCase();
    const matches = value => !search || JSON.stringify(value).toLowerCase().includes(search);
    const end = Math.max(result?.now_s || 0, events.at(-1)?.time_s || 0, 0.000001);
    const timeline = $('timeline'); timeline.replaceChildren();
    const lanes = [...stageEvents].filter(([key, list]) => (!req || list[0].request === req) && (!node || stageMap.get(key)?.node_id === node || list.some(e => e.node_id === node)) && matches([key, stageMap.get(key), list]));
    if (!lanes.length) empty(timeline, 'No stage transitions match these filters.');
    else {
      const chart = el('div', undefined, 'timeline');
      chart.style.width = `${100 * Number($('zoom').value)}%`;
      const axisRow = el('div', undefined, 'time-row'), axis = el('div', undefined, 'axis');
      for (let i = 0; i <= 5; i++) axis.append(el('span', seconds(end * i / 5)));
      axisRow.append(el('span', 'Stage / assigned node', 'lane-label'), axis); chart.append(axisRow);
      // Bound DOM work; filters narrow lanes, and the final snapshot is paginated below.
      for (const [key, list] of lanes.slice(0, 300)) {
        const row = el('div', undefined, 'time-row'), label = el('div', `${key} · ${stageMap.get(key)?.node_id || 'unassigned'}`, 'lane-label');
        label.title = label.textContent;
        const track = el('div', undefined, 'track');
        list.forEach((event, i) => {
          const terminal = ['SUCCEEDED', 'CANCELLED', 'FAILED'].includes(event.status);
          const stop = terminal ? event.time_s : list[i + 1]?.time_s ?? result.now_s;
          const info = {stage: key, status: event.status, start_s: event.time_s, end_s: stop, duration_s: stop - event.time_s, node: stageMap.get(key)?.node_id, event: event.raw};
          const segment = el('button', undefined, 'segment');
          segment.style.left = `${Math.min(99.7, event.time_s / end * 100)}%`;
          segment.style.width = `${Math.max(0, (stop - event.time_s) / end * 100)}%`;
          segment.style.background = colors[event.status] || '#899c99';
          segment.title = `${key} · ${event.status} · ${seconds(event.time_s)} → ${seconds(stop)}`;
          segment.setAttribute('aria-label', segment.title); segment.onclick = () => detail('Stage interval', info); track.append(segment);
        });
        row.append(label, track); chart.append(row);
      }
      timeline.append(chart);
      if (lanes.length > 300) timeline.append(el('p', `Showing 300 of ${lanes.length} lanes. Filter by request or node to narrow the timeline.`, 'muted'));
    }
    const requests = (result?.state.requests || []).filter(r => (!req || r.request_id === req) && (!node || stageStates.some(s => s.request_id === r.request_id && s.node_id === node) || sc?.requests.some(s => s.id === r.request_id && s.receiver === node)) && matches(r));
    table($('request-state'), ['Request', 'Status', 'Arrival', 'Deadline', 'Completed', 'Reason'], requests.map(r => [r.request_id, r.status, seconds(r.arrival_s), seconds(r.deadline_s), seconds(r.completed_s), r.reason]), i => detail('Request snapshot', requests[i]));
    const stages = stageStates.filter(s => (!req || s.request_id === req) && (!node || s.node_id === node) && matches(s));
    table($('stage-state'), ['Stage', 'Node', 'Status', 'Started', 'Completed', 'Remaining FLOPs', 'Reason'], stages.map(s => [`${s.request_id}/${s.stage_id}`, s.node_id, s.status, seconds(s.started_s), seconds(s.completed_s), fmt(s.remaining_flops), s.reason]), i => detail('Stage snapshot & durations', stages[i]));
    // Only request-scoped output artifacts have reliable ownership; global transfers stay visible.
    const shared = new Set((sc?.artifacts || []).map(a => a.id));
    const filteredTransfers = transfers.filter(t => (!req || shared.has(t.artifact) || !t.artifact.includes('/') || t.artifact.startsWith(req + '/')) && (!node || t.src === node || t.dst === node) && matches(t));
    table($('transfers'), ['Artifact', 'Route', 'Size', 'Start', 'End', 'Duration'], filteredTransfers.map(t => [t.artifact, `${t.src} → ${t.dst}`, bytes(t.size_bytes), t.start_s == null ? 'Unknown' : seconds(t.start_s), t.end_s == null ? 'Unknown (no end event)' : seconds(t.end_s), t.start_s != null && t.end_s != null ? seconds(t.end_s - t.start_s) : 'Unknown']), i => detail('Transfer', filteredTransfers[i]));
    const kind = $('event-filter').value;
    const filteredEvents = events.filter(e => (!kind || e.kind === kind) && (!req || e.request === req || e.entity.startsWith(req + '/') || e.kind.startsWith('transfer_') && (shared.has(e.entity) || !e.entity.includes('/'))) && (!node || [e.node_id, e.src, e.dst, stageMap.get(e.stageKey)?.node_id].includes(node)) && matches(e.raw));
    table($('events'), ['Sequence', 'Time', 'Event', 'Entity', 'Details'], filteredEvents.map(e => [e.sequence, seconds(e.time_s), e.kind, e.entity, JSON.stringify(e.raw.details)]), i => detail('Event', filteredEvents[i].raw));
    const filteredCommands = commands.filter(c => c.commands.some(action => (!req || action.request_id === req || action.kind === 'defer') && (!node || action.node_id === node || stageMap.get(`${action.request_id}/${action.stage_id}`)?.node_id === node || action.kind === 'defer')) && matches(c));
    table($('commands'), ['Time', 'Decision', 'Policy', 'Commands'], filteredCommands.map(c => [seconds(c.time_s), c.decision_id, c.policy_id, c.commands.map(a => `${a.kind}${a.request_id ? ' ' + a.request_id : ''}${a.stage_id ? '/' + a.stage_id : ''}${a.node_id ? ' → ' + a.node_id : ''}${a.until_s != null ? ' until ' + seconds(a.until_s) : ''}`).join('; ')]), i => detail('Decision & state snapshot', filteredCommands[i]), 'No decision records. Pass --commands commands.json to include them.');
  }
  ['request-filter', 'node-filter', 'event-filter', 'zoom'].forEach(id => $(id).onchange = renderTrace);
  let searchTimer;
  $('search').oninput = () => { clearTimeout(searchTimer); searchTimer = setTimeout(renderTrace, 120); };
  $('reset').onclick = () => { ['request-filter', 'node-filter', 'event-filter', 'search'].forEach(id => $(id).value = ''); $('zoom').value = '1'; renderTrace(); };
  renderTrace();
})();
