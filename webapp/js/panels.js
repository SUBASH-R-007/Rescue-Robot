// Right-hand panels of the Simulate view: "What happened", "Plan", "Numbers", "Settings",
// plus the map legend. They only render state; every change goes through callbacks.
import {
  escapeHtml, fmtPos, fmtNum, fmtMs, fmtPct, fmtPctAuto, statusInfo, outcomeText, normOutcome,
  ACTION_ARROWS, ACTION_NAMES, samePos, humanize, CELL, outcomeSplit, tonedSentences, toneStyle,
} from './format.js';
import { PlanTreeView, countNodes } from './plantree.js';
import { icon, infoTip, swatch } from './ui.js';
import { MapRenderer } from './render.js';

// --- explanation ---------------------------------------------------------------------------
/** The explanation list items of a state: each sentence with its tone's glyph (the tone comes
 * from Python, explanation_tones; shape + colour, never colour alone). */
export function explanationItems(state) {
  const sentences = tonedSentences(state);
  if (!sentences.length) sentences.push({ text: 'Nothing happened yet.', tone: 'info' });
  return sentences.map(({ text, tone }) => {
    const { glyph, cls } = toneStyle(tone);
    return `<li><span class="ex-ico tone-${cls}" aria-hidden="true">${glyph}</span><span>${escapeHtml(text)}</span></li>`;
  }).join('');
}

/** The badge of the last move (arrow, direction and outcome), or '' at t = 0 / while replaying
 * a state without its own move. */
export function moveBadge(state) {
  const last = state.trace?.length ? state.trace[state.trace.length - 1] : null;
  if (!last || last.t !== state.t) return '';
  const o = normOutcome(last.outcome);
  const tone = o === 'slipped' ? 'warn' : o?.startsWith('skid') ? 'bad' : last.bump ? 'bad' : 'info';
  return `<span class="badge tone-${tone}">${ACTION_ARROWS[last.action] ?? ''} ${escapeHtml(ACTION_NAMES[last.action] ?? last.action ?? '')} · ${escapeHtml(outcomeText(last.outcome, last.blocked))}</span>`;
}

const LOG_GLYPH = { slip: '↺', skid: '▲', replan: '◆', success: '✓', failure: '■', warn: '!', info: '•' };

// --- metrics -------------------------------------------------------------------------------
/** Metric labels and plain-English definitions (shown on hover). Order = display order. */
export const METRICS = [
  ['steps', 'Steps', 'Moves the robot has tried so far. Every attempt counts, even one where it slipped.'],
  ['rescued', 'Rescued', 'Survivors brought to safety so far.'],
  ['replans', 'Replans', 'Times the agent threw its plan away and planned again.'],
  ['slips', 'Slips', 'Moves where the wheels slipped and the robot stayed where it was.'],
  ['skids', 'Skids', 'Moves where the robot slid sideways (left or right) instead.'],
  ['slips_handled', 'Slips handled', 'Slips and skids the agent coped with without giving up.'],
  ['bumps', 'Collisions', 'Times the robot drove into debris it did not know about. With a working sensor AND-OR never does.'],
  ['blocked_moves', 'Blocked moves', 'Moves into a wall or into debris the robot already knew about. It just stays put.'],
  ['unplanned_states', 'Unplanned states', 'Times the robot ended up in a cell its plan had not prepared for.'],
  ['cells_explored', 'Cells explored', 'Cells the robot has seen with its sensor.'],
  ['first_detection_step', 'First found at step', 'Step at which the robot first saw a survivor (search mode).'],
  ['shortest_path', 'Shortest path', 'Moves an all-knowing robot without slips would need on the true map.'],
  ['wasted_moves', 'Wasted moves', 'Steps minus the shortest path (only once the mission succeeded).'],
  ['planning_ms', 'Planning time', 'Total time spent planning, measured here in your browser (WebAssembly is slower than desktop Python).'],
  ['nodes_expanded', 'Nodes expanded', 'Search nodes the planner looked at in total: a measure of planning effort.'],
  ['plan_size', 'Plan size', 'States covered by the current plan (AND-OR) or cells on the path (A*).'],
  ['aftershock_events', 'Aftershocks', 'Aftershocks so far: new debris that fell where the robot could not see.'],
  ['fallback_replans', 'Fallbacks', 'Times the noisy-sensor agent relaxed its map because suspected debris walled it in.'],
  ['planner_fallbacks', 'Planner fallbacks', 'Times AND-OR handed over to the reference planner (should stay 0).'],
];
const METRIC_INDEX = new Map(METRICS.map((m, i) => [m[0], i]));

/** Display value of a metric. */
export function metricValue(key, value, state) {
  if (key === 'steps' && state?.budget) return `${fmtNum(value)} / ${state.budget}`;
  if (key === 'rescued' && state?.survivors) return `${fmtNum(value)} / ${state.survivors.length}`;
  if (key === 'planning_ms') return fmtMs(value);
  if (key === 'wasted_moves') return value == null ? '–' : fmtNum(value, Number.isInteger(value) ? 0 : 1);
  if (typeof value === 'boolean') return value ? 'yes' : 'no';
  if (typeof value === 'number') return fmtNum(value, Number.isInteger(value) ? 0 : 2);
  return value == null ? '–' : String(value);
}

/** Metric rows shown for a metrics dict: the known ones (in order), no identity fields. */
export function metricRows(metrics) {
  const rows = [];
  for (const [key, value] of Object.entries(metrics ?? {})) {
    if (!METRIC_INDEX.has(key)) continue;
    if (key === 'first_detection_step' && value == null) continue;
    if ((key === 'fallback_replans' || key === 'planner_fallbacks' || key === 'aftershock_events') && !value) continue;
    const [, label, def] = METRICS[METRIC_INDEX.get(key)];
    rows.push({ key, label, def, value });
  }
  rows.sort((a, b) => METRIC_INDEX.get(a.key) - METRIC_INDEX.get(b.key));
  return rows;
}

// --- What happened -----------------------------------------------------------------------
/** The "What happened" panel. `log` is the whole recorded event log (while replaying an
 * earlier step it also lists the later events, dimmed, so they can be jumped to). */
export function renderWhat(root, state, { onSeek, log: fullLog = null }) {
  // A stable skeleton (built once) whose parts are updated: keyboard focus on a log line
  // survives every step, and the panel is no live region (announcements go through the
  // page's one status region, sim.js).
  if (!root.querySelector('.what-step')) {
    root.innerHTML = `
      <section class="panel-section what-step">
        <h3 class="sr-only">What happened</h3>
        <div class="step-head"></div>
        <ul class="explain"></ul>
      </section>
      <section class="panel-section">
        <h4 class="panel-title">${icon('chat')}Event log <span class="muted">(click a line to jump there)</span></h4>
        <div class="log-box"></div>
      </section>`;
    root.querySelector('.log-box').addEventListener('click', (ev) => {
      const b = ev.target.closest('button[data-t]');
      if (b) root._onSeek?.(Number(b.dataset.t));
    });
  }
  root._onSeek = onSeek;
  const st = statusInfo(state.status);
  let head = state.t === 0 ? '<h4 class="step-title">Before the first move</h4>' : `<h4 class="step-title">Step ${state.t}</h4>${moveBadge(state)}`;
  if (state.status !== 'running') head += `<span class="badge tone-${st.tone}">${st.glyph} ${escapeHtml(st.label)}</span>`;
  root.querySelector('.step-head').innerHTML = head;
  root.querySelector('.explain').innerHTML = explanationItems(state);
  const log = (fullLog ?? state.log ?? []);
  const logBox = root.querySelector('.log-box');
  const focused = logBox.contains(document.activeElement) ? document.activeElement.dataset.i : null;
  const scrollBefore = logBox.querySelector('.log')?.scrollTop ?? 0;
  const logItems = log.map((e, i) => `<li class="k-${escapeHtml(e.kind)}${e.t === state.t ? ' current' : ''}${e.t > state.t ? ' future' : ''}"><button type="button" data-t="${e.t}" data-i="${i}" title="Go to step ${e.t}"${e.t === state.t ? ' aria-current="step"' : ''}>
      <span class="lt">t=${e.t}</span><span class="lg" aria-hidden="true">${LOG_GLYPH[e.kind] ?? '•'}</span><span>${escapeHtml(e.display ?? e.text)}</span></button></li>`).join('');
  logBox.innerHTML = log.length ? `<ul class="log">${logItems}</ul>` : '<p class="muted">Slips, skids, replans and rescues will be listed here.</p>';
  const box = logBox.querySelector('.log');
  if (!box) return;
  if (focused != null) {
    // Keyboard users stay on the line they activated.
    box.querySelector(`button[data-i="${focused}"]`)?.focus({ preventScroll: true });
    box.scrollTop = scrollBefore;
    return;
  }
  const cur = box.querySelector('.current');
  if (cur) box.scrollTop = Math.max(0, cur.offsetTop - box.clientHeight / 2);
  else box.scrollTop = box.scrollHeight;
}

// --- Plan --------------------------------------------------------------------------------
export class PlanPanel {
  constructor(root, { onHoverCell, onShowArrows }) {
    this.root = root;
    this.onHoverCell = onHoverCell;
    this.onShowArrows = onShowArrows;
    this.view = null;
  }

  /** Forget the shown plan (a new mission starts; its plan ids start at 0 again). */
  reset() {
    this.view = null;
    this.root.innerHTML = '';
    this.onHoverCell(null);
  }

  /** The badges above a plan tree. */
  static planHead(plan) {
    const nodes = countNodes(plan.tree);
    return `<span class="badge tone-purple">Plan #${escapeHtml(plan.id)}</span>
        ${plan.t != null ? `<span class="badge plain">made at step ${plan.t}${plan.made_at ? ` in ${fmtPos(plan.made_at)}` : ''}</span>` : ''}
        <span class="badge plain">${nodes} nodes shown</span>
        ${plan.stats ? `<span class="badge plain" title="Size of the whole plan (the tree below is cut at depth 8)">${plan.stats.act_nodes} decisions · ${plan.stats.loops} loop-backs</span>` : ''}`;
  }

  render(state) {
    const kind = state.agent?.kind;
    const plan = state.plan;
    const path = state.path;
    const parts = [];
    const treeBox = this.root.querySelector('.plan-tree');
    if (plan?.tree && treeBox && this.view) {
      // Same layout as before: update in place, so the tree keeps keyboard focus.
      this.root.querySelector('.plan-head').innerHTML = PlanPanel.planHead(plan);
      this.root.querySelector('.plan-pretty').textContent = plan.pretty ?? '';
      this.view.show(plan.tree, `${plan.id}`, state.robot);
      return;
    }
    if (plan?.tree) {
      parts.push(`<div class="plan-head">${PlanPanel.planHead(plan)}</div>
        <p class="muted" style="margin-bottom:8px">A contingency plan has a branch for <b>every</b> outcome of every move:
        arrived, slipped (stays put) or skidded sideways. ↺ means “go back to an earlier step of the plan”.
        Hover a row (or move through it with the arrow keys) to see its cell on the map.</p>
        <div class="plan-pretty" title="The plan in the notation of the proposal deck">${escapeHtml(plan.pretty ?? '')}</div>
        <div class="plan-tree" role="tree" aria-label="Plan tree"></div>`);
    } else if (path?.length) {
      const here = path.findIndex((p) => samePos(p, state.robot));
      const chips = path.map((p, i) => `<span class="${i < here ? 'done' : i === here ? 'here' : ''}" data-r="${p[0]}" data-c="${p[1]}">${fmtPos(p)}</span>`).join('');
      parts.push(`<div class="plan-head"><span class="badge tone-info">Path: ${path.length - 1} moves</span></div>
        <p class="muted">A* plans one shortest path and assumes every move works. When something goes wrong
        ${state.agent?.name === 'astar_fixed' ? 'this agent gives up.' : 'it plans a new path from wherever the robot ended up.'}</p>
        <div class="path-list" aria-label="Planned path">${chips}</div>`);
    } else if (kind === 'expected' || Object.keys(state.policy ?? {}).length) {
      parts.push(`<p>This agent follows a <b>policy</b>: one best action per cell, chosen to need the fewest steps
        on average. It has no AND-OR tree; switch on the plan arrows to see the policy on the map.</p>
        <button class="btn small" type="button" data-arrows>${icon('arrows')}Show plan arrows</button>`);
    } else if (state.plan_message) {
      parts.push(`<p class="note warn">${escapeHtml(state.plan_message)}</p>`);
    } else if (state.status !== 'running') {
      parts.push(`<p class="muted">The mission is over: no plan is needed any more.</p>`);
    } else {
      parts.push('<p class="muted">No plan yet.</p>');
    }
    this.onHoverCell(null); // the old rows (and their highlight) are gone
    this.root.innerHTML = parts.join('');
    this.root.querySelector('[data-arrows]')?.addEventListener('click', () => this.onShowArrows());
    const pathBox = this.root.querySelector('.path-list');
    if (pathBox) {
      pathBox.addEventListener('pointerover', (ev) => {
        const chip = ev.target.closest('[data-r]');
        this.onHoverCell(chip ? [Number(chip.dataset.r), Number(chip.dataset.c)] : null);
      });
      pathBox.addEventListener('pointerleave', () => this.onHoverCell(null));
    }
    const box = this.root.querySelector('.plan-tree');
    if (box) {
      const keep = this.view;
      this.view = new PlanTreeView(box, { onHoverCell: this.onHoverCell });
      if (keep && keep.treeKey === `${plan.id}`) {
        this.view.treeKey = keep.treeKey;
        this.view.collapsed = keep.collapsed;
      }
      this.view.show(plan.tree, `${plan.id}`, state.robot);
    } else {
      this.view = null;
    }
  }
}

// --- Numbers -----------------------------------------------------------------------------
export function renderNumbers(root, state) {
  const rows = metricRows(state.metrics);
  const cards = rows.map((m) => `<div class="metric${m.key === 'steps' ? ' hl' : ''}">
      <div class="mv">${escapeHtml(metricValue(m.key, m.value, state))}</div>
      <div class="ml">${escapeHtml(m.label)}${infoTip(m.def, m.label)}</div></div>`).join('');
  const st = statusInfo(state.status);
  root.innerHTML = `
    <section class="panel-section">
      <div class="step-head"><span class="badge tone-${st.tone}">${st.glyph} ${escapeHtml(st.label)}</span>
      <span class="badge plain">${escapeHtml(state.agent?.label ?? state.agent?.name ?? '')}</span></div>
      <div class="metric-grid">${cards || '<p class="muted">No numbers yet.</p>'}</div>
    </section>
    <p class="muted" style="margin-top:16px;font-size:13px">These are the same metrics the experiment harness records for every mission
    (one row of <code>results/raw.csv</code>). Hover or focus ⓘ for a definition.</p>`;
}

// --- Settings ----------------------------------------------------------------------------
/** Tips of the settings; `slip` is built from the catalog (outcome split, terrain p). */
const TIPS = {
  mode: 'Rescue: the robot knows where the survivor is. The other modes are the extensions: an unknown location, several survivors, or a sensor that sometimes lies.',
  map: 'The disaster zone. Bundled maps come from the repository; generated maps are made by the scenario generators from a seed, so the same seed always gives the same map.',
  agent: 'The planning strategy that drives the robot. Try the same map and seed with different agents (or use the Compare tab).',
  radius: 'How far the robot sees (Manhattan distance, including its own cell). 1 = the four neighbours. 0 means it only finds debris by bumping into it.',
  replan: 'Eager: replan whenever the sensor finds debris the map did not expect. Lazy: replan only if that debris touches the current plan.',
  aftershock: 'Stress test: each step, with probability q, k new debris cells fall somewhere the robot cannot see (never cutting off the survivor).',
  seed: 'Decides the luck: which moves slip or skid. The same seed replays exactly the same mission, here and on the command line.',
  mapSeed: 'Decides how the generated map looks. Seeds 0–29 are pre-generated; other seeds run the generator in your browser.',
  k: 'Number of survivors to rescue. The robot visits the nearest known survivor first.',
  fn: 'False negative: the chance that the sensor reports real debris as open.',
  fp: 'False positive: the chance that the sensor reports an open cell as debris (a “phantom wall”).',
};

/** The slip tip, with the catalog's numbers (config.py), never copied ones. */
export function slipTip(params) {
  const f = params?.outcome_fractions ?? {};
  const high = params?.terrain_high_p?.value;
  return `How unreliable the wheels are. With slip p a move arrives with probability 1 − p, stays put with ${f.stay}·p and skids left or right with ${f.skid / 2}·p each. Water and mud (~) always use p = ${high}.`;
}

/** Spoken values of the settings sliders (aria-valuetext). */
export const SLIDER_TEXT = {
  slip_p: (v) => `slip p ${Number(v).toFixed(2)}`,
  sensor_radius: (v) => `${v} cell${Number(v) === 1 ? '' : 's'}`,
  aftershock_q: (v) => `${fmtPct(Number(v))} chance per step`,
  aftershock_k: (v) => `${v} debris cell${Number(v) === 1 ? '' : 's'} per aftershock`,
  k: (v) => `${v} survivor${Number(v) === 1 ? '' : 's'}`,
  false_negative: (v) => `${fmtPct(Number(v))} of real debris missed`,
  false_positive: (v) => `${fmtPct(Number(v))} of open cells reported as debris`,
};

/** "Title (r×c)" unless the title already names the size. */
export function withSize(title, rows, cols) {
  return String(title).includes(`${rows}×${cols}`) || String(title).includes(`${rows}x${cols}`) ? title : `${title} (${rows}×${cols})`;
}

/** Settings that belong to one mode only: reset to the new mode's defaults on a mode switch
 * (the core mode rejects them, and 5 survivors carried into "Noisy sensors" would surprise). */
const MODE_PARAMS = ['k', 'false_negative', 'false_positive'];

/** The config patch for switching `config` to `mode`: an agent of that mode and its defaults. */
export function modePatch(catalog, config, mode) {
  const agents = (catalog.agents ?? []).filter((a) => !a.modes || a.modes.includes(mode));
  const info = (catalog.modes ?? []).find((m) => m.name === mode);
  const agent = agents.some((a) => a.name === config.agent) ? config.agent : (info?.default_agent ?? agents[0]?.name);
  const patch = { mode, agent };
  for (const key of MODE_PARAMS) {
    // The core mode takes none of them; an extension mode starts from its own defaults.
    patch[key] = mode === 'core' ? undefined : (info?.defaults?.[key] ?? catalog.params?.[key]?.default);
  }
  return patch;
}

function rangeField(id, label, tip, { min, max, step, value, format, spoken = format }) {
  return `<div class="field"><label class="field-label" for="${id}">${escapeHtml(label)}${infoTip(tip, label)}</label>
    <div class="field-row"><input type="range" id="${id}" min="${min}" max="${max}" step="${step}" value="${value}" aria-valuetext="${escapeHtml(spoken(value))}">
    <output for="${id}" id="${id}-out" aria-hidden="true">${escapeHtml(format(value))}</output></div></div>`;
}

export class SettingsPanel {
  constructor(root, app, { onApply }) {
    this.root = root;
    this.app = app;
    this.onApply = onApply;
    this.note = '';
  }

  /** Render for `config` (the active mission's settings). */
  render(config, note = '') {
    const { catalog } = this.app;
    const P = catalog.params ?? {};
    const modes = catalog.modes ?? [{ name: 'core', title: 'Rescue' }];
    const mode = config.mode ?? 'core';
    const agents = (catalog.agents ?? []).filter((a) => !a.modes || a.modes.includes(mode));
    const map = config.map ?? {};
    const mapValue = map.source === 'scenario' ? `scenario:${map.scenario}` : map.source === 'ascii' ? 'ascii' : `file:${map.name}`;
    const scen = (catalog.scenarios ?? []).find((s) => s.name === map.scenario);
    const sizes = scen?.sizes ?? [10, 15, 20];
    const slip = config.slip_p ?? P.slip_p?.default ?? 0.2;
    const split = outcomeSplit(slip, P.outcome_fractions);
    const pct = fmtPctAuto;
    const range = (key, fallback) => ({ ...fallback, ...(P[key] ?? {}) });
    const rSlip = range('slip_p', { min: 0, max: 0.4, step: 0.05 });
    const rRad = range('sensor_radius', { min: 0, max: 3, step: 1 });
    const rQ = range('aftershock_q', { min: 0, max: 1, step: 0.05 });
    const rK = range('aftershock_k', { min: 0, max: 5, step: 1 });
    const rSurv = range('k', { min: 1, max: 6, step: 1 });
    const rFn = range('false_negative', { min: 0, max: 0.45, step: 0.05 });
    const rFp = range('false_positive', { min: 0, max: 0.45, step: 0.05 });
    const mapOptions = [
      `<optgroup label="Bundled maps">${(catalog.maps ?? []).map((m) => `<option value="file:${escapeHtml(m.name)}">${escapeHtml(withSize(m.title ?? m.name, m.rows, m.cols))}</option>`).join('')}</optgroup>`,
      `<optgroup label="Generated maps">${(catalog.scenarios ?? []).map((s) => `<option value="scenario:${escapeHtml(s.name)}">${escapeHtml(s.title ?? humanize(s.name))}</option>`).join('')}</optgroup>`,
      this.app.customMap ? '<optgroup label="Your map"><option value="ascii">Map from the editor</option></optgroup>' : '',
    ].join('');
    const fileMap = map.source === 'file' ? (catalog.maps ?? []).find((m) => m.name === map.name) : null;
    const describe = map.source === 'scenario'
      ? scen?.description
      : map.source === 'ascii' ? 'The map you drew in the map editor.' : fileMap?.description;
    const source = fileMap?.source && fileMap.source !== fileMap.description
      ? `<details class="field-source"><summary>Source</summary><p>${escapeHtml(fileMap.source)}</p></details>` : '';
    this.root.innerHTML = `
      ${note ? `<p class="note settings-note" role="status">${escapeHtml(note)}</p>` : ''}
      <section class="panel-section">
        <h4 class="panel-title">${icon('wand')}Mission</h4>
        <div class="field"><span class="field-label">Mode${infoTip(TIPS.mode, 'Mode')}</span>
          <div class="segmented" role="group" aria-label="Mode">${modes.map((m) => `<button type="button" data-mode="${escapeHtml(m.name)}" aria-pressed="${m.name === mode}" title="${escapeHtml(m.description ?? '')}">${escapeHtml(m.title ?? m.name)}</button>`).join('')}</div>
          <span class="field-hint">${escapeHtml(modes.find((m) => m.name === mode)?.description ?? '')}</span></div>
        <div class="field"><label class="field-label" for="set-map">Map${infoTip(TIPS.map, 'Map')}</label>
          <select id="set-map">${mapOptions}</select>
          ${describe ? `<span class="field-hint">${escapeHtml(describe)}</span>` : ''}${source}
          ${map.source === 'scenario' ? `<div class="field-row">
            <label class="field-hint" for="set-size">Size</label><select id="set-size">${sizes.map((n) => `<option value="${n}" ${n === map.rows ? 'selected' : ''}>${n}×${n}</option>`).join('')}</select>
            <label class="field-hint" for="set-mapseed">Map seed${infoTip(TIPS.mapSeed, 'Map seed')}</label><input type="number" id="set-mapseed" min="0" step="1" value="${map.seed ?? 0}"></div>` : ''}
        </div>
        ${mode === 'multi' ? rangeField('set-k', 'Survivors (k)', TIPS.k, { ...rSurv, value: config.k ?? rSurv.default ?? 3, format: (v) => String(v), spoken: SLIDER_TEXT.k }) : ''}
        ${mode === 'noisy' ? rangeField('set-fn', 'Missed debris (false negatives)', TIPS.fn, { ...rFn, value: config.false_negative ?? rFn.default ?? 0.1, format: (v) => fmtPct(Number(v)), spoken: SLIDER_TEXT.false_negative }) : ''}
        ${mode === 'noisy' ? rangeField('set-fp', 'Phantom debris (false positives)', TIPS.fp, { ...rFp, value: config.false_positive ?? rFp.default ?? 0, format: (v) => fmtPct(Number(v)), spoken: SLIDER_TEXT.false_positive }) : ''}
      </section>
      <section class="panel-section">
        <h4 class="panel-title">${icon('robot')}Agent${infoTip(TIPS.agent, 'Agent')}</h4>
        <div class="choice-list" role="group" aria-label="Agent">${agents.map((a) => `<button type="button" class="choice" data-agent="${escapeHtml(a.name)}" aria-pressed="${a.name === config.agent}">
          <span class="radio" aria-hidden="true"></span><span><strong>${escapeHtml(a.label ?? a.name)}</strong><span class="desc">${escapeHtml(a.description ?? '')}${a.needs_scipy ? ' Its first use downloads extra code (a few seconds).' : ''}</span></span></button>`).join('')}</div>
      </section>
      <section class="panel-section">
        <h4 class="panel-title">${icon('gear')}World</h4>
        ${rangeField('set-slip', 'Wheel slip p', slipTip(P), { ...rSlip, value: slip, format: (v) => Number(v).toFixed(2), spoken: SLIDER_TEXT.slip_p })}
        <div class="split-bar" aria-hidden="true"><span style="width:${split.arrive * 100}%;background:var(--c-ok)"></span><span style="width:${split.stay * 100}%;background:var(--c-slip)"></span><span style="width:${(split.skidL + split.skidR) * 100}%;background:var(--c-skid)"></span></div>
        <div class="split-legend" id="split-legend"><span>→ arrives <b>${pct(split.arrive)}</b></span><span>↺ stays <b>${pct(split.stay)}</b></span><span>▲ skids <b>${pct(split.skidL)}</b> left + <b>${pct(split.skidR)}</b> right</span></div>
        ${rangeField('set-radius', 'Sensor radius', TIPS.radius, { ...rRad, value: config.sensor_radius ?? rRad.default ?? 1, format: SLIDER_TEXT.sensor_radius })}
        <div class="field"><span class="field-label">Replanning${infoTip(TIPS.replan, 'Replanning')}</span>
          <div class="segmented" role="group" aria-label="Replan mode">${['eager', 'lazy'].map((m) => `<button type="button" data-replan="${m}" aria-pressed="${(config.replan_mode ?? 'eager') === m}">${m === 'eager' ? 'Eager' : 'Lazy'}</button>`).join('')}</div></div>
        ${rangeField('set-q', 'Aftershock chance q', TIPS.aftershock, { ...rQ, value: config.aftershock_q ?? 0, format: (v) => fmtPct(Number(v)), spoken: SLIDER_TEXT.aftershock_q })}
        ${rangeField('set-ak', 'Debris per aftershock k', TIPS.aftershock, { ...rK, value: config.aftershock_k ?? rK.default ?? 1, format: (v) => String(v), spoken: SLIDER_TEXT.aftershock_k })}
        <div class="field"><label class="field-label" for="set-seed">Luck seed${infoTip(TIPS.seed, 'Luck seed')}</label>
          <div class="field-row seed-row"><input type="number" id="set-seed" min="0" step="1" value="${config.seed ?? ''}" placeholder="default">
          <button class="btn small" type="button" id="set-seed-rand">${icon('dice')}Roll</button></div></div>
      </section>`;
    const sel = this.root.querySelector('#set-map');
    sel.value = mapValue;
    this._wire(config);
  }

  _wire(config) {
    const $ = (sel) => this.root.querySelector(sel);
    const apply = (patch, why) => this.onApply({ ...config, ...patch }, why);
    this.root.querySelectorAll('[data-mode]').forEach((b) => b.addEventListener('click', () => {
      const mode = b.dataset.mode;
      if (mode === (config.mode ?? 'core')) return;
      apply(modePatch(this.app.catalog, config, mode), `Mode changed to “${b.textContent.trim()}”`);
    }));
    $('#set-map').addEventListener('change', (ev) => {
      const v = ev.target.value;
      let map;
      if (v === 'ascii') map = { source: 'ascii', text: this.app.customMap };
      else if (v.startsWith('scenario:')) {
        const scenario = v.slice(9);
        const size = config.map?.rows && config.map.source === 'scenario' ? config.map.rows : 15;
        map = { source: 'scenario', scenario, rows: size, cols: size, seed: config.map?.seed ?? 0 };
      } else map = { source: 'file', name: v.slice(5) };
      apply({ map, seed: null }, 'New map');
    });
    $('#set-size')?.addEventListener('change', (ev) => {
      const n = Number(ev.target.value);
      apply({ map: { ...config.map, rows: n, cols: n }, seed: null }, `Map size ${n}×${n}`);
    });
    $('#set-mapseed')?.addEventListener('change', (ev) => {
      const s = Math.max(0, Math.floor(Number(ev.target.value) || 0));
      apply({ map: { ...config.map, seed: s }, seed: null }, `Map seed ${s}`);
    });
    this.root.querySelectorAll('[data-agent]').forEach((b) => b.addEventListener('click', () => {
      if (b.dataset.agent !== config.agent) apply({ agent: b.dataset.agent }, `Agent: ${b.querySelector('strong').textContent}`);
    }));
    this.root.querySelectorAll('[data-replan]').forEach((b) => b.addEventListener('click', () => {
      if (b.dataset.replan !== (config.replan_mode ?? 'eager')) apply({ replan_mode: b.dataset.replan }, `Replanning: ${b.textContent}`);
    }));
    const live = (id, key, fmt, label) => {
      const input = $(id);
      if (!input) return;
      const out = $(`${id}-out`);
      input.addEventListener('input', () => {
        out.textContent = fmt(Number(input.value));
        input.setAttribute('aria-valuetext', (SLIDER_TEXT[key] ?? fmt)(Number(input.value)));
        if (key === 'slip_p') this._updateSplit(Number(input.value));
      });
      input.addEventListener('change', () => apply({ [key]: Number(input.value) }, `${label}: ${fmt(Number(input.value))}`));
    };
    live('#set-slip', 'slip_p', (v) => v.toFixed(2), 'Wheel slip p');
    live('#set-radius', 'sensor_radius', (v) => `${v} cell${v === 1 ? '' : 's'}`, 'Sensor radius');
    live('#set-q', 'aftershock_q', (v) => fmtPct(v), 'Aftershock chance');
    live('#set-ak', 'aftershock_k', (v) => String(v), 'Debris per aftershock');
    live('#set-k', 'k', (v) => String(v), 'Survivors');
    live('#set-fn', 'false_negative', (v) => fmtPct(v), 'False negatives');
    live('#set-fp', 'false_positive', (v) => fmtPct(v), 'False positives');
    $('#set-seed').addEventListener('change', (ev) => {
      const raw = ev.target.value.trim();
      const seed = raw === '' ? null : Math.max(0, Math.floor(Number(raw) || 0));
      apply({ seed }, seed == null ? 'Default seed' : `Luck seed ${seed}`);
    });
    $('#set-seed-rand').addEventListener('click', () => {
      const seed = Math.floor(Math.random() * 10000);
      apply({ seed }, `Luck seed ${seed}`);
    });
  }

  _updateSplit(p) {
    const s = outcomeSplit(p, this.app.catalog.params?.outcome_fractions);
    const pct = fmtPctAuto;
    const bar = this.root.querySelector('.split-bar');
    const spans = bar.querySelectorAll('span');
    spans[0].style.width = `${s.arrive * 100}%`;
    spans[1].style.width = `${s.stay * 100}%`;
    spans[2].style.width = `${(s.skidL + s.skidR) * 100}%`;
    this.root.querySelector('#split-legend').innerHTML = `<span>→ arrives <b>${pct(s.arrive)}</b></span><span>↺ stays <b>${pct(s.stay)}</b></span><span>▲ skids <b>${pct(s.skidL)}</b> left + <b>${pct(s.skidR)}</b> right</span>`;
  }
}

// --- legend ------------------------------------------------------------------------------
const LEGEND_SIZE = 22;

/** Paint a legend swatch with the real renderer (so the legend always matches the map). */
function legendCanvas(state, cols = 1, opts = {}) {
  const c = document.createElement('canvas');
  c.style.width = `${LEGEND_SIZE * cols}px`;
  c.style.height = `${LEGEND_SIZE}px`;
  const r = new MapRenderer(c, { compact: true, interactive: false, size: { width: LEGEND_SIZE * cols, height: LEGEND_SIZE } });
  r.setOptions(opts);
  r.setState({ rows: 1, cols, cells: [Array(cols).fill(CELL.OPEN)], belief: [Array(cols).fill(CELL.OPEN)], high_slip: [Array(cols).fill(false)], ...state });
  return { canvas: c, renderer: r };
}

/** Legend items: label + state painting it. */
export function legendItems(mode) {
  const open = (n = 1) => [Array(n).fill(CELL.OPEN)];
  const items = [
    ['Unknown (not seen yet)', { belief: [[CELL.UNKNOWN]] }],
    ['Open', { belief: open() }],
    ['Debris', { belief: [[CELL.DEBRIS]], cells: [[CELL.DEBRIS]] }],
    ['Water / mud (slips more)', { high_slip: [[true]] }],
    ['Robot', { robot: [0, 0] }],
    ['Survivor', { survivors: [[0, 0]], belief: [[CELL.SURVIVOR]], cells: [[CELL.SURVIVOR]] }],
    ['Rescued', { survivors: [[0, 0]], rescued: [[0, 0]], belief: [[CELL.SURVIVOR]], cells: [[CELL.SURVIVOR]] }],
    ['Start', { start: [0, 0] }],
    ['Moved', { trace: [{ t: 1, from: [0, 0], to: [0, 1], action: 'E', outcome: 'arrived' }], cols: 2, belief: open(2), cells: open(2), high_slip: [[false, false]] }],
    ['Slipped (stayed)', { trace: [{ t: 1, from: [0, 0], to: [0, 0], action: 'E', outcome: 'slipped' }] }],
    ['Skidded sideways', { trace: [{ t: 1, from: [0, 0], to: [0, 1], action: 'N', outcome: 'skid-R' }], cols: 2, belief: open(2), cells: open(2), high_slip: [[false, false]] }],
    ['Blocked / bump', { trace: [{ t: 1, from: [0, 0], to: [0, 0], action: 'E', outcome: 'arrived', blocked: true, bump: true }] }],
  ];
  if (mode !== 'core') {
    items.push(['Current target', { target: [0, 0] }]);
  }
  if (mode === 'noisy') {
    items.push(['Suspected debris (shading = probability)', { p_debris: [[0.2]] }]);
  }
  return items;
}

export function renderLegend(root, mode) {
  root.innerHTML = '';
  for (const [label, state] of legendItems(mode)) {
    const cols = state.cols ?? 1;
    const { canvas } = legendCanvas(state, cols);
    const item = document.createElement('span');
    item.className = 'legend-item';
    item.append(canvas);
    item.insertAdjacentHTML('beforeend', `<span>${escapeHtml(label)}</span>`);
    root.append(item);
  }
}

export { swatch };
