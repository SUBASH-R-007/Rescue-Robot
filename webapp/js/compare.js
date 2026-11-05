// Compare view: 2–3 agents on the same map with the same seed (common random numbers), played
// in lock-step, with status badges, a shared timeline, metrics side by side and a summary.
import { MapRenderer } from './render.js';
import { MissionSession } from './session.js';
import {
  escapeHtml, fmtNum, fmtMs, humanize, statusInfo, plural, keySentences, toneStyle,
} from './format.js';
import { icon, infoTip, showError } from './ui.js';
import { withSize } from './panels.js';

/** Panes allowed. */
export const MIN_PANES = 2;
export const MAX_PANES = 3;
/** Steps per second of the speed slider (same scale as Simulate). */
export const CMP_SPEEDS = [0.5, 1, 2, 4, 8, 16, 30];
const DEFAULT_SPEED = 3;
/** Size of a generated map picked in the Compare bar when the current map is not generated. */
const DEFAULT_GENERATED_SIZE = 15;
/** Wheel-slip values offered in the Compare bar. */
const SLIP_CHOICES = [0, 0.1, 0.2, 0.3, 0.4];

/**
 * Rows of the side-by-side table: key, label, lower-is-better (null = never ranked), format.
 * Planning time is wall-clock time in this browser (noisy): shown, never ranked.
 */
export const CMP_METRICS = [
  ['steps', 'Steps', true, (v) => fmtNum(v)],
  ['rescued', 'Rescued', false, (v) => fmtNum(v)],
  ['replans', 'Replans', true, (v) => fmtNum(v)],
  ['slips', 'Slips', null, (v) => fmtNum(v)],
  ['skids', 'Skids', null, (v) => fmtNum(v)],
  ['bumps', 'Collisions', true, (v) => fmtNum(v)],
  ['unplanned_states', 'Unplanned states', true, (v) => fmtNum(v)],
  ['wasted_moves', 'Wasted moves', true, (v) => (v == null ? '–' : fmtNum(v))],
  ['cells_explored', 'Cells explored', null, (v) => fmtNum(v)],
  ['planning_ms', 'Planning time (wall-clock, not ranked)', null, (v) => fmtMs(v)],
];

/** What the ★ means (the table's caption). */
export const BEST_CAPTION = '★ = best value among the agents that rescued the survivor, shown once every agent has finished. Slips, skids, cells explored and planning time are not ranked.';

/** Default agents of a mode: the catalog's "main" agents (first three), else the first ones. */
export function defaultAgents(catalog, mode, wanted = null) {
  const all = (catalog?.agents ?? []).filter((a) => !a.modes || a.modes.includes(mode));
  const names = all.map((a) => a.name);
  if (Array.isArray(wanted)) {
    const ok = wanted.filter((n) => names.includes(n)).slice(0, MAX_PANES);
    if (ok.length >= MIN_PANES) return ok;
  }
  const main = all.filter((a) => a.main !== false).map((a) => a.name);
  const pick = (main.length >= MIN_PANES ? main : names).slice(0, MAX_PANES);
  return pick;
}

/** Index of the best value among `values` (null entries ignored); -1 if no unique best. */
export function bestIndex(values, lowerIsBetter) {
  if (lowerIsBetter === null || lowerIsBetter === undefined) return -1;
  let best = -1;
  let tie = false;
  values.forEach((v, i) => {
    if (v === null || v === undefined || Number.isNaN(Number(v))) return;
    if (best < 0) { best = i; tie = false; return; }
    const a = Number(v);
    const b = Number(values[best]);
    if (a === b) tie = true;
    else if (lowerIsBetter ? a < b : a > b) { best = i; tie = false; }
  });
  const defined = values.filter((v) => v !== null && v !== undefined).length;
  return tie || defined < 2 ? -1 : best;
}

/** "A", "A and B", "A, B and C". */
export function joinNames(names) {
  if (names.length <= 1) return names[0] ?? '';
  return `${names.slice(0, -1).join(', ')} and ${names[names.length - 1]}`;
}

/**
 * One sentence summarising a comparison so far. `panes`: [{label, status, t, steps, k}].
 * e.g. "A* (no replan) gave up at step 4; AND-OR and A* + replan are both still running."
 */
export function summarize(panes) {
  if (!panes.length) return '';
  const groups = new Map();
  for (const p of panes) {
    if (!groups.has(p.status)) groups.set(p.status, []);
    groups.get(p.status).push(p);
  }
  const parts = [];
  const names = (list) => joinNames(list.map((p) => p.label));
  const rescued = groups.get('rescued') ?? [];
  if (rescued.length) {
    const sorted = [...rescued].sort((a, b) => a.t - b.t);
    const what = (p) => ((p.k ?? 1) > 1 ? `all ${p.k} survivors` : 'the survivor');
    if (sorted.length === 1) parts.push(`${sorted[0].label} rescued ${what(sorted[0])} in ${plural(sorted[0].t, 'step')}`);
    else if (sorted.every((p) => p.t === sorted[0].t)) parts.push(`${names(sorted)} each rescued ${what(sorted[0])} in ${plural(sorted[0].t, 'step')}`);
    else parts.push(`${sorted.map((p) => `${p.label} rescued ${what(p)} in ${plural(p.t, 'step')}`).join(', ')}`);
  }
  for (const [status, verb] of [['aborted', 'gave up'], ['budget', 'ran out of steps'], ['unreachable', 'found no route']]) {
    const list = groups.get(status) ?? [];
    if (!list.length) continue;
    if (list.length === 1) parts.push(`${list[0].label} ${verb} at step ${list[0].t}`);
    else if (list.every((p) => p.t === list[0].t)) parts.push(`${names(list)} ${list.length === 2 ? 'both' : 'all'} ${verb} at step ${list[0].t}`);
    else parts.push(list.map((p) => `${p.label} ${verb} at step ${p.t}`).join(', '));
  }
  const running = groups.get('running') ?? [];
  if (running.length === panes.length) {
    const t = Math.max(...running.map((p) => p.t));
    parts.push(t === 0
      ? `${names(running)} are ready at the start: press Play to run them side by side`
      : `${names(running)} are ${running.length === 2 ? 'both' : 'all'} still running (step ${t})`);
  } else if (running.length === 1) parts.push(`${running[0].label} is still running`);
  else if (running.length > 1) parts.push(`${names(running)} are ${running.length === 2 ? 'both' : 'all'} still running`);
  let sentence = parts.join('; ');
  if (rescued.length >= 2 && rescued.length === panes.length) {
    const sorted = [...rescued].sort((a, b) => a.t - b.t);
    if (sorted[0].t < sorted[1].t) sentence += `. ${sorted[0].label} was the fastest`;
  }
  return `${sentence[0].toUpperCase()}${sentence.slice(1)}.`;
}

/** One agent's pane. */
class Pane {
  constructor(view, index, agent) {
    this.view = view;
    this.index = index;
    this.agent = agent;
    this.session = null;
    this.el = document.createElement('div');
    this.el.className = 'cmp-pane card';
    this.el.id = `cmp-pane-${index}`;
    this.el.setAttribute('role', 'tabpanel');
    this.el.innerHTML = `
      <div class="cmp-pane-head"><h3></h3><span class="cmp-status"></span></div>
      <div class="canvas-box"><canvas role="img"></canvas></div>
      <ul class="explain compact cmp-last"></ul>`;
    this.renderer = new MapRenderer(this.el.querySelector('canvas'), { compact: false });
  }

  get state() { return this.session?.view ?? null; }

  /** Close the mission and stop the renderer's listeners (the pane is being replaced). */
  dispose() {
    this.session?.dispose();
    this.renderer.destroy();
    this.disposed = true;
  }

  async start(config) {
    this.session?.dispose();
    const session = new MissionSession(this.view.app.backend, { ...config, agent: this.agent });
    this.session = session;
    const state = await session.start();
    if (this.disposed || session !== this.session) return;
    this.renderer.setState(state);
    this.render();
  }

  async step() {
    const session = this.session;
    if (!session || session.done && session.atLive) return;
    const prev = session.view;
    const { state } = await session.forward();
    if (this.disposed || session !== this.session) return;
    this.renderer.setState(state, { animate: true, prev });
    this.render();
  }

  async finish() {
    const session = this.session;
    if (!session || session.done) return;
    const state = await session.finish();
    if (this.disposed || session !== this.session) return;
    this.renderer.setState(state);
    this.render();
  }

  /** Show recorded step t (clamped to this pane's own last step). */
  async seek(t) {
    const session = this.session;
    if (!session) return;
    const state = await session.seek(Math.min(t, session.maxT));
    if (this.disposed || session !== this.session) return;
    this.renderer.setState(state);
    this.render();
  }

  render() {
    const s = this.state;
    const info = (this.view.app.catalog.agents ?? []).find((a) => a.name === this.agent);
    const head = this.el.querySelector('h3');
    head.textContent = info?.label ?? this.agent;
    head.title = info?.description ?? '';
    const canvas = this.el.querySelector('canvas');
    if (!s) {
      canvas.setAttribute('aria-label', `Map of ${info?.label ?? this.agent}`);
      return;
    }
    const st = statusInfo(s.status);
    canvas.setAttribute('aria-label', `Map of ${info?.label ?? this.agent}, step ${s.t}, ${st.label}`);
    this.el.querySelector('.cmp-status').innerHTML = `<span class="badge tone-${st.tone}">${st.glyph} ${escapeHtml(st.label)}</span>
      <span class="badge plain">step ${s.t}</span>`;
    // Two sentences: the move and the one that matters most (the rescue or the give-up once
    // the mission is over), picked by the tones Python gives each sentence.
    const sentences = s.t === 0 ? keySentences(s, 1) : keySentences(s, 2);
    this.el.querySelector('.cmp-last').innerHTML = sentences.map(({ text, tone }) => {
      const { glyph, cls } = toneStyle(tone);
      return `<li><span class="ex-ico tone-${cls}" aria-hidden="true">${glyph}</span><span>${escapeHtml(text)}</span></li>`;
    }).join('') || '<li>Ready.</li>';
  }
}

export class CompareView {
  constructor(app, root) {
    this.app = app;
    this.root = root;
    this.panes = [];
    this.playing = false;
    this._playGen = 0;
    this.speedIndex = DEFAULT_SPEED;
    this.config = null;
    this.agents = null;
    this.active = 0; // the pane shown on narrow screens
    this.root.innerHTML = `
      <h2 class="sr-only">Compare agents</h2>
      <div class="cmp-bar card" id="cmp-controls">
        <div class="ctl-group" role="toolbar" aria-label="Comparison controls">
          <button class="ctl-btn primary" id="cmp-play" aria-label="Play all (Space)"><svg aria-hidden="true"><use href="#i-play"/></svg><span class="ctl-label">Play</span></button>
          <button class="ctl-btn" id="cmp-step" aria-label="One step for every agent (N)"><svg aria-hidden="true"><use href="#i-step"/></svg><span class="ctl-label">Step</span></button>
          <button class="ctl-btn" id="cmp-finish" aria-label="Run every agent to the end"><svg aria-hidden="true"><use href="#i-end"/></svg><span class="ctl-label">Finish</span></button>
          <button class="ctl-btn" id="cmp-reset" aria-label="Reset (R)"><svg aria-hidden="true"><use href="#i-reset"/></svg><span class="ctl-label">Reset</span></button>
        </div>
        <label class="speed"><span class="speed-label">Speed</span>
          <input type="range" id="cmp-speed" min="0" max="${CMP_SPEEDS.length - 1}" step="1" value="${DEFAULT_SPEED}">
          <output id="cmp-speed-out" aria-hidden="true"></output></label>
        <label class="cmp-scrub"><span class="speed-label">Step</span>
          <input type="range" id="cmp-scrub" min="0" max="0" step="1" value="0" aria-label="Replay a step in every pane">
          <output id="cmp-scrub-out" aria-hidden="true">0</output></label>
      </div>
      <div class="cmp-bar card cmp-setup">
        <div class="cmp-agents" id="cmp-agents" aria-label="Agents to compare"></div>
        <div class="cmp-world" id="cmp-world"></div>
      </div>
      <p class="cmp-summary card" id="cmp-summary" role="status" aria-live="polite"></p>
      <div class="cmp-switch segmented" id="cmp-switch" role="tablist" aria-label="Pane"></div>
      <div class="cmp-panes" id="cmp-panes"></div>
      <div class="card cmp-table-wrap"><table class="cmp-table" id="cmp-table"></table></div>`;
    const $ = (id) => this.root.querySelector(`#${id}`);
    $('cmp-play').addEventListener('click', () => this.togglePlay());
    $('cmp-step').addEventListener('click', () => { this.pause(); this.stepAll(); });
    $('cmp-finish').addEventListener('click', () => this.finishAll());
    $('cmp-reset').addEventListener('click', () => this.restart());
    $('cmp-speed').addEventListener('input', (ev) => this.setSpeed(Number(ev.target.value)));
    $('cmp-scrub').addEventListener('input', (ev) => { this.pause(); this.seekAll(Number(ev.target.value)); });
    this.setSpeed(DEFAULT_SPEED);
  }

  /** Called by the tab: (re)start when the Simulate mission changed since last time. */
  onShow() {
    const sim = this.app.views.simulate?.config;
    if (!sim) return;
    const key = JSON.stringify({ ...sim, agent: null });
    if (key !== this._simKey || !this.panes.length) {
      this._simKey = key;
      this.startWith(sim, this.agents);
    } else {
      for (const p of this.panes) p.renderer.resize();
    }
  }

  /** Compare `agents` on `config` (map, seed and parameters; the agent is replaced). */
  async startWith(config, agents = null) {
    this.pause();
    this.config = { ...config };
    this._simKey = JSON.stringify({ ...config, agent: null });
    const mode = config.mode ?? 'core';
    this.agents = defaultAgents(this.app.catalog, mode, agents);
    await this.restart();
  }

  async restart() {
    this.pause();
    if (!this.config) return;
    for (const p of this.panes) p.dispose();
    const box = this.root.querySelector('#cmp-panes');
    box.innerHTML = '';
    box.style.setProperty('--panes', String(this.agents.length));
    this.active = Math.min(this.active, this.agents.length - 1);
    const panes = this.agents.map((a, i) => new Pane(this, i, a));
    this.panes = panes;
    for (const p of panes) box.append(p.el);
    this._renderAgents();
    this._renderWorld();
    this._renderSwitch();
    try {
      for (const p of panes) await p.start(this.config);
    } catch (err) {
      if (panes === this.panes) showError(err, 'Could not start the comparison');
    }
    if (panes === this.panes) this._update();
  }

  _renderAgents() {
    const mode = this.config.mode ?? 'core';
    const options = (this.app.catalog.agents ?? []).filter((a) => !a.modes || a.modes.includes(mode));
    const box = this.root.querySelector('#cmp-agents');
    box.innerHTML = `<span class="field-label">Agents${infoTip('Every pane runs the same map with the same world seed, so all agents draw the same random numbers at each step (common random numbers). On water and mud a draw can still turn out differently for two agents, because the slip chance there is higher. Differences come from the planning strategy.', 'Agents')}</span>
      ${this.agents.map((name, i) => `<span class="cmp-agent-pick">
        <label class="sr-only" for="cmp-agent-${i}">Agent ${i + 1}</label>
        <select id="cmp-agent-${i}" data-i="${i}">${options.map((a) => `<option value="${escapeHtml(a.name)}" ${a.name === name ? 'selected' : ''}>${escapeHtml(a.label ?? a.name)}</option>`).join('')}</select>
        ${this.agents.length > MIN_PANES ? `<button type="button" class="icon-btn" data-remove="${i}" aria-label="Remove agent ${i + 1}" title="Remove this pane">${icon('close')}</button>` : ''}
      </span>`).join('')}
      ${this.agents.length < MAX_PANES && options.length > this.agents.length ? `<button type="button" class="btn small" id="cmp-add">${icon('plus')}Add agent</button>` : ''}`;
    box.querySelectorAll('select').forEach((sel) => sel.addEventListener('change', () => {
      this.agents[Number(sel.dataset.i)] = sel.value;
      this.restart();
    }));
    box.querySelectorAll('[data-remove]').forEach((b) => b.addEventListener('click', () => {
      this.agents.splice(Number(b.dataset.remove), 1);
      this.restart();
    }));
    box.querySelector('#cmp-add')?.addEventListener('click', () => {
      const unused = options.find((a) => !this.agents.includes(a.name)) ?? options[0];
      this.agents.push(unused.name);
      this.restart();
    });
  }

  /** Map, luck seed and wheel slip, changeable here (the panes restart with them). */
  _renderWorld() {
    const c = this.config;
    const cat = this.app.catalog;
    const map = c.map ?? {};
    const value = map.source === 'scenario' ? `scenario:${map.scenario}` : map.source === 'ascii' ? 'ascii' : `file:${map.name}`;
    const options = [
      `<optgroup label="Bundled maps">${(cat.maps ?? []).map((m) => `<option value="file:${escapeHtml(m.name)}">${escapeHtml(withSize(m.title ?? m.name, m.rows, m.cols))}</option>`).join('')}</optgroup>`,
      `<optgroup label="Generated maps">${(cat.scenarios ?? []).map((s) => `<option value="scenario:${escapeHtml(s.name)}">${escapeHtml(s.title ?? humanize(s.name))}</option>`).join('')}</optgroup>`,
      map.source === 'ascii' ? '<optgroup label="Your map"><option value="ascii">Map from the editor</option></optgroup>' : '',
    ].join('');
    const mode = c.mode ?? 'core';
    const modeTitle = mode === 'core' ? '' : (cat.modes ?? []).find((m) => m.name === mode)?.title ?? humanize(mode);
    const seed = this.panes[0]?.state?.seed ?? c.seed ?? '';
    const slip = c.slip_p ?? cat.params?.slip_p?.default ?? 0.2;
    const slips = SLIP_CHOICES.includes(slip) ? SLIP_CHOICES : [...SLIP_CHOICES, slip].sort();
    const box = this.root.querySelector('#cmp-world');
    box.innerHTML = `${modeTitle ? `<span class="badge plain">Mode: ${escapeHtml(modeTitle)}</span>` : ''}
      <label class="cmp-field"><span>Map</span><select id="cmp-map">${options}</select></label>
      ${map.source === 'scenario' ? `<label class="cmp-field"><span>Map seed</span><input type="number" id="cmp-mapseed" min="0" step="1" value="${map.seed ?? 0}"></label>` : ''}
      <label class="cmp-field"><span>Luck seed</span><input type="number" id="cmp-seed" min="0" step="1" value="${escapeHtml(String(seed))}"></label>
      <label class="cmp-field"><span>Wheel slip p</span><select id="cmp-slip">${slips.map((p) => `<option value="${p}" ${p === slip ? 'selected' : ''}>${p.toFixed(2)}</option>`).join('')}</select></label>`;
    box.querySelector('#cmp-map').value = value;
    const apply = (patch) => {
      this.config = { ...this.config, ...patch };
      this.restart();
    };
    box.querySelector('#cmp-map').addEventListener('change', (ev) => {
      const v = ev.target.value;
      if (v === 'ascii') return;
      if (v.startsWith('scenario:')) {
        const n = map.source === 'scenario' ? map.rows : DEFAULT_GENERATED_SIZE;
        apply({ map: { source: 'scenario', scenario: v.slice(9), rows: n, cols: n, seed: map.seed ?? 0 }, seed: null });
      } else apply({ map: { source: 'file', name: v.slice(5) }, seed: null });
    });
    box.querySelector('#cmp-mapseed')?.addEventListener('change', (ev) => {
      apply({ map: { ...map, seed: Math.max(0, Math.floor(Number(ev.target.value) || 0)) }, seed: null });
    });
    box.querySelector('#cmp-seed').addEventListener('change', (ev) => {
      const raw = ev.target.value.trim();
      apply({ seed: raw === '' ? null : Math.max(0, Math.floor(Number(raw) || 0)) });
    });
    box.querySelector('#cmp-slip').addEventListener('change', (ev) => apply({ slip_p: Number(ev.target.value) }));
  }

  /** Narrow screens show one pane at a time: a tab per agent. */
  _renderSwitch() {
    const label = (name) => (this.app.catalog.agents ?? []).find((a) => a.name === name)?.label ?? name;
    const box = this.root.querySelector('#cmp-switch');
    box.innerHTML = this.agents.map((name, i) => `<button type="button" role="tab" id="cmp-tab-${i}" aria-controls="cmp-pane-${i}" aria-selected="${i === this.active}" tabindex="${i === this.active ? 0 : -1}" data-pane="${i}">${escapeHtml(label(name))}</button>`).join('');
    box.querySelectorAll('[data-pane]').forEach((b) => {
      b.addEventListener('click', () => this.showPane(Number(b.dataset.pane)));
      b.addEventListener('keydown', (ev) => {
        const d = { ArrowRight: 1, ArrowLeft: -1 }[ev.key];
        if (!d) return;
        ev.preventDefault();
        ev.stopPropagation();
        const next = (Number(b.dataset.pane) + d + this.agents.length) % this.agents.length;
        this.showPane(next);
        box.querySelector(`[data-pane="${next}"]`).focus();
      });
    });
    this.panes.forEach((p, i) => {
      p.el.setAttribute('aria-labelledby', `cmp-tab-${i}`);
      p.el.classList.toggle('active', i === this.active);
    });
  }

  showPane(i) {
    this.active = i;
    this.root.querySelectorAll('#cmp-switch [data-pane]').forEach((b) => {
      const sel = Number(b.dataset.pane) === i;
      b.setAttribute('aria-selected', String(sel));
      b.tabIndex = sel ? 0 : -1;
    });
    this.panes.forEach((p, k) => p.el.classList.toggle('active', k === i));
    this.panes[i]?.renderer.resize();
  }

  // --- playing -------------------------------------------------------------------------
  get allDone() {
    return this.panes.every((p) => p.session?.done && p.session.atLive);
  }

  async stepAll() {
    if (this._stepping) return;
    this._stepping = true;
    const panes = this.panes;
    try {
      // A pane replaying an earlier step catches up first (no new Python step until then).
      await Promise.all(panes.map((p) => p.step()));
    } catch (err) {
      if (panes === this.panes) {
        this.pause();
        showError(err, 'A step failed');
      }
    } finally {
      this._stepping = false;
    }
    if (panes === this.panes) this._update();
  }

  async finishAll() {
    this.pause();
    const panes = this.panes;
    try {
      await Promise.all(panes.map((p) => p.finish()));
      await Promise.all(panes.map((p) => p.seek(p.session?.maxT ?? 0)));
    } catch (err) {
      if (panes === this.panes) showError(err, 'Could not finish the comparison');
    }
    if (panes === this.panes) this._update();
  }

  /** Show step t in every pane (each pane stops at its own last step). */
  async seekAll(t) {
    this._seekTarget = t;
    if (this._seeking) return;
    this._seeking = true;
    const panes = this.panes;
    try {
      while (this._seekTarget != null) {
        const target = this._seekTarget;
        this._seekTarget = null;
        await Promise.all(panes.map((p) => p.seek(target)));
      }
    } catch (err) {
      if (panes === this.panes) showError(err, 'Could not show that step');
    } finally {
      this._seeking = false;
    }
    if (panes === this.panes) this._update();
  }

  togglePlay() {
    if (this.playing) this.pause();
    else this.play();
  }

  async play() {
    if (!this.panes.length) return;
    if (this.allDone) await this.restart();
    this.playing = true;
    this._syncPlay();
    const gen = ++this._playGen; // one loop per play(): an older sleeping loop stops
    while (this.playing && gen === this._playGen && !this.allDone) {
      const t0 = performance.now();
      await this.stepAll();
      if (gen !== this._playGen) return;
      const wait = 1000 / CMP_SPEEDS[this.speedIndex] - (performance.now() - t0);
      if (wait > 0) await new Promise((r) => setTimeout(r, wait));
    }
    if (gen === this._playGen) this.pause();
  }

  pause() {
    this.playing = false;
    this._playGen += 1;
    this._syncPlay();
  }

  setSpeed(i) {
    this.speedIndex = Math.max(0, Math.min(CMP_SPEEDS.length - 1, i));
    const v = CMP_SPEEDS[this.speedIndex];
    const input = this.root.querySelector('#cmp-speed');
    input.value = String(this.speedIndex);
    const text = `${v} step${v === 1 ? '' : 's'}/s`;
    this.root.querySelector('#cmp-speed-out').textContent = text;
    input.setAttribute('aria-valuetext', text.replace('/s', ' per second'));
  }

  _syncPlay() {
    const b = this.root.querySelector('#cmp-play');
    b.querySelector('use').setAttribute('href', this.playing ? '#i-pause' : '#i-play');
    b.querySelector('.ctl-label').textContent = this.playing ? 'Pause' : 'Play';
    b.setAttribute('aria-label', this.playing ? 'Pause (Space)' : 'Play all (Space)');
  }

  handleKey(ev) {
    const k = ev.key;
    if (k === ' ') { this.togglePlay(); return true; }
    if (k === 'n' || k === 'N' || k === 'ArrowRight') { this.pause(); this.stepAll(); return true; }
    if (k === 'r' || k === 'R') { this.restart(); return true; }
    return false;
  }

  // --- summary and table -----------------------------------------------------------------
  _update() {
    const label = (p) => (this.app.catalog.agents ?? []).find((a) => a.name === p.agent)?.label ?? p.agent;
    const rows = this.panes.filter((p) => p.state).map((p) => ({
      label: label(p), status: p.state.status, t: p.state.t, k: p.state.survivors?.length ?? 1,
    }));
    this.root.querySelector('#cmp-summary').innerHTML = `${icon('columns')} ${escapeHtml(summarize(rows))}`;
    const done = this.allDone;
    this.root.querySelector('#cmp-step').disabled = done;
    this.root.querySelector('#cmp-finish').disabled = this.panes.every((p) => p.session?.done);
    // The shared timeline spans the longest mission; it shows the furthest pane's step.
    const maxT = Math.max(0, ...this.panes.map((p) => p.session?.maxT ?? 0));
    const shown = Math.max(0, ...this.panes.map((p) => p.state?.t ?? 0));
    const scrub = this.root.querySelector('#cmp-scrub');
    scrub.max = String(maxT);
    scrub.value = String(shown);
    scrub.setAttribute('aria-valuetext', `step ${shown} of ${maxT}`);
    this.root.querySelector('#cmp-scrub-out').textContent = `${shown} / ${maxT}`;
    const head = `<caption>${escapeHtml(BEST_CAPTION)}</caption><thead><tr><th scope="col">Metric</th>${this.panes.map((p) => `<th scope="col">${escapeHtml(label(p))}</th>`).join('')}</tr></thead>`;
    const body = CMP_METRICS.map(([key, name, lower, fmt]) => {
      const values = this.panes.map((p) => {
        const v = p.state?.metrics?.[key];
        return v === undefined ? null : v;
      });
      if (values.every((v) => v === null)) return '';
      // "Best" only once every agent has finished (a running agent's count is not final), and
      // only among the agents that succeeded (giving up early is not "fewest steps").
      const ranked = values.map((v, i) => (this.panes[i].state?.status === 'rescued' ? v : null));
      const best = done ? bestIndex(ranked, lower) : -1;
      return `<tr><th scope="row">${escapeHtml(name)}</th>${values.map((v, i) => (i === best
        ? `<td class="best">${escapeHtml(fmt(v))} <span aria-hidden="true">★</span><span class="sr-only">(best)</span></td>`
        : `<td>${escapeHtml(fmt(v))}</td>`)).join('')}</tr>`;
    }).join('');
    const status = `<tr><th scope="row">Status</th>${this.panes.map((p) => {
      const st = statusInfo(p.state?.status);
      return `<td>${st.glyph} ${escapeHtml(st.label)}</td>`;
    }).join('')}</tr>`;
    this.root.querySelector('#cmp-table').innerHTML = `${head}<tbody>${status}${body}</tbody>`;
  }
}
