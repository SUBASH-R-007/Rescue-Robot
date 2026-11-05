// Simulate view: one mission, controls with the timeline, map and the side panels.
import { MapRenderer } from './render.js';
import { Timeline } from './timeline.js';
import { MissionSession } from './session.js';
import { PlanPanel, SettingsPanel, renderWhat, renderNumbers, renderLegend, moveBadge } from './panels.js';
import { branchesAt, edgeSentence } from './plantree.js';
import {
  escapeHtml, fmtPos, fmtNum, statusInfo, CELL, posKey, ACTION_NAMES, ACTION_ARROWS, humanize,
  keySentences, toneStyle,
} from './format.js';
import { icon, showError, toast, announce } from './ui.js';

/** Speeds of the slider, steps per second. */
export const SPEEDS = [0.5, 1, 2, 4, 8, 16, 30];
/** Show the "Planning…" chip when a call takes longer than this (ms). */
const BUSY_DELAY_MS = 250;
/** While playing, screen readers hear a summary every this many steps (and on pause / end). */
const ANNOUNCE_EVERY = 10;
/** Size of the generated maps "New map" rolls from a bundled map. */
const NEW_MAP_SIZE = 10;

const CELL_NAMES = {
  [CELL.UNKNOWN]: 'Unknown: not seen yet (assumed open when planning)',
  [CELL.OPEN]: 'Open',
  [CELL.DEBRIS]: 'Debris',
  [CELL.SURVIVOR]: 'Survivor',
};

/** "4 steps/s", and the spoken form. */
export function speedText(v) {
  return `${v} step${v === 1 ? '' : 's'}/s`;
}

/** One line for screen readers and the canvas label: step, robot, survivors, status. */
export function stateSummary(state) {
  if (!state) return 'Map of the disaster zone';
  const st = statusInfo(state.status);
  const survivors = (state.survivors ?? []).filter((p) => state.survivors_known !== false
    || state.belief?.[p[0]]?.[p[1]] === CELL.SURVIVOR);
  const where = survivors.length ? `, survivor${survivors.length > 1 ? 's' : ''} at ${survivors.map(fmtPos).join(', ')}` : ', survivor not found yet';
  return `Step ${state.t}: robot at ${fmtPos(state.robot)}${where}. ${st.label}.`;
}

export class SimView {
  constructor(app) {
    this.app = app;
    this.session = null;
    this.config = null;
    this.playing = false;
    this._playGen = 0;
    this.speedIndex = 3;
    this.showTrue = false;
    this.showPolicy = false;
    this.use3d = false;
    this.renderer3d = null;
    this._loading3d = null;
    this.hover = null;
    this.panel = 'what';
    this.$ = (id) => document.getElementById(id);
    this.renderer = new MapRenderer(this.$('sim-canvas'));
    this.timeline = new Timeline(this.$('sim-timeline'), { onScrub: (t) => this.seek(t) });
    this.planPanel = new PlanPanel(this.$('panel-plan'), {
      onHoverCell: (cell) => this._setOptions({ highlight: cell }),
      onShowArrows: () => this.setPolicy(true),
    });
    this.settings = new SettingsPanel(this.$('panel-settings'), app, {
      // The note is shown at the top of the Settings panel itself, so no toast is needed there.
      onApply: (config, why) => this.start(config, {
        note: `${why}. The mission restarted with the new settings.`, quiet: this.panel === 'settings',
      }),
    });
    this._wire();
  }

  _wire() {
    const on = (id, fn) => this.$(id).addEventListener('click', fn);
    on('btn-play', () => this.togglePlay());
    on('btn-step', () => this.stepForward());
    on('btn-back', () => this.stepBack());
    on('btn-reset', () => this.reset());
    on('btn-newmap', () => this.newMap());
    on('btn-finish', () => this.finish());
    on('tg-true', () => this.setTrue(!this.showTrue));
    on('tg-policy', () => this.setPolicy(!this.showPolicy));
    on('tg-3d', () => this.set3d(!this.use3d));
    const speed = this.$('speed');
    speed.addEventListener('input', () => this.setSpeed(Number(speed.value)));
    this.setSpeed(this.speedIndex);
    // Side tabs (arrow keys move between them, as in a WAI-ARIA tab list).
    const tabs = [...document.querySelectorAll('.side-tabs [role=tab]')];
    tabs.forEach((tab, i) => {
      tab.addEventListener('click', () => this.showPanel(tab.dataset.panel));
      tab.addEventListener('keydown', (ev) => {
        const d = { ArrowRight: 1, ArrowLeft: -1 }[ev.key];
        if (!d) return;
        ev.preventDefault();
        ev.stopPropagation();
        const next = tabs[(i + d + tabs.length) % tabs.length];
        next.focus();
        this.showPanel(next.dataset.panel);
      });
    });
    // Map hover tooltip.
    const canvas = this.$('sim-canvas');
    canvas.addEventListener('pointermove', (ev) => this._hover(ev));
    canvas.addEventListener('pointerleave', () => this._hover(null));
    canvas.addEventListener('pointerdown', (ev) => { if (ev.pointerType !== 'mouse') this._hover(ev); });
    // The key (legend) popover closes on Escape and on a click elsewhere.
    const pop = this.$('legend-pop');
    document.addEventListener('click', (ev) => { if (pop.open && !pop.contains(ev.target)) pop.open = false; });
    pop.addEventListener('keydown', (ev) => {
      if (ev.key === 'Escape' && pop.open) {
        ev.stopPropagation();
        pop.open = false;
        pop.querySelector('summary').focus();
      }
    });
  }

  // --- mission lifecycle ---------------------------------------------------------------
  /**
   * Start a mission with `config`. `note` is shown in Settings and (unless `quiet`) as one
   * toast; `warning` stays visible above the map until the next mission.
   */
  async start(config, { note = '', quiet = false, warning = '' } = {}) {
    this.pause();
    this.config = config;
    this.session?.dispose();
    const session = new MissionSession(this.app.backend, config);
    this.session = session;
    this._lastAnnounced = -1;
    this.planPanel.reset();
    this._setOptions({ highlight: null });
    this.settings.render(config, note);
    renderLegend(this.$('legend'), config.mode ?? 'core');
    const banner = this.$('map-warning');
    banner.hidden = !warning;
    banner.textContent = warning;
    this._busy(true, 'Building the map and the first plan…');
    try {
      const state = await session.start();
      if (session !== this.session) return;
      this._paint(state, { animate: false });
      this._update(state);
      if (note && !quiet && !warning) toast({ title: 'Mission restarted', text: note, tone: 'info', timeout: 3500 });
    } catch (err) {
      if (session === this.session) showError(err, 'Could not start this mission');
    } finally {
      if (session === this.session) this._busy(false);
    }
  }

  reset() {
    if (this.config) this.start(this.config, { note: 'Reset: same map, same seed, so the same mission replays.' });
  }

  newMap() {
    const cfg = this.config;
    const map = cfg.map ?? {};
    const scenarios = this.app.catalog.scenarios ?? [];
    // "New map" stays inside the pre-generated library (no scipy download).
    const seeds = this.app.catalog.params?.map_size?.library_seeds ?? 1;
    let next;
    if (map.source === 'scenario') {
      next = { ...map, seed: ((map.seed ?? 0) + 1) % seeds };
    } else {
      const pick = scenarios[Math.floor(Math.random() * scenarios.length)] ?? { name: 'random_rubble' };
      next = { source: 'scenario', scenario: pick.name, rows: NEW_MAP_SIZE, cols: NEW_MAP_SIZE, seed: Math.floor(Math.random() * seeds) };
    }
    const title = scenarios.find((s) => s.name === next.scenario)?.title ?? humanize(next.scenario);
    this.start({ ...cfg, map: next, seed: null }, { note: `New map: ${title}, ${next.rows}×${next.cols}, map seed ${next.seed}.` });
  }

  // --- stepping ------------------------------------------------------------------------
  async stepForward() {
    const session = this.session;
    if (!session || this._stepping) return;
    this._stepping = true;
    const timer = setTimeout(() => this._busy(true, 'Planning…'), BUSY_DELAY_MS);
    try {
      const prev = session.view;
      const { state } = await session.forward();
      if (session !== this.session) return; // the mission was replaced meanwhile
      this._paint(state, { animate: true, prev });
      this._update(state);
    } catch (err) {
      if (session !== this.session) return;
      this.pause();
      showError(err, 'The step failed');
    } finally {
      clearTimeout(timer);
      if (session === this.session) this._busy(false);
      this._stepping = false;
    }
  }

  async stepBack() {
    if (!this.session || this.session.t === 0) return;
    this.pause();
    await this.seek(this.session.t - 1);
  }

  async seek(t) {
    const session = this.session;
    if (!session) return;
    this._seekTarget = t;
    if (this._seeking) return;
    this._seeking = true;
    try {
      while (this._seekTarget !== null && this._seekTarget !== undefined) {
        const target = this._seekTarget;
        this._seekTarget = null;
        const state = await session.seek(target);
        if (session !== this.session) return;
        this._paint(state, { animate: false });
        this._update(state);
      }
    } catch (err) {
      if (session === this.session) showError(err, 'Could not show that step');
    } finally {
      this._seeking = false;
      this._seekTarget = null;
    }
  }

  async finish() {
    const session = this.session;
    if (!session) return;
    this.pause();
    if (!session.atLive) await this.seek(session.maxT);
    if (session !== this.session || session.done) return;
    this._busy(true, 'Running the mission to the end…');
    try {
      const state = await session.finish();
      if (session !== this.session) return;
      this._paint(state, { animate: false });
      this._update(state);
    } catch (err) {
      if (session === this.session) showError(err, 'Could not finish the mission');
    } finally {
      if (session === this.session) this._busy(false);
    }
  }

  togglePlay() {
    if (this.playing) this.pause();
    else this.play();
  }

  play() {
    if (!this.session) return;
    if (this.session.done && this.session.atLive) {
      // Finished: play again from the start (replay of the recorded mission).
      this.seek(0);
    }
    this.playing = true;
    this._syncPlay();
    // One loop per play(): a loop still sleeping from an earlier play() sees a newer
    // generation and stops, so playback never runs at twice the chosen speed.
    const gen = ++this._playGen;
    const loop = async () => {
      while (this.playing && gen === this._playGen) {
        const t0 = performance.now();
        await this.stepForward();
        if (gen !== this._playGen) return;
        if (!this.session || (this.session.done && this.session.atLive)) {
          this.pause();
          break;
        }
        const wait = 1000 / SPEEDS[this.speedIndex] - (performance.now() - t0);
        if (wait > 0) await new Promise((r) => setTimeout(r, wait));
      }
    };
    loop();
  }

  pause() {
    const was = this.playing;
    this.playing = false;
    this._playGen += 1;
    this._syncPlay();
    if (was && this.session?.view) this._announce(this.session.view, true);
  }

  setSpeed(i) {
    this.speedIndex = Math.max(0, Math.min(SPEEDS.length - 1, i));
    const input = this.$('speed');
    input.value = String(this.speedIndex);
    const text = speedText(SPEEDS[this.speedIndex]);
    this.$('speed-value').textContent = text;
    input.setAttribute('aria-valuetext', text.replace('/s', ' per second'));
  }

  setTrue(on) {
    this.showTrue = on;
    this.$('tg-true').setAttribute('aria-pressed', String(on));
    this._setOptions({ showTrue: on });
  }

  setPolicy(on) {
    this.showPolicy = on;
    this.$('tg-policy').setAttribute('aria-pressed', String(on));
    this._setOptions({ showPolicy: on });
  }

  showPanel(name) {
    this.panel = name;
    for (const tab of document.querySelectorAll('.side-tabs [role=tab]')) {
      const sel = tab.dataset.panel === name;
      tab.setAttribute('aria-selected', String(sel));
      tab.tabIndex = sel ? 0 : -1;
    }
    for (const p of ['what', 'plan', 'numbers', 'settings']) this.$(`panel-${p}`).hidden = p !== name;
    // A highlight belongs to the panel that set it (plan rows, path chips).
    this._setOptions({ highlight: null });
    if (this.session?.view) this._renderPanel(this.session.view);
  }

  // --- rendering -----------------------------------------------------------------------
  _syncPlay() {
    const b = this.$('btn-play');
    b.querySelector('use').setAttribute('href', this.playing ? '#i-pause' : '#i-play');
    b.querySelector('.ctl-label').textContent = this.playing ? 'Pause' : 'Play';
    b.setAttribute('aria-label', this.playing ? 'Pause (Space)' : 'Play (Space)');
  }

  _busy(on, text = 'Planning…') {
    this.$('sim-busy').hidden = !on;
    this.$('sim-busy-text').textContent = text;
  }

  /** Screen readers: one summary line, throttled while playing (every ANNOUNCE_EVERY steps,
   * on pause and at the end). The canvas label always describes the state on screen. */
  _announce(state, force = false) {
    const summary = stateSummary(state);
    this.$('sim-canvas').setAttribute('aria-label', `Map of the disaster zone. ${summary}`);
    const ended = state.status !== 'running';
    if (!force && this.playing && !ended && state.t % ANNOUNCE_EVERY !== 0) return;
    if (state.t === this._lastAnnounced && !force) return;
    this._lastAnnounced = state.t;
    const first = keySentences(state, 2).map((s) => s.text).join(' ');
    announce(this.playing && !ended ? summary : `${summary} ${first}`);
  }

  _update(state) {
    const s = this.session;
    this.timeline.update({ t: state.t, maxT: s.maxT, done: s.done, trace: s.live?.trace ?? state.trace, log: s.live?.log ?? state.log, budget: state.budget });
    const st = statusInfo(state.status);
    const agent = state.agent?.label ?? state.agent?.name ?? '';
    const map = this.config?.map ?? {};
    const mapName = map.source === 'scenario'
      ? `${humanize(map.scenario)} ${map.rows}×${map.cols} · map seed ${map.seed ?? 0}`
      : map.source === 'ascii' ? 'Your map' : (this.app.catalog.maps?.find((m) => m.name === map.name)?.title ?? map.name);
    const replay = state.t < s.maxT ? `<span class="badge tone-warn">⏪ replaying step ${state.t} of ${s.maxT}</span>` : '';
    this.$('map-badges').innerHTML = `<span class="badge tone-${st.tone}">${st.glyph} ${escapeHtml(st.label)}</span>
      <span class="badge plain">${icon('robot')}${escapeHtml(agent)}</span>
      <span class="badge plain">${icon('map')}${escapeHtml(mapName ?? '')}</span>${replay}`;
    this._caption(state);
    const done = s.done && s.atLive;
    this.$('btn-step').disabled = done;
    this.$('btn-finish').disabled = s.done;
    this.$('btn-back').disabled = state.t === 0;
    this._renderPanel(state);
    if (this.hover) this._showTip(this.hover);
    this._announce(state);
    this.app.onSimState?.(state);
  }

  /** The compact "last step" caption under the map (shown on narrow screens, where the side
   * panel is below the fold): the move badge and the step's key sentences. */
  _caption(state) {
    const items = keySentences(state, 2).map(({ text, tone }) => {
      const { glyph, cls } = toneStyle(tone);
      return `<li><span class="ex-ico tone-${cls}">${glyph}</span><span>${escapeHtml(text)}</span></li>`;
    }).join('');
    const head = state.t === 0 ? '<strong>Before the first move</strong>' : `<strong>Step ${state.t}</strong>${moveBadge(state)}`;
    this.$('sim-caption').innerHTML = `<div class="cap-head">${head}</div><ul class="explain compact">${items}</ul>`;
  }

  _renderPanel(state) {
    switch (this.panel) {
      case 'what': renderWhat(this.$('panel-what'), state, { onSeek: (t) => { this.pause(); this.seek(t); }, log: this.session?.live?.log }); break;
      case 'plan': this.planPanel.render(state); break;
      case 'numbers': renderNumbers(this.$('panel-numbers'), state); break;
      default: break;
    }
  }

  // --- hover tooltip -------------------------------------------------------------------
  _hover(ev) {
    if (this.use3d) return;
    if (!ev) {
      if (!this.hover) return;
      this.hover = null;
      this.renderer.setOptions({ hover: null });
      this.$('cell-tip').hidden = true;
      return;
    }
    const cell = this.renderer.cellFromEvent(ev);
    if (!cell) {
      this._hover(null);
      return;
    }
    const same = this.hover && this.hover.cell[0] === cell[0] && this.hover.cell[1] === cell[1];
    this.hover = { cell, x: ev.clientX, y: ev.clientY };
    if (same) {
      this._placeTip(this.hover); // only the pointer moved: no redraw, no new tooltip text
      return;
    }
    this.renderer.setOptions({ hover: cell });
    this._showTip(this.hover);
  }

  _showTip(hover) {
    const state = this.session?.view;
    const tip = this.$('cell-tip');
    if (!state) return;
    tip.innerHTML = cellTipHtml(state, hover.cell, { showTrue: this.showTrue });
    tip.hidden = false;
    this._placeTip(hover);
  }

  _placeTip({ x, y }) {
    const tip = this.$('cell-tip');
    if (tip.hidden) return;
    const w = tip.offsetWidth;
    const h = tip.offsetHeight;
    let left = x + 16;
    if (left + w > window.innerWidth - 8) left = x - w - 16;
    let top = y + 16;
    if (top + h > window.innerHeight - 8) top = y - h - 16;
    tip.style.left = `${Math.max(8, left)}px`;
    tip.style.top = `${Math.max(8, top)}px`;
  }

  /** Keyboard shortcuts of this view. Returns true if handled. */
  // --- renderers -----------------------------------------------------------------------
  /** The renderer currently on screen. */
  _active() {
    return this.use3d && this.renderer3d ? this.renderer3d : this.renderer;
  }

  /** Draw a state on whichever renderer is showing. */
  _paint(state, opts = {}) {
    this._active().setState(state, opts);
  }

  /** Options both renderers share, so a switch keeps the toggles it was left with. */
  _setOptions(opts) {
    this.renderer.setOptions(opts);
    this.renderer3d?.setOptions(opts);
  }

  /**
   * Switch between the flat map and the 3D view. three.js and `render3d.js` are fetched the
   * first time only; if that fails the view stays flat and says so.
   */
  async set3d(on) {
    if (on === this.use3d) return;
    const btn = this.$('tg-3d');
    if (on && !this.renderer3d) {
      if (this._loading3d) return;
      btn.disabled = true;
      this._busy(true, 'Loading the 3D view…');
      try {
        this._loading3d = import('./render3d.js');
        const { Renderer3D } = await this._loading3d;
        this.renderer3d = new Renderer3D(this.$('sim-canvas-3d'));
        this.renderer3d.setOptions({ showTrue: this.showTrue, showPolicy: this.showPolicy });
      } catch (err) {
        showError(err, 'Could not load the 3D view');
        return;
      } finally {
        this._loading3d = null;
        btn.disabled = false;
        this._busy(false);
      }
    }
    this.use3d = on;
    btn.setAttribute('aria-pressed', String(on));
    this.$('sim-canvas').hidden = on;
    this.$('sim-canvas-3d').hidden = !on;
    if (on) this.renderer3d.resize();
    else this.renderer.resize();
    const state = this.session?.view;
    if (state) this._paint(state, { animate: false });
    announce(on ? 'Showing the mission in 3D.' : 'Showing the flat map.');
  }

  handleKey(ev) {
    const k = ev.key;
    if (k === ' ' || k === 'Spacebar') { this.togglePlay(); return true; }
    if (k === 'n' || k === 'N' || k === 'ArrowRight') { this.pause(); this.stepForward(); return true; }
    if (k === 'b' || k === 'B' || k === 'ArrowLeft') { this.stepBack(); return true; }
    if (k === 'r' || k === 'R') { this.reset(); return true; }
    if (k === 't' || k === 'T') { this.setTrue(!this.showTrue); return true; }
    if (k === 'p' || k === 'P') { this.setPolicy(!this.showPolicy); return true; }
    if (k === 'd' || k === 'D') { this.set3d(!this.use3d); return true; }
    if (k === 'Home') { this.pause(); this.seek(0); return true; }
    if (k === 'End') { this.pause(); this.seek(this.session?.maxT ?? 0); return true; }
    return false;
  }
}

/** HTML of the hover card for `cell` (1-indexed everywhere). */
export function cellTipHtml(state, cell, { showTrue = false } = {}) {
  const [r, c] = cell;
  const belief = state.belief?.[r]?.[c];
  const truth = state.cells?.[r]?.[c];
  const water = state.high_slip?.[r]?.[c];
  const rows = [];
  const isRobot = state.robot && state.robot[0] === r && state.robot[1] === c;
  const isStart = state.start && state.start[0] === r && state.start[1] === c;
  const isTarget = state.target && state.target[0] === r && state.target[1] === c;
  const tags = [isRobot && 'robot', isStart && 'start', isTarget && 'target'].filter(Boolean);
  rows.push(['Robot thinks', CELL_NAMES[belief] ?? '–']);
  if (showTrue) rows.push(['Really', CELL_NAMES[truth] ?? '–']);
  if (water) rows.push(['Terrain', 'Water / mud: slips more']);
  const p = state.p_debris?.[r]?.[c];
  if (p != null) rows.push(['P(debris)', p < 0.001 ? '< 0.1 %' : `${fmtNum(p * 100, p < 0.1 ? 1 : 0)} %`]);
  const action = state.policy?.[posKey(cell)];
  if (action) rows.push(['Plan here', `${ACTION_ARROWS[action]} move ${ACTION_NAMES[action]}`]);
  const branches = state.plan?.tree ? branchesAt(state.plan.tree, cell) : [];
  const branchList = branches.length
    ? `<ul>${branches.map((b) => `<li>${escapeHtml(edgeSentence(b.edge))}: ${escapeHtml(b.text)}</li>`).join('')}</ul>`
    : '';
  return `<h4>Cell ${fmtPos(cell)}${tags.length ? ` <span class="badge plain">${tags.join(' · ')}</span>` : ''}</h4>
    <dl>${rows.map(([k, v]) => `<dt>${escapeHtml(k)}</dt><dd>${escapeHtml(v)}</dd>`).join('')}</dl>${branchList}`;
}
