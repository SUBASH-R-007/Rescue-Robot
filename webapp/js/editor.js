// Map editor: paint debris / water, place the start and survivors, resize, check the map with
// the Python validator and send it to the Simulate view. Edits show instantly (mapcore's local
// preview) and are confirmed by webapi.edit_map, whose canonical text replaces the preview.
import { MapRenderer } from './render.js';
import { lineCells } from './geometry.js';
import {
  CH, DRAG_TOOLS, applyOpLocal, asciiSize, asciiToState, blankMap, charAt, mapStats, opsForTool,
  parseAscii, stateToAscii,
} from './mapcore.js';
import { escapeHtml, fmtPos, humanize, plural } from './format.js';
import { icon, infoTip, showError, toast } from './ui.js';
import { withSize } from './panels.js';

/** Editor tools: id, label, hint, icon (the water hint gets the catalog's terrain p). */
export const TOOLS = [
  { id: 'debris', label: 'Debris', hint: 'Click or drag to drop rubble', icon: 'brush', key: '1' },
  { id: 'erase', label: 'Eraser', hint: 'Back to open floor', icon: 'eraser', key: '2' },
  { id: 'water', label: 'Water / mud', hint: 'Slippery: wheels slip with p = {high}', icon: 'water', key: '3' },
  { id: 'start', label: 'Start', hint: 'Where the robot begins', icon: 'start', key: '4' },
  { id: 'survivor', label: 'Survivor', hint: 'Move the survivor here', icon: 'person', key: '5' },
  { id: 'extra', label: 'Extra survivor', hint: 'Add or remove one (several-survivors mode)', icon: 'people', key: '6' },
];
/** Undo steps kept. */
const UNDO_LIMIT = 50;
/** Default size of a new blank map. */
const BLANK_SIZE = 10;

/**
 * Why a map text cannot be loaded into the editor (null if it can): size limits from the
 * catalog (webmaps: at most max_text characters and map_size.max rows / columns) and rows of
 * equal length. Checked before anything is drawn or sent to Python.
 */
export function loadProblem(text, limits) {
  if (text.length > limits.maxText) return `The text is too long (${text.length} characters; at most ${limits.maxText}).`;
  const rows = parseAscii(text);
  if (!rows.length) return 'The text has no map rows.';
  if (rows.length > limits.max || rows[0].length > limits.max) {
    return `The map is ${rows.length}×${rows[0].length}; the web app takes at most ${limits.max}×${limits.max}.`;
  }
  const ragged = rows.findIndex((r) => r.length !== rows[0].length);
  if (ragged >= 0) return `Map row ${ragged + 1} has ${rows[ragged].length} cells; the first row has ${rows[0].length}.`;
  return null;
}

export class EditorView {
  constructor(app, root) {
    this.app = app;
    this.root = root;
    this.tool = 'debris';
    this.ascii = blankMap(BLANK_SIZE, BLANK_SIZE);
    this.remote = this.ascii; // the text Python has confirmed
    this.chain = Promise.resolve();
    this.pending = 0;
    this.undo = [];
    this.check = null; // last validation: {ok, solvable, errors, warnings}
    this.title = 'Blank canvas';
    this.stroke = null;
    this.generation = 0; // bumped by load(): edits of an earlier map are dropped
    const size = app.catalog?.params?.map_size ?? {};
    this.limits = { min: size.editor_min, max: size.max, maxText: size.max_text };
    this._build();
    this._render();
    this._validate();
  }

  // --- layout --------------------------------------------------------------------------
  _build() {
    const cat = this.app.catalog ?? {};
    const sizes = cat.params?.map_size?.library ?? cat.library?.sizes ?? [10, 15, 20];
    const maps = cat.maps ?? [];
    const scenarios = cat.scenarios ?? [];
    const high = cat.params?.terrain_high_p?.value;
    this.root.innerHTML = `
      <h2 class="sr-only">Map editor</h2>
      <div class="ed-layout">
        <div class="ed-tools card" role="toolbar" aria-label="Map editor tools">
          <h4 class="panel-title">${icon('brush')}Tools</h4>
          <div class="tool-list">${TOOLS.map((t) => {
            const hint = t.hint.replace('{high}', String(high));
            return `<button type="button" class="tool" data-tool="${t.id}" aria-pressed="${t.id === this.tool}" title="${escapeHtml(hint)} (key ${t.key})">
            <span class="sw" aria-hidden="true">${icon(t.icon)}</span><span>${escapeHtml(t.label)}<small>${escapeHtml(hint)}</small></span></button>`;
          }).join('')}</div>
          <div class="stack">
            <button type="button" class="btn small" id="ed-undo" disabled>${icon('reset')}Undo</button>
            <button type="button" class="btn small" id="ed-clear">${icon('eraser')}Clear debris</button>
          </div>
          <h4 class="panel-title">${icon('expand')}Size</h4>
          <div class="size-row">
            <label class="sr-only" for="ed-rows">Rows</label><input type="number" id="ed-rows" min="${this.limits.min}" max="${this.limits.max}" step="1">
            <span aria-hidden="true">×</span>
            <label class="sr-only" for="ed-cols">Columns</label><input type="number" id="ed-cols" min="${this.limits.min}" max="${this.limits.max}" step="1">
            <button type="button" class="btn small" id="ed-resize">Resize</button>
          </div>
          <button type="button" class="btn small" id="ed-blank">${icon('plus')}New blank map</button>
        </div>

        <div class="ed-canvas-card card">
          <div class="map-bar"><div class="map-badges" id="ed-badges"></div></div>
          <div class="canvas-box" id="ed-canvas-box">
            <canvas id="ed-canvas" role="img" aria-label="Map being edited"></canvas>
          </div>
        </div>

        <div class="ed-side">
          <div class="card">
            <h4 class="panel-title">${icon('check')}Check <span class="muted">(updates as you edit)</span></h4>
            <div class="stack">
              <button type="button" class="btn primary" id="ed-use">${icon('play')}Use this map</button>
            </div>
            <div class="ed-check">
              <div class="ed-status unknown" id="ed-status" role="status" aria-live="polite">Not checked yet</div>
              <ul class="ed-msgs" id="ed-msgs"></ul>
            </div>
          </div>
          <div class="card">
            <h4 class="panel-title">${icon('map')}Start from${infoTip('Load any bundled map or a generated map as a starting point, then edit it.', 'Start from')}</h4>
            <div class="stack">
              <label class="field-hint" for="ed-load-file">Bundled map</label>
              <select id="ed-load-file"><option value="">Choose a map…</option>${maps.map((m) => `<option value="${escapeHtml(m.name)}">${escapeHtml(withSize(m.title ?? m.name, m.rows, m.cols))}</option>`).join('')}</select>
              <label class="field-hint" for="ed-load-scen">Generated map</label>
              <select id="ed-load-scen">${scenarios.map((s) => `<option value="${escapeHtml(s.name)}">${escapeHtml(s.title ?? humanize(s.name))}</option>`).join('')}</select>
              <div class="size-row">
                <label class="field-hint" for="ed-load-size">Size</label>
                <select id="ed-load-size">${sizes.map((n) => `<option value="${n}">${n}×${n}</option>`).join('')}</select>
                <label class="field-hint" for="ed-load-seed">Seed</label>
                <input type="number" id="ed-load-seed" min="0" step="1" value="0">
              </div>
              <button type="button" class="btn small" id="ed-load-gen">${icon('dice')}Load generated map</button>
              <button type="button" class="btn small" id="ed-load-sim">${icon('robot')}Copy the Simulate map</button>
            </div>
          </div>
          <div class="card">
            <h4 class="panel-title">${icon('book')}Text form${infoTip('The same ASCII format as the files in maps/: . open, # debris, ~ water or mud, S start, P survivor. You can paste a map here.', 'Text form')}</h4>
            <label class="sr-only" for="ed-ascii">Map as text</label>
            <textarea id="ed-ascii" class="ed-ascii" spellcheck="false" autocomplete="off"></textarea>
            <button type="button" class="btn small" id="ed-ascii-apply" style="margin-top:8px">Apply text</button>
          </div>
        </div>
      </div>`;
    const $ = (sel) => this.root.querySelector(sel);
    this.canvas = $('#ed-canvas');
    this.renderer = new MapRenderer(this.canvas);
    this.renderer.setOptions({ editor: true, showPath: false });
    this.root.querySelector('.tool-list').addEventListener('click', (ev) => {
      const b = ev.target.closest('[data-tool]');
      if (b) this.setTool(b.dataset.tool);
    });
    $('#ed-undo').addEventListener('click', () => this.undoLast());
    $('#ed-clear').addEventListener('click', () => this._edit([{ op: 'clear' }], { record: true }));
    $('#ed-resize').addEventListener('click', () => {
      const rows = Number($('#ed-rows').value);
      const cols = Number($('#ed-cols').value);
      const { min, max } = this.limits;
      if (!(rows >= min && rows <= max && cols >= min && cols <= max)) {
        toast({ title: 'Size out of range', text: `Rows and columns must be between ${min} and ${max}.`, tone: 'warn' });
        return;
      }
      this._edit([{ op: 'resize', rows, cols }], { record: true });
    });
    $('#ed-blank').addEventListener('click', () => {
      const { rows, cols } = asciiSize(this.ascii);
      this.load(blankMap(rows || BLANK_SIZE, cols || BLANK_SIZE), 'Blank canvas');
    });
    $('#ed-use').addEventListener('click', () => this.use());
    $('#ed-load-file').addEventListener('change', (ev) => {
      const m = maps.find((x) => x.name === ev.target.value);
      if (m?.ascii) this.load(m.ascii, m.title ?? m.name);
      else if (m) this._loadSpec({ source: 'file', name: m.name }, m.title ?? m.name);
      ev.target.value = '';
    });
    $('#ed-load-gen').addEventListener('click', () => {
      const scenario = $('#ed-load-scen').value;
      const n = Number($('#ed-load-size').value);
      const seed = Math.max(0, Math.floor(Number($('#ed-load-seed').value) || 0));
      const title = scenarios.find((s) => s.name === scenario)?.title ?? humanize(scenario);
      this._loadSpec({ source: 'scenario', scenario, rows: n, cols: n, seed }, `${title} ${n}×${n}, seed ${seed}`);
    });
    $('#ed-load-sim').addEventListener('click', () => {
      const st = this.app.views.simulate?.session?.view;
      if (st) this.load(stateToAscii(st), 'Map from Simulate');
    });
    $('#ed-ascii-apply').addEventListener('click', () => this.load($('#ed-ascii').value, 'Your text map'));
    // Painting with mouse, pen or finger.
    this.canvas.addEventListener('pointerdown', (ev) => this._down(ev));
    this.canvas.addEventListener('pointermove', (ev) => this._move(ev));
    this.canvas.addEventListener('pointerup', (ev) => this._up(ev));
    this.canvas.addEventListener('pointercancel', (ev) => this._up(ev));
    this.canvas.addEventListener('pointerleave', () => this.renderer.setOptions({ hover: null }));
    this.canvas.style.touchAction = 'none';
  }

  onShow() {
    this.renderer.resize();
  }

  /** Keyboard shortcuts: 1–6 pick a tool. */
  handleKey(ev) {
    const tool = TOOLS.find((t) => t.key === ev.key);
    if (tool) {
      this.setTool(tool.id);
      return true;
    }
    return false;
  }

  setTool(id) {
    this.tool = id;
    for (const b of this.root.querySelectorAll('[data-tool]')) b.setAttribute('aria-pressed', String(b.dataset.tool === id));
  }

  /** Replace the whole map (new blank, loaded map, pasted text). Returns false (and says why)
   * when the text cannot be loaded. Edits still on their way to Python belong to the old
   * map: the new generation drops their answers. */
  load(text, title = 'Your map') {
    const problem = loadProblem(String(text ?? ''), this.limits);
    if (problem) {
      toast({ title: 'This map cannot be loaded', text: problem, tone: 'bad', timeout: 6000 });
      return false;
    }
    this.generation += 1;
    this.pending = 0;
    this.stroke = null;
    this._pushUndo();
    this.title = title;
    this.ascii = text;
    this.remote = text;
    this._render();
    this._validate();
    return true;
  }

  async _loadSpec(spec, title) {
    try {
      const text = await this.app.backend.mapAscii(spec, 'multi');
      this.load(text, title);
    } catch (err) {
      showError(err, 'Could not load that map');
    }
  }

  // --- painting ------------------------------------------------------------------------
  _cell(ev) {
    return this.renderer.cellFromEvent(ev);
  }

  _down(ev) {
    const cell = this._cell(ev);
    if (!cell || ev.button > 0) return;
    ev.preventDefault();
    this.canvas.setPointerCapture(ev.pointerId);
    this._pushUndo();
    this.stroke = { last: cell, ops: [] };
    this._paint([cell]);
    if (!DRAG_TOOLS.has(this.tool)) this._endStroke();
  }

  _move(ev) {
    const cell = this._cell(ev);
    this.renderer.setOptions({ hover: cell });
    if (!this.stroke || !cell) return;
    const { last } = this.stroke;
    if (last[0] === cell[0] && last[1] === cell[1]) return;
    this._paint(lineCells(last, cell).slice(1));
    this.stroke.last = cell;
  }

  _up(ev) {
    try { this.canvas.releasePointerCapture(ev.pointerId); } catch { /* not captured */ }
    if (this.stroke) this._endStroke();
  }

  _paint(cells) {
    const ops = [];
    for (const [r, c] of cells) {
      for (const op of opsForTool(this.tool, charAt(this.ascii, r, c), r, c)) {
        this.ascii = applyOpLocal(this.ascii, op);
        ops.push(op);
      }
    }
    if (!ops.length) return;
    this.stroke.ops.push(...ops);
    this._render();
  }

  _endStroke() {
    const { ops } = this.stroke;
    this.stroke = null;
    if (!ops.length) {
      this.undo.pop(); // nothing changed
      this._syncUndo();
      return;
    }
    this._send(ops);
  }

  /** Apply ops locally and in Python (record = add an undo step first). */
  _edit(ops, { record = false } = {}) {
    if (record) this._pushUndo();
    for (const op of ops) this.ascii = applyOpLocal(this.ascii, op);
    this._render();
    this._send(ops);
  }

  /** Send ops to webapi.edit_map one after another; the last answer is authoritative. A map
   * loaded meanwhile (a new generation) makes the remaining ops and their answers void. */
  _send(ops) {
    const generation = this.generation;
    this.pending += 1;
    this.chain = this.chain.then(async () => {
      let result = null;
      const warnings = [];
      try {
        for (const op of ops) {
          if (generation !== this.generation) return;
          const answer = await this.app.backend.editMap(this.remote, op);
          if (generation !== this.generation) return;
          result = answer;
          this.remote = result.ascii;
          if (result.changed === false && result.warnings?.length) warnings.push(...result.warnings);
        }
      } catch (err) {
        if (generation === this.generation) showError(err, 'The edit was refused');
      }
      if (generation !== this.generation) return;
      this.pending -= 1;
      if (this.pending === 0) {
        this.ascii = this.remote;
        if (result) this._showCheck({ ok: result.valid ?? true, solvable: result.solvable, errors: result.errors ?? [], warnings: result.warnings ?? [], problems: result.problems ?? [] });
        this._render();
        if (warnings.length) toast({ title: 'Not changed', text: [...new Set(warnings)][0], tone: 'warn', timeout: 3500 });
      }
    });
  }

  _pushUndo() {
    this.undo.push({ ascii: this.ascii, title: this.title });
    if (this.undo.length > UNDO_LIMIT) this.undo.shift();
    this._syncUndo();
  }

  undoLast() {
    if (!this.undo.length || this.pending) return;
    const { ascii, title } = this.undo.pop();
    this.ascii = ascii;
    this.title = title;
    this.remote = this.ascii;
    this._syncUndo();
    this._render();
    this._validate();
  }

  _syncUndo() {
    const b = this.root.querySelector('#ed-undo');
    if (b) b.disabled = this.undo.length === 0;
  }

  // --- checking and using ----------------------------------------------------------------
  /** The mode a map needs: several survivors -> 'multi', else 'core'. */
  modeFor(text = this.ascii) {
    return mapStats(text).survivors > 1 ? 'multi' : 'core';
  }

  async _validate(announce = false) {
    const generation = this.generation;
    try {
      const result = await this.app.backend.validateMap(this.ascii, this.modeFor());
      if (generation !== this.generation) return result;
      this._showCheck(result);
      if (announce) {
        const tone = !result.ok ? 'bad' : result.solvable === false ? 'warn' : 'good';
        const title = !result.ok ? 'This map cannot be used yet' : result.solvable === false ? 'Valid, but no route' : 'Map looks good';
        toast({ title, text: [...(result.errors ?? []), ...(result.warnings ?? [])][0] ?? 'Every survivor can be reached from the start.', tone, timeout: 3500 });
      }
      return result;
    } catch (err) {
      showError(err, 'Could not check the map');
      return null;
    }
  }

  _showCheck(result) {
    this.check = result;
    const status = this.root.querySelector('#ed-status');
    const msgs = this.root.querySelector('#ed-msgs');
    let cls;
    let text;
    if (!result.ok) {
      cls = 'bad';
      text = `${icon('alert')}Not usable yet`;
    } else if (result.solvable === false) {
      cls = 'bad';
      text = `${icon('alert')}Valid, but a survivor is walled in`;
    } else {
      cls = 'ok';
      text = `${icon('check')}Ready: every survivor is reachable`;
    }
    status.className = `ed-status ${cls}`;
    status.innerHTML = text;
    const lines = [...(result.errors ?? []), ...(result.warnings ?? [])];
    msgs.innerHTML = lines.map((m) => `<li>${escapeHtml(m)}</li>`).join('');
    // Problem cells are outlined on the map; an unusable or unsolvable map demotes the button.
    this.renderer.setOptions({ problems: result.problems ?? [] });
    const use = this.root.querySelector('#ed-use');
    use.disabled = !result.ok;
    use.classList.toggle('primary', result.ok && result.solvable !== false);
    this._armed = false;
    use.lastChild.textContent = result.ok && result.solvable === false ? 'Use anyway…' : 'Use this map';
  }

  /** Send the map to Simulate (several survivors switch to the several-survivors mode). */
  async use() {
    const armed = this._armed && this._armedFor === this.ascii; // second click, same map
    const check = await this._validate();
    if (!check?.ok) {
      toast({ title: 'Fix the map first', text: check?.errors?.[0] ?? 'The map is not valid.', tone: 'bad' });
      return;
    }
    const sim = this.app.views.simulate;
    const base = sim.config ?? {};
    const survivors = mapStats(this.ascii).survivors;
    const mode = this.modeFor();
    const agents = (this.app.catalog.agents ?? []).filter((a) => a.modes?.includes(mode));
    const agent = agents.some((a) => a.name === base.agent) ? base.agent : agents[0]?.name;
    const config = {
      mode,
      agent,
      map: { source: 'ascii', name: 'custom', text: this.ascii },
      slip_p: base.slip_p,
      sensor_radius: base.sensor_radius,
      replan_mode: base.replan_mode,
    };
    if (mode === 'multi') config.k = survivors;
    for (const key of Object.keys(config)) if (config[key] === undefined) delete config[key];
    let warning = '';
    if (check.solvable === false) {
      // Confirm with a second click (the button says so; no blocking browser dialog).
      const button = this.root.querySelector('#ed-use');
      if (!armed) {
        this._armed = true;
        this._armedFor = this.ascii;
        button.lastChild.textContent = 'Click again to use it anyway';
        return;
      }
      warning = 'A survivor is walled in on this map, so the robot will end with “No route”. Clear some debris in the map editor to give it a chance.';
    }
    this.app.customMap = this.ascii;
    await this.app.switchView('simulate');
    const where = mode === 'multi' ? ` It has ${plural(survivors, 'survivor')}, so the “Several survivors” mode is used.` : '';
    await sim.start(config, { note: `Your map is loaded.${where}`, warning });
  }

  // --- drawing -------------------------------------------------------------------------
  _render() {
    const state = asciiToState(this.ascii);
    this.renderer.setState(state);
    const stats = mapStats(this.ascii);
    const { rows, cols } = asciiSize(this.ascii);
    const $ = (sel) => this.root.querySelector(sel);
    if (document.activeElement !== $('#ed-rows')) $('#ed-rows').value = String(rows);
    if (document.activeElement !== $('#ed-cols')) $('#ed-cols').value = String(cols);
    if (document.activeElement !== $('#ed-ascii')) $('#ed-ascii').value = this.ascii;
    const start = state.start ? `start ${fmtPos(state.start)}` : 'no start';
    $('#ed-badges').innerHTML = `<span class="badge plain">${icon('map')}${escapeHtml(this.title)}</span>
      <span class="badge plain">${rows}×${cols}</span>
      <span class="badge plain">${escapeHtml(start)}</span>
      <span class="badge plain">${plural(stats.survivors, 'survivor')}</span>
      <span class="badge plain">${plural(stats.debris, 'debris cell')}</span>
      ${stats.water ? `<span class="badge plain">${plural(stats.water, 'water cell')}</span>` : ''}`;
  }
}

export { CH };
