// Canvas renderer of a mission state: fog of war, debris hatch, water waves, P(debris)
// shading, true-map overlay, policy arrows, A* path, the executed trace (colour AND glyph
// per outcome), start, survivors, target flag and the robot. Crisp at any devicePixelRatio.
import { computeLayout, cellAt, cellOrigin, cellCenter, backingSize, DELTA } from './geometry.js';
import { CELL, normOutcome, samePos, parseKey } from './format.js';

/** Robot glide between cells, ms (0 with prefers-reduced-motion). */
const GLIDE_MS = 160;
/** Duration of the "newly sensed debris" pulse, ms. */
const PULSE_MS = 1400;
/** P(debris) below this is not shaded (log scale up to PROB_SHADE_MAX). */
const PROB_SHADE_MIN = 1e-3;
const PROB_SHADE_MAX = 0.5;
const PROB_LEVELS = 5;
/** Cell sizes (CSS px) from which the fog "?" and the START label are drawn; smaller fog cells
 * get a dot instead, smaller start cells only the dashed frame. */
const FOG_GLYPH_MIN = 16;
const FOG_DOT_MIN = 5;
const START_LABEL_MIN = 28;

const LEFT_OF = { N: 'W', E: 'N', S: 'E', W: 'S' };
const RIGHT_OF = { N: 'E', E: 'S', S: 'W', W: 'N' };

/** Shading level 0..PROB_LEVELS of a debris probability (log scale, like the Pygame app). */
export function probLevel(p) {
  if (p == null || !(p > PROB_SHADE_MIN)) return 0;
  const x = Math.log(p / PROB_SHADE_MIN) / Math.log(PROB_SHADE_MAX / PROB_SHADE_MIN);
  return Math.max(1, Math.min(PROB_LEVELS, Math.ceil(PROB_LEVELS * x)));
}

/** Direction the robot actually tried to move in for a trace entry (skids turn sideways). */
export function attemptedDir(entry) {
  const o = normOutcome(entry.outcome);
  if (o === 'skid-L') return LEFT_OF[entry.action];
  if (o === 'skid-R') return RIGHT_OF[entry.action];
  return entry.action;
}

const TOKENS = [
  'map-bg', 'map-floor', 'map-grid', 'map-fog', 'map-fog-ink', 'map-debris', 'map-debris-ink',
  'map-water', 'map-water-ink', 'map-ruler', 'map-robot', 'map-robot-ink', 'map-survivor',
  'map-survivor-ink', 'map-start', 'map-target', 'map-policy', 'map-path', 'map-hover',
  'map-true', 'map-prob', 'c-ok', 'c-slip', 'c-skid', 'c-bump', 'c-replan', 'c-rescue',
];

export class MapRenderer {
  constructor(canvas, { compact = false, interactive = true, size = null } = {}) {
    this.canvas = canvas;
    this.ctx = canvas.getContext('2d');
    this.compact = compact;
    this.state = null;
    this.opts = { showTrue: false, showPolicy: false, showPath: true, hover: null, highlight: null, editor: false, problems: null };
    this.layout = null;
    this.colors = {};
    this.pulse = { cells: [], start: 0 };
    this.glide = null;
    this._raf = 0;
    this.reducedMotion = matchMedia('(prefers-reduced-motion: reduce)');
    this.fixedSize = size;
    this.refreshTheme();
    if (!size) {
      this._onTheme = () => this.refreshTheme();
      window.addEventListener('rr-theme', this._onTheme);
      this._ro = new ResizeObserver(() => this.resize());
      this._ro.observe(canvas.parentElement);
      this._watchDpr();
    }
    this.resize();
    if (interactive) canvas.addEventListener('contextmenu', (e) => e.preventDefault());
  }

  /** Stop listening (theme, size, pixel ratio) and drop the state: call before discarding a
   * renderer (Compare panes are rebuilt on every reset or agent change). */
  destroy() {
    if (this._onTheme) window.removeEventListener('rr-theme', this._onTheme);
    this._ro?.disconnect();
    this._dprQuery?.removeEventListener('change', this._dprHandler);
    if (this._raf) cancelAnimationFrame(this._raf);
    if (this._drawRaf) cancelAnimationFrame(this._drawRaf);
    this._raf = 0;
    this._drawRaf = 0;
    this.state = null;
    this.destroyed = true;
  }

  _watchDpr() {
    this._dprQuery?.removeEventListener('change', this._dprHandler);
    this._dprHandler = () => { this.resize(); this._watchDpr(); };
    this._dprQuery = matchMedia(`(resolution: ${window.devicePixelRatio}dppx)`);
    this._dprQuery.addEventListener('change', this._dprHandler);
  }

  /** Re-read the CSS colour tokens (after a theme switch). */
  refreshTheme() {
    const css = getComputedStyle(document.documentElement);
    for (const t of TOKENS) this.colors[t] = css.getPropertyValue(`--${t}`).trim() || '#888';
    this.font = css.getPropertyValue('--font-ui').trim() || 'system-ui, sans-serif';
    this.draw();
  }

  resize() {
    const box = this.fixedSize ?? this.canvas.parentElement.getBoundingClientRect();
    const cssW = Math.max(1, Math.floor(box.width));
    const cssH = Math.max(1, Math.floor(box.height));
    const { width, height, ratio } = backingSize(cssW, cssH, window.devicePixelRatio || 1);
    if (this.canvas.width !== width || this.canvas.height !== height) {
      this.canvas.width = width;
      this.canvas.height = height;
    }
    this.canvas.style.width = `${cssW}px`;
    this.canvas.style.height = `${cssH}px`;
    this.cssW = cssW;
    this.cssH = cssH;
    this.dpr = ratio;
    this.draw();
  }

  /** New state; `animate` glides the robot from its previous cell and pulses new debris. */
  setState(state, { animate = false, prev = null } = {}) {
    const old = this.state;
    this.state = state;
    const motion = animate && !this.reducedMotion.matches;
    if (motion && old?.robot && state?.robot && !samePos(old.robot, state.robot)) {
      this.glide = { from: old.robot, to: state.robot, start: performance.now() };
    } else {
      this.glide = null;
    }
    const before = prev ?? old;
    if (animate && before && state && before.rows === state.rows && before.cols === state.cols) {
      const cells = [];
      for (let r = 0; r < state.rows; r += 1) {
        for (let c = 0; c < state.cols; c += 1) {
          if (state.belief[r][c] === CELL.DEBRIS && before.belief[r][c] !== CELL.DEBRIS) cells.push([r, c]);
        }
      }
      this.pulse = { cells, start: performance.now() };
    } else if (!animate) {
      this.pulse = { cells: [], start: 0 };
    }
    this._kick();
  }

  /** Change display options; redraws (once per frame) only when something changed. */
  setOptions(opts) {
    let changed = false;
    for (const [key, value] of Object.entries(opts)) {
      if (!sameOption(this.opts[key], value)) {
        this.opts[key] = value;
        changed = true;
      }
    }
    if (changed) this._scheduleDraw();
  }

  _scheduleDraw() {
    if (this._raf || this._drawRaf || this.destroyed) return; // a frame is already coming
    this._drawRaf = requestAnimationFrame(() => {
      this._drawRaf = 0;
      this.draw();
    });
  }

  /** [r, c] under a pointer event, or null. */
  cellFromEvent(ev) {
    if (!this.layout) return null;
    const rect = this.canvas.getBoundingClientRect();
    return cellAt(this.layout, ev.clientX - rect.left, ev.clientY - rect.top);
  }

  /** CSS rectangle of a cell relative to the canvas (for tooltips / the tour). */
  cellRect(r, c) {
    if (!this.layout) return null;
    const o = cellOrigin(this.layout, r, c);
    return { x: o.x, y: o.y, size: this.layout.cell };
  }

  _kick() {
    if (this._raf) return;
    const loop = () => {
      this._raf = 0;
      this.draw();
      const now = performance.now();
      const gliding = this.glide && now - this.glide.start < GLIDE_MS;
      const pulsing = this.pulse.cells.length && now - this.pulse.start < PULSE_MS;
      if (gliding || pulsing) this._raf = requestAnimationFrame(loop);
    };
    this._raf = requestAnimationFrame(loop);
  }

  draw() {
    const { ctx, state } = this;
    if (!ctx || !this.cssW || this.destroyed) return;
    ctx.setTransform(this.dpr, 0, 0, this.dpr, 0, 0);
    ctx.clearRect(0, 0, this.cssW, this.cssH);
    if (!state || !state.rows) {
      this.layout = null;
      return;
    }
    const L = computeLayout(state.rows, state.cols, this.cssW, this.cssH, {
      ruler: !this.compact, maxCell: this.compact ? 64 : 96, minCell: 4,
    });
    this.layout = L;
    this._cells(L);
    this._gridLines(L);
    if (!this.compact) this._rulers(L);
    if (this.opts.showTrue && !this.opts.editor) this._trueOverlay(L);
    if (this.opts.showPolicy) this._policy(L);
    if (this.opts.showPath && state.path) this._path(L);
    this._trace(L);
    this._pulse(L);
    this._start(L);
    this._target(L);
    this._survivors(L);
    this._robot(L);
    this._problems(L);
    this._hover(L);
  }

  // --- cells ----------------------------------------------------------------------------
  _cells(L) {
    const { ctx, state, colors } = this;
    const s = L.cell;
    const view = this.opts.editor ? state.cells : state.belief;
    ctx.fillStyle = colors['map-bg'];
    ctx.fillRect(L.ox, L.oy, L.gridW, L.gridH);
    for (let r = 0; r < state.rows; r += 1) {
      for (let c = 0; c < state.cols; c += 1) {
        const x = L.ox + c * s;
        const y = L.oy + r * s;
        const code = view[r][c];
        const water = state.high_slip?.[r]?.[c];
        if (code === CELL.UNKNOWN) {
          ctx.fillStyle = colors['map-fog'];
          ctx.fillRect(x, y, s, s);
          if (water) this._waves(x, y, s, 0.6);
          // Unknown cells always carry a mark in fog ink (>= 3:1 against fog and floor), so
          // explored and unexplored ground differ by more than a subtle fill.
          ctx.fillStyle = colors['map-fog-ink'];
          if (s >= FOG_GLYPH_MIN) {
            ctx.font = `700 ${Math.round(s * 0.42)}px ${this.font}`;
            ctx.textAlign = 'center';
            ctx.textBaseline = 'middle';
            ctx.fillText('?', x + s / 2, y + s / 2 + 1);
          } else if (s >= FOG_DOT_MIN) {
            ctx.beginPath();
            ctx.arc(x + s / 2, y + s / 2, Math.max(1, s * 0.12), 0, Math.PI * 2);
            ctx.fill();
          }
        } else if (code === CELL.DEBRIS) {
          this._debris(x, y, s, 1);
        } else {
          ctx.fillStyle = colors['map-floor'];
          ctx.fillRect(x, y, s, s);
          if (water) this._waves(x, y, s, 1);
          const p = state.p_debris?.[r]?.[c];
          const level = probLevel(p);
          if (level) this._probShade(x, y, s, level);
        }
      }
    }
  }

  _debris(x, y, s, alpha) {
    const { ctx, colors } = this;
    ctx.save();
    ctx.globalAlpha = alpha;
    ctx.fillStyle = colors['map-debris'];
    ctx.fillRect(x, y, s, s);
    ctx.beginPath();
    ctx.rect(x, y, s, s);
    ctx.clip();
    ctx.strokeStyle = colors['map-debris-ink'];
    ctx.lineWidth = Math.max(1, s * 0.07);
    const gap = Math.max(4, s / 4);
    ctx.beginPath();
    for (let d = -s; d < s * 2; d += gap) {
      ctx.moveTo(x + d, y + s);
      ctx.lineTo(x + d + s, y);
    }
    ctx.stroke();
    ctx.restore();
  }

  _waves(x, y, s, alpha) {
    const { ctx, colors } = this;
    ctx.save();
    ctx.globalAlpha = alpha;
    ctx.fillStyle = colors['map-water'];
    ctx.fillRect(x, y, s, s);
    ctx.strokeStyle = colors['map-water-ink'];
    ctx.lineWidth = Math.max(1, s * 0.06);
    ctx.lineCap = 'round';
    const amp = s * 0.07;
    for (const fy of [0.36, 0.66]) {
      ctx.beginPath();
      const yy = y + s * fy;
      for (let i = 0; i <= 12; i += 1) {
        const xx = x + s * 0.18 + (s * 0.64 * i) / 12;
        const yv = yy + Math.sin((i / 12) * Math.PI * 2) * amp;
        if (i === 0) ctx.moveTo(xx, yv);
        else ctx.lineTo(xx, yv);
      }
      ctx.stroke();
    }
    ctx.restore();
  }

  _probShade(x, y, s, level) {
    const { ctx, colors } = this;
    ctx.save();
    ctx.globalAlpha = 0.1 + 0.1 * level;
    ctx.fillStyle = colors['map-prob'];
    ctx.fillRect(x, y, s, s);
    ctx.globalAlpha = 0.55;
    ctx.strokeStyle = colors['map-prob'];
    ctx.lineWidth = Math.max(1, s * 0.04);
    ctx.beginPath();
    ctx.rect(x, y, s, s);
    ctx.clip();
    const gap = s / (level + 1);
    ctx.beginPath();
    for (let d = gap; d < s * 2; d += gap) {
      ctx.moveTo(x + d, y);
      ctx.lineTo(x, y + d);
    }
    ctx.stroke();
    ctx.restore();
  }

  _gridLines(L) {
    const { ctx, colors, dpr } = this;
    if (L.cell < 8) return;
    ctx.save();
    ctx.strokeStyle = colors['map-grid'];
    ctx.lineWidth = 1 / dpr;
    ctx.beginPath();
    const snap = (v) => (Math.round(v * dpr) + 0.5) / dpr;
    for (let r = 0; r <= L.rows; r += 1) {
      const y = snap(L.oy + r * L.cell);
      ctx.moveTo(L.ox, y);
      ctx.lineTo(L.ox + L.gridW, y);
    }
    for (let c = 0; c <= L.cols; c += 1) {
      const x = snap(L.ox + c * L.cell);
      ctx.moveTo(x, L.oy);
      ctx.lineTo(x, L.oy + L.gridH);
    }
    ctx.stroke();
    ctx.restore();
  }

  _rulers(L) {
    if (!L.ruler) return;
    const { ctx, colors } = this;
    ctx.save();
    ctx.fillStyle = colors['map-ruler'];
    const size = Math.max(9, Math.min(12, Math.round(L.cell * 0.4)));
    ctx.font = `500 ${size}px ${this.font}`;
    ctx.textAlign = 'center';
    ctx.textBaseline = 'middle';
    const every = L.cell < 20 ? 5 : 1;
    for (let c = 0; c < L.cols; c += 1) {
      if ((c + 1) % every && c !== 0) continue;
      ctx.fillText(String(c + 1), L.ox + (c + 0.5) * L.cell, L.oy - L.ruler / 2);
    }
    ctx.textAlign = 'right';
    for (let r = 0; r < L.rows; r += 1) {
      if ((r + 1) % every && r !== 0) continue;
      ctx.fillText(String(r + 1), L.ox - 6, L.oy + (r + 0.5) * L.cell);
    }
    ctx.restore();
  }

  _trueOverlay(L) {
    const { ctx, state, colors } = this;
    const s = L.cell;
    for (let r = 0; r < state.rows; r += 1) {
      for (let c = 0; c < state.cols; c += 1) {
        const truth = state.cells[r][c];
        const belief = state.belief[r][c];
        const { x, y } = cellOrigin(L, r, c);
        if (truth === CELL.DEBRIS && belief !== CELL.DEBRIS) {
          this._debris(x, y, s, 0.42);
          ctx.save();
          ctx.strokeStyle = colors['map-true'];
          ctx.lineWidth = Math.max(1.5, s * 0.06);
          ctx.setLineDash([Math.max(2, s * 0.12), Math.max(2, s * 0.1)]);
          ctx.strokeRect(x + 2, y + 2, s - 4, s - 4);
          ctx.restore();
        } else if (truth !== CELL.DEBRIS && belief === CELL.DEBRIS) {
          // A phantom: believed debris that is really free (noisy sensors).
          ctx.save();
          ctx.strokeStyle = colors['map-true'];
          ctx.lineWidth = Math.max(1.5, s * 0.06);
          ctx.setLineDash([2, 3]);
          ctx.strokeRect(x + 3, y + 3, s - 6, s - 6);
          ctx.restore();
        }
      }
    }
  }

  // --- plan overlays -------------------------------------------------------------------
  _arrow(cx, cy, dir, len, color, width) {
    const { ctx } = this;
    const [dr, dc] = DELTA[dir] ?? [0, 0];
    const x1 = cx - (dc * len) / 2;
    const y1 = cy - (dr * len) / 2;
    const x2 = cx + (dc * len) / 2;
    const y2 = cy + (dr * len) / 2;
    ctx.save();
    ctx.strokeStyle = color;
    ctx.fillStyle = color;
    ctx.lineWidth = width;
    ctx.lineCap = 'round';
    ctx.beginPath();
    ctx.moveTo(x1, y1);
    ctx.lineTo(x2, y2);
    ctx.stroke();
    const head = Math.max(3, len * 0.38);
    const ang = Math.atan2(y2 - y1, x2 - x1);
    ctx.beginPath();
    ctx.moveTo(x2 + Math.cos(ang) * head * 0.3, y2 + Math.sin(ang) * head * 0.3);
    ctx.lineTo(x2 - Math.cos(ang - 0.5) * head, y2 - Math.sin(ang - 0.5) * head);
    ctx.lineTo(x2 - Math.cos(ang + 0.5) * head, y2 - Math.sin(ang + 0.5) * head);
    ctx.closePath();
    ctx.fill();
    ctx.restore();
  }

  _policy(L) {
    const { state, colors } = this;
    for (const [key, dir] of Object.entries(state.policy ?? {})) {
      const [r, c] = parseKey(key);
      const { x, y } = cellCenter(L, r, c);
      this._arrow(x, y, dir, L.cell * 0.46, colors['map-policy'], Math.max(1.5, L.cell * 0.07));
    }
  }

  _path(L) {
    const { ctx, state, colors } = this;
    const path = state.path;
    if (!path || path.length < 2) return;
    // Show the remaining path: from the robot's cell (or its first match) to the end.
    let i0 = path.findIndex((p) => samePos(p, state.robot));
    if (i0 < 0) i0 = 0;
    ctx.save();
    ctx.strokeStyle = colors['map-path'];
    ctx.lineWidth = Math.max(2, L.cell * 0.09);
    ctx.setLineDash([Math.max(2, L.cell * 0.08), Math.max(3, L.cell * 0.14)]);
    ctx.lineCap = 'round';
    ctx.beginPath();
    path.slice(i0).forEach((p, i) => {
      const { x, y } = cellCenter(L, p[0], p[1]);
      if (i === 0) ctx.moveTo(x, y);
      else ctx.lineTo(x, y);
    });
    ctx.stroke();
    ctx.restore();
  }

  // --- trace ---------------------------------------------------------------------------
  _trace(L) {
    const { ctx, state, colors } = this;
    const trace = state.trace ?? [];
    const s = L.cell;
    const w = Math.max(2, s * 0.09);
    const slipCount = new Map();
    trace.forEach((e, i) => {
      const last = i === trace.length - 1;
      const o = normOutcome(e.outcome);
      const a = cellCenter(L, e.from[0], e.from[1]);
      const b = cellCenter(L, e.to[0], e.to[1]);
      ctx.save();
      ctx.globalAlpha = last ? 1 : 0.85;
      ctx.lineCap = 'round';
      if (o === 'slipped') {
        const key = `${e.from[0]},${e.from[1]}`;
        slipCount.set(key, (slipCount.get(key) ?? 0) + 1);
      } else if (e.blocked || e.bump) {
        const dir = attemptedDir(e);
        const [dr, dc] = DELTA[dir] ?? [0, 0];
        const ex = a.x + dc * s * 0.45;
        const ey = a.y + dr * s * 0.45;
        const color = e.bump ? colors['c-bump'] : colors['c-skid'];
        ctx.strokeStyle = color;
        ctx.lineWidth = w;
        ctx.beginPath();
        ctx.moveTo(a.x, a.y);
        ctx.lineTo(ex, ey);
        ctx.stroke();
        // Bar across the stub end: the move was stopped there.
        const bar = s * 0.16;
        ctx.beginPath();
        ctx.moveTo(ex + dr * bar, ey - dc * bar);
        ctx.lineTo(ex - dr * bar, ey + dc * bar);
        ctx.stroke();
        if (e.bump) this._cross(ex + dc * s * 0.12, ey + dr * s * 0.12, s * 0.14, colors['c-bump']);
      } else if (o === 'skid-L' || o === 'skid-R') {
        ctx.strokeStyle = colors['c-skid'];
        ctx.lineWidth = w;
        ctx.setLineDash([w * 1.2, w * 1.4]);
        ctx.beginPath();
        ctx.moveTo(a.x, a.y);
        ctx.lineTo(b.x, b.y);
        ctx.stroke();
        ctx.setLineDash([]);
        // Triangle at the midpoint: the skid glyph.
        const mx = (a.x + b.x) / 2;
        const my = (a.y + b.y) / 2;
        const ang = Math.atan2(b.y - a.y, b.x - a.x);
        const t = Math.max(4, s * 0.16);
        ctx.fillStyle = colors['c-skid'];
        ctx.beginPath();
        ctx.moveTo(mx + Math.cos(ang) * t, my + Math.sin(ang) * t);
        ctx.lineTo(mx + Math.cos(ang + 2.4) * t, my + Math.sin(ang + 2.4) * t);
        ctx.lineTo(mx + Math.cos(ang - 2.4) * t, my + Math.sin(ang - 2.4) * t);
        ctx.closePath();
        ctx.fill();
      } else {
        ctx.strokeStyle = colors['c-ok'];
        ctx.lineWidth = last ? w * 1.25 : w;
        ctx.beginPath();
        ctx.moveTo(a.x, a.y);
        ctx.lineTo(b.x, b.y);
        ctx.stroke();
        ctx.fillStyle = colors['c-ok'];
        ctx.beginPath();
        ctx.arc(a.x, a.y, w * 0.9, 0, Math.PI * 2);
        ctx.fill();
      }
      ctx.restore();
    });
    // Slips: a loop glyph (↺) in the cell's corner, with a count when it slipped repeatedly.
    for (const [key, n] of slipCount) {
      const [r, c] = parseKey(key);
      const { x, y } = cellOrigin(L, r, c);
      this._loop(x + s * 0.24, y + s * 0.24, Math.max(3, s * 0.13), colors['c-slip'], n);
    }
  }

  _loop(cx, cy, rad, color, n) {
    const { ctx } = this;
    ctx.save();
    ctx.strokeStyle = color;
    ctx.fillStyle = color;
    ctx.lineWidth = Math.max(1.5, rad * 0.45);
    ctx.beginPath();
    ctx.arc(cx, cy, rad, -Math.PI * 0.35, Math.PI * 1.45);
    ctx.stroke();
    const ax = cx + Math.cos(-Math.PI * 0.35) * rad;
    const ay = cy + Math.sin(-Math.PI * 0.35) * rad;
    const h = rad * 0.75;
    ctx.beginPath();
    ctx.moveTo(ax + h, ay);
    ctx.lineTo(ax - h * 0.3, ay - h * 0.9);
    ctx.lineTo(ax - h * 0.2, ay + h * 0.8);
    ctx.closePath();
    ctx.fill();
    if (n > 1 && rad >= 4) {
      ctx.font = `700 ${Math.round(rad * 1.5)}px ${this.font}`;
      ctx.textAlign = 'left';
      ctx.textBaseline = 'middle';
      ctx.fillText(`×${n}`, cx + rad * 1.5, cy);
    }
    ctx.restore();
  }

  _cross(cx, cy, h, color) {
    const { ctx } = this;
    ctx.save();
    ctx.strokeStyle = color;
    ctx.lineWidth = Math.max(2, h * 0.45);
    ctx.lineCap = 'round';
    ctx.beginPath();
    ctx.moveTo(cx - h, cy - h);
    ctx.lineTo(cx + h, cy + h);
    ctx.moveTo(cx + h, cy - h);
    ctx.lineTo(cx - h, cy + h);
    ctx.stroke();
    ctx.restore();
  }

  _pulse(L) {
    const { ctx, colors } = this;
    const { cells, start } = this.pulse;
    if (!cells.length) return;
    const age = performance.now() - start;
    const still = this.reducedMotion.matches;
    if (!still && age > PULSE_MS) return;
    const k = still ? 1 : 1 - age / PULSE_MS;
    ctx.save();
    ctx.strokeStyle = colors['c-replan'];
    for (const [r, c] of cells) {
      const { x, y } = cellOrigin(L, r, c);
      const grow = still ? 0 : (1 - k) * L.cell * 0.15;
      ctx.globalAlpha = still ? 0.9 : 0.25 + 0.75 * k;
      ctx.lineWidth = Math.max(2, L.cell * 0.1);
      ctx.strokeRect(x - grow, y - grow, L.cell + 2 * grow, L.cell + 2 * grow);
    }
    ctx.restore();
  }

  // --- markers -------------------------------------------------------------------------
  _start(L) {
    const { ctx, state, colors } = this;
    if (!state.start) return;
    const { x, y } = cellOrigin(L, state.start[0], state.start[1]);
    const s = L.cell;
    ctx.save();
    ctx.strokeStyle = colors['map-start'];
    ctx.lineWidth = Math.max(1.5, s * 0.06);
    ctx.setLineDash([Math.max(2, s * 0.1), Math.max(2, s * 0.08)]);
    ctx.strokeRect(x + s * 0.1, y + s * 0.1, s * 0.8, s * 0.8);
    ctx.setLineDash([]);
    if (s >= START_LABEL_MIN) {
      ctx.fillStyle = colors['map-start'];
      ctx.font = `700 ${Math.round(s * 0.22)}px ${this.font}`;
      ctx.textAlign = 'right';
      ctx.textBaseline = 'bottom';
      ctx.fillText('START', x + s * 0.88, y + s * 0.9);
    }
    ctx.restore();
  }

  _target(L) {
    const { ctx, state, colors } = this;
    if (!state.target) return;
    const { x, y } = cellOrigin(L, state.target[0], state.target[1]);
    const s = L.cell;
    ctx.save();
    ctx.strokeStyle = colors['map-target'];
    ctx.lineWidth = Math.max(2, s * 0.07);
    ctx.setLineDash([Math.max(3, s * 0.14), Math.max(2, s * 0.09)]);
    ctx.strokeRect(x + 1.5, y + 1.5, s - 3, s - 3);
    ctx.setLineDash([]);
    // Flag: pole + pennant in the top-right corner.
    const px = x + s * 0.72;
    const py = y + s * 0.12;
    ctx.lineWidth = Math.max(1.5, s * 0.05);
    ctx.beginPath();
    ctx.moveTo(px, py);
    ctx.lineTo(px, py + s * 0.38);
    ctx.stroke();
    ctx.fillStyle = colors['map-target'];
    ctx.beginPath();
    ctx.moveTo(px, py);
    ctx.lineTo(px + s * 0.2, py + s * 0.08);
    ctx.lineTo(px, py + s * 0.17);
    ctx.closePath();
    ctx.fill();
    ctx.restore();
  }

  _survivors(L) {
    const { ctx, state, colors } = this;
    const s = L.cell;
    const rescued = state.rescued ?? [];
    for (const p of state.survivors ?? []) {
      const known = this.opts.editor || this.opts.showTrue || state.belief[p[0]][p[1]] === CELL.SURVIVOR;
      if (!known) continue;
      const { x, y } = cellCenter(L, p[0], p[1]);
      const saved = rescued.some((q) => samePos(q, p));
      ctx.save();
      ctx.globalAlpha = state.belief[p[0]][p[1]] === CELL.SURVIVOR || this.opts.editor ? 1 : 0.55;
      // Person: head + shoulders, in a soft disc.
      ctx.fillStyle = colors['map-survivor'];
      ctx.beginPath();
      ctx.arc(x, y, s * 0.36, 0, Math.PI * 2);
      ctx.fill();
      ctx.fillStyle = colors['map-survivor-ink'];
      ctx.beginPath();
      ctx.arc(x, y - s * 0.1, s * 0.1, 0, Math.PI * 2);
      ctx.fill();
      ctx.beginPath();
      ctx.ellipse(x, y + s * 0.15, s * 0.18, s * 0.12, 0, Math.PI, 0);
      ctx.fill();
      ctx.restore();
      if (saved) {
        const bx = x + s * 0.27;
        const by = y + s * 0.27;
        const br = Math.max(5, s * 0.17);
        ctx.save();
        ctx.fillStyle = colors['c-rescue'];
        ctx.beginPath();
        ctx.arc(bx, by, br, 0, Math.PI * 2);
        ctx.fill();
        ctx.strokeStyle = '#fff';
        ctx.lineWidth = Math.max(1.5, br * 0.35);
        ctx.lineCap = 'round';
        ctx.lineJoin = 'round';
        ctx.beginPath();
        ctx.moveTo(bx - br * 0.45, by);
        ctx.lineTo(bx - br * 0.1, by + br * 0.38);
        ctx.lineTo(bx + br * 0.5, by - br * 0.35);
        ctx.stroke();
        ctx.restore();
      }
    }
  }

  _robot(L) {
    const { ctx, state, colors } = this;
    if (!state.robot) return;
    let [r, c] = state.robot;
    if (this.glide) {
      const k = Math.min(1, (performance.now() - this.glide.start) / GLIDE_MS);
      const e = 1 - (1 - k) ** 3;
      r = this.glide.from[0] + (this.glide.to[0] - this.glide.from[0]) * e;
      c = this.glide.from[1] + (this.glide.to[1] - this.glide.from[1]) * e;
      if (k >= 1) this.glide = null;
    }
    const s = L.cell;
    // On a survivor's cell the robot shrinks into the top-left corner so both stay visible.
    const onSurvivor = !this.glide && (state.survivors ?? []).some((p) => samePos(p, state.robot));
    const shift = onSurvivor ? -s * 0.26 : 0;
    const x = L.ox + (c + 0.5) * s + shift;
    const y = L.oy + (r + 0.5) * s + shift;
    const last = state.trace?.[state.trace.length - 1];
    const dir = last?.action ?? 'E';
    const [dr, dc] = DELTA[dir] ?? [0, 1];
    const body = s * (onSurvivor ? 0.2 : 0.3);
    const top = y - body * 0.8; // the body sits a little low: room for the antenna
    ctx.save();
    // Antenna (a stalk with a ball), visor and wheels: a robot, not a dice face.
    ctx.strokeStyle = colors['map-robot'];
    ctx.fillStyle = colors['map-robot'];
    ctx.lineWidth = Math.max(1.2, s * 0.045);
    ctx.lineCap = 'round';
    ctx.beginPath();
    ctx.moveTo(x, top);
    ctx.lineTo(x, top - body * 0.45);
    ctx.stroke();
    ctx.beginPath();
    ctx.arc(x, top - body * 0.5, Math.max(1.2, body * 0.16), 0, Math.PI * 2);
    ctx.fill();
    ctx.shadowColor = 'rgba(0,0,0,0.25)';
    ctx.shadowBlur = Math.max(2, s * 0.08);
    roundRect(ctx, x - body, top, body * 2, body * 1.7, body * 0.45);
    ctx.fill();
    ctx.shadowColor = 'transparent';
    ctx.strokeStyle = colors['map-robot-ink'];
    ctx.lineWidth = Math.max(1.2, s * 0.035);
    ctx.stroke();
    // The visor's eyes look in the direction of the last action.
    const cy = top + body * 0.72;
    ctx.fillStyle = colors['map-robot-ink'];
    roundRect(ctx, x - body * 0.74, cy - body * 0.3, body * 1.48, body * 0.6, body * 0.3);
    ctx.fill();
    ctx.fillStyle = colors['map-robot'];
    const ex = dc * body * 0.22;
    const ey = dr * body * 0.1;
    const er = Math.max(1, body * 0.15);
    for (const side of [-1, 1]) {
      ctx.beginPath();
      ctx.arc(x + ex + side * body * 0.34, cy + ey, er, 0, Math.PI * 2);
      ctx.fill();
    }
    ctx.fillStyle = colors['map-robot-ink'];
    for (const side of [-1, 1]) {
      roundRect(ctx, x + side * body * 0.55 - body * 0.27, top + body * 1.62, body * 0.54, body * 0.26, body * 0.12);
      ctx.fill();
    }
    ctx.restore();
  }

  /** Editor: outline the cells the validator flagged (walled-in survivor, unknown character). */
  _problems(L) {
    const cells = this.opts.problems;
    if (!cells?.length) return;
    const { ctx, colors } = this;
    ctx.save();
    ctx.strokeStyle = colors['c-bump'];
    const w = Math.max(2, L.cell * 0.1);
    ctx.lineWidth = w;
    for (const [r, c] of cells) {
      if (r >= L.rows || c >= L.cols) continue;
      const { x, y } = cellOrigin(L, r, c);
      ctx.strokeRect(x + w / 2, y + w / 2, L.cell - w, L.cell - w);
    }
    ctx.restore();
  }

  _hover(L) {
    const { ctx, colors } = this;
    for (const [cell, width, dash] of [[this.opts.hover, 2, []], [this.opts.highlight, 3, [4, 3]]]) {
      if (!cell) continue;
      const { x, y } = cellOrigin(L, cell[0], cell[1]);
      ctx.save();
      ctx.strokeStyle = colors['map-hover'];
      ctx.lineWidth = width;
      ctx.setLineDash(dash);
      ctx.strokeRect(x + width / 2, y + width / 2, L.cell - width, L.cell - width);
      ctx.restore();
    }
  }
}

/** Equality of option values: positions and lists by value, everything else by identity. */
function sameOption(a, b) {
  if (a === b) return true;
  if (Array.isArray(a) && Array.isArray(b)) return JSON.stringify(a) === JSON.stringify(b);
  return false;
}

function roundRect(ctx, x, y, w, h, r) {
  ctx.beginPath();
  ctx.moveTo(x + r, y);
  ctx.arcTo(x + w, y, x + w, y + h, r);
  ctx.arcTo(x + w, y + h, x, y + h, r);
  ctx.arcTo(x, y + h, x, y, r);
  ctx.arcTo(x, y, x + w, y, r);
  ctx.closePath();
}
