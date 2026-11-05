// Timeline scrubber: pure mapping helpers (tested in Node) + the DOM widget.
import { normOutcome, escapeHtml } from './format.js';

/** Before the mission ends the track spans at least this many steps (room to grow). */
export const MIN_SPAN = 20;
/** Horizontal padding inside the track, CSS px (so the end ticks are not clipped). */
export const TRACK_PAD = 10;

/** Tick kinds in drawing order, with their human label and glyph (shape differs per kind). */
export const TICK_KINDS = Object.freeze({
  slip: { label: 'Slip', glyph: '●' },
  skid: { label: 'Skid', glyph: '▲' },
  bump: { label: 'Bump', glyph: '✕' },
  replan: { label: 'Replan', glyph: '◆' },
  found: { label: 'Survivor found', glyph: '★' },
  rescue: { label: 'Rescue', glyph: '✓' },
  fail: { label: 'Mission failed', glyph: '■' },
});

/** Steps shown by the track: the recorded steps, padded to MIN_SPAN while running. */
export function trackSpan(maxT, done) {
  if (done) return Math.max(1, maxT);
  return Math.max(MIN_SPAN, maxT, 1);
}

/** x (CSS px, relative to the track) of step t. */
export function tToX(t, span, width, pad = TRACK_PAD) {
  const usable = Math.max(1, width - 2 * pad);
  return pad + (Math.max(0, Math.min(span, t)) / span) * usable;
}

/** Step under x, rounded and clamped to [0, maxT] (only recorded steps can be shown). */
export function xToT(x, span, width, maxT, pad = TRACK_PAD) {
  const usable = Math.max(1, width - 2 * pad);
  const t = Math.round(((x - pad) / usable) * span);
  return Math.max(0, Math.min(maxT, t));
}

/** Tick kind of an event-log entry: its `tag` (set by Python: replan / found / rescue /
 * fail), or null for no tick. The text is never parsed. */
export function logTick(entry) {
  return entry?.tag && entry.tag in TICK_KINDS ? entry.tag : null;
}

/**
 * Tick marks of a mission: one per (t, kind), sorted by t then kind order. Slips / skids /
 * bumps come from the trace, replans / rescues / failures from the event log.
 */
export function buildTicks(trace = [], log = []) {
  const seen = new Set();
  const ticks = [];
  const add = (t, kind) => {
    const key = `${t}:${kind}`;
    if (kind && !seen.has(key)) {
      seen.add(key);
      ticks.push({ t, kind });
    }
  };
  for (const e of trace) {
    const o = normOutcome(e.outcome);
    if (o === 'slipped') add(e.t, 'slip');
    else if (o === 'skid-L' || o === 'skid-R') add(e.t, 'skid');
    if (e.bump) add(e.t, 'bump');
  }
  for (const entry of log) {
    const kind = logTick(entry);
    if (entry.t > 0 || kind === 'fail') add(entry.t, kind);
  }
  const order = Object.keys(TICK_KINDS);
  ticks.sort((a, b) => a.t - b.t || order.indexOf(a.kind) - order.indexOf(b.kind));
  return ticks;
}

/** Group ticks by step so stacked marks get distinct lanes: [{t, kinds: [...]}]. */
export function stackTicks(ticks) {
  const out = [];
  for (const tick of ticks) {
    const last = out[out.length - 1];
    if (last && last.t === tick.t) last.kinds.push(tick.kind);
    else out.push({ t: tick.t, kinds: [tick.kind] });
  }
  return out;
}

/** DOM widget: a slider-like track with tick marks; calls onScrub(t) while dragging. */
export class Timeline {
  constructor(root, { onScrub }) {
    this.root = root;
    this.onScrub = onScrub;
    this.maxT = 0;
    this.t = 0;
    this.done = false;
    this.ticks = [];
    root.classList.add('timeline');
    root.innerHTML = `
      <div class="tl-track" role="slider" tabindex="0" aria-label="Mission timeline: drag to replay a step"
           aria-valuemin="0" aria-valuemax="0" aria-valuenow="0">
        <div class="tl-rail"><div class="tl-recorded"></div></div>
        <div class="tl-ticks" aria-hidden="true"></div>
        <div class="tl-thumb" aria-hidden="true"></div>
      </div>
      <div class="tl-labels"><span class="tl-now"></span><span class="tl-end"></span></div>`;
    this.track = root.querySelector('.tl-track');
    this.recorded = root.querySelector('.tl-recorded');
    this.tickLayer = root.querySelector('.tl-ticks');
    this.thumb = root.querySelector('.tl-thumb');
    this.nowLabel = root.querySelector('.tl-now');
    this.endLabel = root.querySelector('.tl-end');
    this._dragging = false;
    this.track.addEventListener('pointerdown', (ev) => this._down(ev));
    this.track.addEventListener('pointermove', (ev) => this._move(ev));
    this.track.addEventListener('pointerup', (ev) => this._up(ev));
    this.track.addEventListener('pointercancel', (ev) => this._up(ev));
    this.track.addEventListener('keydown', (ev) => this._key(ev));
    new ResizeObserver(() => this._layout()).observe(this.track);
  }

  /** Update from a mission: t shown, recorded maxT, done flag, trace and log. */
  update({ t, maxT, done, trace, log, budget }) {
    this.t = t;
    this.maxT = maxT;
    this.done = done;
    this.budget = budget;
    this.ticks = stackTicks(buildTicks(trace, log));
    this._layout();
  }

  _span() {
    return trackSpan(this.maxT, this.done);
  }

  _layout() {
    const width = this.track.clientWidth;
    if (!width) return;
    const span = this._span();
    const xNow = tToX(this.t, span, width);
    const xMax = tToX(this.maxT, span, width);
    this.thumb.style.transform = `translateX(${xNow}px)`;
    this.recorded.style.width = `${xMax}px`;
    this.track.setAttribute('aria-valuemax', String(this.maxT));
    this.track.setAttribute('aria-valuenow', String(this.t));
    this.track.setAttribute('aria-valuetext', `step ${this.t} of ${this.maxT}`);
    const html = [];
    for (const group of this.ticks) {
      const x = tToX(group.t, span, width);
      group.kinds.forEach((kind, lane) => {
        const info = TICK_KINDS[kind];
        html.push(`<span class="tick tick-${kind}" style="left:${x}px;--lane:${lane}" title="Step ${group.t}: ${escapeHtml(info.label)}">${info.glyph}</span>`);
      });
    }
    this.tickLayer.innerHTML = html.join('');
    this.nowLabel.textContent = this.t === this.maxT ? `Step ${this.t}` : `Replaying step ${this.t} of ${this.maxT}`;
    this.endLabel.textContent = this.done ? `Finished after ${this.maxT}` : (this.budget ? `Budget ${this.budget} steps` : '');
  }

  _tFromEvent(ev) {
    const rect = this.track.getBoundingClientRect();
    return xToT(ev.clientX - rect.left, this._span(), rect.width, this.maxT);
  }

  _down(ev) {
    this._dragging = true;
    this.track.setPointerCapture(ev.pointerId);
    this.track.classList.add('dragging');
    this._emit(this._tFromEvent(ev));
  }

  _move(ev) {
    if (this._dragging) this._emit(this._tFromEvent(ev));
  }

  _up(ev) {
    if (!this._dragging) return;
    this._dragging = false;
    this.track.classList.remove('dragging');
    try { this.track.releasePointerCapture(ev.pointerId); } catch { /* already released */ }
  }

  _key(ev) {
    const steps = { ArrowLeft: -1, ArrowRight: 1, PageDown: -10, PageUp: 10 };
    if (ev.key in steps) {
      ev.preventDefault();
      ev.stopPropagation();
      this._emit(Math.max(0, Math.min(this.maxT, this.t + steps[ev.key])));
    } else if (ev.key === 'Home' || ev.key === 'End') {
      ev.preventDefault();
      this._emit(ev.key === 'Home' ? 0 : this.maxT);
    }
  }

  _emit(t) {
    if (t === this.t) return;
    this.t = t;
    this._layout();
    this.onScrub(t);
  }
}
