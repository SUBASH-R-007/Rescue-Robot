// One mission in the Python worker, seen from the UI: live stepping, scrubbing back through
// recorded steps (state_at) and a small state cache. No DOM; tested in Node with a fake
// backend.
//
// Lean states (docs/plan.md §3.16): the worker sends each trace / log entry, grid version and
// plan once; this class keeps them and rebuilds full states, so payloads stay small however
// long a mission runs. Full states (the mock backend) pass through unchanged.

/** States kept in the scrub cache (they share the trace, log, grids and plans). */
export const CACHE_LIMIT = 240;

/** Error type of a call made on a session that was disposed (the UI ignores it quietly). */
export const DISPOSED = 'SessionDisposed';

/** True when a state's mission has ended. */
export function isDone(state) {
  if (!state) return false;
  if (typeof state.done === 'boolean') return state.done;
  return state.status !== 'running';
}

/** A rejected call of a disposed session. */
function disposedError() {
  const err = new Error('This mission was replaced');
  err.type = DISPOSED;
  return err;
}

export class MissionSession {
  constructor(backend, config) {
    this.backend = backend;
    this.config = config;
    this.sid = null;
    this.live = null; // newest state (t = maxT)
    this.view = null; // state on screen (t <= maxT)
    this.prevView = null;
    this.cache = new Map();
    this._chain = Promise.resolve();
    this.disposed = false;
    // What the worker has sent once (lean states refer to it).
    this.trace = [];
    this.log = [];
    this.grids = new Map();
    this.plans = new Map();
    this.fullConfig = null;
  }

  get maxT() { return this.live?.t ?? 0; }
  get t() { return this.view?.t ?? 0; }
  get done() { return isDone(this.live); }
  /** True when the view shows the newest step (stepping forward runs the mission). */
  get atLive() { return this.t === this.maxT; }

  /** Run fn after every earlier call (no interleaving on the single worker); calls queued
   * after dispose() reject with a DISPOSED error instead of reaching Python. */
  _serial(fn) {
    const guarded = () => (this.disposed ? Promise.reject(disposedError()) : fn());
    const next = this._chain.then(guarded, guarded);
    this._chain = next.catch(() => {});
    return next;
  }

  /** A full state from a lean one (or the full state itself). */
  hydrate(raw) {
    if (!raw?.lean) return raw;
    if (raw.trace_new?.length) this.trace.push(...raw.trace_new);
    if (raw.log_new?.length) this.log.push(...raw.log_new);
    if ('cells' in raw) this.grids.set(raw.grid_version, { cells: raw.cells, high_slip: raw.high_slip });
    if ('plan' in raw) this.plans.set(raw.plan_id, raw.plan);
    if (raw.config) this.fullConfig = raw.config;
    const { trace_new: _t, log_new: _l, lean: _x, has_plan: hasPlan, ...rest } = raw;
    const grid = this.grids.get(raw.grid_version) ?? {};
    return {
      ...rest,
      cells: grid.cells,
      high_slip: grid.high_slip,
      trace: this.trace.slice(0, raw.trace_end),
      log: this.log.slice(0, raw.log_end),
      plan: hasPlan ? this.plans.get(raw.plan_id) ?? null : null,
      config: this.fullConfig,
    };
  }

  _remember(state) {
    this.cache.delete(state.t);
    this.cache.set(state.t, state);
    // Evict the least recently stored step, never t = 0 or the live step.
    while (this.cache.size > CACHE_LIMIT) {
      let victim = null;
      for (const key of this.cache.keys()) {
        if (key !== 0 && key !== this.maxT) { victim = key; break; }
      }
      if (victim === null) break;
      this.cache.delete(victim);
    }
  }

  _show(state) {
    this.prevView = this.view;
    this.view = state;
    return state;
  }

  /** Create the mission in Python; resolves with the t = 0 state. */
  start() {
    return this._serial(async () => {
      const { sid, state: raw } = await this.backend.create(this.config);
      this.sid = sid;
      if (this.disposed) {
        this.backend.dispose(sid).catch(() => {});
        throw disposedError();
      }
      const state = this.hydrate(raw);
      this.live = state;
      this.cache.clear();
      this._remember(state);
      return this._show(state);
    });
  }

  /** One step forward: replay a recorded step, or run a new one. Resolves {state, ran}. */
  forward() {
    return this._serial(async () => {
      if (this.t < this.maxT) {
        return { state: this._show(await this._fetch(this.t + 1)), ran: false };
      }
      if (this.done) return { state: this.view, ran: false };
      const result = await this.backend.step(this.sid);
      const state = this.hydrate(result.state);
      if (result.explanation && !state.explanation) state.explanation = result.explanation;
      this.live = state;
      this._remember(state);
      return { state: this._show(state), ran: true, event: result.event };
    });
  }

  /** Run until the end (or maxSteps more steps) in one call. */
  finish(maxSteps = 100000) {
    return this._serial(async () => {
      const { state: raw } = await this.backend.run(this.sid, maxSteps);
      const state = this.hydrate(raw);
      this.live = state;
      this._remember(state);
      return this._show(state);
    });
  }

  /** Show recorded step t (clamped to 0..maxT). */
  seek(t) {
    return this._serial(async () => {
      const target = Math.max(0, Math.min(this.maxT, t));
      if (target === this.t) return this.view;
      return this._show(await this._fetch(target));
    });
  }

  back() {
    return this.seek(this.t - 1);
  }

  async _fetch(t) {
    if (this.cache.has(t)) return this.cache.get(t);
    const state = this.hydrate(await this.backend.stateAt(this.sid, t));
    this._remember(state);
    return state;
  }

  dispose() {
    if (this.disposed) return;
    const sid = this.sid;
    // Close after every call already queued (they finish first; later ones are refused).
    this._chain = this._chain.then(() => (sid != null ? this.backend.dispose(sid) : null)).catch(() => {});
    this.disposed = true;
  }
}
