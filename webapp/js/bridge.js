// Bridge between the UI and the Python core: a promise-based RPC over the Pyodide worker
// (or the mock backend with ?mock=1). Every method resolves with plain JSON.

/** A Python error surfaced to the UI. */
export class BridgeError extends Error {
  constructor(message, detail = '', type = 'Error') {
    super(message);
    this.name = 'BridgeError';
    this.detail = detail;
    this.type = type;
  }
}

/** A call still unanswered after this long (ms) is reported once through onSlow (the UI offers
 * to restart; Python cannot be interrupted, so restarting means reloading the page). */
export const SLOW_CALL_MS = 20000;

class WorkerBackend {
  constructor(onProgress, onSlow) {
    this.worker = new Worker(new URL('./worker.js', import.meta.url), { type: 'module' });
    this.pending = new Map();
    this.nextId = 1;
    this.onSlow = onSlow;
    this.worker.onmessage = (ev) => {
      const msg = ev.data;
      if ('progress' in msg) {
        onProgress?.({ text: msg.progress, detail: msg.detail });
        return;
      }
      const entry = this.pending.get(msg.id);
      if (!entry) return;
      clearTimeout(entry.timer);
      this.pending.delete(msg.id);
      if (msg.ok) entry.resolve(msg.result);
      else entry.reject(new BridgeError(msg.error?.message ?? 'Python error', msg.error?.details, msg.error?.type));
    };
    this.worker.onerror = (ev) => {
      const err = new BridgeError('The Python worker crashed', ev.message, 'WorkerCrashed');
      for (const entry of this.pending.values()) {
        clearTimeout(entry.timer);
        entry.reject(err);
      }
      this.pending.clear();
    };
  }

  /** One rescue.webapi.dispatch call (method names and params are the Python ones). */
  call(method, params = {}) {
    const id = this.nextId++;
    return new Promise((resolve, reject) => {
      // init downloads Pyodide: the loading screen has its own slow-network notice.
      const timer = method === 'init' ? 0 : setTimeout(() => this.onSlow?.(method), SLOW_CALL_MS);
      this.pending.set(id, { resolve, reject, timer });
      this.worker.postMessage({ id, method, params });
    });
  }

  init() { return this.call('init'); }
  // Mission calls ask for lean states (session.js rebuilds the full ones).
  async create(config) {
    const { id, state } = await this.call('open', { config, lean: true });
    return { sid: id, state };
  }
  step(sid) { return this.call('step', { id: sid, lean: true }); }
  run(sid, maxSteps) { return this.call('run', { id: sid, max_steps: maxSteps ?? null, lean: true }); }
  stateAt(sid, t) { return this.call('state_at', { id: sid, t, lean: true }); }
  dispose(sid) { return this.call('close', { id: sid }); }
  editMap(ascii, op) { return this.call('edit_map', { ascii, op }); }
  validateMap(ascii, mode = 'core') { return this.call('validate_map', { ascii, mode }); }
  mapAscii(spec, mode = 'core') { return this.call('map_ascii', { spec, mode }); }

  /** The committed results (results.json, written by the site build); fetched once. */
  results() {
    this._results ??= fetch('results.json', { cache: 'no-cache' }).then((res) => {
      if (!res.ok) throw new BridgeError('results.json is missing', `HTTP ${res.status}: build the site with "python -m rescue.webapp".`);
      return res.json();
    });
    this._results.catch(() => { this._results = null; });
    return this._results;
  }
}

/** Create the backend: the mock with ?mock=1, else the Pyodide worker. */
export async function createBackend({ mock, onProgress, onSlow }) {
  if (mock) {
    // Development only: mock.js and its fixtures are not part of the built site.
    const { createMockBackend } = await import('./mock.js');
    return createMockBackend(onProgress);
  }
  return new WorkerBackend(onProgress, onSlow);
}
