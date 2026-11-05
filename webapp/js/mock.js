// Mock backend (only with ?mock=1, for developing the UI without Pyodide): replays missions
// recorded from the real Python core (webapp/tests/fixtures/missions.json, delta-coded frames)
// with the worker bridge's interface. A config without a recording replays the closest one.
// The fixtures are not published, so the mock works only when webapp/ itself is served.
import { applyOpLocal, parseAscii, stateToAscii, CH } from './mapcore.js';

const FIXTURES = new URL('../tests/fixtures/', import.meta.url);
/** Simulated latency of a step, ms. */
const STEP_DELAY_MS = 15;
/** Delay of each loading stage, ms. */
const LOAD_DELAY_MS = 220;

const wait = (ms) => new Promise((r) => setTimeout(r, ms));

/** Rebuild full states from a recording's delta-coded frames. */
export function expandMission(rec) {
  const states = [];
  let belief = null;
  let pDebris = null;
  let policy = {};
  let envelope = [];
  const trace = [];
  const log = [];
  for (const f of rec.frames) {
    if ('belief' in f) belief = f.belief;
    if ('p_debris' in f) pDebris = f.p_debris;
    if ('policy' in f) policy = f.policy;
    if ('envelope' in f) envelope = f.envelope;
    trace.push(...(f.trace ?? []));
    log.push(...(f.log ?? []));
    const plan = f.plan == null ? null : rec.plans[String(f.plan)] ?? null;
    states.push({
      ...rec.base,
      frame: f.frame,
      t: f.t,
      status: f.status,
      done: f.done,
      belief,
      p_debris: pDebris,
      robot: f.robot,
      rescued: f.rescued,
      target: f.target,
      target_mode: f.target_mode,
      plan_id: f.plan_id,
      policy,
      envelope: plan || f.path ? envelope : [],
      plan,
      path: f.path,
      plan_message: f.plan_message,
      metrics: f.metrics,
      panel: f.panel,
      explanation: f.explanation,
      event: f.event,
      trace: trace.slice(),
      log: log.slice(),
    });
  }
  return states;
}

/** Score how well a recording matches a config (higher is better). */
export function matchScore(recConfig, config) {
  let score = 0;
  const mode = config.mode ?? 'core';
  if ((recConfig.mode ?? 'core') === mode) score += 100;
  if (recConfig.agent === (config.agent ?? (mode === 'core' ? 'andor' : 'search_andor'))) score += 10;
  const a = recConfig.map ?? {};
  const b = config.map ?? {};
  if (a.source === b.source) score += 2;
  if (a.name && a.name === b.name) score += 3;
  if (a.scenario && a.scenario === b.scenario) score += 3;
  return score;
}

function solvable(text) {
  const g = parseAscii(text);
  const rows = g.length;
  const cols = g[0]?.length ?? 0;
  let start = null;
  const goals = [];
  g.forEach((row, r) => row.forEach((ch, c) => {
    if (ch === CH.START) start = [r, c];
    if (ch === CH.SURVIVOR) goals.push(`${r},${c}`);
  }));
  if (!start || !goals.length) return null;
  const seen = new Set([`${start[0]},${start[1]}`]);
  const queue = [start];
  while (queue.length) {
    const [r, c] = queue.shift();
    for (const [dr, dc] of [[-1, 0], [0, 1], [1, 0], [0, -1]]) {
      const nr = r + dr;
      const nc = c + dc;
      const k = `${nr},${nc}`;
      if (nr < 0 || nc < 0 || nr >= rows || nc >= cols || seen.has(k) || g[nr][nc] === CH.DEBRIS) continue;
      seen.add(k);
      queue.push([nr, nc]);
    }
  }
  return goals.every((k) => seen.has(k));
}

function check(text, mode = 'core') {
  const g = parseAscii(text);
  const flat = g.flat();
  const errors = [];
  if (!g.length || g.some((row) => row.length !== g[0].length)) errors.push('Every row must have the same length.');
  if (flat.filter((ch) => ch === CH.START).length !== 1) errors.push('The map needs exactly one start (S).');
  const survivors = flat.filter((ch) => ch === CH.SURVIVOR).length;
  if (!survivors) errors.push('The map needs a survivor (P).');
  if (mode === 'core' && survivors > 1) errors.push('Several survivors need the “Several survivors” mode.');
  const ok = !errors.length;
  const solv = ok ? solvable(text) : null;
  return { ok, errors, warnings: solv === false ? ['A survivor cannot be reached from the start (mock check).'] : [], solvable: solv };
}

/** Mock bridge with the worker bridge's interface. */
export function createMockBackend(onProgress) {
  let data = null;
  let resultsData = null;
  let nextId = 1;
  const sessions = new Map();
  const load = async () => {
    if (!data) {
      const res = await fetch(new URL('missions.json', FIXTURES));
      if (!res.ok) throw new Error(`mock fixtures missing (HTTP ${res.status}); serve webapp/ itself for ?mock=1`);
      data = await res.json();
      data.expanded = new Map();
    }
    return data;
  };
  const recordingFor = (config) => {
    let best = data.missions[0];
    let bestScore = -1;
    for (const m of data.missions) {
      const s = matchScore(m.config, config);
      if (s > bestScore) { best = m; bestScore = s; }
    }
    if (!data.expanded.has(best)) data.expanded.set(best, expandMission(best));
    return data.expanded.get(best);
  };
  return {
    async init() {
      for (const text of ['Starting Python…', 'Loading numpy…', 'Loading the rescue robot…']) {
        onProgress?.({ text, detail: 'mock mode: recorded missions, no Python' });
        await wait(LOAD_DELAY_MS);
      }
      await load();
      return { catalog: data.catalog, version: { commit: 'mock', pyodide: 'mock', numpy: '–', python: '–' } };
    },
    async create(config) {
      await load();
      const states = recordingFor(config);
      const sid = nextId++;
      sessions.set(sid, { states, t: 0 });
      return { sid, state: states[0] };
    },
    async step(sid) {
      const s = sessions.get(sid);
      await wait(STEP_DELAY_MS);
      if (s.t < s.states.length - 1) s.t += 1;
      const state = s.states[s.t];
      return { event: state.event, explanation: state.explanation, state };
    },
    async run(sid, maxSteps = 100000) {
      const s = sessions.get(sid);
      s.t = Math.min(s.states.length - 1, s.t + maxSteps);
      return { events: [], state: s.states[s.t] };
    },
    async stateAt(sid, t) {
      const s = sessions.get(sid);
      return s.states[Math.max(0, Math.min(s.t, t))];
    },
    async dispose(sid) {
      sessions.delete(sid);
    },
    async editMap(ascii, op) {
      const next = applyOpLocal(ascii, op);
      const c = check(next, 'multi');
      return { ascii: next, changed: next !== ascii, warnings: c.warnings, errors: c.errors, valid: c.ok, solvable: c.solvable };
    },
    async validateMap(ascii, mode = 'core') {
      return check(ascii, mode);
    },
    async mapAscii(spec) {
      await load();
      const file = data.catalog.maps.find((m) => m.name === spec?.name);
      if (spec?.source === 'file' && file?.ascii) return file.ascii;
      return stateToAscii(recordingFor({ mode: 'core', map: spec })[0]);
    },
    async results() {
      if (!resultsData) {
        const res = await fetch(new URL('results.json', FIXTURES));
        if (!res.ok) throw new Error('mock results missing');
        resultsData = await res.json();
        // Plot files are build products (assets/): the mock shows their titles only.
        for (const img of [...(resultsData.plots ?? []), ...(resultsData.figures ?? [])]) img.src = null;
      }
      return resultsData;
    },
  };
}
