// Web Worker that runs the Python core with Pyodide (a module worker). The UI thread never
// blocks on Python: every call is a message {id, method, params} answered by
// {id, ok, result} or {id, ok: false, error: {type, message, details}}. While loading (and
// while scipy is fetched on demand) the worker also posts {progress, detail}.
//
// Python side: rescue.webapi.dispatch, one JSON-in / JSON-out entry point (docs/plan.md §3.16).

/** Fallback Pyodide release; version.json (written by `python -m rescue.webapp`) wins. It must
 * equal config.WEB_PYODIDE_VERSION (tests/test_webapp_build.py checks this). */
const DEFAULT_PYODIDE_VERSION = '0.28.3';
const cdnFolder = (v) => `https://cdn.jsdelivr.net/pyodide/v${v}/full/`;
/** Where rescue.zip is unpacked in Pyodide's file system (the maps/ and the map library
 * inside the zip are found relative to the working directory). */
const APP_DIR = '/home/pyodide/app';

let pyodide = null;
let dispatch = null;
let scipyLoaded = false;

const post = (msg) => self.postMessage(msg);
const progress = (text, detail = '') => post({ progress: text, detail });

/** An error carrying the Python exception's type, message and traceback. */
class PyError extends Error {
  constructor({ type = 'Error', message = 'Python error', details = '' }) {
    super(message);
    this.type = type;
    this.details = details;
  }
}

async function readVersion() {
  try {
    const res = await fetch(new URL('../version.json', self.location), { cache: 'no-cache' });
    if (res.ok) return await res.json();
  } catch { /* served without a build: use the fallback */ }
  return {};
}

function call(method, params = {}) {
  const reply = JSON.parse(dispatch(JSON.stringify({ method, params })));
  if (!reply.ok) throw new PyError(reply.error ?? {});
  return reply.result;
}

/** Methods that only read (or create a fresh session): safe to run again after scipy loads. A
 * step or run may already have advanced the world, so it is never retried. */
const IDEMPOTENT = new Set(['open', 'map_ascii', 'needs_scipy', 'catalog', 'state', 'state_at', 'version',
  'normalize_config', 'validate_map', 'edit_map', 'report']);

/** What the UI shows while scipy downloads, by reason (needs_scipy: agent label / map flag). */
function scipyMessage(need) {
  if (need?.agent) return [`Loading scipy for ${need.agent}…`, 'downloaded only the first time it is needed (a few MB)'];
  if (need?.map) return ['Loading the map generator…', 'this map is not pre-generated, so scipy is downloaded once (a few MB)'];
  return ['Loading scipy…', 'downloaded only the first time it is needed (a few MB)'];
}

async function ensureScipy(need) {
  if (scipyLoaded) return;
  const [text, detail] = scipyMessage(need);
  progress(text, detail);
  try {
    await pyodide.loadPackage('scipy');
  } catch (err) {
    throw new PyError({
      type: 'ScipyUnavailable',
      message: 'scipy could not be downloaded',
      details: String(err?.message ?? err),
    });
  } finally {
    progress('', '');
  }
  scipyLoaded = true;
}

async function init() {
  const version = await readVersion();
  const pv = version.pyodide || DEFAULT_PYODIDE_VERSION;
  const indexURL = version.pyodide_url || cdnFolder(pv);
  progress('Starting Python…', `Pyodide ${pv}: CPython compiled to WebAssembly`);
  const { loadPyodide } = await import(`${indexURL}pyodide.mjs`);
  pyodide = await loadPyodide({ indexURL });
  progress('Loading numpy…', 'the only third-party package the simulator needs');
  await pyodide.loadPackage('numpy');
  progress('Loading the rescue robot…', 'unpacking the same Python package the command line uses');
  const zip = await fetch(new URL('../rescue.zip', self.location));
  if (!zip.ok) {
    throw new PyError({
      type: 'BuildMissing',
      message: `rescue.zip was not found (HTTP ${zip.status}). Build and serve the site with "python -m rescue.webapp".`,
    });
  }
  pyodide.FS.mkdirTree(APP_DIR);
  pyodide.unpackArchive(await zip.arrayBuffer(), 'zip', { extractDir: APP_DIR });
  pyodide.runPython(`
import os, sys
sys.path.insert(0, ${JSON.stringify(APP_DIR)})
os.chdir(${JSON.stringify(APP_DIR)})
from rescue.webapi import dispatch as _rescue_dispatch
`);
  dispatch = pyodide.globals.get('_rescue_dispatch');
  const info = call('version');
  const catalog = call('catalog');
  return { catalog, version: { ...version, pyodide: pv, python: info.python, numpy: info.numpy, api: info.api } };
}

/** Methods whose config may need scipy (a generated map outside the library, or LAO*). */
async function maybeScipy(method, params) {
  if (scipyLoaded) return;
  let config = null;
  if (method === 'open') config = params.config;
  else if (method === 'map_ascii') config = { map: params.spec, mode: params.mode ?? 'core' };
  if (!config) return;
  const need = call('needs_scipy', { config });
  if (need.scipy) await ensureScipy(need);
}

async function handle(method, params) {
  if (method === 'init') return init();
  if (!dispatch) throw new PyError({ message: 'Python is not ready yet' });
  await maybeScipy(method, params);
  try {
    return call(method, params);
  } catch (err) {
    // Safety net: anything else that turns out to import scipy loads it; a read-only call is
    // retried once, a step is not (the mission may have moved on: it must be reset).
    if (!scipyLoaded && err.type === 'ModuleNotFoundError' && /scipy/.test(err.message)) {
      await ensureScipy();
      if (IDEMPOTENT.has(method)) return call(method, params);
      throw new PyError({
        type: 'ResetNeeded',
        message: 'This step needed extra code (scipy), which is now loaded. Press Reset to run the mission again.',
        details: err.details,
      });
    }
    throw err;
  }
}

// Calls are answered strictly in order (Python is single-threaded anyway).
let queue = Promise.resolve();
self.onmessage = (ev) => {
  const { id, method, params } = ev.data;
  queue = queue.then(async () => {
    try {
      post({ id, ok: true, result: await handle(method, params ?? {}) });
    } catch (err) {
      post({ id, ok: false, error: { type: err.type ?? err.name ?? 'Error', message: String(err.message ?? err), details: err.details ?? err.stack ?? '' } });
    }
  });
};
