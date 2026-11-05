// Entry point: loading screen, Python bridge, views, welcome card, theme, help, shortcuts.
import { createBackend } from './bridge.js';
import { SimView } from './sim.js';
import { escapeHtml } from './format.js';
import { icon, initTips, initDialogs, store, isTyping, showError, showProgress, toast } from './ui.js';
import { renderLegend } from './panels.js';

const params = new URLSearchParams(location.search);
const MOCK = params.get('mock') === '1';
const THEMES = ['light', 'dark', 'contrast'];
const THEME_ICON = { light: 'sun', dark: 'moon', contrast: 'contrast' };
const THEME_NAME = { light: 'light', dark: 'dark', contrast: 'high contrast' };
/** Seconds without loading progress before the slow-network notice appears. */
const LOADING_SLOW_MS = 20000;
/** localStorage keys: the chosen theme, and whether the welcome card was dismissed. */
const THEME_KEY = 'rr-theme';
const WELCOMED_KEY = 'rr-welcomed';
const PRESET_ICON = {
  slide9: 'robot', head_to_head: 'columns', hidden_survivor: 'search', several_survivors: 'people',
  noisy_sensors: 'signal', blank_canvas: 'brush',
};

/** A tiny map drawing for a preset card (cells: . open, # debris, S start, P survivor, ? fog, ~ water). */
const PRESET_ART = {
  slide9: ['S...', '....', '.#..', '..#P'],
  head_to_head: ['S.#.', '.##.', '...P', '#...'],
  hidden_survivor: ['S??', '.??', '?.?'],
  several_survivors: ['S..P', '.#..', 'P..P'],
  noisy_sensors: ['S.#.', '.?#.', '..?P'],
  blank_canvas: ['S...', '....', '...P'],
};

const app = {
  backend: null,
  catalog: null,
  version: null,
  customMap: null,
  views: {},
  current: 'simulate',
};
window.rescueApp = app; // handy in the console; not used by the code

// --- loading -----------------------------------------------------------------------------
const LOADING_STEPS = ['Starting Python', 'Loading numpy', 'Loading the rescue robot'];
let slowTimer = 0;
/** True when this page comes from a build (version.json exists): no offline demo there. */
let builtSite = null;
fetch('version.json', { cache: 'no-cache' }).then((r) => { builtSite = r.ok; }).catch(() => { builtSite = false; });

/** Mark loading steps < idx done (their label becomes e.g. "Python ready"), idx active. */
function markSteps(idx) {
  document.querySelectorAll('.loading-steps li').forEach((li, i) => {
    li.classList.toggle('done', i < idx);
    li.classList.toggle('active', i === idx);
    const label = li.querySelector('.ls-text');
    if (i < idx && li.dataset.done) label.textContent = li.dataset.done;
  });
}

function setLoading(text, detail = '') {
  const idx = LOADING_STEPS.findIndex((s) => text.startsWith(s));
  if (idx >= 0) markSteps(idx);
  document.getElementById('loading-detail').textContent = detail || '';
  // Any progress restarts the slow-network clock.
  clearTimeout(slowTimer);
  document.getElementById('loading-slow').hidden = true;
  slowTimer = setTimeout(loadingSlow, LOADING_SLOW_MS);
}

function fallbackLink() {
  return builtSite === false ? '<a class="btn small ghost" href="?mock=1">Open the offline demo instead</a>' : '';
}

function loadingSlow() {
  const box = document.getElementById('loading-slow');
  box.hidden = false;
  box.innerHTML = `<p>This is taking longer than usual. The first visit downloads about 10 MB (Python and numpy);
    a slow or blocked connection to cdn.jsdelivr.net stops here.</p>
    <p class="loading-actions"><button class="btn small" type="button" data-retry>Retry</button>${fallbackLink()}</p>`;
  box.querySelector('[data-retry]').addEventListener('click', () => location.reload());
}

function loadingFailed(err) {
  clearTimeout(slowTimer);
  document.getElementById('loading-slow').hidden = true;
  const box = document.getElementById('loading-error');
  box.hidden = false;
  box.innerHTML = `<strong>The simulator could not start.</strong>
    <p>${escapeHtml(err?.message ?? String(err))}</p>
    <p>It needs a network connection the first time (Pyodide is downloaded from cdn.jsdelivr.net) and must be
    served over http (for example <code>python -m rescue.webapp</code>), not opened as a file.</p>
    ${err?.detail ? `<details><summary>Technical details</summary><pre>${escapeHtml(err.detail)}</pre></details>` : ''}
    <p class="loading-actions"><button class="btn small" type="button" data-retry>Try again</button>${fallbackLink()}</p>`;
  box.querySelector('[data-retry]').addEventListener('click', () => location.reload());
  document.querySelectorAll('.loading-steps li.active').forEach((li) => li.classList.remove('active'));
}

// --- progress chip for on-demand loads (scipy) -------------------------------------------
/** Worker progress: the loading screen while starting, then the global chip (any view), which
 * the worker's closing empty message hides again. */
function onProgress({ text, detail }) {
  if (!app.catalog) {
    if (text) setLoading(text, detail);
    return;
  }
  showProgress(text, detail);
}

/** A Python call that takes very long: Python cannot be interrupted, so offer a reload. */
function onSlow() {
  if (app.slowToastOpen) return;
  app.slowToastOpen = true;
  const close = toast({
    title: 'Python is still busy',
    text: 'This step is taking unusually long (a very large map or a slow device). Wait, or reload the page to start fresh.',
    tone: 'warn',
    timeout: 0,
  });
  const box = document.getElementById('toasts').lastElementChild;
  const btn = document.createElement('button');
  btn.className = 'btn small';
  btn.type = 'button';
  btn.textContent = 'Reload';
  btn.addEventListener('click', () => location.reload());
  box?.querySelector('div')?.append(btn);
  box?.querySelector('.icon-btn')?.addEventListener('click', () => { app.slowToastOpen = false; close(); });
}

// --- views -------------------------------------------------------------------------------
async function view(name) {
  if (app.views[name]) return app.views[name];
  let v;
  switch (name) {
    case 'compare': v = new (await import('./compare.js')).CompareView(app, document.getElementById('view-compare')); break;
    case 'editor': v = new (await import('./editor.js')).EditorView(app, document.getElementById('view-editor')); break;
    case 'results': v = new (await import('./results.js')).ResultsView(app, document.getElementById('view-results')); break;
    case 'learn': v = new (await import('./learn.js')).LearnView(app, document.getElementById('view-learn')); break;
    default: v = app.views.simulate;
  }
  app.views[name] = v;
  return v;
}

async function switchView(name, { focus = false } = {}) {
  app.current = name;
  for (const tab of document.querySelectorAll('.view-tabs [role=tab]')) {
    const sel = tab.dataset.view === name;
    tab.setAttribute('aria-selected', String(sel));
    tab.tabIndex = sel ? 0 : -1;
    if (sel && focus) tab.focus();
  }
  for (const sec of document.querySelectorAll('main > .view')) sec.hidden = sec.id !== `view-${name}`;
  if (name !== 'simulate') app.views.simulate?.pause();
  if (name !== 'compare') app.views.compare?.pause?.();
  try {
    const v = await view(name);
    v.onShow?.();
  } catch (err) {
    showError(err, `Could not open ${name}`);
  }
  if (params.get('view') !== name) {
    const url = new URL(location.href);
    if (name === 'simulate') url.searchParams.delete('view');
    else url.searchParams.set('view', name);
    history.replaceState(null, '', url);
  }
}
app.switchView = switchView;

function wireTabs() {
  const tabs = [...document.querySelectorAll('.view-tabs [role=tab]')];
  tabs.forEach((tab, i) => {
    tab.addEventListener('click', () => switchView(tab.dataset.view));
    tab.addEventListener('keydown', (ev) => {
      const d = { ArrowRight: 1, ArrowLeft: -1 }[ev.key];
      if (!d) return;
      ev.preventDefault();
      ev.stopPropagation();
      switchView(tabs[(i + d + tabs.length) % tabs.length].dataset.view, { focus: true });
    });
  });
}

// --- theme -------------------------------------------------------------------------------
function applyTheme(theme) {
  document.documentElement.dataset.theme = theme;
  const btn = document.getElementById('theme-btn');
  btn.querySelector('use').setAttribute('href', `#i-${THEME_ICON[theme]}`);
  btn.setAttribute('aria-label', `Theme: ${THEME_NAME[theme]} (click to switch)`);
  btn.title = `Theme: ${THEME_NAME[theme]}. Click for ${THEME_NAME[THEMES[(THEMES.indexOf(theme) + 1) % THEMES.length]]}.`;
  window.dispatchEvent(new Event('rr-theme'));
  if (app.views.simulate?.config) renderLegend(document.getElementById('legend'), app.views.simulate.config.mode ?? 'core');
}

function wireTheme() {
  applyTheme(document.documentElement.dataset.theme || 'light');
  document.getElementById('theme-btn').addEventListener('click', () => {
    const cur = document.documentElement.dataset.theme;
    const next = THEMES[(THEMES.indexOf(cur) + 1) % THEMES.length];
    try { localStorage.setItem(THEME_KEY, next); } catch { /* ignore */ }
    applyTheme(next);
  });
  // Until the visitor picks a theme, follow the system's light / dark setting (also while the
  // loading screen is up).
  const dark = matchMedia('(prefers-color-scheme: dark)');
  dark.addEventListener('change', () => {
    let chosen = null;
    try { chosen = localStorage.getItem(THEME_KEY); } catch { /* storage blocked */ }
    if (!chosen) applyTheme(dark.matches ? 'dark' : 'light');
  });
}

// --- help --------------------------------------------------------------------------------
function renderHelp() {
  const v = app.version ?? {};
  const keys = [
    ['Space', 'Play / pause'], ['N or →', 'One step forward'], ['B or ←', 'One step back'],
    ['Home / End', 'First / newest step'], ['R', 'Reset the mission (same seed)'], ['T', 'Show / hide the true map'],
    ['P', 'Show / hide the plan arrows'], ['D', 'Flat map / 3D view'], ['?', 'This help'],
    ['Esc', 'Close a dialog'],
  ];
  document.getElementById('help-body').innerHTML = `
    <section><h3>What am I looking at?</h3>
      <p>A rescue robot (the rounded box with an antenna and two eyes) must reach a survivor (the person in a circle) on
      a grid it only partly knows. Cells marked “?” (a dot on big maps) have not been seen yet; hatched cells are debris;
      wavy cells are water or mud. The <b>Key</b> button above the map shows every symbol. Each move can go wrong: the
      wheels may <b>slip</b> (the robot stays put) or <b>skid</b> sideways. The <b>AND-OR</b> agent plans ahead for every
      outcome; the A* agents plan one path and fix it when it breaks. Every coordinate is written (row, column), counting
      from 1.</p></section>
    <section><h3>Keyboard shortcuts</h3>
      <dl class="keys">${keys.map(([k, d]) => `<dt>${k.split(' or ').map((x) => `<kbd>${escapeHtml(x)}</kbd>`).join(' or ')}</dt><dd>${escapeHtml(d)}</dd>`).join('')}</dl></section>
    <section><h3>Mouse</h3>
      <p>Hover a cell for what the robot believes is there and what its plan does there. Drag the timeline to replay any
      step. Click an event-log line to jump to it. Hover a plan-tree row to find its cell on the map. In the
      <b>3D view</b> drag to orbit the site and scroll to zoom (cell hover is a flat-map feature).</p></section>
    <section class="stack"><h3>Start again</h3>
      <div class="field-row"><button class="btn" id="help-welcome">${icon('wand')}Show the welcome screen</button>
      <button class="btn" id="help-tour">${icon('help')}Take the tour</button></div></section>
    <section><h3>About</h3>
      <p class="muted">The simulation is the project's Python package running in your browser with Pyodide${v.pyodide ? ` ${escapeHtml(v.pyodide)}` : ''}
      (Python ${escapeHtml(v.python ?? '?')}, numpy ${escapeHtml(v.numpy ?? '?')})${v.commit ? `, built from commit <code>${escapeHtml(String(v.commit).slice(0, 7))}</code>` : ''}.
      The same seed gives the same mission as <code>python -m rescue.demo</code> and <code>python -m rescue.sim</code>.
      CS23533 Foundations of AI mini project, Rajalakshmi Engineering College.</p></section>`;
  document.getElementById('help-welcome').addEventListener('click', () => {
    document.getElementById('help-dialog').close();
    openWelcome();
  });
  document.getElementById('help-tour').addEventListener('click', () => {
    document.getElementById('help-dialog').close();
    startTour();
  });
}

function openHelp() {
  renderHelp();
  document.getElementById('help-dialog').showModal();
}

// --- welcome + presets -------------------------------------------------------------------
function presetKey(p) {
  return p.id ?? p.name ?? '';
}

function miniArt(rows) {
  if (!rows) return '';
  const n = rows.length;
  const m = rows[0].length;
  const cell = Math.min(56 / m, 40 / n);
  const fill = { '.': 'var(--map-floor)', '#': 'var(--map-debris)', S: 'var(--map-floor)', P: 'var(--map-survivor)', '?': 'var(--map-fog)', '~': 'var(--map-water)' };
  const rects = [];
  rows.forEach((row, r) => [...row].forEach((ch, c) => {
    rects.push(`<rect x="${c * cell}" y="${r * cell}" width="${cell - 1}" height="${cell - 1}" rx="1.5" fill="${fill[ch] ?? fill['.']}" stroke="var(--map-grid)" stroke-width=".5"/>`);
    if (ch === 'S') rects.push(`<rect x="${c * cell + cell * 0.2}" y="${r * cell + cell * 0.2}" width="${cell * 0.5}" height="${cell * 0.5}" rx="2" fill="var(--map-robot)"/>`);
  }));
  return `<svg class="mini" viewBox="0 0 ${m * cell} ${n * cell}" aria-hidden="true">${rects.join('')}</svg>`;
}

function renderPresets() {
  const grid = document.getElementById('preset-grid');
  const presets = app.catalog.presets ?? [];
  grid.innerHTML = presets.map((p, i) => {
    const key = presetKey(p);
    const ico = PRESET_ICON[key] ?? 'robot';
    return `<button class="preset" type="button" data-preset="${i}"${p.note ? ` title="${escapeHtml(p.note)}"` : ''}>
      <span class="preset-art"><span class="pi">${icon(ico)}</span>${miniArt(PRESET_ART[key])}</span>
      <strong>${escapeHtml(p.title ?? key)}</strong><span>${escapeHtml(p.description ?? '')}</span></button>`;
  }).join('');
  grid.onclick = (ev) => {
    const b = ev.target.closest('[data-preset]');
    if (!b) return;
    document.getElementById('welcome').close();
    applyPreset(presets[Number(b.dataset.preset)]);
  };
}

/** Start a preset: Simulate (default), Compare (the preset lists agents) or the map editor. */
async function applyPreset(p) {
  if (!p) return;
  const config = { ...(p.config ?? {}) };
  const title = p.title ?? presetKey(p);
  if (p.edit) {
    await switchView('editor');
    if (config.map?.source === 'ascii' && config.map.text) app.views.editor?.load(config.map.text, title);
    return;
  }
  if (Array.isArray(p.compare_agents)) {
    await app.views.simulate.start(config, { note: `Preset “${title}”.`, quiet: true });
    const cmp = await view('compare');
    await cmp.startWith(app.views.simulate.config ?? config, p.compare_agents);
    await switchView('compare');
    return;
  }
  await switchView('simulate');
  await app.views.simulate.start(config, { note: `Preset “${title}”.`, quiet: true });
}
app.applyPreset = applyPreset;

function openWelcome() {
  renderPresets();
  const dlg = document.getElementById('welcome');
  dlg.showModal();
  // Returning visitors are not greeted again (Help › "Show the welcome screen" brings it back).
  dlg.addEventListener('close', () => store.set(WELCOMED_KEY, true), { once: true });
}

async function startTour() {
  for (const dlg of document.querySelectorAll('dialog[open]')) dlg.close();
  await switchView('simulate');
  const { runTour } = await import('./tour.js');
  runTour(app);
}

// --- keyboard ----------------------------------------------------------------------------
function wireKeys() {
  document.addEventListener('keydown', (ev) => {
    if (ev.defaultPrevented || ev.ctrlKey || ev.metaKey || ev.altKey) return;
    if (document.querySelector('dialog[open]') || !document.getElementById('tour-layer').hidden) return;
    if (isTyping(ev)) return;
    if (ev.key === '?') {
      ev.preventDefault();
      openHelp();
      return;
    }
    // Let focused buttons / sliders keep their own Space and arrow keys.
    const t = ev.target;
    if ((ev.key === ' ' || ev.key === 'Enter') && t instanceof HTMLElement && t.closest('button, a, [role=slider], [role=treeitem]')) return;
    if (ev.key.startsWith('Arrow') && t instanceof HTMLElement && t.closest('[role=slider], [role=tree], [role=tablist], input')) return;
    const v = app.views[app.current];
    if (v?.handleKey?.(ev)) ev.preventDefault();
  });
}

// --- boot --------------------------------------------------------------------------------
async function boot() {
  initTips();
  initDialogs();
  wireTabs();
  wireTheme();
  wireKeys();
  document.getElementById('help-btn').addEventListener('click', openHelp);
  document.getElementById('btn-tour').addEventListener('click', () => {
    document.getElementById('welcome').close();
    startTour();
  });
  setLoading('Starting Python…', MOCK ? 'mock mode: recorded missions, no Python' : '');
  try {
    app.backend = await createBackend({ mock: MOCK, onProgress, onSlow });
    const { catalog, version } = await app.backend.init();
    app.catalog = catalog;
    app.version = version;
  } catch (err) {
    loadingFailed(err);
    return;
  }
  clearTimeout(slowTimer);
  markSteps(LOADING_STEPS.length);
  app.views.simulate = new SimView(app);
  const presets = catalog().presets ?? [];
  const wanted = params.get('preset');
  const first = presets.find((p) => presetKey(p) === wanted) ?? presets[0];
  await app.views.simulate.start(first?.config ?? { mode: 'core', map: { source: 'file', name: 'deck_demo' }, agent: 'andor' }, { quiet: true });
  document.getElementById('loading').classList.add('gone');
  const startView = params.get('view');
  if (wanted && first) await applyPreset(first);
  else if (startView && startView !== 'simulate') await switchView(startView);
  if (!wanted && !startView && params.get('welcome') !== '0' && !store.get(WELCOMED_KEY, false)) openWelcome();
}

function catalog() {
  return app.catalog ?? {};
}

boot();
