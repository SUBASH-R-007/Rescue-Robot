// Small DOM helpers shared by every view: icons, toasts, info tooltips, dialogs, storage.
import { escapeHtml } from './format.js';

/** Inline SVG icon from the sprite in index.html. */
export function icon(name, cls = '') {
  return `<svg aria-hidden="true"${cls ? ` class="${cls}"` : ''}><use href="#i-${name}"/></svg>`;
}

let tipCount = 0;

/**
 * An (i) button whose tooltip explains a concept: hover, keyboard focus, click or tap opens
 * it. Screen readers get a short name plus the text as the button's description.
 */
export function infoTip(text, label = 'More information') {
  tipCount += 1;
  const id = `tipd-${tipCount}`;
  return `<button type="button" class="info-btn" data-tip="${escapeHtml(text)}" aria-label="About ${escapeHtml(label)}" aria-describedby="${id}" aria-expanded="false">${icon('info')}</button><span class="sr-only" id="${id}">${escapeHtml(text)}</span>`;
}

const TOAST_ICONS = { good: 'check', bad: 'alert', warn: 'alert', info: 'info' };

/** Show a toast; returns a function that closes it. */
export function toast({ title, text = '', tone = 'info', detail = '', timeout = 5000 }) {
  const box = document.getElementById('toasts');
  const el = document.createElement('div');
  el.className = `toast tone-${tone}`;
  el.setAttribute('role', tone === 'bad' ? 'alert' : 'status');
  el.innerHTML = `${icon(TOAST_ICONS[tone] ?? 'info')}
    <div><strong>${escapeHtml(title)}</strong>${text ? `<p>${escapeHtml(text)}</p>` : ''}
    ${detail ? `<details><summary>Technical details</summary><pre>${escapeHtml(detail)}</pre></details>` : ''}</div>
    <button class="icon-btn" aria-label="Dismiss">${icon('close')}</button>`;
  const close = () => {
    el.remove();
  };
  el.querySelector('button').addEventListener('click', close);
  box.append(el);
  while (box.children.length > 4) box.firstElementChild.remove();
  if (timeout && !detail) setTimeout(close, timeout);
  return close;
}

/** Friendly texts of known error types (BridgeError.type). */
const FRIENDLY_ERRORS = {
  ScipyUnavailable: 'The extra code this needs (scipy) could not be downloaded. Check the network connection and try again; bundled maps and generated maps with seeds 0–29 at the library sizes work without it.',
  WorkerCrashed: 'The Python worker stopped. Reload the page to start it again.',
};

/** Toast for an error from the bridge (Python message + traceback in the details). */
export function showError(err, context = 'Something went wrong') {
  if (err?.type === 'SessionDisposed') return; // a call for a mission that was replaced
  console.warn(context, err);
  const text = FRIENDLY_ERRORS[err?.type] ?? err?.message ?? String(err);
  toast({ title: context, text, detail: err?.detail ?? '', tone: 'bad', timeout: 0 });
}

/** Say `text` to screen readers through the page's one polite status region. */
export function announce(text) {
  const region = document.getElementById('sr-status');
  if (!region) return;
  region.textContent = '';
  // A new text node in a persistent live region is announced reliably.
  requestAnimationFrame(() => { region.textContent = text; });
}

/** The global chip for on-demand downloads (scipy): shown with text, hidden with ''. */
export function showProgress(text, detail = '') {
  const chip = document.getElementById('progress-chip');
  if (!chip) return;
  chip.hidden = !text;
  chip.querySelector('.pc-text').textContent = text;
  chip.querySelector('.pc-detail').textContent = detail;
}

/** A click within this many ms of the pointer / focus that opened a tip keeps it open (a tap
 * fires pointerover, focusin and click in a row). */
const TIP_OPEN_GRACE_MS = 400;

/** Global [data-tip] tooltips on hover, keyboard focus, click and tap. */
export function initTips() {
  const tip = document.getElementById('tip');
  let anchor = null;
  let shownAt = 0;
  const show = (el) => {
    if (anchor !== el) shownAt = performance.now();
    anchor?.setAttribute?.('aria-expanded', 'false');
    anchor = el;
    if (el.classList.contains('info-btn')) el.setAttribute('aria-expanded', 'true');
    tip.textContent = el.dataset.tip;
    tip.hidden = false;
    const r = el.getBoundingClientRect();
    const tw = tip.offsetWidth;
    const th = tip.offsetHeight;
    let x = r.left + r.width / 2 - tw / 2;
    x = Math.max(8, Math.min(window.innerWidth - tw - 8, x));
    let y = r.bottom + 8;
    if (y + th > window.innerHeight - 8) y = r.top - th - 8;
    tip.style.left = `${x}px`;
    tip.style.top = `${Math.max(8, y)}px`;
  };
  const hide = () => {
    anchor?.setAttribute?.('aria-expanded', 'false');
    anchor = null;
    tip.hidden = true;
  };
  document.addEventListener('pointerover', (ev) => {
    const el = ev.target.closest?.('[data-tip]');
    if (el && el !== anchor) show(el);
    else if (!el && anchor) hide();
  });
  document.addEventListener('focusin', (ev) => {
    const el = ev.target.closest?.('[data-tip]');
    if (el) show(el);
    else if (anchor) hide();
  });
  document.addEventListener('focusout', hide);
  document.addEventListener('scroll', hide, true);
  document.addEventListener('keydown', (ev) => { if (ev.key === 'Escape') hide(); });
  // Click / tap on an (i): open its tip; a second click (not the one that just opened it
  // through pointerover / focus) closes it again.
  document.addEventListener('click', (ev) => {
    const el = ev.target.closest?.('.info-btn[data-tip]');
    if (!el) return;
    ev.preventDefault();
    const justOpened = performance.now() - shownAt < TIP_OPEN_GRACE_MS;
    if (anchor === el && !tip.hidden && !justOpened) hide();
    else show(el);
  });
}

/** Wire [data-close] buttons and backdrop clicks of every <dialog>. */
export function initDialogs() {
  for (const dlg of document.querySelectorAll('dialog')) {
    dlg.addEventListener('click', (ev) => {
      if (ev.target.closest('[data-close]')) dlg.close();
      else if (ev.target === dlg) {
        // Click on the backdrop (outside the dialog box).
        const r = dlg.getBoundingClientRect();
        const inside = ev.clientX >= r.left && ev.clientX <= r.right && ev.clientY >= r.top && ev.clientY <= r.bottom;
        if (!inside) dlg.close();
      }
    });
  }
}

/** localStorage that never throws (private windows, blocked storage). */
export const store = {
  get(key, fallback = null) {
    try {
      const v = localStorage.getItem(key);
      return v === null ? fallback : JSON.parse(v);
    } catch {
      return fallback;
    }
  },
  set(key, value) {
    try {
      localStorage.setItem(key, JSON.stringify(value));
    } catch { /* storage unavailable: the app works without it */ }
  },
};

/** True when keyboard shortcuts should be ignored (typing in a field). */
export function isTyping(ev) {
  const t = ev.target;
  return t instanceof HTMLElement && (t.isContentEditable || /^(INPUT|SELECT|TEXTAREA)$/.test(t.tagName));
}

/** Debounce. */
export function debounce(fn, ms) {
  let id = 0;
  return (...args) => {
    clearTimeout(id);
    id = setTimeout(() => fn(...args), ms);
  };
}

/** A small canvas swatch drawn by `paint(ctx, size)` (legend, editor tools). */
export function swatch(paint, size = 22) {
  const c = document.createElement('canvas');
  const dpr = window.devicePixelRatio || 1;
  c.width = Math.round(size * dpr);
  c.height = Math.round(size * dpr);
  c.style.width = `${size}px`;
  c.style.height = `${size}px`;
  const ctx = c.getContext('2d');
  ctx.scale(dpr, dpr);
  paint(ctx, size);
  return c;
}
