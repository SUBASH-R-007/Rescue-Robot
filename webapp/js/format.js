// Pure formatting helpers (no DOM). Everything a person reads is 1-indexed.

/** Cell codes of the Python core (grid.Cell). */
export const CELL = Object.freeze({ UNKNOWN: 0, OPEN: 1, DEBRIS: 2, SURVIVOR: 3 });

/** Format an internal 0-indexed [r, c] for humans: [0, 0] -> "(1,1)". */
export function fmtPos(p) {
  if (!Array.isArray(p) || p.length < 2) return '–';
  return `(${p[0] + 1},${p[1] + 1})`;
}

/** Key of a position in policy dicts: [2, 3] -> "2,3" (0-indexed, like the bridge). */
export function posKey(p) {
  return `${p[0]},${p[1]}`;
}

/** Inverse of posKey: "2,3" -> [2, 3]. */
export function parseKey(key) {
  const [r, c] = String(key).split(',').map(Number);
  return [r, c];
}

/** True when two positions are the same cell. */
export function samePos(a, b) {
  return Array.isArray(a) && Array.isArray(b) && a[0] === b[0] && a[1] === b[1];
}

/** Compass names of the four actions. */
export const ACTION_NAMES = Object.freeze({ N: 'north', E: 'east', S: 'south', W: 'west' });
/** Arrow glyphs of the four actions. */
export const ACTION_ARROWS = Object.freeze({ N: '↑', E: '→', S: '↓', W: '←' });

/** Normalise an outcome label from the bridge ("arrived", "slipped", "skid-L", "SKID_R"...). */
export function normOutcome(outcome) {
  if (outcome == null) return null;
  const s = String(outcome).toLowerCase().replace('_', '-');
  if (s.startsWith('arriv')) return 'arrived';
  if (s.startsWith('slip')) return 'slipped';
  if (s === 'skid-l') return 'skid-L';
  if (s === 'skid-r') return 'skid-R';
  return s;
}

/** Short human label of an outcome: "arrived" -> "moved as planned". */
export function outcomeText(outcome, blocked = false) {
  const o = normOutcome(outcome);
  let text;
  switch (o) {
    case 'arrived': text = 'moved as planned'; break;
    case 'slipped': text = 'slipped (stayed put)'; break;
    case 'skid-L': text = 'skidded left'; break;
    case 'skid-R': text = 'skidded right'; break;
    default: text = o ?? '';
  }
  return blocked && o !== 'slipped' ? `${text}, blocked` : text;
}

/** Status of a mission -> label, tone (CSS) and glyph (never colour alone). */
export function statusInfo(status) {
  switch (status) {
    case 'running': return { label: 'Running', tone: 'info', glyph: '▶' };
    case 'rescued': return { label: 'Rescued', tone: 'good', glyph: '✓' };
    case 'aborted': return { label: 'Gave up', tone: 'bad', glyph: '✕' };
    case 'budget': return { label: 'Out of steps', tone: 'bad', glyph: '⌛' };
    case 'unreachable': return { label: 'No route', tone: 'bad', glyph: '⊘' };
    default: return { label: String(status ?? '…'), tone: 'muted', glyph: '•' };
  }
}

/** Number formatting with a fixed number of decimals, '–' for null / NaN. */
export function fmtNum(x, digits = 0) {
  if (x === null || x === undefined || Number.isNaN(Number(x))) return '–';
  const n = Number(x);
  if (!Number.isFinite(n)) return n > 0 ? '∞' : '−∞';
  return n.toFixed(digits);
}

/** Probability as a percentage: 0.2 -> "20 %". */
export function fmtPct(x, digits = 0) {
  if (x === null || x === undefined || Number.isNaN(Number(x))) return '–';
  return `${(Number(x) * 100).toFixed(digits)} %`;
}

/** Milliseconds, adaptive precision: 0.42 -> "0.42 ms", 12.3 -> "12 ms", 1534 -> "1.5 s". */
export function fmtMs(ms) {
  if (ms === null || ms === undefined || Number.isNaN(Number(ms))) return '–';
  const n = Number(ms);
  if (n >= 1000) return `${(n / 1000).toFixed(1)} s`;
  if (n >= 10) return `${Math.round(n)} ms`;
  return `${n.toFixed(2)} ms`;
}

/** "1 replan" / "2 replans". */
export function plural(n, word, pluralWord = `${word}s`) {
  return `${n} ${n === 1 ? word : pluralWord}`;
}

/** Escape text for innerHTML. */
export function escapeHtml(text) {
  return String(text ?? '')
    .replaceAll('&', '&amp;')
    .replaceAll('<', '&lt;')
    .replaceAll('>', '&gt;')
    .replaceAll('"', '&quot;')
    .replaceAll("'", '&#39;');
}

/** Verdict glyph of a claims table -> label and tone. */
export function verdictInfo(verdict) {
  const v = String(verdict ?? '').trim();
  if (v.includes('✅') || /^supported$/i.test(v)) return { label: 'Supported', tone: 'good', glyph: '✓' };
  if (v.includes('❌') || /^refuted$/i.test(v)) return { label: 'Refuted', tone: 'bad', glyph: '✕' };
  if (v.includes('⚠') || /inconclusive/i.test(v)) return { label: 'Inconclusive', tone: 'warn', glyph: '!' };
  if (v.includes('ℹ') || /^descriptive$/i.test(v)) return { label: 'Measurement', tone: 'muted', glyph: 'i' };
  return { label: v || 'Not measured', tone: 'muted', glyph: '?' };
}

/**
 * The headline number of a "Measured" cell: the text before the first ';' (and before a
 * " over " / " — " clause), trimmed. "100.0% [99.8%, 100.0%] (2400/2400); Wilson…" ->
 * "100.0% [99.8%, 100.0%] (2400/2400)".
 */
export function keyNumber(measured) {
  let s = String(measured ?? '').trim();
  const semi = s.indexOf(';');
  if (semi >= 0) s = s.slice(0, semi);
  const over = s.indexOf(' over ');
  if (over > 0) s = s.slice(0, over);
  return s.replaceAll('`', '').trim();
}

/** Strip markdown backticks and bold markers for plain display. */
export function plainMd(text) {
  return String(text ?? '').replaceAll('`', '').replace(/\*\*(.*?)\*\*/g, '$1');
}

/** Clamp x into [lo, hi]. */
export function clamp(x, lo, hi) {
  return Math.min(hi, Math.max(lo, x));
}

/** Pretty-print a plan string from the bridge (already 1-indexed); collapse whitespace. */
export function tidyPlan(text) {
  return String(text ?? '').replace(/\s+/g, ' ').trim();
}

/** Title-case an identifier: "collapsed_building" -> "Collapsed building". */
export function humanize(id) {
  const s = String(id ?? '').replaceAll('_', ' ').trim();
  return s ? s[0].toUpperCase() + s.slice(1) : s;
}

/** Inline Markdown (already escaped text): `code` and **bold**. */
function mdInline(text) {
  return escapeHtml(text)
    .replace(/`([^`]+)`/g, '<code>$1</code>')
    .replace(/\*\*(.+?)\*\*/g, '<strong>$1</strong>');
}

/**
 * Minimal Markdown → HTML for the reports' detail texts: paragraphs, pipe tables, "- " lists,
 * `code` and **bold**. Everything is escaped first (the text is data, not markup).
 */
export function renderMarkdown(text) {
  const blocks = String(text ?? '').replace(/\r\n/g, '\n').split(/\n\s*\n/);
  const out = [];
  for (const block of blocks) {
    const lines = block.split('\n').map((l) => l.trim()).filter(Boolean);
    if (!lines.length) continue;
    if (lines.every((l) => l.startsWith('|'))) {
      const cells = (l) => l.replace(/^\||\|$/g, '').split('|').map((c) => c.trim());
      const rows = lines.filter((l) => !/^\|[\s:|-]+\|?$/.test(l)).map(cells);
      const [head, ...body] = rows;
      out.push(`<div class="md-table"><table><thead><tr>${head.map((c) => `<th>${mdInline(c)}</th>`).join('')}</tr></thead><tbody>${body.map((r) => `<tr>${r.map((c) => `<td>${mdInline(c)}</td>`).join('')}</tr>`).join('')}</tbody></table></div>`);
    } else if (lines.every((l) => /^[-*] /.test(l))) {
      out.push(`<ul>${lines.map((l) => `<li>${mdInline(l.slice(2))}</li>`).join('')}</ul>`);
    } else if (/^#{1,6} /.test(lines[0])) {
      out.push(`<p><strong>${mdInline(lines[0].replace(/^#+ /, ''))}</strong></p>`);
      if (lines.length > 1) out.push(`<p>${mdInline(lines.slice(1).join(' '))}</p>`);
    } else {
      out.push(`<p>${mdInline(lines.join(' '))}</p>`);
    }
  }
  return out.join('');
}

/** Prose paragraphs of a Markdown text (no tables, lists or headings), backticks kept. */
export function mdParagraphs(text) {
  return String(text ?? '').replace(/\r\n/g, '\n').split(/\n\s*\n/)
    .map((b) => b.trim())
    .filter((b) => b && !b.startsWith('|') && !b.startsWith('#') && !/^[-*] /.test(b))
    .map((b) => b.split('\n').map((l) => l.trim()).join(' '));
}

/** Percentage with as many decimals as needed (up to 2): 0.8 -> "80 %", 0.025 -> "2.5 %",
 * 0.15 (float noise 15.000000000000002) -> "15 %". */
export function fmtPctAuto(x) {
  if (x === null || x === undefined || Number.isNaN(Number(x))) return '–';
  const v = Math.round(Number(x) * 10000) / 100;
  let digits = 0;
  while (digits < 2 && Math.abs(v * 10 ** digits - Math.round(v * 10 ** digits)) > 1e-6) digits += 1;
  return `${v.toFixed(digits)} %`;
}

/**
 * Probabilities of the four outcomes of a move at slip parameter p. `fractions` comes from the
 * catalog (params.outcome_fractions = config.py's SLIP_STAY_FRACTION / SKID_FRACTION): the
 * robot stays with stay·p and skids with skid·p, split equally left and right.
 */
export function outcomeSplit(p, fractions) {
  const stay = fractions.stay * p;
  const skid = fractions.skid * p;
  return { arrive: 1 - stay - skid, stay, skidL: skid / 2, skidR: skid / 2 };
}

/** Glyph of each explanation tone (webtext.TONES; shape as well as colour) and its CSS class. */
export const TONE_STYLE = Object.freeze({
  move: { glyph: '→', cls: 'ok' },
  slip: { glyph: '↺', cls: 'slip' },
  skid: { glyph: '▲', cls: 'skid' },
  blocked: { glyph: '▮', cls: 'skid' },
  bump: { glyph: '✕', cls: 'bad' },
  aftershock: { glyph: '≋', cls: 'bad' },
  sense: { glyph: '◉', cls: 'sense' },
  replan: { glyph: '◆', cls: 'replan' },
  search: { glyph: '⌕', cls: 'ok' },
  found: { glyph: '★', cls: 'good' },
  good: { glyph: '✓', cls: 'good' },
  bad: { glyph: '■', cls: 'bad' },
  next: { glyph: '»', cls: 'muted' },
  info: { glyph: '•', cls: 'muted' },
});

/** Style of a tone (unknown tones look like plain information). */
export function toneStyle(tone) {
  return TONE_STYLE[tone] ?? TONE_STYLE.info;
}

/**
 * The sentences of a state with their tones: [{text, tone}]. States without tones (older
 * recordings) get the neutral tone; the UI never guesses a tone from the text.
 */
export function tonedSentences(state) {
  const texts = state?.explanation ?? [];
  const tones = state?.explanation_tones ?? [];
  return texts.map((text, i) => ({ text, tone: tones[i] ?? 'info' }));
}

/** Tones that carry a mission's end (shown first in short summaries). */
const END_TONES = new Set(['good', 'bad']);
/** Tones worth a second line in a short summary while the mission runs. */
const NOTABLE_TONES = ['bump', 'aftershock', 'replan', 'found', 'slip', 'skid', 'blocked', 'sense', 'search'];

/**
 * At most `n` sentences that best summarise a step: the move sentence, then (once the mission
 * is over) the end sentence, else the most notable reaction. Keeps the original order.
 */
export function keySentences(state, n = 2) {
  const all = tonedSentences(state);
  if (all.length <= n) return all;
  const picked = new Set([0]);
  const done = state?.status && state.status !== 'running';
  const pick = (pred) => {
    for (let i = all.length - 1; i >= 0 && picked.size < n; i -= 1) {
      if (!picked.has(i) && pred(all[i])) { picked.add(i); return; }
    }
  };
  if (done) pick((s) => END_TONES.has(s.tone));
  for (const tone of NOTABLE_TONES) if (picked.size < n) pick((s) => s.tone === tone);
  pick(() => true);
  return [...picked].sort((a, b) => a - b).map((i) => all[i]);
}
