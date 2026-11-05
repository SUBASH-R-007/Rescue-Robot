// Pure ASCII map helpers for the editor (no DOM). The ASCII format is the core's
// (grid.py): '.' open, '#' debris, 'S' start, 'P' survivor, '~' open + high slip;
// ';' starts a comment line. The Python bridge (webapi.edit_map / validate_map) is the
// authority; these helpers give instant feedback while painting and pick the edit ops.
import { CELL } from './format.js';

export const CH = Object.freeze({ OPEN: '.', DEBRIS: '#', START: 'S', SURVIVOR: 'P', WATER: '~' });

// Size limits are not copied here: the editor reads catalog.params.map_size (webmaps.py).

/** Parse ASCII into an array of row arrays (comments / blank lines dropped). */
export function parseAscii(text) {
  return String(text ?? '')
    .split(/\r?\n/)
    .map((line) => line.trim())
    .filter((line) => line && !line.startsWith(';'))
    .map((line) => [...line]);
}

/** Rows back to ASCII (one line per row, trailing newline). */
export function toAscii(rows) {
  return `${rows.map((r) => r.join('')).join('\n')}\n`;
}

/** {rows, cols} of an ASCII map. */
export function asciiSize(text) {
  const rows = parseAscii(text);
  return { rows: rows.length, cols: rows[0]?.length ?? 0 };
}

/** Character at [r, c] ('' outside). */
export function charAt(text, r, c) {
  return parseAscii(text)[r]?.[c] ?? '';
}

/** ASCII of a bridge state (true map): cells, high_slip, start, survivors. */
export function stateToAscii(state) {
  const rows = [];
  for (let r = 0; r < state.rows; r += 1) {
    const row = [];
    for (let c = 0; c < state.cols; c += 1) {
      const code = state.cells[r][c];
      let ch = CH.OPEN;
      if (code === CELL.DEBRIS) ch = CH.DEBRIS;
      else if (code === CELL.SURVIVOR) ch = CH.SURVIVOR;
      else if (state.high_slip?.[r]?.[c]) ch = CH.WATER;
      row.push(ch);
    }
    rows.push(row);
  }
  if (state.start) rows[state.start[0]][state.start[1]] = CH.START;
  return toAscii(rows);
}

/** A render-ready pseudo state of an ASCII map (the editor shows the true map, no fog). */
export function asciiToState(text) {
  const grid = parseAscii(text);
  const rows = grid.length;
  const cols = grid[0]?.length ?? 0;
  const cells = [];
  const highSlip = [];
  const survivors = [];
  let start = null;
  grid.forEach((row, r) => {
    cells.push(row.map((ch) => (ch === CH.DEBRIS ? CELL.DEBRIS : ch === CH.SURVIVOR ? CELL.SURVIVOR : CELL.OPEN)));
    highSlip.push(row.map((ch) => ch === CH.WATER));
    row.forEach((ch, c) => {
      if (ch === CH.START) start = [r, c];
      if (ch === CH.SURVIVOR) survivors.push([r, c]);
    });
  });
  return {
    rows, cols, cells, belief: cells, high_slip: highSlip, start, survivors,
    robot: null, rescued: [], trace: [], policy: {}, path: null, target: null, p_debris: null,
  };
}

/** A blank rows x cols map with S top-left and P bottom-right. */
export function blankMap(rows, cols) {
  const grid = Array.from({ length: rows }, () => Array(cols).fill(CH.OPEN));
  grid[0][0] = CH.START;
  grid[rows - 1][cols - 1] = CH.SURVIVOR;
  return toAscii(grid);
}

/** Editor tools -> the bridge ops (webapi.edit_map) that give the cell under the brush the
 * tool's content. Returns [] when the cell already has it (so dragging never toggles back and
 * forth) or when the cell is protected (paint never covers the start or a survivor). */
export function opsForTool(tool, ch, r, c) {
  const cell = [r, c];
  const endpoint = ch === CH.START || ch === CH.SURVIVOR;
  const paint = (value) => (ch === value || endpoint ? [] : [{ op: 'paint', cell, value }]);
  switch (tool) {
    case 'debris': return paint(CH.DEBRIS);
    case 'water': return paint(CH.WATER);
    case 'erase': return paint(CH.OPEN);
    case 'start': return ch === CH.START || ch === CH.SURVIVOR ? [] : [{ op: 'set_start', cell }];
    case 'survivor': return endpoint ? [] : [{ op: 'set_survivor', cell }];
    case 'extra':
      if (ch === CH.SURVIVOR) return [{ op: 'remove_survivor', cell }];
      if (ch === CH.START) return [];
      return [{ op: 'add_survivor', cell }];
    default:
      return [];
  }
}

/** Tools that paint while dragging (the others act once per click). */
export const DRAG_TOOLS = new Set(['debris', 'water', 'erase']);

/**
 * Local preview of an op (mirrors webapi.edit_map, whose result replaces it). Returns new
 * ASCII; refused edits (e.g. debris on the start) leave the map unchanged.
 */
export function applyOpLocal(text, op) {
  const grid = parseAscii(text);
  const rows = grid.length;
  const cols = grid[0]?.length ?? 0;
  if (op.op === 'resize') return resizeLocal(text, op.rows, op.cols);
  if (op.op === 'clear') {
    return toAscii(grid.map((row) => row.map((ch) => (ch === CH.START || ch === CH.SURVIVOR ? ch : CH.OPEN))));
  }
  const [r, c] = op.cell ?? [-1, -1];
  if (r < 0 || c < 0 || r >= rows || c >= cols) return text;
  const ch = grid[r][c];
  const endpoint = ch === CH.START || ch === CH.SURVIVOR;
  const clearAll = (x) => { for (const row of grid) for (let j = 0; j < cols; j += 1) if (row[j] === x) row[j] = CH.OPEN; };
  switch (op.op) {
    case 'paint':
      if (!endpoint && [CH.OPEN, CH.DEBRIS, CH.WATER].includes(op.value)) grid[r][c] = op.value;
      break;
    case 'toggle_debris':
      if (!endpoint) grid[r][c] = ch === CH.DEBRIS ? CH.OPEN : CH.DEBRIS;
      break;
    case 'toggle_high_slip':
      if (!endpoint) grid[r][c] = ch === CH.WATER ? CH.OPEN : CH.WATER;
      break;
    case 'set_start':
      if (ch !== CH.SURVIVOR) { clearAll(CH.START); grid[r][c] = CH.START; }
      break;
    case 'set_survivor':
      if (ch !== CH.START) { clearAll(CH.SURVIVOR); grid[r][c] = CH.SURVIVOR; }
      break;
    case 'add_survivor':
      if (!endpoint) grid[r][c] = CH.SURVIVOR;
      break;
    case 'remove_survivor':
      if (ch === CH.SURVIVOR && grid.flat().filter((x) => x === CH.SURVIVOR).length > 1) grid[r][c] = CH.OPEN;
      break;
    default:
      break;
  }
  return toAscii(grid);
}

/** Cells holding `ch`, row by row. */
function findAll(grid, ch) {
  const out = [];
  grid.forEach((row, r) => row.forEach((x, c) => { if (x === ch) out.push([r, c]); }));
  return out;
}

/** First open / water cell not in `avoid` (keys "r,c"), else the first cell not in it
 * (webmaps._first_free). */
function firstFree(grid, avoid) {
  for (const pass of [true, false]) {
    for (let r = 0; r < grid.length; r += 1) {
      for (let c = 0; c < grid[r].length; c += 1) {
        const free = !pass || grid[r][c] === CH.OPEN || grid[r][c] === CH.WATER;
        if (free && !avoid.has(`${r},${c}`)) return [r, c];
      }
    }
  }
  return null;
}

/**
 * Resize like webmaps._resize (the preview must match Python's answer): crop / pad with open
 * cells; a start or survivor that is cut off moves to the first free cell, unless one is
 * still on the map. Sizes are taken as given (the editor checks the catalog's limits).
 */
export function resizeLocal(text, rows, cols) {
  const old = parseAscii(text);
  const oldRows = old.length;
  const oldCols = old[0]?.length ?? 0;
  const grid = Array.from({ length: rows }, (_, r) =>
    Array.from({ length: cols }, (__, c) => (r < oldRows && c < oldCols ? old[r][c] : CH.OPEN)));
  const lost = [];
  for (let r = 0; r < oldRows; r += 1) {
    for (let c = 0; c < oldCols; c += 1) {
      if ((r >= rows || c >= cols) && (old[r][c] === CH.START || old[r][c] === CH.SURVIVOR)) lost.push(old[r][c]);
    }
  }
  for (const ch of lost) {
    if (findAll(grid, ch).length) continue; // keep only what still fits (at least one each)
    const avoid = new Set([...findAll(grid, CH.START), ...findAll(grid, CH.SURVIVOR)].map(([r, c]) => `${r},${c}`));
    const spot = firstFree(grid, avoid);
    if (spot) grid[spot[0]][spot[1]] = ch;
  }
  return toAscii(grid);
}

/** Counts of each cell character. */
export function mapStats(text) {
  const counts = { open: 0, debris: 0, water: 0, survivors: 0, starts: 0 };
  for (const ch of parseAscii(text).flat()) {
    if (ch === CH.OPEN) counts.open += 1;
    else if (ch === CH.DEBRIS) counts.debris += 1;
    else if (ch === CH.WATER) counts.water += 1;
    else if (ch === CH.SURVIVOR) counts.survivors += 1;
    else if (ch === CH.START) counts.starts += 1;
  }
  return counts;
}
