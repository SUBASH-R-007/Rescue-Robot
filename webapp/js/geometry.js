// Pure canvas / grid geometry (no DOM): layout of a rows x cols grid inside a box, hit tests
// and device-pixel-ratio sizing. All coordinates here are CSS pixels.

/** Largest and smallest cell edge (CSS px) the map view uses. */
export const MAX_CELL = 96;
export const MIN_CELL = 6;
/** Space for the 1-indexed rulers (row numbers left, column numbers on top), CSS px. */
export const RULER = 22;
/** Cells smaller than this hide the rulers (numbers would not fit). */
export const RULER_MIN_CELL = 14;

/**
 * Fit a rows x cols grid into a width x height box, centred, with square cells.
 * Returns { cell, ox, oy, gridW, gridH, ruler } where (ox, oy) is the top-left corner of
 * cell (0, 0) and `ruler` the ruler band actually reserved (0 when the cells are too small).
 */
export function computeLayout(rows, cols, width, height, opts = {}) {
  const maxCell = opts.maxCell ?? MAX_CELL;
  const minCell = opts.minCell ?? MIN_CELL;
  const wantRuler = opts.ruler ?? true;
  const fit = (band) => Math.floor(Math.min((width - band) / cols, (height - band) / rows));
  let ruler = wantRuler ? RULER : 0;
  let cell = fit(ruler);
  if (ruler && cell < RULER_MIN_CELL) {
    ruler = 0;
    cell = fit(0);
  }
  cell = Math.max(minCell, Math.min(maxCell, cell));
  const gridW = cell * cols;
  const gridH = cell * rows;
  const ox = Math.round(ruler + (width - ruler - gridW) / 2);
  const oy = Math.round(ruler + (height - ruler - gridH) / 2);
  return { rows, cols, cell, ox, oy, gridW, gridH, ruler };
}

/** The [r, c] under CSS point (x, y), or null outside the grid. */
export function cellAt(layout, x, y) {
  const c = Math.floor((x - layout.ox) / layout.cell);
  const r = Math.floor((y - layout.oy) / layout.cell);
  if (r < 0 || c < 0 || r >= layout.rows || c >= layout.cols) return null;
  return [r, c];
}

/** Top-left CSS corner of cell [r, c]. */
export function cellOrigin(layout, r, c) {
  return { x: layout.ox + c * layout.cell, y: layout.oy + r * layout.cell };
}

/** CSS centre of cell [r, c]. */
export function cellCenter(layout, r, c) {
  return {
    x: layout.ox + (c + 0.5) * layout.cell,
    y: layout.oy + (r + 0.5) * layout.cell,
  };
}

/**
 * Backing-store size for a canvas of cssW x cssH at device pixel ratio dpr: integers, at
 * least 1, so the bitmap maps 1:1 onto physical pixels (crisp lines at 125 % / 150 %).
 */
export function backingSize(cssW, cssH, dpr) {
  const ratio = dpr > 0 && Number.isFinite(dpr) ? dpr : 1;
  return {
    width: Math.max(1, Math.round(cssW * ratio)),
    height: Math.max(1, Math.round(cssH * ratio)),
    ratio,
  };
}

/** Snap a CSS coordinate to the centre of a physical pixel (for 1-px lines). */
export function crisp(x, dpr = 1) {
  return (Math.round(x * dpr) + 0.5) / dpr;
}

/** Cells on the straight segment between two cell centres, for drag painting (Bresenham). */
export function lineCells(a, b) {
  const out = [];
  let [r0, c0] = a;
  const [r1, c1] = b;
  const dr = Math.abs(r1 - r0);
  const dc = Math.abs(c1 - c0);
  const sr = r0 < r1 ? 1 : -1;
  const sc = c0 < c1 ? 1 : -1;
  let err = dc - dr;
  for (;;) {
    out.push([r0, c0]);
    if (r0 === r1 && c0 === c1) break;
    const e2 = 2 * err;
    if (e2 > -dr) { err -= dr; c0 += sc; }
    if (e2 < dc) { err += dc; r0 += sr; }
  }
  return out;
}

/** Direction vector of an action letter. */
export const DELTA = Object.freeze({ N: [-1, 0], E: [0, 1], S: [1, 0], W: [0, -1] });
