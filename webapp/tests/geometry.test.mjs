// node --test webapp/tests : canvas / grid geometry and devicePixelRatio sizing.
import { test } from 'node:test';
import assert from 'node:assert/strict';
import {
  computeLayout, cellAt, cellCenter, cellOrigin, backingSize, crisp, lineCells, RULER, MAX_CELL,
} from '../js/geometry.js';

test('layout fits square cells, centred, with rulers', () => {
  const L = computeLayout(4, 4, 500, 400);
  assert.equal(L.ruler, RULER);
  assert.equal(L.cell, Math.floor((400 - RULER) / 4));
  assert.ok(L.ox >= RULER && L.oy >= RULER);
  assert.ok(L.ox + L.gridW <= 500 && L.oy + L.gridH <= 400);
});

test('tiny cells drop the rulers; huge boxes cap the cell size', () => {
  assert.equal(computeLayout(40, 40, 300, 300).ruler, 0);
  assert.equal(computeLayout(2, 2, 2000, 2000).cell, MAX_CELL);
});

test('hit testing round-trips through cell centres and rejects outside points', () => {
  const L = computeLayout(10, 15, 800, 600);
  for (const [r, c] of [[0, 0], [9, 14], [4, 7]]) {
    const { x, y } = cellCenter(L, r, c);
    assert.deepEqual(cellAt(L, x, y), [r, c]);
    const o = cellOrigin(L, r, c);
    assert.deepEqual(cellAt(L, o.x + 0.01, o.y + 0.01), [r, c]);
  }
  assert.equal(cellAt(L, L.ox - 1, L.oy + 5), null);
  assert.equal(cellAt(L, L.ox + L.gridW + 1, L.oy + 5), null);
});

test('backing store is an integer multiple of the CSS size (crisp at 125 % / 150 %)', () => {
  assert.deepEqual(backingSize(401, 300, 1.25), { width: 501, height: 375, ratio: 1.25 });
  assert.deepEqual(backingSize(100, 50, 2), { width: 200, height: 100, ratio: 2 });
  assert.deepEqual(backingSize(0, 0, NaN), { width: 1, height: 1, ratio: 1 });
  assert.equal(crisp(10, 1), 10.5);
  assert.equal(crisp(10, 2), 10.25);
});

test('lineCells walks a connected line for drag painting', () => {
  const cells = lineCells([0, 0], [3, 5]);
  assert.deepEqual(cells[0], [0, 0]);
  assert.deepEqual(cells[cells.length - 1], [3, 5]);
  for (let i = 1; i < cells.length; i += 1) {
    const dr = Math.abs(cells[i][0] - cells[i - 1][0]);
    const dc = Math.abs(cells[i][1] - cells[i - 1][1]);
    assert.ok(dr <= 1 && dc <= 1 && dr + dc >= 1);
  }
  assert.deepEqual(lineCells([2, 2], [2, 2]), [[2, 2]]);
});
