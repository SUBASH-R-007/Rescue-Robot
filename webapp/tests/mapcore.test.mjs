// node --test webapp/tests : the editor's ASCII helpers (local preview of webapi.edit_map).
import { test } from 'node:test';
import assert from 'node:assert/strict';
import {
  parseAscii, toAscii, asciiSize, charAt, asciiToState, stateToAscii, blankMap, opsForTool,
  applyOpLocal, resizeLocal, mapStats, CH,
} from '../js/mapcore.js';

const DECK = 'S...\n....\n.#..\n..#P\n';

test('parse / print round trip, comments dropped', () => {
  assert.equal(toAscii(parseAscii(`; a comment\n${DECK}`)), DECK);
  assert.deepEqual(asciiSize(DECK), { rows: 4, cols: 4 });
  assert.equal(charAt(DECK, 2, 1), '#');
  assert.equal(charAt(DECK, 9, 9), '');
});

test('ASCII -> render state -> ASCII keeps start, survivors, debris and water', () => {
  const text = 'S.~.\n.#..\n...P\n';
  const state = asciiToState(text);
  assert.deepEqual(state.start, [0, 0]);
  assert.deepEqual(state.survivors, [[2, 3]]);
  assert.equal(state.high_slip[0][2], true);
  assert.equal(stateToAscii(state), text);
});

test('tools never paint over the start or a survivor, and dragging does not toggle back', () => {
  assert.deepEqual(opsForTool('debris', CH.START, 0, 0), []);
  assert.deepEqual(opsForTool('debris', CH.DEBRIS, 1, 1), []);
  assert.deepEqual(opsForTool('debris', CH.OPEN, 1, 1), [{ op: 'paint', cell: [1, 1], value: '#' }]);
  assert.deepEqual(opsForTool('extra', CH.SURVIVOR, 3, 3), [{ op: 'remove_survivor', cell: [3, 3] }]);
  assert.deepEqual(opsForTool('start', CH.OPEN, 2, 2), [{ op: 'set_start', cell: [2, 2] }]);
});

test('local edits mirror the bridge ops', () => {
  let t = blankMap(3, 4);
  assert.equal(t, 'S...\n....\n...P\n');
  t = applyOpLocal(t, { op: 'paint', cell: [1, 1], value: '#' });
  t = applyOpLocal(t, { op: 'toggle_high_slip', cell: [1, 2] });
  t = applyOpLocal(t, { op: 'set_survivor', cell: [0, 3] });
  assert.equal(t, 'S..P\n.#~.\n....\n');
  t = applyOpLocal(t, { op: 'add_survivor', cell: [2, 0] });
  assert.equal(mapStats(t).survivors, 2);
  t = applyOpLocal(t, { op: 'remove_survivor', cell: [2, 0] });
  t = applyOpLocal(t, { op: 'remove_survivor', cell: [0, 3] }); // the last survivor stays
  assert.equal(mapStats(t).survivors, 1);
  t = applyOpLocal(t, { op: 'clear' });
  assert.equal(t, 'S..P\n....\n....\n');
  assert.equal(applyOpLocal(t, { op: 'paint', cell: [9, 9], value: '#' }), t);
});

test('resize crops and pads, keeping a start and a survivor', () => {
  const small = resizeLocal(DECK, 2, 2);
  assert.deepEqual(asciiSize(small), { rows: 2, cols: 2 });
  assert.equal(mapStats(small).starts, 1);
  assert.equal(mapStats(small).survivors, 1);
  const big = resizeLocal(DECK, 5, 6);
  assert.equal(big.split('\n')[0], 'S.....');
});
