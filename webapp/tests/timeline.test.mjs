// node --test webapp/tests : timeline mapping (step <-> x) and tick marks.
import { test } from 'node:test';
import assert from 'node:assert/strict';
import { trackSpan, tToX, xToT, buildTicks, stackTicks, logTick, MIN_SPAN, TRACK_PAD } from '../js/timeline.js';

test('the track leaves room to grow while running and fits exactly when done', () => {
  assert.equal(trackSpan(3, false), MIN_SPAN);
  assert.equal(trackSpan(50, false), 50);
  assert.equal(trackSpan(9, true), 9);
  assert.equal(trackSpan(0, true), 1);
});

test('tToX and xToT are inverse on recorded steps and clamp to [0, maxT]', () => {
  const width = 600;
  const span = trackSpan(9, true);
  for (let t = 0; t <= 9; t += 1) assert.equal(xToT(tToX(t, span, width), span, width, 9), t);
  assert.equal(tToX(0, span, width), TRACK_PAD);
  assert.equal(tToX(9, span, width), width - TRACK_PAD);
  assert.equal(xToT(-50, span, width, 9), 0);
  assert.equal(xToT(5000, span, width, 9), 9);
  // While running, positions past the newest step cannot be scrubbed to.
  assert.equal(xToT(width, trackSpan(4, false), width, 4), 4);
});

test('ticks: slips, skids and bumps from the trace; replans and rescues from the log', () => {
  const trace = [
    { t: 1, outcome: 'arrived' },
    { t: 2, outcome: 'slipped' },
    { t: 3, outcome: 'skid-L', bump: true },
    { t: 4, outcome: 'arrived' },
  ];
  const log = [
    { t: 0, kind: 'info', text: 'andor on deck_demo' },
    { t: 3, kind: 'replan', tag: 'replan', text: 'Debris sensed at (3,2) → replanning' },
    { t: 4, kind: 'success', tag: 'rescue', text: 'Rescue successful' },
    // The tag decides, never the wording (Python tags the lines it writes).
    { t: 5, kind: 'success', tag: 'found', text: 'Something else entirely' },
  ];
  const ticks = buildTicks(trace, log);
  assert.deepEqual(ticks, [
    { t: 2, kind: 'slip' }, { t: 3, kind: 'skid' }, { t: 3, kind: 'bump' }, { t: 3, kind: 'replan' },
    { t: 4, kind: 'rescue' }, { t: 5, kind: 'found' },
  ]);
  assert.deepEqual(stackTicks(ticks)[1], { t: 3, kinds: ['skid', 'bump', 'replan'] });
  assert.equal(logTick({ kind: 'failure', tag: 'fail' }), 'fail');
  assert.equal(logTick({ kind: 'info', text: 'Survivor found at (1,1)' }), null);
  assert.equal(logTick({ kind: 'success', tag: 'bogus' }), null);
});
