// node --test webapp/tests : MissionSession (live steps, scrubbing, cache) over recorded
// missions from the Python core (the mock backend's data).
import { test } from 'node:test';
import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import { MissionSession, isDone } from '../js/session.js';
import { expandMission, matchScore } from '../js/mock.js';

const fixture = JSON.parse(readFileSync(new URL('./fixtures/missions.json', import.meta.url), 'utf8'));
const deckAndor = () => fixture.missions.find((m) => m.config.map?.name === 'deck_demo' && m.config.agent === 'andor');

/** A backend over recorded states (the mock's logic without fetch / delays). */
function fakeBackend(states) {
  let t = 0;
  const calls = [];
  return {
    calls,
    async create() { calls.push('open'); t = 0; return { sid: 1, state: states[0] }; },
    async step() { calls.push('step'); t = Math.min(t + 1, states.length - 1); return { state: states[t], explanation: states[t].explanation }; },
    async run() { calls.push('run'); t = states.length - 1; return { state: states[t] }; },
    async stateAt(_sid, k) { calls.push(`state_at ${k}`); return states[Math.min(k, t)]; },
    async dispose() { calls.push('close'); },
  };
}

test('recorded missions expand into full states with growing traces and logs', () => {
  for (const rec of fixture.missions) {
    const states = expandMission(rec);
    assert.equal(states.length, rec.frames.length);
    states.forEach((s, i) => {
      assert.equal(s.t, rec.frames[i].t);
      assert.ok(Array.isArray(s.belief) && s.belief.length === s.rows);
      assert.ok(s.trace.every((e) => e.t <= s.t));
      if (i > 0) assert.ok(s.log.length >= states[i - 1].log.length);
    });
  }
});

test('deck_demo + AND-OR + seed 76 is the slide 9 mission: rescued after 9 steps', () => {
  const states = expandMission(deckAndor());
  const last = states[states.length - 1];
  assert.equal(last.status, 'rescued');
  assert.equal(last.t, 9);
  assert.ok(isDone(last));
  assert.ok(!isDone(states[0]));
});

test('stepping, scrubbing back and replaying forward without re-running Python', async () => {
  const rec = deckAndor();
  const backend = fakeBackend(expandMission(rec));
  const s = new MissionSession(backend, rec.config);
  await s.start();
  for (let i = 0; i < 4; i += 1) await s.forward();
  assert.equal(s.t, 4);
  assert.equal(s.maxT, 4);
  assert.ok(s.atLive);
  await s.seek(1);
  assert.equal(s.t, 1);
  assert.ok(!s.atLive);
  const before = backend.calls.filter((c) => c === 'step').length;
  const { ran } = await s.forward(); // replays the recorded step 2
  assert.equal(ran, false);
  assert.equal(s.t, 2);
  assert.equal(backend.calls.filter((c) => c === 'step').length, before);
  await s.seek(99); // clamped to the newest recorded step
  assert.equal(s.t, 4);
  await s.finish();
  assert.ok(s.done);
  assert.equal(s.t, 9);
  const { ran: again } = await s.forward();
  assert.equal(again, false);
});

test('the mock picks the closest recording for a config', () => {
  const pick = (cfg) => fixture.missions.reduce((a, b) => (matchScore(b.config, cfg) > matchScore(a.config, cfg) ? b : a));
  const core = pick({ mode: 'core', agent: 'astar_replan', map: { source: 'file', name: 'deck_demo' } });
  assert.equal(core.config.agent, 'astar_replan');
  assert.equal(core.config.map.name, 'deck_demo');
  assert.equal(pick({ mode: 'noisy', map: { source: 'scenario', scenario: 'random_rubble' } }).config.mode, 'noisy');
});
