// node --test webapp/tests : pure helpers of the views (compare summary, results, tour, learn).
import { test } from 'node:test';
import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import { summarize, bestIndex, joinNames, defaultAgents } from '../js/compare.js';
import { honestFindings, whyParagraph, MEANING } from '../js/results.js';
import { placeBubble, TOUR_STEPS } from '../js/tour.js';
import { outcomeSplit } from '../js/format.js';
import { metricRows, metricValue, withSize, modePatch } from '../js/panels.js';

const fixture = JSON.parse(readFileSync(new URL('./fixtures/missions.json', import.meta.url), 'utf8'));
const results = JSON.parse(readFileSync(new URL('./fixtures/results.json', import.meta.url), 'utf8'));

test('compare summary sentences', () => {
  assert.equal(summarize([
    { label: 'AND-OR', status: 'running', t: 4 },
    { label: 'A* (no replan)', status: 'aborted', t: 4 },
    { label: 'A* + replan', status: 'running', t: 4 },
  ]), 'A* (no replan) gave up at step 4; AND-OR and A* + replan are both still running.');
  assert.equal(summarize([
    { label: 'AND-OR', status: 'rescued', t: 9 },
    { label: 'Fixed A*', status: 'aborted', t: 2 },
    { label: 'A* + replan', status: 'rescued', t: 9 },
  ]), 'AND-OR and A* + replan each rescued the survivor in 9 steps; Fixed A* gave up at step 2.');
  assert.equal(summarize([
    { label: 'X', status: 'rescued', t: 7 },
    { label: 'Y', status: 'rescued', t: 11 },
  ]), 'X rescued the survivor in 7 steps, Y rescued the survivor in 11 steps. X was the fastest.');
  assert.match(summarize([{ label: 'X', status: 'running', t: 0 }, { label: 'Y', status: 'running', t: 0 }]), /press Play/);
  assert.equal(joinNames(['a', 'b', 'c']), 'a, b and c');
});

test('best value: unique minimum (or maximum), ignoring missing values', () => {
  assert.equal(bestIndex([9, null, 7], true), 2);
  assert.equal(bestIndex([3, 3, 5], true), -1);
  assert.equal(bestIndex([1, 2], false), 1);
  assert.equal(bestIndex([4, null, null], true), -1);
  assert.equal(bestIndex([1, 2], null), -1);
});

test('default compare agents come from the catalog', () => {
  assert.deepEqual(defaultAgents(fixture.catalog, 'core'), ['andor', 'astar_fixed', 'astar_replan']);
  assert.deepEqual(defaultAgents(fixture.catalog, 'search'), ['search_andor', 'search_astar']);
  assert.deepEqual(defaultAgents(fixture.catalog, 'core', ['andor', 'vi_expected']), ['andor', 'vi_expected']);
});

test('honest findings list every claim that is not supported, with the report’s own text', () => {
  const items = honestFindings(results.reports);
  const flagged = results.reports.flatMap((r) => r.claims).filter((c) => c.verdict === 'refuted' || c.verdict === 'inconclusive');
  for (const c of flagged) assert.ok(items.some((i) => i.id === c.id && !i.note), c.id);
  for (const item of items) if (item.text) assert.ok(results.reports.some((r) => r.claims.some((c) => c.details.replace(/\s+/g, ' ').includes(item.text.slice(0, 60)))));
  assert.equal(whyParagraph('Intro.\n\nWhy: because.\n\n| a |\n|---|'), 'Why: because.');
  // Every core and extension claim has a plain-language question.
  for (const r of results.reports.filter((x) => !x.descriptive)) for (const c of r.claims) assert.ok(MEANING[c.id], c.id);
});

test('tour bubbles stay inside the viewport', () => {
  assert.equal(TOUR_STEPS.length, 6);
  const vw = 390;
  const vh = 844;
  for (const rect of [{ left: 0, top: 0, right: 390, bottom: 800, width: 390, height: 800 }, { left: 300, top: 700, right: 380, bottom: 740, width: 80, height: 40 }]) {
    const p = placeBubble(rect, 340, 200, vw, vh);
    assert.ok(p.left >= 0 && p.left + 340 <= vw);
    assert.ok(p.top >= 0 && p.top + 200 <= vh);
  }
  assert.equal(placeBubble({ left: 10, top: 10, right: 100, bottom: 50, width: 90, height: 40 }, 200, 100, 1280, 800).side, 'below');
});

test('slip split: 0.80 / 0.15 / 0.025 / 0.025 at p = 0.2', () => {
  // The fractions come from the catalog (config.py SLIP_STAY_FRACTION / SKID_FRACTION); the
  // front end must not hard-code them.
  const fractions = fixture.catalog.params.outcome_fractions;
  const o = outcomeSplit(0.2, fractions);
  assert.ok(Math.abs(o.arrive - 0.8) < 1e-12 && Math.abs(o.stay - 0.15) < 1e-12);
  assert.ok(Math.abs(o.skidL - 0.025) < 1e-12 && Math.abs(o.skidR - 0.025) < 1e-12);
  assert.ok(Math.abs(o.arrive + o.stay + o.skidL + o.skidR - 1) < 1e-12);
});

test('metric cards: known metrics in order, with budgets', () => {
  const state = { budget: 64, survivors: [[3, 3]] };
  const rows = metricRows({ steps: 2, replans: 0, agent: 'andor', planning_ms: 1.2, aftershock_events: 0 });
  assert.deepEqual(rows.map((r) => r.key), ['steps', 'replans', 'planning_ms']);
  assert.equal(metricValue('steps', 2, state), '2 / 64');
  assert.equal(metricValue('planning_ms', 1.2, state), '1.20 ms');
  assert.equal(withSize('Slide 9 demo (4×4)', 4, 4), 'Slide 9 demo (4×4)');
  assert.equal(withSize('Building', 15, 20), 'Building (15×20)');
});

test('a mode switch resets the mode-only settings (core takes none of them)', () => {
  const catalog = {
    agents: [
      { name: 'andor', modes: ['core'] }, { name: 'astar_replan', modes: ['core'] },
      { name: 'search_andor', modes: ['search', 'multi', 'noisy'] }, { name: 'search_astar', modes: ['search', 'multi', 'noisy'] },
    ],
    modes: [
      { name: 'core', default_agent: 'andor', defaults: { k: 1, false_negative: 0, false_positive: 0 } },
      { name: 'multi', default_agent: 'search_andor', defaults: { k: 3, false_negative: 0, false_positive: 0 } },
      { name: 'noisy', default_agent: 'search_andor', defaults: { k: 1, false_negative: 0.1, false_positive: 0 } },
    ],
    params: { k: { default: 3 } },
  };
  const multi = { mode: 'multi', agent: 'search_astar', k: 5, false_negative: 0, false_positive: 0 };
  assert.deepEqual(modePatch(catalog, multi, 'noisy'), { mode: 'noisy', agent: 'search_astar', k: 1, false_negative: 0.1, false_positive: 0 });
  const toCore = modePatch(catalog, { ...multi, mode: 'noisy', false_positive: 0.2 }, 'core');
  assert.equal(toCore.agent, 'andor');
  // JSON (the worker message) drops the undefined keys, so the core mode gets none of them.
  assert.deepEqual(JSON.parse(JSON.stringify({ ...multi, ...toCore })), { mode: 'core', agent: 'andor' });
  assert.equal(modePatch(catalog, { mode: 'core', agent: 'andor' }, 'multi').k, 3);
});
