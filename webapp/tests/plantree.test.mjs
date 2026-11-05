// node --test webapp/tests : plan-tree flattening, folding and the current node.
import { test } from 'node:test';
import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import {
  flattenTree, findCurrentNode, nodeById, ancestorIds, countNodes, defaultCollapsed, branchesAt,
  edgeSentence, edgeTone,
} from '../js/plantree.js';

/** The deck's root plan (recorded from the Python core, maps/deck_demo.txt). */
const fixture = JSON.parse(readFileSync(new URL('./fixtures/missions.json', import.meta.url), 'utf8'));
const deck = fixture.missions.find((m) => m.config.map?.name === 'deck_demo' && m.config.agent === 'andor');
const tree = deck.plans['0'].tree;

test('the recorded root plan is the deck’s: E → {arrived: S-plan, slipped: retry E, skid→(2,1): E-plan}', () => {
  assert.equal(deck.plans['0'].pretty, 'E → {arrived: S-plan, slipped: retry E, skid→(2,1): E-plan}');
  assert.equal(tree.kind, 'act');
  assert.equal(tree.action, 'E');
  assert.deepEqual(tree.children.map((c) => c.edge), ['arrived', 'slipped', 'skid→(2,1)']);
});

test('flattening without folds lists every node in pre-order with depths', () => {
  const rows = flattenTree(tree);
  assert.equal(rows.length, countNodes(tree));
  assert.equal(rows[0].id, '0');
  assert.equal(rows[0].depth, 0);
  assert.equal(rows[1].id, '0.0');
  assert.equal(rows[1].depth, 1);
  for (const row of rows) assert.equal(nodeById(tree, row.id).text, row.text);
});

test('folded nodes hide their subtree and report how many nodes are hidden', () => {
  const rows = flattenTree(tree, new Set(['0.0']));
  const folded = rows.find((r) => r.id === '0.0');
  assert.ok(folded.collapsed);
  assert.equal(folded.descendants, countNodes(nodeById(tree, '0.0')) - 1);
  assert.ok(!rows.some((r) => r.id.startsWith('0.0.')));
});

test('the current node is the shallowest decision at the robot’s cell, and stays unfolded', () => {
  assert.equal(findCurrentNode(tree, [0, 0]), '0');
  const id = findCurrentNode(tree, [0, 1]);
  assert.equal(id, '0.0');
  assert.deepEqual(nodeById(tree, id).state, [0, 1]);
  assert.equal(findCurrentNode(tree, [9, 9]), null);
  assert.deepEqual(ancestorIds('0.1.2'), ['0', '0.1']);
  const deep = flattenTree(tree).find((r) => r.depth >= 3 && r.kind === 'act');
  const collapsed = defaultCollapsed(tree, deep.id);
  for (const a of ancestorIds(deep.id)) assert.ok(!collapsed.has(a));
});

test('outcome branches and edge wording', () => {
  const branches = branchesAt(tree, [0, 0]);
  assert.deepEqual(branches.map((b) => b.edge), ['arrived', 'slipped', 'skid→(2,1)']);
  assert.equal(edgeSentence('skid→(2,1)'), 'if it skids to (2,1)');
  assert.equal(edgeSentence('slipped'), 'if it slips');
  assert.equal(edgeTone('skid-L'), 'skid');
  assert.equal(edgeTone('arrived'), 'ok');
});
