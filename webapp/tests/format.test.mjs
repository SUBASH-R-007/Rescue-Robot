// node --test webapp/tests : pure formatting helpers (everything a person reads is 1-indexed).
import { test } from 'node:test';
import assert from 'node:assert/strict';
import {
  fmtPos, posKey, parseKey, samePos, normOutcome, outcomeText, statusInfo, fmtNum, fmtPct,
  fmtPctAuto, fmtMs, plural, escapeHtml, verdictInfo, keyNumber, plainMd, humanize, clamp,
  renderMarkdown, mdParagraphs,
} from '../js/format.js';

test('positions are shown 1-indexed', () => {
  assert.equal(fmtPos([0, 0]), '(1,1)');
  assert.equal(fmtPos([2, 9]), '(3,10)');
  assert.equal(fmtPos(null), '–');
  assert.equal(posKey([2, 3]), '2,3');
  assert.deepEqual(parseKey('2,3'), [2, 3]);
  assert.ok(samePos([1, 2], [1, 2]));
  assert.ok(!samePos([1, 2], [2, 1]));
  assert.ok(!samePos(null, [0, 0]));
});

test('outcome labels from the bridge are normalised', () => {
  assert.equal(normOutcome('arrived'), 'arrived');
  assert.equal(normOutcome('SKID_L'), 'skid-L');
  assert.equal(normOutcome('skid-R'), 'skid-R');
  assert.equal(normOutcome(null), null);
  assert.equal(outcomeText('slipped'), 'slipped (stayed put)');
  assert.equal(outcomeText('skid-L', true), 'skidded left, blocked');
  assert.equal(outcomeText('slipped', true), 'slipped (stayed put)');
});

test('every status has a label, a tone and a glyph (never colour alone)', () => {
  for (const s of ['running', 'rescued', 'aborted', 'budget', 'unreachable']) {
    const info = statusInfo(s);
    assert.ok(info.label && info.tone && info.glyph, s);
  }
  assert.equal(statusInfo('aborted').label, 'Gave up');
});

test('numbers', () => {
  assert.equal(fmtNum(3.14159, 2), '3.14');
  assert.equal(fmtNum(null), '–');
  assert.equal(fmtNum(Infinity), '∞');
  assert.equal(fmtPct(0.2), '20 %');
  assert.equal(fmtMs(0.421), '0.42 ms');
  assert.equal(fmtMs(12.3), '12 ms');
  assert.equal(fmtMs(1534), '1.5 s');
  assert.equal(plural(1, 'step'), '1 step');
  assert.equal(plural(2, 'step'), '2 steps');
  assert.equal(clamp(5, 0, 3), 3);
});

test('fmtPctAuto uses only the decimals it needs (slip split at p = 0.2)', () => {
  assert.equal(fmtPctAuto(1 - 0.2), '80 %');
  assert.equal(fmtPctAuto(0.75 * 0.2), '15 %');
  assert.equal(fmtPctAuto(0.125 * 0.2), '2.5 %');
  assert.equal(fmtPctAuto(0.125 * 0.15), '1.88 %');
  assert.equal(fmtPctAuto(undefined), '–');
});

test('escapeHtml escapes markup', () => {
  assert.equal(escapeHtml('<a href="x">&\'</a>'), '&lt;a href=&quot;x&quot;&gt;&amp;&#39;&lt;/a&gt;');
});

test('verdicts map to label, tone and glyph', () => {
  assert.equal(verdictInfo('supported').tone, 'good');
  assert.equal(verdictInfo('✅').label, 'Supported');
  assert.equal(verdictInfo('refuted').glyph, '✕');
  assert.equal(verdictInfo('inconclusive').tone, 'warn');
  assert.equal(verdictInfo('descriptive').label, 'Measurement');
});

test('keyNumber keeps the headline of a Measured cell', () => {
  assert.equal(keyNumber('100.0% [99.8%, 100.0%] (2400/2400); Wilson lower bound 99.8%'), '100.0% [99.8%, 100.0%] (2400/2400)');
  assert.equal(keyNumber('-3.09 moves [-3.81, -2.37] over 2400 maps'), '-3.09 moves [-3.81, -2.37]');
  assert.equal(plainMd('`andor` is **bold**'), 'andor is bold');
  assert.equal(humanize('collapsed_building'), 'Collapsed building');
});

test('renderMarkdown renders tables, lists and inline code, escaping everything', () => {
  const html = renderMarkdown('Hello `andor` <b>\n\n| a | b |\n|---|---|\n| 1 | 2 |\n\n- one\n- two');
  assert.match(html, /<p>Hello <code>andor<\/code> &lt;b&gt;<\/p>/);
  assert.match(html, /<th>a<\/th><th>b<\/th>/);
  assert.match(html, /<td>1<\/td><td>2<\/td>/);
  assert.match(html, /<ul><li>one<\/li><li>two<\/li><\/ul>/);
  assert.deepEqual(mdParagraphs('Why: one\ntwo\n\n| x |\n|---|\n\n- item\n\nlast'), ['Why: one two', 'last']);
});
