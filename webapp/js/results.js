// Results view: the committed experiment results (results.json, parsed at build time from
// results/HYPOTHESES.md, results/ext/EXTENSIONS.md and results/lao/LAO.md) as claim cards, the
// plots in a lightbox, and the honest findings. Every number shown comes from results.json.
import {
  escapeHtml, keyNumber, mdParagraphs, plainMd, renderMarkdown, verdictInfo,
} from './format.js';
import { icon, showError } from './ui.js';

/** Plain-language question behind each claim (no numbers: those come from results.json). */
export const MEANING = {
  H1: 'Does the AND-OR robot almost always reach the survivor, even when its wheels slip a lot?',
  H2: 'Does the contingency plan really cover every outcome? An “unplanned state” would mean the robot landed somewhere its plan had no answer for.',
  H3: 'Does planning for slips in advance waste fewer moves than repairing an A* path after every surprise?',
  H4: 'Does a robot that never replans fail as soon as one move goes wrong?',
  H5: 'Does AND-OR need to throw its plan away less often than A* + replan?',
  H6: 'A reference check: is the expected-steps-optimal policy (value iteration) faster than AND-OR?',
  H7: 'Is the H3 advantage really due to contingency planning, or to how each planner breaks ties between equally short routes?',
  X1: 'Can the robot find and rescue a survivor whose location nobody knows?',
  X2: 'How much does not knowing where the survivor is cost, compared with knowing it?',
  X3: 'Is “nearest survivor first” almost as good as the best possible visiting order?',
  X4: 'With a sensor that sometimes misses debris, the robot can now bump into things. Does it still succeed?',
  X5: 'With a noisy sensor, does AND-OR bump into debris no more often than A*?',
  X6: 'Does the robot still succeed when the sensor sometimes reports debris that is not there?',
};

/** What a verdict means, in one sentence. */
export const VERDICT_MEANING = {
  supported: 'The measurements back this claim.',
  refuted: 'The measurements contradict this claim. It is reported as measured; nothing was tuned to change it.',
  inconclusive: 'The data cannot decide this claim either way.',
  descriptive: 'A measurement, not a claim with a verdict.',
};

/** The paragraph of a claim's details that explains it best: one starting "Why", else none. */
export function whyParagraph(details) {
  return mdParagraphs(details).find((p) => /^why\b/i.test(p)) ?? null;
}

/**
 * Honest findings: every claim that was not supported, with its explanation paragraph, plus
 * the paragraphs of other claims that refer to one of them (e.g. H3's note pointing to H7).
 */
export function honestFindings(reports) {
  const claims = reports.flatMap((r) => r.claims.map((c) => ({ ...c, report: r.id })));
  const flagged = claims.filter((c) => c.verdict === 'refuted' || c.verdict === 'inconclusive');
  const ids = new Set(flagged.map((c) => c.id));
  const items = flagged.map((c) => ({ id: c.id, claim: c.claim, verdict: c.verdict, measured: c.measured, text: whyParagraph(c.details) }));
  for (const c of claims) {
    if (ids.has(c.id)) continue;
    const refs = mdParagraphs(c.details).filter((p) => [...ids].some((id) => new RegExp(`\\b${id}\\b`).test(p)));
    for (const p of refs) items.push({ id: c.id, claim: c.claim, verdict: c.verdict, measured: null, text: p, note: true });
  }
  return items;
}

export class ResultsView {
  constructor(app, root) {
    this.app = app;
    this.root = root;
    this.data = null;
    this.root.innerHTML = '<div class="card res-hero"><p class="muted">Loading the results…</p></div>';
    this._load();
  }

  async _load() {
    try {
      this.data = await this.app.backend.results();
      this._render();
    } catch (err) {
      this.root.innerHTML = `<div class="card res-hero"><h2>Results are not available</h2><p>${escapeHtml(err.message)}</p></div>`;
      showError(err, 'Could not load the results');
    }
  }

  _render() {
    const { reports = [], plots = [] } = this.data;
    const totals = { supported: 0, refuted: 0, inconclusive: 0 };
    for (const r of reports) for (const c of r.claims) if (c.verdict in totals) totals[c.verdict] += 1;
    const badge = (verdict, n) => {
      const v = verdictInfo(verdict);
      return `<span class="badge tone-${v.tone}">${v.glyph} ${n} ${escapeHtml(v.label.toLowerCase())}</span>`;
    };
    const findings = honestFindings(reports);
    this.root.innerHTML = `
      <section class="card res-hero">
        <h2>What the experiments found</h2>
        <p>Every claim of the project proposal was tested on thousands of generated maps with the same Python code this page runs.
        Each card shows the claim, what it means in plain words, the verdict and the key number, exactly as written in the
        committed reports. Refuted claims stay on the page: the experiments were never tuned to make the proposed planner win.</p>
        <div class="res-counts">${badge('supported', totals.supported)}${badge('refuted', totals.refuted)}${totals.inconclusive ? badge('inconclusive', totals.inconclusive) : ''}</div>
      </section>
      ${findings.length ? `<section class="card findings" aria-labelledby="res-honest">
        <h3 id="res-honest">${icon('alert')} Honest findings</h3>
        <ul>${findings.map((f) => `<li><strong>${escapeHtml(f.id)}</strong> ${f.note ? '(context) ' : `<span class="badge tone-${verdictInfo(f.verdict).tone}">${verdictInfo(f.verdict).glyph} ${escapeHtml(verdictInfo(f.verdict).label)}</span> `}${escapeHtml(plainMd(f.claim))}${f.text ? `<div class="claim-text md">${renderMarkdown(f.text)}</div>` : ''}</li>`).join('')}</ul>
      </section>` : ''}
      ${reports.map((r) => this._report(r, plots.filter((p) => p.group === r.id))).join('')}`;
    this.root.addEventListener('click', (ev) => {
      const b = ev.target.closest('[data-plot]');
      if (!b) return;
      const plot = plots.find((p) => p.id === b.dataset.plot);
      if (plot?.src) openLightbox(plot);
    });
  }

  _report(r, plots) {
    const meta = r.meta ?? {};
    const facts = [meta['Date of run'] && `run ${meta['Date of run']}`, meta['Git commit'] && `commit ${plainMd(meta['Git commit'])}`].filter(Boolean).join(' · ');
    return `<section class="res-section" aria-labelledby="res-${escapeHtml(r.id)}">
      <h3 id="res-${escapeHtml(r.id)}">${escapeHtml(r.title)}</h3>
      <p class="src">From <code>${escapeHtml(r.source)}</code>${facts ? ` (${escapeHtml(facts)})` : ''}. Reproduce: <code>${escapeHtml(r.command)}</code></p>
      ${r.intro ? `<p class="claim-text" style="margin-bottom:12px">${escapeHtml(plainMd(r.intro))}</p>` : ''}
      <div class="claim-grid">${r.claims.map((c) => this._claim(c)).join('')}</div>
      ${plots.length ? `<h4 class="panel-title" style="margin-top:16px">${icon('chart')}Plots (click to enlarge)</h4>
        <div class="plot-grid">${plots.map((p) => `<button type="button" class="plot" data-plot="${escapeHtml(p.id)}" aria-label="Enlarge plot: ${escapeHtml(plainMd(p.title))}">
          ${p.src ? `<img src="${escapeHtml(p.src)}" alt="" loading="lazy">` : '<span class="muted">(image in the built site)</span>'}<span>${escapeHtml(plainMd(p.title))}</span></button>`).join('')}</div>` : ''}
    </section>`;
  }

  _claim(c) {
    const v = verdictInfo(c.verdict);
    const meaning = MEANING[c.id];
    return `<article class="claim card tone-${v.tone}">
      <div class="claim-head"><span class="claim-id">${escapeHtml(c.id)}</span>
        <span class="badge tone-${v.tone}">${v.glyph} ${escapeHtml(v.label)}</span></div>
      <p class="claim-q">${escapeHtml(meaning ?? plainMd(c.claim))}</p>
      ${meaning ? `<p class="claim-text"><b>Claim:</b> ${escapeHtml(plainMd(c.claim))}</p>` : ''}
      <p class="claim-text">${escapeHtml(VERDICT_MEANING[c.verdict] ?? '')}</p>
      <div class="claim-num" title="Key number, as written in the report">${escapeHtml(keyNumber(c.measured))}</div>
      ${c.details || c.measured !== keyNumber(c.measured) ? `<details><summary>Full measurement and method</summary>
        <div class="md"><p>${escapeHtml(plainMd(c.measured))}</p>${renderMarkdown(c.details)}</div></details>` : ''}
    </article>`;
  }
}

/** Show an image (results plot or figure) in the lightbox dialog. */
export function openLightbox(item) {
  const dlg = document.getElementById('lightbox');
  document.getElementById('lightbox-title').textContent = plainMd(item.title);
  const img = document.getElementById('lightbox-img');
  img.src = item.src;
  img.alt = plainMd(item.title);
  document.getElementById('lightbox-caption').textContent = item.caption ? plainMd(item.caption) : '';
  dlg.showModal();
}
