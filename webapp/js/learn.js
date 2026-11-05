// Learn view: a short illustrated explanation (static HTML + inline SVG) of the problem, the
// slip/skid model, AND-OR contingency plans, how the agents react, and the extensions.
import { escapeHtml, fmtPctAuto, plainMd, outcomeSplit } from './format.js';
import { icon } from './ui.js';
import { openLightbox } from './results.js';


const LESSONS = [
  ['problem', 'The rescue problem'],
  ['slips', 'Slips and skids'],
  ['andor', 'AND-OR contingency plans'],
  ['agents', 'How the agents react'],
  ['extensions', 'The extensions'],
  ['figures', 'Figures from the report'],
];

/** The outcome diagram for a move east from a cell, at slip p. */
function slipDiagram(p, fractions) {
  const s = outcomeSplit(p, fractions);
  const pct = fmtPctAuto;
  const cell = 64;
  const gap = 10;
  const ox = 190; // robot cell
  const oy = 84;
  const box = (x, y, label) => `<rect x="${x}" y="${y}" width="${cell}" height="${cell}" rx="8" fill="var(--map-floor)" stroke="var(--border-strong)" stroke-width="1.5"/>${label ? `<text class="svg-muted" x="${x + cell / 2}" y="${y + cell - 7}" text-anchor="middle" font-size="10">${label}</text>` : ''}`;
  const cx = ox + cell / 2;
  const cy = oy + cell / 2 - 6;
  const tx = ox + cell * 2 + gap * 2 + 8; // text column right of the target
  return `<svg class="diagram slip-diagram" viewBox="0 0 600 ${oy * 2 + cell}" role="img" aria-label="A move east: arrives with ${pct(s.arrive)}, stays with ${pct(s.stay)}, skids left or right with ${pct(s.skidL)} each">
    <defs>
      <marker id="lm-ok" viewBox="0 0 10 10" refX="8" refY="5" markerWidth="5" markerHeight="5" orient="auto-start-reverse"><path d="M0 0 10 5 0 10z" fill="var(--c-ok)"/></marker>
      <marker id="lm-skid" viewBox="0 0 10 10" refX="8" refY="5" markerWidth="5" markerHeight="5" orient="auto-start-reverse"><path d="M0 0 10 5 0 10z" fill="var(--c-skid)"/></marker>
    </defs>
    ${box(ox, oy, 'robot')}
    ${box(ox + cell + gap, oy, 'east')}
    ${box(ox, oy - cell - gap, '')}
    ${box(ox, oy + cell + gap, '')}
    <rect x="${cx - 15}" y="${cy - 15}" width="30" height="30" rx="8" fill="var(--map-robot)"/>
    <circle cx="${cx + 7}" cy="${cy - 5}" r="3" fill="var(--map-robot-ink)"/><circle cx="${cx + 7}" cy="${cy + 5}" r="3" fill="var(--map-robot-ink)"/>
    <line x1="${cx + 18}" y1="${cy}" x2="${ox + cell * 2 + gap - 12}" y2="${cy}" stroke="var(--c-ok)" stroke-width="4" marker-end="url(#lm-ok)"/>
    <line x1="${cx}" y1="${cy - 18}" x2="${cx}" y2="${oy - gap - 18}" stroke="var(--c-skid)" stroke-width="3.5" stroke-dasharray="5 4" marker-end="url(#lm-skid)"/>
    <line x1="${cx}" y1="${cy + 18}" x2="${cx}" y2="${oy + cell + gap + 22}" stroke="var(--c-skid)" stroke-width="3.5" stroke-dasharray="5 4" marker-end="url(#lm-skid)"/>
    <text class="svg-text" x="${tx}" y="${cy + 2}" font-size="15" font-weight="700">→ arrives ${pct(s.arrive)}</text>
    <text class="svg-muted" x="${tx}" y="${cy + 20}" font-size="12">1 − p</text>
    <text x="${ox - 20}" y="${cy + 2}" text-anchor="end" font-size="15" font-weight="700" fill="var(--c-slip)" font-family="var(--font-ui)">↺ stays ${pct(s.stay)}</text>
    <text class="svg-muted" x="${ox - 20}" y="${cy + 20}" text-anchor="end" font-size="12">${fractions.stay} · p (a slip)</text>
    <text class="svg-text" x="${ox + cell + gap}" y="${oy - gap - cell / 2 + 5}" font-size="14" font-weight="700">▲ skids left ${pct(s.skidL)}</text>
    <text class="svg-text" x="${ox + cell + gap}" y="${oy + cell + gap + cell / 2 + 5}" font-size="14" font-weight="700">▲ skids right ${pct(s.skidR)}</text>
    <text class="svg-muted" x="${ox + cell + gap}" y="${oy + cell + gap + cell / 2 + 23}" font-size="12">skids: 0.25 · p, half left, half right</text>
  </svg>`;
}

/** The deck's root plan: E → {arrived: S-plan, slipped: retry E, skid→(2,1): E-plan}. */
function planDiagram() {
  const node = (x, y, w, title, sub, tone) => `<g>
    <rect x="${x}" y="${y}" width="${w}" height="54" rx="12" fill="var(--surface)" stroke="${tone}" stroke-width="2.5"/>
    <text class="svg-text" x="${x + w / 2}" y="${y + 23}" text-anchor="middle" font-size="15" font-weight="700">${title}</text>
    <text class="svg-muted" x="${x + w / 2}" y="${y + 42}" text-anchor="middle" font-size="12">${sub}</text></g>`;
  const edge = (x1, y1, x2, y2, label, color, lx, ly) => `<path d="M${x1} ${y1} C ${x1} ${(y1 + y2) / 2}, ${x2} ${(y1 + y2) / 2}, ${x2} ${y2}" fill="none" stroke="${color}" stroke-width="2.5"/>
    <rect x="${lx - 4}" y="${ly - 15}" width="${label.length * 7.4 + 8}" height="21" rx="10" fill="var(--surface-2)"/>
    <text x="${lx}" y="${ly}" font-size="13" font-weight="700" fill="${color}" font-family="var(--font-ui)">${label}</text>`;
  return `<svg class="diagram" viewBox="0 0 640 250" role="img" aria-label="Plan tree: at (1,1) move east. If it arrives, follow the south plan from (1,2). If it slips, retry east. If it skids to (2,1), follow the east plan from there.">
    ${node(220, 14, 200, 'At (1,1): move E →', 'OR node: the robot chooses', 'var(--accent)')}
    <circle cx="320" cy="96" r="9" fill="var(--purple)"/>
    <text class="svg-muted" x="336" y="100" font-size="12">AND node: nature chooses the outcome</text>
    <line x1="320" y1="68" x2="320" y2="87" stroke="var(--border-strong)" stroke-width="2.5"/>
    ${edge(320, 105, 95, 178, 'arrived', 'var(--c-ok)', 112, 148)}
    ${edge(320, 105, 320, 178, 'slipped', 'var(--c-slip)', 330, 148)}
    ${edge(320, 105, 545, 178, 'skid→(2,1)', 'var(--c-skid)', 452, 148)}
    ${node(10, 178, 170, 'S-plan from (1,2)', 'keep going', 'var(--c-ok)')}
    ${node(235, 178, 170, '↺ retry E at (1,1)', 'loop back: same decision', 'var(--c-slip)')}
    ${node(460, 178, 170, 'E-plan from (2,1)', 'a new route', 'var(--c-skid)')}
  </svg>`;
}

export class LearnView {
  constructor(app, root) {
    this.app = app;
    this.root = root;
    const p = app.catalog?.params?.slip_p?.default ?? 0.2;
    const high = app.catalog?.params?.terrain_high_p?.value;
    const fractions = app.catalog?.params?.outcome_fractions;
    const s = outcomeSplit(p, fractions);
    const pct = fmtPctAuto;
    root.innerHTML = `
      <div class="learn">
        <nav class="learn-toc card" aria-label="Lessons">${LESSONS.map(([id, title], i) => `<a href="#learn-${id}">${i + 1}. ${escapeHtml(title)}</a>`).join('')}</nav>
        <div class="learn-body">
          <section class="lesson card" id="learn-problem">
            <h2><span class="num">1</span>The rescue problem</h2>
            <p>After an earthquake a small robot must drive through a damaged building to a survivor. The building is a grid of cells.
            The robot knows the floor plan only roughly: cells it has not seen yet are <b>unknown</b> (“?” on the map), and the planner
            optimistically assumes they are open. Its sensor sees every cell within a short distance (radius 1 = the four neighbours and
            its own cell) after every move, so it discovers debris as it goes.</p>
            <p>Two things make this hard: the map can be wrong (hidden debris), and the robot's own moves are unreliable. A good plan has
            to cope with both. Coordinates on this site are written (row, column), counting from 1.</p>
          </section>

          <section class="lesson card" id="learn-slips">
            <h2><span class="num">2</span>Slips and skids</h2>
            <p>Every move is <b>nondeterministic</b>: the robot asks to move one cell, but the wheels decide what happens. With slip
            parameter p = ${p} a move east has four possible outcomes:</p>
            <figure>${slipDiagram(p, fractions)}<figcaption>The four outcomes of one move at p = ${p}: ${pct(s.arrive)} / ${pct(s.stay)} / ${pct(s.skidL)} / ${pct(s.skidR)}.</figcaption></figure>
            <ul>
              <li><b>Arrives</b> (probability 1 − p): the robot reaches the cell it wanted.</li>
              <li><b>Slips</b> (${fractions.stay} · p): the wheels spin and it stays where it is.</li>
              <li><b>Skids</b> (${fractions.skid} · p): it slides sideways, to the left or the right of the direction it wanted, half the time each.</li>
              <li>If an outcome would end in a wall or in known debris, the robot simply stays put (a “blocked” move).</li>
              <li>Water and mud cells (wavy cells, “~”) are extra slippery: there the slip parameter is ${high}, whatever p is.</li>
            </ul>
          </section>

          <section class="lesson card" id="learn-andor">
            <h2><span class="num">3</span>AND-OR contingency plans</h2>
            <p>An ordinary path planner such as A* returns one sequence of moves and hopes each move works. An <b>AND-OR search</b>
            (Russell &amp; Norvig, AIMA §4.3) instead returns a <b>contingency plan</b>: a tree that says what to do after <i>every</i>
            possible outcome.</p>
            <ul>
              <li><b>OR nodes</b> are the robot's choices: which direction to move. The planner needs <i>one</i> action that works.</li>
              <li><b>AND nodes</b> are nature's choices: arrived, slipped or skidded. The plan must work for <i>all</i> of them.</li>
              <li>A branch may loop back (“retry”): after a slip the robot is in the same cell and simply tries again. Because every
              loop can still reach the survivor, the plan is a <b>cyclic</b> solution: the robot gets there with probability 1.</li>
            </ul>
            <figure>${planDiagram()}<figcaption>The root of the plan on the slide 9 map (maps/deck_demo.txt): move east; if it arrives, follow the
            south plan; if it slips, retry east; if it skids to (2,1), follow the east plan from there. (A skid north would hit the wall, so it
            merges with the slip.)</figcaption></figure>
            <p>When the sensor shows debris where the plan assumed open floor, the plan may be wrong, so the AND-OR agent plans again from where
            it stands. <b>Eager</b> replanning does this for any surprise; <b>lazy</b> replanning only when the debris touches the current plan.</p>
            <p class="try"><button type="button" class="btn primary" data-preset="slide9">${icon('play')}Watch it on the slide 9 map</button></p>
          </section>

          <section class="lesson card" id="learn-agents">
            <h2><span class="num">4</span>How the agents react</h2>
            <p>The same slip hits every agent at the same step (they share the random numbers), so the Compare tab shows only the difference in strategy.</p>
            <table>
              <thead><tr><th scope="col">Agent</th><th scope="col">What it plans</th><th scope="col">After a slip or skid</th><th scope="col">After surprise debris</th></tr></thead>
              <tbody>
                <tr><th scope="row">AND-OR contingency</th><td>A tree with a branch for every outcome</td><td>Follows the branch it already has: no replanning</td><td>Replans (eager or lazy)</td></tr>
                <tr><th scope="row">Fixed A*</th><td>One shortest path, once</td><td>Gives up (the mission is aborted)</td><td>Gives up if the debris is on its path</td></tr>
                <tr><th scope="row">A* + replan</th><td>One shortest path</td><td>Runs A* again from wherever it ended up</td><td>Runs A* again</td></tr>
                <tr><th scope="row">D* Lite</th><td>One shortest path, repaired incrementally</td><td>Repairs its previous search</td><td>Repairs its previous search</td></tr>
                <tr><th scope="row">Value iteration / LAO*</th><td>A policy: the action with the fewest expected steps in every cell</td><td>The policy already covers the new cell</td><td>Recomputes the policy</td></tr>
              </tbody>
            </table>
            <p class="try"><button type="button" class="btn" data-preset="head_to_head">${icon('columns')}Compare them side by side</button></p>
          </section>

          <section class="lesson card" id="learn-extensions">
            <h2><span class="num">5</span>The extensions</h2>
            <ul>
              <li><b>Hidden survivor</b>: nobody knows where the survivor is. The robot drives to the nearest <i>frontier</i> (a known open cell next
              to unknown ones) until its sensor spots the survivor, then plans a rescue.</li>
              <li><b>Several survivors</b>: k survivors with known positions; the robot always heads for the nearest one not yet rescued.</li>
              <li><b>Noisy sensors</b>: the sensor sometimes misses debris (false negatives) or reports debris that is not there (false positives).
              The robot keeps a probability of debris for every cell (red shading on the map) and updates it with Bayes' rule after each reading.
              Now the robot <i>can</i> bump into debris it was told was clear.</li>
              <li><b>Aftershocks</b> (a stress test in every mode): new debris falls where the robot cannot see, but never cuts the survivor off.</li>
            </ul>
            <p class="try"><button type="button" class="btn" data-preset="hidden_survivor">${icon('search')}Try the hidden survivor</button>
            <button type="button" class="btn" data-preset="noisy_sensors">${icon('signal')}Try noisy sensors</button></p>
          </section>

          <section class="lesson card" id="learn-figures">
            <h2><span class="num">6</span>Figures from the report</h2>
            <p class="muted">Pictures produced by the desktop simulator (the same Python core). Click one to enlarge it.</p>
            <div class="plot-grid" id="learn-figs" style="margin-top:12px"><p class="muted">Loading…</p></div>
          </section>
        </div>
      </div>`;
    root.addEventListener('click', (ev) => {
      const b = ev.target.closest('[data-preset]');
      if (b) {
        const preset = (app.catalog.presets ?? []).find((x) => x.id === b.dataset.preset);
        if (preset) app.applyPreset(preset);
        return;
      }
      const f = ev.target.closest('[data-fig]');
      const fig = f && this.figures?.[Number(f.dataset.fig)];
      if (fig?.src) openLightbox(fig);
    });
    root.querySelector('.learn-toc').addEventListener('click', (ev) => {
      const a = ev.target.closest('a[href^="#learn-"]');
      if (!a) return;
      ev.preventDefault();
      const target = root.querySelector(a.getAttribute('href'));
      const reduce = matchMedia('(prefers-reduced-motion: reduce)').matches;
      target?.scrollIntoView({ behavior: reduce ? 'auto' : 'smooth', block: 'start' });
    });
    this._figures();
  }

  async _figures() {
    const box = this.root.querySelector('#learn-figs');
    try {
      const data = await this.app.backend.results();
      this.figures = data.figures ?? [];
      box.innerHTML = this.figures.map((f, i) => `<button type="button" class="plot" data-fig="${i}" aria-label="Enlarge figure: ${escapeHtml(plainMd(f.title))}">
        ${f.src ? `<img src="${escapeHtml(f.src)}" alt="" loading="lazy">` : '<span class="muted">(image in the built site)</span>'}<span>${escapeHtml(plainMd(f.title))}</span></button>`).join('') || '<p class="muted">No figures in this build.</p>';
    } catch {
      box.innerHTML = '<p class="muted">Figures are added by the site build (python -m rescue.webapp).</p>';
    }
  }
}
