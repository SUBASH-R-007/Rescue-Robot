// The 30-second tour: a spotlight with a speech bubble, stepping through the Simulate view.
import { clamp } from './format.js';

/** Gap between the highlighted element and the bubble, CSS px. */
const BUBBLE_GAP = 14;
/** Padding of the spotlight around its target, CSS px. */
const SPOT_PAD = 6;

/** Tour steps: target selector, title, text, and the side panel to show first (if any). */
export const TOUR_STEPS = [
  {
    target: '#sim-canvas-box',
    title: 'The disaster zone',
    text: 'The robot (the rounded box with an antenna) must reach the survivor (the person in a circle). Cells marked “?” are still unknown; hatched cells are debris. Hover or tap any cell to see what the robot believes is there and what its plan does there; the Key button lists every symbol.',
  },
  {
    target: '#sim-controls',
    title: 'Drive the mission',
    text: 'Play runs the robot step by step, Step moves once, Back rewinds. Reset replays the same mission (same seed, same luck); New map rolls a different disaster zone.',
  },
  {
    target: '#sim-timeline',
    title: 'Scrub through time',
    text: 'Every step is recorded. Drag along the timeline under the buttons to replay any moment. Marks show slips (●), skids (▲), replans (◆) and the rescue (✓).',
  },
  {
    target: '#panel-what',
    panel: 'what',
    title: 'What just happened?',
    text: 'After each step this box explains in plain English what the robot tried, what really happened (did the wheels slip?), and how the planner reacted.',
  },
  {
    target: '.side',
    panel: 'plan',
    title: 'The plan',
    text: 'The AND-OR planner prepares a branch for every outcome: arrived, slipped, or skidded sideways. The highlighted row is where the robot is now; hover a row to find its cell on the map.',
  },
  {
    target: '#st-settings',
    panel: 'settings',
    title: 'Try your own experiment',
    text: 'Pick another planner, make the wheels slip more, add aftershocks or noisy sensors. Every change restarts the mission. Press ? any time for help and keyboard shortcuts.',
  },
];

/**
 * Where to put a bubble of size (bw, bh) next to a target rect in a viewport (vw, vh):
 * below if it fits, else above, else to the side with more room; always inside the viewport.
 * Returns {left, top, side}.
 */
export function placeBubble(rect, bw, bh, vw, vh, gap = BUBBLE_GAP) {
  const margin = 12;
  const cx = rect.left + rect.width / 2;
  let side;
  let left;
  let top;
  if (rect.bottom + gap + bh <= vh - margin) {
    side = 'below';
    top = rect.bottom + gap;
    left = cx - bw / 2;
  } else if (rect.top - gap - bh >= margin) {
    side = 'above';
    top = rect.top - gap - bh;
    left = cx - bw / 2;
  } else if (vw - rect.right >= rect.left) {
    side = 'right';
    left = rect.right + gap;
    top = rect.top + rect.height / 2 - bh / 2;
  } else {
    side = 'left';
    left = rect.left - gap - bw;
    top = rect.top + rect.height / 2 - bh / 2;
  }
  return {
    left: clamp(left, margin, Math.max(margin, vw - bw - margin)),
    top: clamp(top, margin, Math.max(margin, vh - bh - margin)),
    side,
  };
}

/** Run the tour over the Simulate view. Resolves when it is closed. */
export function runTour(app) {
  const layer = document.getElementById('tour-layer');
  const spot = document.getElementById('tour-spot');
  const bubble = document.getElementById('tour-bubble');
  const $ = (id) => document.getElementById(id);
  const sim = app.views.simulate;
  const panelBefore = sim?.panel ?? 'what';
  const opener = document.activeElement;
  let index = 0;

  return new Promise((resolve) => {
    const place = () => {
      const step = TOUR_STEPS[index];
      const el = document.querySelector(step.target);
      if (!el) return;
      el.scrollIntoView({ block: 'nearest', inline: 'nearest' });
      const r = el.getBoundingClientRect();
      const vw = window.innerWidth;
      const vh = window.innerHeight;
      // Clip the spotlight to the viewport (a tall element on a phone).
      const top = Math.max(4, r.top - SPOT_PAD);
      const bottom = Math.min(vh - 4, r.bottom + SPOT_PAD);
      const rect = { left: Math.max(4, r.left - SPOT_PAD), top, right: Math.min(vw - 4, r.right + SPOT_PAD), bottom };
      rect.width = rect.right - rect.left;
      rect.height = Math.max(0, rect.bottom - rect.top);
      Object.assign(spot.style, { left: `${rect.left}px`, top: `${rect.top}px`, width: `${rect.width}px`, height: `${rect.height}px` });
      const pos = placeBubble(rect, bubble.offsetWidth, bubble.offsetHeight, vw, vh);
      // A bubble that would cover a tall target on a small screen goes to the bottom edge.
      bubble.style.left = `${pos.left}px`;
      bubble.style.top = `${pos.top}px`;
    };

    const show = () => {
      const step = TOUR_STEPS[index];
      if (step.panel && sim) sim.showPanel(step.panel);
      $('tour-count').textContent = `Step ${index + 1} of ${TOUR_STEPS.length}`;
      $('tour-title').textContent = step.title;
      $('tour-text').textContent = step.text;
      $('tour-prev').disabled = index === 0;
      $('tour-next').textContent = index === TOUR_STEPS.length - 1 ? 'Finish' : 'Next';
      requestAnimationFrame(place);
      $('tour-next').focus();
    };

    const close = () => {
      layer.hidden = true;
      window.removeEventListener('resize', place);
      window.removeEventListener('scroll', place, true);
      document.removeEventListener('keydown', onKey, true);
      $('tour-next').onclick = null;
      $('tour-prev').onclick = null;
      $('tour-skip').onclick = null;
      if (sim) sim.showPanel(panelBefore === 'settings' ? 'what' : panelBefore);
      if (opener instanceof HTMLElement && document.contains(opener)) opener.focus();
      resolve();
    };

    const go = (d) => {
      const next = index + d;
      if (next >= TOUR_STEPS.length) close();
      else if (next >= 0) {
        index = next;
        show();
      }
    };

    function onKey(ev) {
      if (ev.key === 'Escape') { ev.preventDefault(); close(); }
      else if (ev.key === 'ArrowRight') { ev.preventDefault(); go(1); }
      else if (ev.key === 'ArrowLeft') { ev.preventDefault(); go(-1); }
      else if (ev.key === 'Tab') {
        // Keep focus inside the bubble.
        const items = [...bubble.querySelectorAll('button:not(:disabled)')];
        const i = items.indexOf(document.activeElement);
        ev.preventDefault();
        items[(i + (ev.shiftKey ? -1 : 1) + items.length) % items.length]?.focus();
      }
    }

    $('tour-next').onclick = () => go(1);
    $('tour-prev').onclick = () => go(-1);
    $('tour-skip').onclick = close;
    layer.hidden = false;
    window.addEventListener('resize', place);
    window.addEventListener('scroll', place, true);
    document.addEventListener('keydown', onKey, true);
    show();
  });
}
