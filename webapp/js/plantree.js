// Plan tree: pure helpers (flattening, current node, outcome branches) + the DOM view.
// Nodes come from Plan.to_tree(depth): {state: [r, c], kind, action, edge, text, children};
// texts are already 1-indexed.
import { escapeHtml, samePos, ACTION_ARROWS } from './format.js';

/** Levels expanded by default (deeper nodes start folded unless they lead to the robot). */
export const DEFAULT_OPEN_DEPTH = 2;

/** Id of a child: "0" -> "0.2". The root's id is "0". */
export function childId(parentId, index) {
  return `${parentId}.${index}`;
}

/** Ancestor ids of a node id, root first: "0.1.2" -> ["0", "0.1"]. */
export function ancestorIds(id) {
  const parts = String(id).split('.');
  const out = [];
  for (let i = 1; i < parts.length; i += 1) out.push(parts.slice(0, i).join('.'));
  return out;
}

/** Node with the given id, or null. */
export function nodeById(tree, id) {
  if (!tree) return null;
  const parts = String(id).split('.').map(Number);
  let node = tree;
  for (const index of parts.slice(1)) {
    node = node.children?.[index];
    if (!node) return null;
  }
  return node;
}

/** Number of nodes in a tree. */
export function countNodes(tree) {
  if (!tree) return 0;
  let n = 1;
  for (const child of tree.children ?? []) n += countNodes(child);
  return n;
}

/**
 * Breadth-first search for the shallowest node where the robot at `pos` is, i.e. the node
 * whose decision applies now: an 'act' (or 'goal') node with state == pos. Returns its id.
 */
export function findCurrentNode(tree, pos) {
  if (!tree || !pos) return null;
  const queue = [{ node: tree, id: '0' }];
  while (queue.length) {
    const { node, id } = queue.shift();
    if ((node.kind === 'act' || node.kind === 'goal') && samePos(node.state, pos)) return id;
    (node.children ?? []).forEach((child, i) => queue.push({ node: child, id: childId(id, i) }));
  }
  return null;
}

/** branchesAt results per plan tree (trees are immutable; hover asks the same cells again). */
const BRANCH_CACHE = new WeakMap();

/** Outcome branches of the plan at `pos`: [{edge, text, kind}] (empty if none). Cached. */
export function branchesAt(tree, pos) {
  if (!tree || !pos) return [];
  let perTree = BRANCH_CACHE.get(tree);
  if (!perTree) {
    perTree = new Map();
    BRANCH_CACHE.set(tree, perTree);
  }
  const key = `${pos[0]},${pos[1]}`;
  if (!perTree.has(key)) perTree.set(key, computeBranches(tree, pos));
  return perTree.get(key);
}

function computeBranches(tree, pos) {
  const id = findCurrentNode(tree, pos);
  const node = id ? nodeById(tree, id) : null;
  if (!node || node.kind !== 'act') return [];
  return (node.children ?? []).map((c) => ({ edge: c.edge ?? c.label ?? '', text: c.text, kind: c.kind }));
}

/** Default folded set: every node with children deeper than DEFAULT_OPEN_DEPTH, except the
 * ancestors of `keepOpenId` (so the current node is visible). */
export function defaultCollapsed(tree, keepOpenId = null, openDepth = DEFAULT_OPEN_DEPTH) {
  const keep = new Set(keepOpenId ? ancestorIds(keepOpenId) : []);
  const collapsed = new Set();
  const walk = (node, id, depth) => {
    const kids = node.children ?? [];
    if (kids.length && depth >= openDepth && !keep.has(id)) collapsed.add(id);
    kids.forEach((c, i) => walk(c, childId(id, i), depth + 1));
  };
  if (tree) walk(tree, '0', 0);
  return collapsed;
}

/** Visible rows of the tree given the set of folded node ids (pre-order). */
export function flattenTree(tree, collapsed = new Set()) {
  const rows = [];
  const walk = (node, id, depth) => {
    const kids = node.children ?? [];
    const folded = collapsed.has(id);
    rows.push({
      id,
      depth,
      kind: node.kind,
      text: node.text,
      edge: node.edge ?? node.label ?? null,
      action: node.action ?? null,
      state: node.state,
      hasChildren: kids.length > 0,
      collapsed: folded && kids.length > 0,
      descendants: folded ? countNodes(node) - 1 : 0,
    });
    if (!folded) kids.forEach((c, i) => walk(c, childId(id, i), depth + 1));
  };
  if (tree) walk(tree, '0', 0);
  return rows;
}

/** Tone of an edge label: arrived / slipped / skid. */
export function edgeTone(edge) {
  const e = String(edge ?? '');
  if (e.startsWith('slip')) return 'slip';
  if (e.startsWith('skid')) return 'skid';
  if (e.startsWith('arriv')) return 'ok';
  return 'muted';
}

/** Plain-English edge label: "skid→(2,1)" -> "if it skids to (2,1)". */
export function edgeSentence(edge) {
  const e = String(edge ?? '');
  if (e.startsWith('arriv')) return 'if it arrives';
  if (e.startsWith('slip')) return 'if it slips';
  if (e.startsWith('skid')) {
    const target = e.split('→')[1];
    return target ? `if it skids to ${target}` : 'if it skids';
  }
  return e;
}

const KIND_GLYPH = { goal: '★', loop: '↺', ref: '↪', truncated: '…' };

/** Parent id of a node id ("0.1.2" -> "0.1"; the root has none). */
export function parentId(id) {
  const i = String(id).lastIndexOf('.');
  return i < 0 ? null : String(id).slice(0, i);
}

/**
 * The row a WAI-ARIA tree key moves to (or the fold change it makes) from row `id` among the
 * visible `rows` (flattenTree): {focus: id} and/or {toggle: id}. ArrowUp/Down: previous / next
 * row; Home/End: first / last; ArrowRight: unfold a folded row, else go to its first child;
 * ArrowLeft: fold an open row, else go to its parent; Enter/Space: fold or unfold.
 */
export function treeKeyAction(rows, id, key) {
  const i = rows.findIndex((r) => r.id === id);
  if (i < 0) return null;
  const row = rows[i];
  switch (key) {
    case 'ArrowDown': return i + 1 < rows.length ? { focus: rows[i + 1].id } : null;
    case 'ArrowUp': return i > 0 ? { focus: rows[i - 1].id } : null;
    case 'Home': return { focus: rows[0].id };
    case 'End': return { focus: rows[rows.length - 1].id };
    case 'ArrowRight':
      if (!row.hasChildren) return null;
      if (row.collapsed) return { toggle: id, focus: id };
      return { focus: childId(id, 0) };
    case 'ArrowLeft':
      if (row.hasChildren && !row.collapsed) return { toggle: id, focus: id };
      return parentId(id) ? { focus: parentId(id) } : null;
    case 'Enter':
    case ' ':
      return row.hasChildren ? { toggle: id, focus: id } : null;
    default: return null;
  }
}

/** DOM view of a plan tree with folding, current-node highlight and hover callbacks. Keyboard:
 * the WAI-ARIA tree pattern with a roving tabindex; re-rendering keeps the focused row. */
export class PlanTreeView {
  constructor(root, { onHoverCell }) {
    this.root = root;
    this.onHoverCell = onHoverCell;
    this.tree = null;
    this.treeKey = null;
    this.collapsed = new Set();
    this.currentId = null;
    this.focusId = null; // the row that holds tabindex=0
    root.addEventListener('click', (ev) => {
      const row = ev.target.closest('[data-id]');
      if (!row) return;
      this.focusId = row.dataset.id;
      if (row.dataset.fold) this._toggle(row.dataset.id);
    });
    root.addEventListener('keydown', (ev) => {
      const row = ev.target.closest('[data-id]');
      if (!row) return;
      const action = treeKeyAction(flattenTree(this.tree, this.collapsed), row.dataset.id, ev.key);
      if (!action) return;
      ev.preventDefault();
      ev.stopPropagation();
      if (action.focus) this.focusId = action.focus;
      if (action.toggle) this._toggle(action.toggle);
      else this._focusRow(action.focus);
    });
    root.addEventListener('pointerover', (ev) => {
      const row = ev.target.closest('[data-r]');
      this.onHoverCell(row ? [Number(row.dataset.r), Number(row.dataset.c)] : null);
    });
    root.addEventListener('pointerleave', () => this.onHoverCell(null));
    root.addEventListener('focusin', (ev) => {
      const row = ev.target.closest('[data-id]');
      if (!row) return;
      this.focusId = row.dataset.id;
      this._syncTabindex();
      this.onHoverCell(row.dataset.r != null ? [Number(row.dataset.r), Number(row.dataset.c)] : null);
    });
    // Leaving the tree clears the map highlight (no stale marker on the map).
    root.addEventListener('focusout', (ev) => {
      if (!root.contains(ev.relatedTarget)) this.onHoverCell(null);
    });
  }

  /** Show `tree` (key changes when a new plan arrives) with the robot at `pos`. */
  show(tree, key, pos) {
    const fresh = key !== this.treeKey;
    this.tree = tree;
    this.currentId = findCurrentNode(tree, pos);
    if (fresh) {
      this.treeKey = key;
      this.collapsed = defaultCollapsed(tree, this.currentId);
      this.focusId = null;
    } else if (this.currentId) {
      for (const a of ancestorIds(this.currentId)) this.collapsed.delete(a);
    }
    this._render(true);
  }

  _toggle(id) {
    if (this.collapsed.has(id)) this.collapsed.delete(id);
    else this.collapsed.add(id);
    this._render(false);
  }

  _focusRow(id) {
    const el = id ? this.root.querySelector(`[data-id="${CSS.escape(id)}"]`) : null;
    if (!el) return;
    this._syncTabindex();
    el.focus({ preventScroll: false });
  }

  /** Roving tabindex: exactly one row is reachable with Tab (the focused, current or first). */
  _syncTabindex() {
    const rows = [...this.root.querySelectorAll('.pt-row')];
    if (!rows.length) return;
    let keep = rows.find((r) => r.dataset.id === this.focusId)
      ?? rows.find((r) => r.dataset.id === this.currentId) ?? rows[0];
    if (!keep) keep = rows[0];
    for (const r of rows) r.tabIndex = r === keep ? 0 : -1;
  }

  _render(scroll) {
    if (!this.tree) {
      this.root.innerHTML = '';
      return;
    }
    const hadFocus = this.root.contains(document.activeElement);
    const rows = flattenTree(this.tree, this.collapsed);
    if (this.focusId && !rows.some((r) => r.id === this.focusId)) {
      // The focused row was folded away: focus its nearest visible ancestor.
      this.focusId = ancestorIds(this.focusId).reverse().find((a) => rows.some((r) => r.id === a)) ?? null;
    }
    const html = rows.map((row) => {
      const current = row.id === this.currentId;
      const glyph = row.kind === 'act' ? (ACTION_ARROWS[row.action] ?? '•') : (KIND_GLYPH[row.kind] ?? '•');
      const fold = row.hasChildren
        ? `<span class="pt-fold" aria-hidden="true">${row.collapsed ? '▸' : '▾'}</span>`
        : '<span class="pt-fold pt-leaf" aria-hidden="true"></span>';
      const edge = row.edge
        ? `<span class="pt-edge pt-edge-${edgeTone(row.edge)}" title="${escapeHtml(edgeSentence(row.edge))}">${escapeHtml(row.edge)}</span>`
        : '<span class="pt-edge pt-edge-root">start</span>';
      const more = row.collapsed ? `<span class="pt-more">+${row.descendants}</span>` : '';
      const state = Array.isArray(row.state) ? ` data-r="${row.state[0]}" data-c="${row.state[1]}"` : '';
      const expanded = row.hasChildren ? ` aria-expanded="${!row.collapsed}"` : '';
      return `<div class="pt-row pt-${row.kind}${current ? ' pt-current' : ''}" role="treeitem" tabindex="-1"
        aria-level="${row.depth + 1}"${expanded}${current ? ' aria-current="location"' : ''} data-id="${row.id}"${row.hasChildren ? ' data-fold="1"' : ''}${state}
        style="--depth:${row.depth}">${fold}${edge}<span class="pt-glyph" aria-hidden="true">${glyph}</span><span class="pt-text">${escapeHtml(row.text)}</span>${current ? '<span class="pt-here">robot is here</span>' : ''}${more}</div>`;
    });
    this.root.innerHTML = html.join('');
    this._syncTabindex();
    if (hadFocus) {
      // Keep keyboard / screen-reader users where they were.
      const el = this.root.querySelector('.pt-row[tabindex="0"]');
      el?.focus({ preventScroll: true });
    } else if (scroll) {
      const cur = this.root.querySelector('.pt-current');
      // Scroll only the tree box (scrollIntoView would also scroll the page on phones).
      if (cur) {
        const box = this.root;
        const top = cur.offsetTop;
        const bottom = top + cur.offsetHeight;
        if (top < box.scrollTop + 8) box.scrollTop = Math.max(0, top - 24);
        else if (bottom > box.scrollTop + box.clientHeight - 8) box.scrollTop = bottom - box.clientHeight + 24;
      }
    }
  }
}
