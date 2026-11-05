// Three.js renderer of a mission state: the same state object the canvas renderer draws
// (render.js), shown as a rover driving a disaster site. Fog of war becomes unscanned ground
// blocks that sink away as the sensor reaches them, debris becomes rubble, the executed trace
// keeps the 2D glyph language (bar = moved, ring = slipped, arrow = skidded, diamond =
// replanned) and the colours come from the same CSS tokens, so all three themes work.
//
// Interface: the subset of MapRenderer that the views use — setState, setOptions, resize,
// destroy — so a view can hold either renderer. Imported lazily (it pulls in three.js).
import * as THREE from 'three';
import { OrbitControls } from 'three/addons/controls/OrbitControls.js';
import { CELL, normOutcome, samePos, parseKey } from './format.js';
import { attemptedDir, probLevel } from './render.js';

/** Rover glide between cells, ms (0 with prefers-reduced-motion). */
const GLIDE_MS = 420;
/** An unscanned block takes this long to sink away once the cell is sensed, ms. */
const FOG_FADE_MS = 320;
/** Opacity of the unscanned blocks, and with the true map shown. */
const FOG_OPACITY = 0.96;
const FOG_OPACITY_TRUE = 0.2;
/** Cell is one world unit; these are fractions of it. */
const TILE_H = 0.14;
const FOG_H = 0.92;
const ROVER_SCALE = 1.18;
/** Camera framing: distance and height are these times the framing span. A small map still
 *  gets framed as if it were MIN_FRAME cells across, or the camera sits inside the scene. */
const CAM_DISTANCE = 1.05;
const CAM_HEIGHT = 0.78;
const MIN_FRAME = 9;
/** Direction vectors in (row, col). */
const DIR = { N: [-1, 0], E: [0, 1], S: [1, 0], W: [0, -1] };
/** Yaw that points the rover (modelled facing +x) along each action. */
const YAW = { E: 0, N: Math.PI / 2, S: -Math.PI / 2, W: Math.PI };
/** Theme tokens this renderer reads (a subset of render.js's TOKENS). */
const TOKENS = [
  'map-bg', 'map-floor', 'map-fog', 'map-fog-ink', 'map-debris', 'map-water', 'map-robot',
  'map-survivor',
  'map-start', 'map-target', 'map-policy', 'map-prob', 'c-ok', 'c-slip', 'c-skid', 'c-replan',
  'c-rescue',
];

const easeInOut = (t) => (t < 0.5 ? 2 * t * t : 1 - Math.pow(-2 * t + 2, 2) / 2);

/** Deterministic 0..1 from an integer, so a map's rubble looks the same every time. */
export function jitter(seed) {
  let t = (seed + 0x6d2b79f5) | 0;
  t = Math.imul(t ^ (t >>> 15), t | 1);
  t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
  return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
}

/** Identity of the map a state is on: a new one means the scene has to be rebuilt. */
export function mapKey(state) {
  if (!state) return '';
  const cells = (state.cells ?? []).map((row) => row.join('')).join('/');
  const water = (state.high_slip ?? []).map((row) => row.map((v) => (v ? 1 : 0)).join('')).join('');
  return `${state.rows}x${state.cols}:${cells}:${water}`;
}

/**
 * Where the rover sits part-way through the step `entry`, as a fraction `p` of the move.
 * A clean move drives across; a slip lurches and settles back; a skid drives at the cell it
 * aimed at and slides into the one it reached. Returns offsets in (row, col).
 */
export function stepPose(entry, p) {
  const out = normOutcome(entry.outcome);
  const [fr, fc] = entry.from;
  const [tr, tc] = entry.to;
  const aim = DIR[attemptedDir(entry)] ?? [0, 0];
  const yaw = YAW[entry.action] ?? 0;
  if (out === 'arrived' && !entry.blocked) {
    const e = easeInOut(p);
    return { r: fr + (tr - fr) * e, c: fc + (tc - fc) * e, yaw, roll: Math.sin(p * Math.PI) * 0.03 };
  }
  if (out === 'slipped' || entry.blocked || (fr === tr && fc === tc)) {
    const reach = entry.blocked ? 0.2 : 0.3;
    const lurch = Math.sin(p * Math.PI) * reach;
    return {
      r: fr + aim[0] * lurch, c: fc + aim[1] * lurch, yaw,
      roll: Math.sin(p * Math.PI) * (entry.blocked ? -0.09 : 0.07),
      spin: true,
    };
  }
  const e = easeInOut(p);
  const fwd = Math.sin(p * Math.PI) * 0.16;
  return {
    r: fr + aim[0] * fwd + (tr - fr) * e,
    c: fc + aim[1] * fwd + (tc - fc) * e,
    yaw,
    roll: Math.sin(p * Math.PI) * (out === 'skid-L' ? 0.16 : -0.16),
  };
}

export class Renderer3D {
  constructor(canvas, { interactive = true } = {}) {
    this.canvas = canvas;
    this.state = null;
    this.opts = { showTrue: false, showPolicy: false, highlight: null };
    this.colors = {};
    this.reducedMotion = matchMedia('(prefers-reduced-motion: reduce)');
    this._mapKey = '';
    this._unknown = new Set();
    this._vanish = new Map();
    this._glide = null;
    this._raf = 0;
    this._disposed = false;
    this._pool = [];

    this.renderer = new THREE.WebGLRenderer({ canvas, antialias: true, alpha: false });
    this.renderer.setPixelRatio(Math.min(window.devicePixelRatio || 1, 2));
    this.renderer.shadowMap.enabled = true;
    this.renderer.shadowMap.type = THREE.PCFSoftShadowMap;
    this.renderer.outputColorSpace = THREE.SRGBColorSpace;

    this.scene = new THREE.Scene();
    this.camera = new THREE.PerspectiveCamera(46, 1, 0.1, 400);
    this.controls = new OrbitControls(this.camera, canvas);
    this.controls.enableDamping = true;
    this.controls.dampingFactor = 0.08;
    this.controls.maxPolarAngle = 1.46;
    this.controls.enabled = interactive;

    this.hemi = new THREE.HemisphereLight(0xffffff, 0x000000, 0.5);
    this.sun = new THREE.DirectionalLight(0xffffff, 1.9);
    this.sun.castShadow = true;
    this.sun.shadow.mapSize.set(2048, 2048);
    this.sun.shadow.bias = -0.0012;
    this.scene.add(this.hemi, this.sun, this.sun.target);

    this.world = new THREE.Group();
    this.scene.add(this.world);

    this._onTheme = () => this.refreshTheme();
    window.addEventListener('rr-theme', this._onTheme);
    this._ro = new ResizeObserver(() => this.resize());
    this._ro.observe(canvas.parentElement);

    this.refreshTheme();
    this.resize();
    this._loop();
  }

  /** Re-read the CSS colour tokens; called on construction and on every theme change. */
  refreshTheme() {
    const css = getComputedStyle(document.documentElement);
    for (const t of TOKENS) {
      const raw = css.getPropertyValue(`--${t}`).trim();
      const col = new THREE.Color();
      try {
        col.setStyle(raw || '#888888');
      } catch {
        col.setStyle('#888888');
      }
      this.colors[t] = col;
    }
    const bg = this.colors['map-bg'];
    this.scene.background = bg.clone();
    // The range is set when the map is built; a later theme change only recolours the fog.
    if (this.scene.fog) this.scene.fog.color.copy(bg);
    else this.scene.fog = new THREE.Fog(bg.clone(), 1, 1);
    // A light theme needs flatter, brighter light than a dark one, or the site turns to mud.
    const light = bg.getHSL({ h: 0, s: 0, l: 0 }).l > 0.5;
    this.hemi.color.setStyle(light ? '#ffffff' : '#9fb0cc');
    this.hemi.groundColor.setStyle(light ? '#c9c2b6' : '#241c15');
    this.hemi.intensity = light ? 1.5 : 0.8;
    this.sun.color.setStyle(light ? '#fff6e8' : '#ffcf9a');
    this.sun.intensity = light ? 1.5 : 2.0;
    this._applyColors();
  }

  /** New state. `animate` glides the rover along the step that led to it. */
  setState(state, { animate = false, prev = null } = {}) {
    const key = mapKey(state);
    const old = this.state;
    this.state = state;
    if (key !== this._mapKey) {
      this._buildMap(state);
      this._mapKey = key;
    }
    const motion = animate && !this.reducedMotion.matches;
    const entry = state?.trace?.length ? state.trace[state.trace.length - 1] : null;
    const movedFrom = prev?.robot ?? old?.robot;
    if (motion && entry && movedFrom && state.robot
        && (!samePos(movedFrom, state.robot) || normOutcome(entry.outcome) !== 'arrived')) {
      this._glide = { entry, start: performance.now() };
    } else {
      this._glide = null;
    }
    this._syncFog(state);
    this._syncTrace(state);
    this._syncMarks(state);
    if (!this._glide) this._placeRover(state.robot, entry ? YAW[entry.action] : null, 0);
  }

  setOptions(opts) {
    Object.assign(this.opts, opts);
    if (this.fog) {
      this.fog.material.opacity = this.opts.showTrue ? FOG_OPACITY_TRUE : FOG_OPACITY;
      this.fog.material.depthWrite = !this.opts.showTrue;
    }
    if (this.policy) this.policy.visible = !!this.opts.showPolicy;
    if (this.state) this._syncMarks(this.state);
  }

  resize() {
    const box = this.canvas.parentElement?.getBoundingClientRect();
    if (!box) return;
    const w = Math.max(1, Math.floor(box.width));
    const h = Math.max(1, Math.floor(box.height));
    this.renderer.setSize(w, h, false);
    this.canvas.style.width = `${w}px`;
    this.canvas.style.height = `${h}px`;
    this.camera.aspect = w / h;
    this.camera.updateProjectionMatrix();
  }

  destroy() {
    this._disposed = true;
    cancelAnimationFrame(this._raf);
    window.removeEventListener('rr-theme', this._onTheme);
    this._ro?.disconnect();
    this._clearWorld();
    this.controls.dispose();
    this.renderer.dispose();
  }

  // --- scene building ---------------------------------------------------------------------

  _clearWorld() {
    this.world.traverse((o) => {
      if (o.geometry) o.geometry.dispose();
      if (o.material) {
        for (const m of Array.isArray(o.material) ? o.material : [o.material]) m.dispose();
      }
    });
    this.world.clear();
  }

  /** (row, col) -> world (x, z), with the map centred on the origin. */
  _x(c) { return c - (this.cols - 1) / 2; }
  _z(r) { return r - (this.rows - 1) / 2; }

  _buildMap(state) {
    this._clearWorld();
    this.rows = state.rows;
    this.cols = state.cols;
    const n = this.rows * this.cols;
    const span = Math.max(this.rows, this.cols);
    const frame = Math.max(span, MIN_FRAME);
    const dummy = new THREE.Object3D();

    this.scene.fog.near = frame * 1.4;
    this.scene.fog.far = frame * 4.2;
    this.controls.minDistance = 3;
    this.controls.maxDistance = frame * 3;
    const s = this.sun.shadow.camera;
    s.left = -span * 0.8; s.right = span * 0.8;
    s.top = span * 0.8; s.bottom = -span * 0.8;
    s.far = span * 5;
    s.updateProjectionMatrix();
    this.sun.position.set(span * 0.7, span * 1.1, span * 0.45);

    // Ground apron under the grid, so the map does not float in the fog.
    this.apron = new THREE.Mesh(
      new THREE.PlaneGeometry(span * 6, span * 6),
      new THREE.MeshStandardMaterial({ roughness: 1 })
    );
    this.apron.rotation.x = -Math.PI / 2;
    this.apron.position.y = -0.02;
    this.apron.receiveShadow = true;
    this.world.add(this.apron);

    // Floor tiles, one instance per cell, coloured per cell (water, debris probability).
    this.tiles = new THREE.InstancedMesh(
      new THREE.BoxGeometry(0.98, TILE_H, 0.98),
      new THREE.MeshStandardMaterial({ roughness: 0.96, metalness: 0.02 }),
      n
    );
    this.tiles.receiveShadow = true;
    this.tiles.instanceColor = new THREE.InstancedBufferAttribute(new Float32Array(n * 3), 3);
    for (let r = 0; r < this.rows; r++) {
      for (let c = 0; c < this.cols; c++) {
        const i = r * this.cols + c;
        dummy.position.set(this._x(c), jitter(i * 31) * 0.025, this._z(r));
        dummy.rotation.set(0, 0, 0);
        dummy.scale.set(1, 1, 1);
        dummy.updateMatrix();
        this.tiles.setMatrixAt(i, dummy.matrix);
      }
    }
    this.world.add(this.tiles);

    // Rubble: a few broken chunks on every debris cell, instanced into one mesh.
    const debris = [];
    for (let r = 0; r < this.rows; r++) {
      for (let c = 0; c < this.cols; c++) {
        if (state.cells?.[r]?.[c] === CELL.DEBRIS) debris.push([r, c]);
      }
    }
    const chunks = [];
    for (const [r, c] of debris) {
      const i = r * this.cols + c;
      const count = 3 + Math.floor(jitter(i * 13 + 5) * 3);
      for (let b = 0; b < count; b++) {
        const a = jitter(i * 97 + b * 17);
        const d = jitter(i * 61 + b * 29);
        chunks.push({ r, c, a, d, h: 0.26 + a * 0.42 });
      }
    }
    this.rubble = new THREE.InstancedMesh(
      new THREE.BoxGeometry(1, 1, 1),
      new THREE.MeshStandardMaterial({ roughness: 0.98 }),
      Math.max(1, chunks.length)
    );
    this.rubble.castShadow = true;
    this.rubble.receiveShadow = true;
    this.rubble.count = chunks.length;
    this.rubble.instanceColor = new THREE.InstancedBufferAttribute(
      new Float32Array(Math.max(1, chunks.length) * 3), 3
    );
    chunks.forEach((k, idx) => {
      dummy.position.set(this._x(k.c) + (k.a - 0.5) * 0.44, k.h / 2 + 0.05,
        this._z(k.r) + (k.d - 0.5) * 0.44);
      dummy.rotation.set(0, k.a * Math.PI, (k.d - 0.5) * 0.22);
      dummy.scale.set(0.3 + k.a * 0.46, k.h, 0.3 + k.d * 0.46);
      dummy.updateMatrix();
      this.rubble.setMatrixAt(idx, dummy.matrix);
    });
    this._chunks = chunks;
    this.world.add(this.rubble);

    // Unscanned ground: one instance per cell the belief has not seen.
    this.fog = new THREE.InstancedMesh(
      new THREE.BoxGeometry(1, FOG_H, 1),
      new THREE.MeshStandardMaterial({
        roughness: 1, transparent: true, opacity: FOG_OPACITY,
      }),
      n
    );
    this.fog.count = 0;
    this.world.add(this.fog);

    this.start = new THREE.Mesh(
      new THREE.RingGeometry(0.28, 0.4, 28),
      new THREE.MeshBasicMaterial({ side: THREE.DoubleSide, transparent: true, opacity: 0.9 })
    );
    this.start.rotation.x = -Math.PI / 2;
    this.start.position.y = 0.09;
    this.world.add(this.start);

    this._buildTraceMeshes(n);
    this._buildPolicy(n);
    this._buildTarget();
    this._buildSurvivors(state);
    this._buildRover();

    this._unknown = new Set();
    this._vanish.clear();

    // Frame the whole site.
    this.controls.target.set(0, 0, 0);
    this.camera.position.set(-frame * CAM_DISTANCE * 0.75, frame * CAM_HEIGHT,
      frame * CAM_DISTANCE);
    this.controls.update();
    this._applyColors();
  }

  _buildTraceMeshes(n) {
    const flat = (geo) => {
      const mesh = new THREE.InstancedMesh(geo, new THREE.MeshBasicMaterial({
        transparent: true, opacity: 0.92, side: THREE.DoubleSide,
      }), n * 2);
      mesh.count = 0;
      this.world.add(mesh);
      return mesh;
    };
    this.trOk = flat(new THREE.PlaneGeometry(0.86, 0.1));
    this.trSlip = flat(new THREE.RingGeometry(0.2, 0.29, 22));
    this.trSkid = flat(new THREE.ConeGeometry(0.15, 0.3, 3));
    this.trReplan = flat(new THREE.CircleGeometry(0.07, 4));
  }

  _buildPolicy(n) {
    this.policy = new THREE.InstancedMesh(
      new THREE.ConeGeometry(0.085, 0.2, 4),
      new THREE.MeshBasicMaterial({ transparent: true, opacity: 0.65 }),
      n
    );
    this.policy.count = 0;
    this.policy.visible = !!this.opts.showPolicy;
    this.world.add(this.policy);
  }

  _buildTarget() {
    this.target = new THREE.Group();
    const pole = new THREE.Mesh(
      new THREE.CylinderGeometry(0.014, 0.014, 0.52, 7),
      new THREE.MeshStandardMaterial({ roughness: 0.6 })
    );
    pole.position.y = 0.3;
    const flag = new THREE.Mesh(
      new THREE.PlaneGeometry(0.22, 0.14),
      new THREE.MeshBasicMaterial({ side: THREE.DoubleSide })
    );
    flag.position.set(0.11, 0.49, 0);
    this.target.add(pole, flag);
    this.target.visible = false;
    this._targetParts = { pole, flag };
    this.world.add(this.target);
  }

  _buildSurvivors(state) {
    this.survivors = new THREE.Group();
    this._survivorNodes = [];
    for (const [r, c] of state.survivors ?? []) {
      const g = new THREE.Group();
      const body = new THREE.Mesh(
        new THREE.CapsuleGeometry(0.11, 0.2, 4, 10),
        new THREE.MeshStandardMaterial({ roughness: 0.6 })
      );
      body.position.y = 0.26;
      body.castShadow = true;
      const halo = new THREE.Mesh(
        new THREE.RingGeometry(0.3, 0.42, 32),
        new THREE.MeshBasicMaterial({ side: THREE.DoubleSide, transparent: true, opacity: 0.8 })
      );
      halo.rotation.x = -Math.PI / 2;
      halo.position.y = 0.1;
      g.add(body, halo);
      g.position.set(this._x(c), 0, this._z(r));
      g.userData = { cell: [r, c], body, halo };
      this.survivors.add(g);
      this._survivorNodes.push(g);
    }
    this.world.add(this.survivors);
  }

  _buildRover() {
    const rover = new THREE.Group();
    this._wheels = [];
    const hull = new THREE.MeshStandardMaterial({ roughness: 0.46, metalness: 0.45 });
    const dark = new THREE.MeshStandardMaterial({ color: 0x3a342d, roughness: 0.7, metalness: 0.3 });
    const rubber = new THREE.MeshStandardMaterial({ color: 0x1d1a17, roughness: 0.95 });
    this._hullMat = hull;

    const chassis = new THREE.Mesh(new THREE.BoxGeometry(0.62, 0.17, 0.44), hull);
    chassis.position.y = 0.235;
    chassis.castShadow = true;
    const deck = new THREE.Mesh(new THREE.BoxGeometry(0.5, 0.025, 0.34),
      new THREE.MeshStandardMaterial({ color: 0x2b3550, roughness: 0.35, metalness: 0.5 }));
    deck.position.y = 0.333;
    deck.castShadow = true;
    const mast = new THREE.Mesh(new THREE.CylinderGeometry(0.018, 0.018, 0.26, 8), dark);
    mast.position.set(0.17, 0.44, 0);
    const head = new THREE.Mesh(new THREE.BoxGeometry(0.13, 0.09, 0.12), hull);
    head.position.set(0.17, 0.6, 0);
    head.castShadow = true;
    rover.add(chassis, deck, mast, head);

    const lamp = new THREE.Mesh(new THREE.SphereGeometry(0.035, 10, 8),
      new THREE.MeshStandardMaterial({ roughness: 0.3 }));
    lamp.position.set(-0.24, 0.35, 0);
    rover.add(lamp);
    this._lamp = lamp;

    const wheelGeo = new THREE.CylinderGeometry(0.105, 0.105, 0.075, 16);
    for (const wx of [-0.23, 0, 0.23]) {
      for (const wz of [-0.245, 0.245]) {
        const arm = new THREE.Mesh(new THREE.BoxGeometry(0.035, 0.1, 0.035), dark);
        arm.position.set(wx, 0.19, wz * 0.82);
        const w = new THREE.Mesh(wheelGeo, rubber);
        w.rotation.x = Math.PI / 2;
        w.position.set(wx, 0.105, wz);
        w.castShadow = true;
        const hub = new THREE.Mesh(new THREE.CylinderGeometry(0.045, 0.045, 0.08, 10), hull);
        hub.rotation.x = Math.PI / 2;
        hub.position.copy(w.position);
        rover.add(arm, w, hub);
        this._wheels.push(w);
      }
    }
    rover.scale.setScalar(ROVER_SCALE);
    this.rover = rover;
    this._prevPos = new THREE.Vector3();
    this.world.add(rover);
  }

  /** Push the theme colours into every material that is built from them. */
  _applyColors() {
    const c = this.colors;
    if (!c['map-floor'] || !this.tiles) return;
    this.apron.material.color.copy(c['map-bg']).lerp(c['map-floor'], 0.35);
    // --map-fog is a flat fill in 2D and can be nearly black; as a lit solid it needs to read
    // as ground, so it is mixed toward the ink the flat view draws its "?" in.
    this.fog.material.color.copy(c['map-fog']).lerp(c['map-fog-ink'], 0.3);
    this.start.material.color.copy(c['map-start']);
    this.trOk.material.color.copy(c['c-ok']);
    this.trSlip.material.color.copy(c['c-slip']);
    this.trSkid.material.color.copy(c['c-skid']);
    this.trReplan.material.color.copy(c['c-replan']);
    this.policy.material.color.copy(c['map-policy']);
    this._targetParts.pole.material.color.copy(c['map-target']);
    this._targetParts.flag.material.color.copy(c['map-target']);
    this._hullMat.color.copy(c['map-robot']);
    this._lamp.material.color.copy(c['map-robot']);
    this._lamp.material.emissive.copy(c['map-robot']);
    for (const g of this._survivorNodes ?? []) {
      g.userData.body.material.color.copy(c['map-survivor']);
      g.userData.halo.material.color.copy(c['map-survivor']);
    }
    const rub = c['map-debris'];
    for (let i = 0; i < (this._chunks ?? []).length; i++) {
      const k = this._chunks[i];
      const col = rub.clone().offsetHSL(0, 0, (k.d - 0.5) * 0.12);
      this.rubble.setColorAt(i, col);
    }
    if (this.rubble.instanceColor) this.rubble.instanceColor.needsUpdate = true;
    this._colorTiles();
  }

  /** Tile colours: floor, water for high-slip cells, a red lean for P(debris) in noisy mode. */
  _colorTiles() {
    const st = this.state;
    if (!this.tiles || !this.rows) return;
    const floor = this.colors['map-floor'];
    const water = this.colors['map-water'];
    const prob = this.colors['map-prob'];
    const col = new THREE.Color();
    for (let r = 0; r < this.rows; r++) {
      for (let c = 0; c < this.cols; c++) {
        const i = r * this.cols + c;
        col.copy(st?.high_slip?.[r]?.[c] ? water : floor);
        col.offsetHSL(0, 0, (jitter(i * 7919) - 0.5) * 0.05);
        const p = st?.p_debris?.[r]?.[c];
        const level = probLevel(p);
        if (level) col.lerp(prob, 0.12 * level);
        this.tiles.setColorAt(i, col);
      }
    }
    if (this.tiles.instanceColor) this.tiles.instanceColor.needsUpdate = true;
  }

  // --- per-state updates ------------------------------------------------------------------

  _syncFog(state) {
    const next = new Set();
    for (let r = 0; r < this.rows; r++) {
      for (let c = 0; c < this.cols; c++) {
        if (state.belief?.[r]?.[c] === CELL.UNKNOWN) next.add(r * this.cols + c);
      }
    }
    const now = performance.now();
    for (const i of this._unknown) {
      if (!next.has(i)) this._vanish.set(i, now);
    }
    for (const i of next) this._vanish.delete(i);   // seeking back re-covers a cell at once
    this._unknown = next;
    this._colorTiles();
  }

  _syncTrace(state) {
    const dummy = new THREE.Object3D();
    const counts = { ok: 0, slip: 0, skid: 0, replan: 0 };
    const put = (mesh, key) => {
      dummy.updateMatrix();
      mesh.setMatrixAt(counts[key]++, dummy.matrix);
    };
    for (const e of state.trace ?? []) {
      const [fr, fc] = e.from;
      const [tr, tc] = e.to;
      const out = normOutcome(e.outcome);
      dummy.scale.set(1, 1, 1);
      if (out === 'arrived' && !e.blocked) {
        dummy.position.set((this._x(fc) + this._x(tc)) / 2, 0.082, (this._z(fr) + this._z(tr)) / 2);
        dummy.rotation.set(-Math.PI / 2, 0, fr === tr ? 0 : Math.PI / 2);
        put(this.trOk, 'ok');
      } else if (out === 'slipped' || e.blocked) {
        dummy.position.set(this._x(fc), 0.084, this._z(fr));
        dummy.rotation.set(-Math.PI / 2, 0, 0);
        put(e.blocked ? this.trSkid : this.trSlip, e.blocked ? 'skid' : 'slip');
      } else {
        const dx = this._x(tc) - this._x(fc);
        const dz = this._z(tr) - this._z(fr);
        dummy.position.set((this._x(fc) + this._x(tc)) / 2, 0.09, (this._z(fr) + this._z(tr)) / 2);
        dummy.rotation.set(-Math.PI / 2, 0, -Math.atan2(dx, -dz));
        put(this.trSkid, 'skid');
      }
      if (e.replanned) {
        dummy.position.set(this._x(tc), 0.1, this._z(tr));
        dummy.rotation.set(-Math.PI / 2, 0, Math.PI / 4);
        put(this.trReplan, 'replan');
      }
    }
    this.trOk.count = counts.ok;
    this.trSlip.count = counts.slip;
    this.trSkid.count = counts.skid;
    this.trReplan.count = counts.replan;
    for (const m of [this.trOk, this.trSlip, this.trSkid, this.trReplan]) {
      m.instanceMatrix.needsUpdate = true;
    }
  }

  /** Start pad, policy arrows, target flag, survivors and what the belief knows of them. */
  _syncMarks(state) {
    if (state.start) this.start.position.set(this._x(state.start[1]), 0.09, this._z(state.start[0]));

    const dummy = new THREE.Object3D();
    let n = 0;
    if (this.opts.showPolicy) {
      for (const key of Object.keys(state.policy ?? {})) {
        const [r, c] = parseKey(key);
        const d = DIR[state.policy[key]];
        if (!d) continue;
        dummy.position.set(this._x(c), 0.56, this._z(r));
        dummy.rotation.set(-Math.PI / 2, 0, -Math.atan2(d[1], -d[0]));
        dummy.scale.set(1, 1, 1);
        dummy.updateMatrix();
        this.policy.setMatrixAt(n++, dummy.matrix);
      }
    }
    this.policy.count = n;
    this.policy.instanceMatrix.needsUpdate = true;
    this.policy.visible = !!this.opts.showPolicy && n > 0;

    if (state.target) {
      this.target.position.set(this._x(state.target[1]), 0, this._z(state.target[0]));
      this.target.visible = true;
    } else {
      this.target.visible = false;
    }

    const rescued = new Set((state.rescued ?? []).map((p) => `${p[0]},${p[1]}`));
    for (const g of this._survivorNodes ?? []) {
      const [r, c] = g.userData.cell;
      const seen = state.survivors_known !== false
        || state.belief?.[r]?.[c] === CELL.SURVIVOR;
      g.visible = seen || this.opts.showTrue;
      const done = rescued.has(`${r},${c}`);
      g.userData.halo.material.color.copy(this.colors[done ? 'c-rescue' : 'map-survivor']);
      g.userData.body.material.color.copy(this.colors[done ? 'c-rescue' : 'map-survivor']);
    }
  }

  _placeRover(cell, yaw, roll) {
    if (!cell || !this.rover) return;
    this.rover.position.set(this._x(cell[1]), 0, this._z(cell[0]));
    if (yaw != null) this.rover.rotation.y = yaw;
    this.rover.rotation.z = roll ?? 0;
  }

  // --- frame loop -------------------------------------------------------------------------

  _loop() {
    const now = performance.now();
    const dt = Math.min((now - (this._last ?? now)) / 1000, 0.05);
    this._last = now;

    if (this._glide) {
      const p = Math.min(1, (now - this._glide.start) / GLIDE_MS);
      const pose = stepPose(this._glide.entry, p);
      this.rover.position.set(this._x(pose.c), 0, this._z(pose.r));
      this.rover.rotation.y = pose.yaw;
      this.rover.rotation.z = pose.roll;
      if (p >= 1) {
        this._glide = null;
        if (this.state?.robot) this._placeRover(this.state.robot, pose.yaw, 0);
      }
    }

    if (this.rover) {
      const moved = this.rover.position.distanceTo(this._prevPos);
      const spinning = this._glide && stepPose(this._glide.entry, 0.5).spin;
      const d = spinning ? Math.max(moved / 0.105, dt * 11) : moved / 0.105;
      for (const w of this._wheels) w.rotation.y += d;
      this._prevPos.copy(this.rover.position);
    }

    this._drawFog(now);

    for (const g of this._survivorNodes ?? []) {
      if (!g.visible) continue;
      const pulse = 0.5 + Math.sin(now / 320) * 0.3;
      g.userData.halo.material.opacity = pulse;
      g.userData.halo.scale.setScalar(1 + Math.sin(now / 320) * 0.08);
    }

    this.controls.update();
    this.renderer.render(this.scene, this.camera);
    if (!this._disposed) this._raf = requestAnimationFrame(() => this._loop());
  }

  /** Unscanned blocks: full height while unknown, sinking away for FOG_FADE_MS once sensed. */
  _drawFog(now) {
    if (!this.fog) return;
    const dummy = new THREE.Object3D();
    let n = 0;
    const place = (i, k) => {
      const r = Math.floor(i / this.cols);
      const c = i % this.cols;
      dummy.position.set(this._x(c), (FOG_H / 2) * k - (1 - k) * 0.5, this._z(r));
      dummy.rotation.set(0, 0, 0);
      dummy.scale.set(1, Math.max(0.001, k), 1);
      dummy.updateMatrix();
      this.fog.setMatrixAt(n++, dummy.matrix);
    };
    for (const i of this._unknown) place(i, 1);
    for (const [i, t0] of this._vanish) {
      const p = (now - t0) / FOG_FADE_MS;
      if (p >= 1) { this._vanish.delete(i); continue; }
      place(i, 1 - easeInOut(p));
    }
    this.fog.count = n;
    this.fog.instanceMatrix.needsUpdate = true;
  }
}
