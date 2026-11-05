# Web app

The web app is a browser front end for the whole project. It runs the **same Python core** as
the command line (`rescue.runner`, the agents, the planners and the extensions) inside the
browser with [Pyodide](https://pyodide.org), so a seed gives the same map and the same mission
in the browser as in `python -m rescue.demo`, `python -m rescue.sim` or the experiment
harness. The page is static: there is no server-side code, and nothing is sent anywhere.

The Pygame simulator (`python -m rescue.sim`) and its pygbag browser build ([web.md](web.md))
are unchanged. The web app is a separate, mouse-first interface: you click instead of using
keyboard shortcuts, and every step comes with a plain-English explanation.

## Run it locally (one command)

```bash
python -m rescue.webapp
```

This builds the site into `build/webapp/` (git-ignored), serves it on
**http://127.0.0.1:8000/** and opens the browser. Stop it with `Ctrl+C`. It needs only the
normal install (`pip install -e .`) and network access on the first page load, which
downloads Pyodide from its CDN (see *Limitations*).

| Option | Effect |
|---|---|
| `--out DIR` | Build into `DIR` instead of `build/webapp/`. An existing build there is replaced. A non-empty directory that is not an earlier build is refused, so a typo cannot delete files. |
| `--port N` | Serve on port `N` (default 8000; `0` picks a free port). |
| `--host ADDR` | Bind address (default `127.0.0.1`). |
| `--no-open` | Do not open a browser. |
| `--build-only` | Build and exit (used by the GitHub Pages workflow). |
| `--skip-build` | Serve an existing build without rebuilding. |

`python scripts/build_webapp.py` is the same command, for people who look in `scripts/`.

Use `http://127.0.0.1:…`, as printed. The address `localhost` usually works too, but
[web.md](web.md) explains why the project avoids it for local web builds.

## What is in the app

The first visit shows the loading steps, then a welcome card with the presets (*Slide 9
demo*, *Planners head-to-head*, *Hidden survivor*, *Several survivors*, *Noisy sensors*,
*Blank canvas*) and a 30-second tour of the screen.

* **3D view**: the toggle above the map (or `D`) shows the same mission as a rover driving the
  site: unscanned cells are solid ground blocks that sink away as the sensor reaches them,
  debris becomes rubble, and the executed trace keeps the flat map's glyphs (bar = moved,
  ring = slipped, arrow = skidded, diamond = replanned). Drag to orbit, scroll to zoom. It
  reads the same theme colours, so light, dark and high contrast all work. three.js and the
  view's own module are fetched the first time the toggle is used, so a visit that never turns
  it on downloads nothing extra. It needs WebGL; the flat map stays the default and keeps the
  features the 3D view has no equivalent for (cell hover, rulers, the A\* path).
* **Simulate**: Play / Back / Step / Finish / Reset / New map, a speed slider and a timeline
  you can drag to replay any step (ticks: slips ● amber, skids ▲ red, replans ◆ purple, the
  rescue ✓ green). The map shows the robot's belief (fog of war “?”), hatched debris, water
  waves, the executed trace (colour *and* glyph per outcome), the start, the target flag and
  the survivors; the *True map* and *Plan arrows* toggles add the hidden debris and the
  policy. Hovering a cell tells what the robot believes, what is really there (when revealed),
  P(debris) in the noisy mode and the plan's action with its outcome branches. The side panel
  has four tabs: *What happened* (plain-English sentences about the last step and the
  clickable event log), *Plan* (the folding AND-OR tree with the robot's node highlighted;
  hovering a row highlights its cell; the path for A\* agents), *Numbers* (metric cards with
  definitions) and *Settings* (mode, map, agent, slip p with its outcome split, sensor radius,
  replan mode, aftershocks, survivors, sensor noise, seed; changing one restarts the mission).
  A legend is always visible.
* **Compare**: two or three agents on the same map with the same outcome stream (common random
  numbers), played in lock-step, with status badges, the metrics side by side and a sentence
  summarising the difference so far.
* **Map editor**: a tool palette (debris, eraser, water/mud, start, survivor, extra survivor)
  with click-and-drag painting, undo, resize, clear, a text form, *Check map* (is every
  survivor reachable?) and *Use this map*; any bundled or generated map can be the starting
  point.
* **Results**: a card per claim in the committed reports (`results/HYPOTHESES.md`,
  `results/ext/EXTENSIONS.md`, `results/lao/LAO.md`): what it means in plain words, the verdict
  and the key number as written in the report, the full method on request, an *Honest
  findings* box (the refuted claims and why) and the plots in a lightbox.
* **Learn**: short illustrated lessons on the slip/skid model, AND-OR contingency plans (the
  deck's root plan drawn as a tree), how each agent reacts and what the extensions add, with
  the figures from `docs/figures/`.

Every coordinate shown in the app is 1-indexed, as in the command-line traces. Keyboard:
Space play/pause, N or → step, B or ← back, Home/End, R reset, T true map, P plan arrows,
? help. Themes: light, dark and high contrast (the sun/moon button).

## How it works

```
 browser main thread                     Web Worker (js/worker.js)
 ───────────────────                     ─────────────────────────
 index.html + js/*.js  ── RPC {id, method, params} ►  Pyodide (CPython 3.13 in WebAssembly)
 (render, panels,      ◄── {id, ok, result | error}   numpy
  plan tree, editor…)                                  rescue.zip → /home/pyodide/app
                                                       rescue.webapi.WebSession
                                                         → runner.Mission / ext.mission.ExtMission
```

| Piece | Role |
|---|---|
| `src/rescue/webapi.py` | The bridge. Pure Python, JSON in and out. `catalog()` lists the maps, scenarios, agents, modes, parameter ranges and presets; `WebSession(config)` wraps `runner.Mission` (core mode) or `ext.mission.ExtMission` (search, multi and noisy modes) exactly as the CLI builds them, and returns the state after each step, including the plain-English explanation. It imports neither pygame, matplotlib nor PyYAML, and imports scipy only when a map has to be generated. |
| `webapp/js/worker.js` | Loads Pyodide from the pinned CDN release, then numpy, unpacks `rescue.zip` and imports `rescue.webapi`. The page talks to it by messages with ids, so the interface never freezes while Python plans. When a map needs the generator, the worker loads scipy on demand and shows *Loading the map generator…*. Python errors reach the page as a short message with the traceback attached. |
| `webapp/` | The front end: `index.html`, `css/app.css` and plain ES modules in `js/`. There is no build step and there are no npm dependencies. |
| `src/rescue/webapp.py` | Builds the static site and serves it (below). |

### What the build writes

`python -m rescue.webapp --build-only` produces:

| File | Contents |
|---|---|
| `index.html`, `css/`, `js/` | Copied from `webapp/` (`webapp/tests/` is left out). |
| `rescue.zip` | The `rescue` package without `viz/` (the Pygame UI), `experiments/` (the harness), `ext/{experiments,claims,plots,demo}.py` and the entry points of the Pygame apps and of the build (`sim`, `compare`, `replay`, `webapp`; `demo.py` stays because the bridge reuses its formatting), plus `maps/*.txt` and a copy of `maps_library.json`. Entries are sorted and carry a fixed timestamp, so building the same tree twice gives the same bytes. The build fails if a packed module imports pygame, matplotlib or imageio. |
| `maps_library.json` | The scenario maps pre-generated with the real generator (`rescue.webmaps.build_map_library`: `mapgen.generate(scenario, n, n, seed)` for the four generated scenarios, n ∈ {10, 12, 15, 20}, seeds 0–29). The browser uses these, so the common case never downloads scipy. Other sizes and seeds load scipy on demand and run the same generator. |
| `results.json` | The claims of the three reports (id, claim, verdict, measured text and the report's details section), the run metadata, the LAO\* tables, and the curated plots and figures with captions taken from the docs (`docs/results.md`, `docs/extensions.md`, `README.md`). The images are copied to `assets/results/` and `assets/figures/`. |
| `version.json` | Package version, bridge API version, git commit, build time (`SOURCE_DATE_EPOCH` is honoured) and the Pyodide release with its Python and numpy versions. The worker reads the Pyodide version from here. |
| `.nojekyll` | Tells GitHub Pages to serve the files unchanged. |

The local server is Python's `http.server`, bound to 127.0.0.1, with the MIME types ES
modules need (`text/javascript`) and `Cache-Control: no-cache`, so a rebuild shows on reload.

### Pinned versions

| | Version |
|---|---|
| Pyodide | 0.28.3 (`config.WEB_PYODIDE_VERSION`) |
| three.js (3D view, on demand) | 0.160.1 (`config.WEB_THREE_VERSION`) |
| CPython in Pyodide | 3.13.2 |
| numpy in Pyodide | 2.2.5 |
| scipy in Pyodide (on demand) | 1.14.1 |

These come from the release's `pyodide-lock.json` on the CDN. To move to another release,
change `WEB_PYODIDE_VERSION` (and the recorded Python and numpy versions) in
`src/rescue/config.py`; the worker's fallback constant must match it (a test checks this).

## Deploy on GitHub Pages

The workflow `.github/workflows/pages.yml` builds the site and publishes it on every push to
`main` (and on a manual run). It installs the package on Python 3.11, runs
`python -m rescue.webapp --build-only --out _site`, uploads `_site` with
`actions/upload-pages-artifact` and deploys it with `actions/deploy-pages`.

To turn it on:

1. On GitHub, open the repository's **Settings → Pages**.
2. Under *Build and deployment*, set **Source** to **GitHub Actions**.
3. Push to `main` (or run the workflow from the *Actions* tab).

The site then appears at `https://<user>.github.io/<repository>/`. Nothing is deployed until
Pages is enabled this way. The same `_site/` folder works on any static host.

## Limitations

* **The first load needs the network.** The browser downloads Pyodide from
  `cdn.jsdelivr.net`: the runtime (`pyodide.asm.wasm` 8.6 MB, 2.7 MB as transferred with
  compression; `python_stdlib.zip` 2.4 MB; the loader 1.1 MB, 0.2 MB compressed) and the
  numpy wheel (3.1 MB). That is about 8.5 MB over the network (15 MB unpacked), plus the site
  itself (about 0.4 MB without the result images: `rescue.zip` 0.17 MB, `maps_library.json`
  0.13 MB, `results.json` 0.06 MB). scipy (13.3 MB, plus OpenBLAS 1.9 MB) is fetched only
  when a map has to be generated or LAO\* is used. The sizes are those of the 0.28.3 files on
  the CDN (measured 2026-10-08). The browser caches these files, but the app does not work
  offline.
* **WebAssembly is slower than desktop CPython.** Small and medium maps plan quickly; large
  maps with the expected-steps planners (`vi_expected`, `lao_expected`) take noticeably longer.
  Planning runs in the worker, so the page stays responsive meanwhile.
* **No recording or run files.** The browser app does not write GIF/MP4 recordings or
  `runs/*.json` files; use the desktop tools for those.
* **The 3D view is Simulate only.** Compare and the map editor stay flat, and the 3D view has
  no cell hover (no picking) and no rulers. It needs WebGL and adds a three.js download
  (about 0.7 MB) the first time it is switched on.
* **Experiments are not run in the browser.** The Results tab shows the committed reports;
  the Monte Carlo harness (`python -m rescue.experiments`) stays a command-line tool.

## Tests

`tests/test_webapp_build.py` builds the site into a temporary directory and checks:

* the zip holds the core, the extensions it needs and the maps, and leaves out `viz/`,
  `experiments/`, the extension harness and the CLI modules; no packed source imports pygame,
  matplotlib or imageio; two builds give identical zips;
* the unpacked zip imports in a fresh interpreter with pygame, matplotlib, scipy, PyYAML and
  imageio blocked, and a `WebSession` on deck_demo reproduces the `python -m rescue.demo`
  trace step for step (same positions, actions and outcomes; rescue after 9 steps);
* `maps_library.json` equals `mapgen.generate(…).to_ascii()` and is the copy inside the zip;
* `results.json` contains every claim of `HYPOTHESES.md` and `EXTENSIONS.md` with its verdict,
  the LAO\* tables, and every curated plot with a caption from the docs;
* `version.json` has the version, commit, build time and Pyodide release; the front end pins
  the same release;
* the server answers HTTP 200 for `index.html` and `rescue.zip` on 127.0.0.1, both in-process
  and through `python -m rescue.webapp --skip-build --port 0 --no-open`;
* the front end's own unit tests (`webapp/tests/`, run with `node --test`) pass when Node.js
  is installed; the test is skipped otherwise.
