# Browser demo (pygbag)

> This page covers the pygbag build of the Pygame simulator. The mouse-first **web app**
> (`python -m rescue.webapp`: Pyodide, plain-English explanations, map editor, comparison and
> results) is described in [webapp.md](webapp.md).

The interactive simulator also runs in a web browser (brief §6.7). The browser build runs the
same code as `python -m rescue.sim --map maps/deck_demo.txt`: CPython 3.12 compiled to
WebAssembly, pygame-ce and numpy, packed by [pygbag](https://pypi.org/project/pygbag/). It opens
`maps/deck_demo.txt` with the demo seed (`config.DEMO_SEED` = 76), paused. The keyboard
controls of the desktop simulator work (README, *Interactive simulator*), except `G` and `Esc`
(see *Known limitations*). Pressing `Space` plays
the same mission as `python -m rescue.demo`: a slip at (1,2), replans after debris is sensed at
(3,2) and (4,3), and a rescue after 9 steps.

## How it works

| Piece | Role |
|---|---|
| `web/main.py` | Entry point. pygbag requires a folder whose `main.py` ends with `asyncio.run(main())`. It builds `SimOptions` for deck_demo (demo seed, autoplay off) and awaits `SimApp(...).run_async()`. Its PEP 723 header and top-level `import numpy` tell pygbag which wheels to fetch. On a desktop, `python web/main.py` runs the same entry point in a normal window. |
| `BaseApp.run_async()` (`src/rescue/viz/sim_app.py`) | The frame loop for asyncio. It does the same per-frame work as the desktop `run()`, through the shared `run_frame()`, and adds `await asyncio.sleep(0)` after each frame. The browser can repaint and deliver input only while Python yields. |
| Lazy imports | Nothing the simulator imports at start-up needs matplotlib, scipy, imageio or PyYAML, none of which the pygbag runtime provides. `config.load_yaml` imports PyYAML when it is called, and the recorder imports imageio only when it writes. `tests/test_web.py` checks this in a fresh interpreter. |
| Bundled fonts | The build copies `DejaVuSans.ttf` and `DejaVuSansMono.ttf` from matplotlib's `mpl-data/fonts/ttf` into `fonts/`, together with their licence (`LICENSE_DEJAVU`, the Bitstream Vera / DejaVu licence). `theme.load_fonts()` picks these files first when they sit in a `fonts/` directory next to the `rescue` package, or in `$RESCUE_FONTS_DIR`, so plan labels keep their `→ ↺ ★ …` glyphs. A normal checkout has no such directory, so the desktop font choice does not change. |
| `scripts/build_web.py` | Stages the app folder and runs pygbag (below). |

## Build

```bash
pip install -e ".[web]"            # pygbag >= 0.9 (the build was verified with 0.9.3)
python scripts/build_web.py        # stage build/web_app/ and run pygbag --build
python scripts/build_web.py --stage-only   # stage only: no pygbag, no network
python scripts/build_web.py --out some/dir # another app folder
```

The script stages `build/web_app/`, which `.gitignore` excludes like the rest of `build/`:

* `main.py`, copied from `web/main.py`;
* `rescue/`, the package without `experiments/`, `ext/`, `viz/ext_sim_app.py` (the
  extension simulator, which needs `ext/`), the web app's modules (`webapi.py`, `webapp.py`,
  `webmaps.py`, `webtext.py`; [webapp.md](webapp.md)) and `__pycache__`, which the simulator
  never imports. `mapgen.py` is kept;
* `maps/*.txt` and `fonts/` (the two DejaVu fonts and their licence).

It then runs `python -m pygbag --build --width 1280 --height 800 --title "Rescue Robot
simulator" build/web_app` and prints the output directory, `build/web_app/build/web/`:

| File | Size (pygbag 0.9.3) |
|---|---|
| `index.html` | 12 KB (pygbag's page template, downloaded from its CDN on the first build) |
| `web_app.apk` | 0.7 MB (a zip of the app folder) |
| `web_app.tar.gz` | 0.7 MB (the same files; the page loads this one) |
| `favicon.png` | 18 KB |

That totals 1.4 MiB. The first build needs network access, because pygbag downloads its page
template and icon (they are cached in `build/web_app/build/web-cache/`).

## Serve locally

```bash
python -m http.server 8765 --bind 127.0.0.1 --directory build/web_app/build/web
```

Open **http://127.0.0.1:8765**, not `http://localhost:8765`. When pygbag's page runs on a
`localhost:8…` address, it assumes it is being served by pygbag's own test server and fetches
the wheels from `http://localhost:8000/cdn/`, the test server's fixed address, whatever port
the page itself is on. Nothing answers there under `http.server`, so the first wheel (numpy)
never loads and the app never starts. On `127.0.0.1`, or on any real host, the wheels
come from the pygbag CDN. Another option is pygbag's own test server, which serves on http://localhost:8000 and
proxies the CDN. Pass it the same options as the build, or it rebuilds the bundle with its
defaults (a 1280×720 canvas and no title): `python -m pygbag --width 1280 --height 800 --title
"Rescue Robot simulator" build/web_app`. This variant was not tried in the verification
browser.

The page loads the Python runtime, then the app. Click the page if it asks for a click: browsers
block audio until the user interacts, and pygbag waits for that click. The simulator then
shows deck_demo, paused. Press `H` for the controls.

## Publish on GitHub Pages

These steps are instructions only; nothing in the repository deploys automatically.

1. Build: `python scripts/build_web.py`.
2. Copy the contents of `build/web_app/build/web/` (`index.html`, `web_app.apk`,
   `web_app.tar.gz` and `favicon.png`) to the root of a `gh-pages` branch, or to a `docs/`
   folder on the default branch. Add an empty `.nojekyll` file so that Pages serves the files
   unchanged.
3. On GitHub, open *Settings → Pages* and choose that branch and folder.
4. Open `https://<user>.github.io/<repository>/`. Pages serves HTTPS on a non-localhost
   address, so the runtime and wheels come from the pygbag CDN.

The same folder works on any static host, such as itch.io (`python -m pygbag --archive
build/web_app` makes the zip that itch.io expects).

## Known limitations

* **No recording.** imageio is not available in the browser, and files could not be saved
  anyway, so `G` only logs "recording unavailable in the browser". Screenshots and GIFs come
  from the desktop build (`scripts/make_figures.py`).
* **No `Esc` to quit.** On the desktop `Esc` ends the frame loop; in the browser that would
  freeze the canvas until a page reload, so there `Esc` only closes the help overlay, and the
  help overlay does not list `Esc` or `G` (`sim_app.BROWSER_HIDDEN_KEYS`).
* **Only map files with `M`.** scipy is not in pygbag's package index, so `rescue.mapgen`
  cannot be imported in the browser. The simulator already treats a missing mapgen as "no
  generators", so `M` cycles through the six map files in `maps/` (deck_demo, deck_figure and
  the four generated samples) and `--scenario` generators are not offered. `Ctrl+S` writes
  into the page's in-memory file system, which is lost on reload.
* **First load downloads the runtime.** On every first visit the page fetches the CPython
  WebAssembly runtime (several MB), the numpy wheel (about 12 MB) and the pygame-ce wheel
  (about 1.5 MB) from `pygame-web.github.io/cdn/`. This needs network access to that CDN; the
  browser caches the files afterwards. The page does not work offline.
* **Performance.** WebAssembly CPython is several times slower than desktop CPython. The 4×4 demo
  plans in a few milliseconds, but large generated maps would plan noticeably slower; the "Plan
  ms" panel value reports the in-browser time. Browsers also pause `requestAnimationFrame` in
  background tabs, so a hidden tab stops stepping.
* **Mouse coordinates at fractional pixel ratios.** pygbag warns "Unsupported device pixel
  ratio" when `window.devicePixelRatio` is not 1. In the verification browser (pixel ratio 1.25,
  Windows display scaling at 125 %) the canvas received correct DOM mouse events, but pygame
  reported a wrong vertical position, so map-editor clicks and plan-tree folding hit the wrong
  row. Keyboard controls are not affected. At a pixel ratio of 1 the mapping is expected to be
  correct; set the browser zoom so that `devicePixelRatio` is 1 (for example 80 % at 125 %
  display scaling). This was not verified, because the verification browser could not change
  its pixel ratio.
* **Python 3.12.** pygbag 0.9.3 ships CPython 3.12. The package needs Python 3.11 or later, so
  the version does not matter. With numpy 2.0.2 in the browser, the demo seed reproduced the
  desktop mission step for step.

## Tests

`tests/test_web.py` runs headless with the SDL dummy drivers. It covers:

* `run_async` runs exactly the requested frames, yields once per frame, saves the screenshot
  and closes the app;
* `run_async` and `run` produce the same mission;
* the interactive `run_async` stops on `Esc` on the desktop, while in the browser `Esc` does not
  quit and the help overlay hides `G` and `Esc`;
* `G` degrades gracefully when recording is unavailable;
* the simulator's import closure contains no matplotlib, scipy, imageio or yaml;
* `load_fonts()` prefers bundled fonts and is unchanged without them;
* `stage()` produces `main.py`, the package (without `experiments/` and `ext/`), the maps and
  the fonts with their licence;
* `main()` prints the output path, with the pygbag call stubbed.

pygbag itself is not run by the test suite.
