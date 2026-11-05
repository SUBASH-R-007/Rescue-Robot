"""Build (and by default serve) the browser app: a thin wrapper around ``python -m rescue.webapp``.

Usage (from the repository root, after ``pip install -e .``)::

    python scripts/build_webapp.py                       # build build/webapp/, serve, open
    python scripts/build_webapp.py --build-only --out _site

Every option of ``python -m rescue.webapp`` is accepted (docs/webapp.md).
"""

from __future__ import annotations

import sys

from rescue.webapp import main

if __name__ == "__main__":
    sys.exit(main())
