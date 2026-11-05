"""The front end's recorded missions (``webapp/tests/fixtures/missions.json``) are current.

The Node tests and the development mock replay them, so a stale file would let the JavaScript
tests pass against data the Python core no longer produces. ``scripts/make_web_fixtures.py``
re-records them.
"""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path
from types import ModuleType
from typing import Any

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
SCRIPT = REPO_ROOT / "scripts" / "make_web_fixtures.py"
#: Catalog keys that depend on the machine (whether a map library file was found).
MACHINE_KEYS = ("library",)


def _script() -> ModuleType:
    spec = importlib.util.spec_from_file_location("make_web_fixtures", SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def committed() -> dict[str, Any]:
    script = _script()
    return json.loads(script.FIXTURE.read_text(encoding="utf-8"))


def test_every_recorded_mission_is_what_the_core_produces_now(committed: dict[str, Any]) -> None:
    script = _script()
    configs = [m["config"] for m in committed["missions"]]
    assert configs == [dict(c) for c in script.MISSIONS]
    for recording in committed["missions"]:
        fresh = json.loads(json.dumps(script.record(dict(recording["config"]))))
        assert fresh == recording, recording["config"]


def test_the_recorded_catalog_is_current(committed: dict[str, Any]) -> None:
    from rescue.webapi import catalog

    now = json.loads(json.dumps(catalog()))
    for key in MACHINE_KEYS:
        now.pop(key, None)
        committed["catalog"].pop(key, None)
    assert committed["catalog"] == now
