"""Tests for rescue.planners.validate (docs/plan.md section 3.7), on hand-built policies."""

from __future__ import annotations

import numpy as np
import pytest

from rescue.grid import Pos
from rescue.model import Action, ProblemModel, UniformSlip
from rescue.planners.validate import ValidationReport, is_strong_cyclic, validate_policy
from tests.helpers_phase2 import ascii_model

N, E, S, W = Action.N, Action.E, Action.S, Action.W

CORRIDOR = "S.P"
#: Skidding S from (1,1) drops the robot into a two-cell side passage.
SIDE_PASSAGE = "S.P\n.##\n.##"


def corridor(p: float) -> tuple[ProblemModel, Pos]:
    return ascii_model(CORRIDOR, UniformSlip(p))


def _check(policy: dict[Pos, Action], model: ProblemModel, start: Pos) -> ValidationReport:
    report = validate_policy(policy, model, start)
    assert isinstance(report, ValidationReport)
    assert isinstance(report.reachable, frozenset)
    assert isinstance(report.missing, frozenset)
    assert isinstance(report.dead, frozenset)
    assert is_strong_cyclic(policy, model, start) is report.ok
    return report


@pytest.mark.parametrize("p", [0.0, 0.2, 0.4])
def test_valid_strong_cyclic(p: float) -> None:
    model, start = corridor(p)
    report = _check({(0, 0): E, (0, 1): E}, model, start)
    assert report.ok
    assert report.missing == frozenset()
    assert report.dead == frozenset()
    assert {(0, 0), (0, 1)} <= report.reachable <= {(0, 0), (0, 1), (0, 2)}


@pytest.mark.parametrize("p", [0.0, 0.2])
def test_missing_state(p: float) -> None:
    model, start = corridor(p)
    report = _check({(0, 0): E}, model, start)
    assert not report.ok
    assert report.missing == frozenset({(0, 1)})
    assert (0, 1) in report.reachable


def test_missing_start() -> None:
    model, start = corridor(0.2)
    report = _check({(0, 1): E}, model, start)
    assert not report.ok
    assert start in report.missing


@pytest.mark.parametrize("p", [0.0, 0.2, 0.4])
def test_dead_loop(p: float) -> None:
    model, start = corridor(p)
    report = _check({(0, 0): E, (0, 1): W}, model, start)
    assert not report.ok
    assert report.missing == frozenset()
    assert report.dead == frozenset({(0, 0), (0, 1)})


def test_wall_bump_self_loop_is_dead() -> None:
    model, start = corridor(0.0)
    report = _check({(0, 0): N, (0, 1): E}, model, start)
    assert not report.ok
    assert report.dead == frozenset({(0, 0)})


def test_dead_loop_reached_only_by_a_skid() -> None:
    policy = {(0, 0): E, (0, 1): E, (1, 0): S, (2, 0): N}
    # p = 0: the side passage is never entered, so the policy is fine.
    model0, start = ascii_model(SIDE_PASSAGE, UniformSlip(0.0))
    assert _check(policy, model0, start).ok
    # p > 0: the E-skid from (1,1) enters a two-state cycle with no exit (weak, not strong).
    model, start = ascii_model(SIDE_PASSAGE, UniformSlip(0.2))
    report = _check(policy, model, start)
    assert not report.ok
    assert report.missing == frozenset()
    assert report.dead == frozenset({(1, 0), (2, 0)})
    assert {(1, 0), (2, 0)} <= report.reachable


def test_start_equals_goal() -> None:
    model, _ = corridor(0.2)
    report = _check({}, model, model.goal)
    assert report.ok
    assert report.missing == frozenset()
    assert report.dead == frozenset()
    assert is_strong_cyclic({(0, 0): W}, model, model.goal)


@pytest.mark.parametrize("p", [0.0, 0.2])
def test_extra_unreachable_entries_are_ignored(p: float) -> None:
    # Column 1 is a wall: the right half is another component full of junk entries.
    passable = np.ones((3, 4), dtype=bool)
    passable[:, 1] = False
    model = ProblemModel(passable, (2, 0), UniformSlip(p))
    policy = {(0, 0): S, (1, 0): S, (0, 3): E, (2, 2): W, (1, 3): N}
    report = _check(policy, model, (0, 0))
    assert report.ok
    assert (0, 3) not in report.reachable
    assert (2, 2) not in report.reachable
    assert report.dead == frozenset()
    assert report.missing == frozenset()


def test_extra_entries_beside_valid_plan_on_same_component() -> None:
    model, start = ascii_model("S..\n...\n..P", UniformSlip(0.0))
    policy = {(0, 0): E, (0, 1): E, (0, 2): S, (1, 2): S, (2, 0): N, (1, 0): N}
    report = _check(policy, model, start)
    assert report.ok
    assert (2, 0) not in report.reachable
