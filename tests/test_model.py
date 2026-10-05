"""Tests for rescue.model (docs/plan.md section 3.3, decisions D3, D4, D5)."""

from __future__ import annotations

import numpy as np
import pytest
from scipy.stats import chisquare

from rescue.belief import BeliefMap
from rescue.config import (
    DEFAULT_SLIP_P,
    MAX_SLIP_P,
    PROB_EPS,
    SKID_FRACTION,
    SLIP_STAY_FRACTION,
)
from rescue.grid import Cell
from rescue.model import (
    ACTIONS,
    OUTCOME_ORDER,
    Action,
    Outcome,
    OutcomeKind,
    ProblemModel,
    TerrainSlip,
    TransitionModel,
    UniformSlip,
    kind_direction,
    outcome_distribution,
    step_pos,
)

A, SL, KL, KR = (
    OutcomeKind.ARRIVED,
    OutcomeKind.SLIPPED,
    OutcomeKind.SKID_L,
    OutcomeKind.SKID_R,
)
N_SAMPLES = 100_000
P_VALUE_FLOOR = 1e-3


def _open(rows: int, cols: int) -> np.ndarray:
    return np.ones((rows, cols), dtype=bool)


# --- Action ------------------------------------------------------------------------------


def test_action_deltas_and_order() -> None:
    assert ACTIONS == (Action.N, Action.E, Action.S, Action.W)
    assert Action.N.delta == (-1, 0)
    assert Action.E.delta == (0, 1)
    assert Action.S.delta == (1, 0)
    assert Action.W.delta == (0, -1)
    for a in ACTIONS:
        assert str(a) == a.name


@pytest.mark.parametrize(
    ("action", "left", "right"),
    [
        (Action.N, Action.W, Action.E),
        (Action.E, Action.N, Action.S),
        (Action.S, Action.E, Action.W),
        (Action.W, Action.S, Action.N),
    ],
)
def test_left_right_table(action: Action, left: Action, right: Action) -> None:
    assert action.left is left
    assert action.right is right
    assert action.left.right is action
    assert action.right.left is action


def test_step_pos() -> None:
    assert step_pos((2, 2), Action.N) == (1, 2)
    assert step_pos((2, 2), Action.E) == (2, 3)
    assert step_pos((2, 2), Action.S) == (3, 2)
    assert step_pos((2, 2), Action.W) == (2, 1)
    assert step_pos((0, 0), Action.N) == (-1, 0)  # no bounds check


def test_kind_direction() -> None:
    for a in ACTIONS:
        assert kind_direction(A, a) is a
        assert kind_direction(SL, a) is None
        assert kind_direction(KL, a) is a.left
        assert kind_direction(KR, a) is a.right


def test_outcome_kind_values_and_order() -> None:
    assert OUTCOME_ORDER == (A, SL, KL, KR)
    assert A.value == "arrived"
    assert SL.value == "slipped"
    assert KL.value == "skid-L"
    assert KR.value == "skid-R"


# --- outcome distribution ----------------------------------------------------------------


@pytest.mark.parametrize("p", [0.0, 0.05, 0.1, 0.2, 0.3, MAX_SLIP_P])
def test_outcome_distribution(p: float) -> None:
    dist = outcome_distribution(p)
    kinds = tuple(k for k, _ in dist)
    assert kinds == OUTCOME_ORDER
    probs = dict(dist)
    assert sum(probs.values()) == pytest.approx(1.0, abs=1e-12)
    assert probs[A] == pytest.approx(1 - p)
    assert probs[SL] == pytest.approx(SLIP_STAY_FRACTION * p)
    assert probs[KL] == pytest.approx(SKID_FRACTION * p / 2)
    assert probs[KR] == pytest.approx(probs[KL])  # skid mass split equally
    assert probs[KL] + probs[KR] == pytest.approx(0.25 * p)


def test_deck_split_at_default_p() -> None:
    probs = dict(outcome_distribution(DEFAULT_SLIP_P))
    assert probs[A] == pytest.approx(0.80)
    assert probs[SL] == pytest.approx(0.15)
    assert probs[KL] == pytest.approx(0.025)
    assert probs[KR] == pytest.approx(0.025)


def test_uniform_slip_basics() -> None:
    t = UniformSlip(0.2)
    assert isinstance(t, TransitionModel)
    assert t.slip_p((0, 0)) == 0.2
    assert t.slip_p((5, 7)) == 0.2
    assert t.max_p() == 0.2
    assert t.distribution((1, 1)) == outcome_distribution(0.2)
    assert t.support((1, 1)) == OUTCOME_ORDER


def test_p_zero_support_and_sampling_deterministic() -> None:
    t = UniformSlip(0.0)
    assert t.support((0, 0)) == (A,)
    for u in [0.0, 0.1, 0.5, 0.8, 0.95, 0.99, 0.999999, np.nextafter(1.0, 0.0)]:
        assert t.sample_kind((0, 0), float(u)) is A
    rng = np.random.default_rng(3)
    assert all(t.sample_kind((2, 2), float(u)) is A for u in rng.random(2_000))


def test_support_filters_by_prob_eps() -> None:
    tiny = UniformSlip(PROB_EPS / 10)
    assert tiny.support((0, 0)) == (A,)


def test_sample_kind_inverse_cdf() -> None:
    t = UniformSlip(0.2)  # cdf: 0.80, 0.95, 0.975, 1.0
    pos = (1, 1)
    assert t.sample_kind(pos, 0.0) is A
    assert t.sample_kind(pos, 0.5) is A
    assert t.sample_kind(pos, 0.79) is A
    assert t.sample_kind(pos, 0.81) is SL
    assert t.sample_kind(pos, 0.94) is SL
    assert t.sample_kind(pos, 0.96) is KL
    assert t.sample_kind(pos, 0.974) is KL
    assert t.sample_kind(pos, 0.976) is KR
    assert t.sample_kind(pos, 0.9999) is KR


def _cumulative(t: TransitionModel, pos: tuple[int, int]) -> list[float]:
    """Running float sums of the distribution, computed exactly as ``sample_kind`` does."""
    out: list[float] = []
    total = 0.0
    for _, prob in t.distribution(pos):
        total += prob
        out.append(total)
    return out


def test_sample_kind_strict_boundaries() -> None:
    """``u`` exactly on a CDF step belongs to the *next* outcome (strict ``u < cumulative``)."""
    t = UniformSlip(0.2)
    pos = (1, 1)
    c1, c2, c3, _ = _cumulative(t, pos)
    assert t.sample_kind(pos, float(np.nextafter(c1, 0.0))) is A
    assert t.sample_kind(pos, c1) is SL
    assert t.sample_kind(pos, float(np.nextafter(c2, 0.0))) is SL
    assert t.sample_kind(pos, c2) is KL
    assert t.sample_kind(pos, float(np.nextafter(c3, 0.0))) is KL
    assert t.sample_kind(pos, c3) is KR
    top = float(np.nextafter(1.0, 0.0))  # largest u that Generator.random() can return
    assert t.sample_kind(pos, top) is KR
    assert UniformSlip(0.0).sample_kind(pos, top) is A


class _ShortMassModel(TransitionModel):
    """Stub whose probabilities sum to slightly less than 1 and give zero mass to skids."""

    def slip_p(self, pos: tuple[int, int]) -> float:
        return 0.5

    def max_p(self) -> float:
        return 0.5

    def distribution(self, pos: tuple[int, int]):  # type: ignore[override]
        return ((A, 0.5), (SL, 0.5 - 1e-9), (KL, 0.0), (KR, 0.0))


def test_sample_kind_fallback_returns_last_supported_kind() -> None:
    t = _ShortMassModel()
    assert t.sample_kind((0, 0), 0.25) is A
    assert t.sample_kind((0, 0), 0.75) is SL
    # u above the (rounded-down) total: the last kind with positive mass, never a skid
    assert t.sample_kind((0, 0), 1.0 - 1e-12) is SL


def test_transition_model_requires_max_p() -> None:
    class OnlySlip(TransitionModel):
        def slip_p(self, pos: tuple[int, int]) -> float:
            return 0.1

    with pytest.raises(TypeError):
        OnlySlip()  # type: ignore[abstract]


_MASK = np.zeros((2, 2), dtype=bool)


@pytest.mark.parametrize(
    "make",
    [
        lambda: UniformSlip(MAX_SLIP_P + 1e-9),
        lambda: UniformSlip(-0.01),
        lambda: UniformSlip(1.0),
        lambda: UniformSlip(float("nan")),
        lambda: TerrainSlip(0.5, 0.2, _MASK),
        lambda: TerrainSlip(-0.1, 0.2, _MASK),
        lambda: TerrainSlip(0.1, MAX_SLIP_P + 0.01, _MASK),
        lambda: TerrainSlip(0.1, -0.2, _MASK),
        lambda: TerrainSlip(0.1, 0.2, np.zeros(3, dtype=bool)),
        lambda: TerrainSlip(0.1, 0.2, np.zeros((2, 2, 2), dtype=bool)),
        lambda: outcome_distribution(1.5),
        lambda: outcome_distribution(-0.1),
    ],
)
def test_transition_models_reject_bad_p(make) -> None:
    with pytest.raises(ValueError):
        make()


@pytest.mark.parametrize("p", [0.0, MAX_SLIP_P])
def test_transition_models_accept_p_range_endpoints(p: float) -> None:
    assert UniformSlip(p).max_p() == p
    assert TerrainSlip(p, p, _MASK).max_p() == p


def test_sample_kind_chi_square_100k() -> None:
    t = UniformSlip(0.2)
    rng = np.random.default_rng(20240607)
    us = rng.random(N_SAMPLES)
    index = {k: i for i, k in enumerate(OUTCOME_ORDER)}
    counts = np.zeros(4)
    for u in us:
        counts[index[t.sample_kind((3, 3), float(u))]] += 1
    expected = N_SAMPLES * np.array([0.80, 0.15, 0.025, 0.025])
    result = chisquare(counts, expected)
    assert result.pvalue > P_VALUE_FLOOR, (counts, result)
    # the two skid sides are symmetric
    assert abs(counts[2] - counts[3]) < 5 * np.sqrt(counts[2] + counts[3])


# --- TerrainSlip -------------------------------------------------------------------------


def test_terrain_slip_uses_high_p_on_mask() -> None:
    mask = np.zeros((3, 4), dtype=bool)
    mask[1, 2] = True
    mask[0, 0] = True
    t = TerrainSlip(0.1, 0.4, mask)
    assert isinstance(t, TransitionModel)
    assert t.slip_p((1, 2)) == 0.4
    assert t.slip_p((0, 0)) == 0.4
    assert t.slip_p((1, 1)) == 0.1
    assert t.slip_p((2, 3)) == 0.1
    assert t.max_p() == pytest.approx(0.4)
    assert t.distribution((1, 2)) == outcome_distribution(0.4)
    assert t.distribution((1, 1)) == outcome_distribution(0.1)
    # same u, different cell -> slip depends on where the robot stands (D4)
    assert t.sample_kind((1, 1), 0.85) is A  # cdf 0.9 on base cells
    assert t.sample_kind((1, 2), 0.85) is SL  # cdf 0.6 / 0.9 on high-slip cells


def test_terrain_slip_max_p_branches() -> None:
    # no high-slip cell on the map: high_p never applies
    assert TerrainSlip(0.1, 0.4, np.zeros((2, 2), dtype=bool)).max_p() == 0.1
    one = np.zeros((2, 2), dtype=bool)
    one[0, 1] = True
    assert TerrainSlip(0.3, 0.1, one).max_p() == 0.3  # base above high
    assert TerrainSlip(0.1, 0.3, one).max_p() == 0.3  # high above base


def test_terrain_slip_mask_is_a_read_only_copy() -> None:
    mask = np.zeros((2, 2), dtype=bool)
    t = TerrainSlip(0.0, 0.4, mask)
    mask[0, 0] = True
    assert t.slip_p((0, 0)) == 0.0
    with pytest.raises(ValueError):
        t.high_mask[0, 0] = True


def test_terrain_slip_zero_base_is_deterministic_off_mask() -> None:
    mask = np.zeros((2, 2), dtype=bool)
    mask[1, 1] = True
    t = TerrainSlip(0.0, 0.3, mask)
    assert t.support((0, 0)) == (A,)
    assert t.support((1, 1)) == OUTCOME_ORDER


# --- ProblemModel.results ----------------------------------------------------------------


def test_results_corner_east_merges_wall_skid_with_slip() -> None:
    m = ProblemModel(_open(3, 3), (2, 2), UniformSlip(0.2))
    res = m.results((0, 0), Action.E)
    assert res == (
        Outcome((0, 1), (A,)),
        Outcome((0, 0), (SL, KL)),
        Outcome((1, 0), (KR,)),
    )


def test_results_corner_north_intended_off_grid_merges_with_slip() -> None:
    m = ProblemModel(_open(3, 3), (2, 2), UniformSlip(0.2))
    res = m.results((0, 0), Action.N)
    assert res == (Outcome((0, 0), (A, SL, KL)), Outcome((0, 1), (KR,)))


def test_results_interior_has_four_distinct_outcomes() -> None:
    m = ProblemModel(_open(3, 3), (2, 2), UniformSlip(0.2))
    res = m.results((1, 1), Action.S)
    assert res == (
        Outcome((2, 1), (A,)),
        Outcome((1, 1), (SL,)),
        Outcome((1, 2), (KL,)),  # S.left = E
        Outcome((1, 0), (KR,)),  # S.right = W
    )


def test_results_debris_collapses_to_stay() -> None:
    passable = _open(3, 3)
    passable[1, 2] = False  # debris east of the centre
    m = ProblemModel(passable, (2, 2), UniformSlip(0.2))
    res = m.results((1, 1), Action.E)
    assert res == (
        Outcome((1, 1), (A, SL)),
        Outcome((0, 1), (KL,)),
        Outcome((2, 1), (KR,)),
    )
    # skid into debris also collapses
    res_n = m.results((1, 1), Action.N)
    assert res_n == (
        Outcome((0, 1), (A,)),
        Outcome((1, 1), (SL, KR)),
        Outcome((1, 0), (KL,)),
    )


def test_results_all_blocked_is_single_stay() -> None:
    passable = np.zeros((1, 1), dtype=bool)
    passable[0, 0] = True
    m = ProblemModel(passable, (0, 0), UniformSlip(0.3))
    for a in ACTIONS:
        assert m.results((0, 0), a) == (Outcome((0, 0), (A, SL, KL, KR)),)


@pytest.mark.parametrize("a", ACTIONS)
def test_results_p_zero_single_outcome(a: Action) -> None:
    m = ProblemModel(_open(4, 4), (3, 3), UniformSlip(0.0))
    res = m.results((1, 1), a)
    assert len(res) == 1
    assert res[0] == Outcome(step_pos((1, 1), a), (A,))
    corner = m.results((0, 0), Action.N)
    assert corner == (Outcome((0, 0), (A,)),)


def test_results_labels_ordered_and_outcomes_by_first_label() -> None:
    rng = np.random.default_rng(11)
    for _ in range(20):
        passable = rng.random((5, 5)) > 0.3
        passable[2, 2] = True
        m = ProblemModel(passable, (4, 4), UniformSlip(0.25))
        for a in ACTIONS:
            res = m.results((2, 2), a)
            all_labels = [lab for o in res for lab in o.labels]
            assert sorted(all_labels, key=OUTCOME_ORDER.index) == list(OUTCOME_ORDER)
            assert len({o.state for o in res}) == len(res)  # merged
            firsts = [OUTCOME_ORDER.index(o.labels[0]) for o in res]
            assert firsts == sorted(firsts)
            for o in res:
                idx = [OUTCOME_ORDER.index(lab) for lab in o.labels]
                assert idx == sorted(idx)
                assert m.is_free(o.state)


def test_results_respects_terrain_support() -> None:
    mask = np.zeros((3, 3), dtype=bool)
    mask[1, 1] = True
    m = ProblemModel(_open(3, 3), (2, 2), TerrainSlip(0.0, 0.2, mask))
    assert len(m.results((0, 1), Action.S)) == 1  # p=0 where the robot stands
    assert len(m.results((1, 1), Action.S)) == 4  # high slip under the wheels


def test_problem_model_basics() -> None:
    passable = _open(3, 4)
    passable[0, 1] = False
    m = ProblemModel(passable, (2, 3), UniformSlip(0.2))
    assert (m.rows, m.cols) == (3, 4)
    assert m.goal == (2, 3)
    assert m.is_goal((2, 3))
    assert not m.is_goal((0, 0))
    assert m.is_free((0, 0))
    assert not m.is_free((0, 1))
    assert not m.is_free((-1, 0))
    assert not m.is_free((3, 0))
    assert not m.is_free((0, 4))
    assert m.actions((0, 0)) == ACTIONS
    states = m.states()
    assert states == sorted(states)
    assert len(states) == 11
    assert (0, 1) not in states


def test_problem_model_passable_is_a_copy() -> None:
    passable = _open(3, 3)
    m = ProblemModel(passable, (2, 2), UniformSlip(0.2))
    passable[0, 1] = False
    assert m.is_free((0, 1))
    assert m.passable[0, 1]
    with pytest.raises(ValueError):
        m.passable[0, 0] = False


def test_problem_model_from_belief_unknown_is_open_debris_collapses() -> None:
    belief = BeliefMap(3, 3, (2, 2))
    belief.update({(0, 1): Cell.DEBRIS, (0, 0): Cell.OPEN})
    m = ProblemModel.from_belief(belief, UniformSlip(0.2))
    assert m.goal == (2, 2)
    assert m.is_free((1, 1))  # unknown assumed open
    assert not m.is_free((0, 1))
    res = m.results((0, 0), Action.E)
    assert res == (Outcome((0, 0), (A, SL, KL)), Outcome((1, 0), (KR,)))
    assert len(m.states()) == 8
