"""Tests for rescue.plan (docs/plan.md sections 3.6 and 3.10), written from the contract."""

from __future__ import annotations

import copy
import dataclasses
import pickle
import sys
from collections import deque
from typing import Any

import numpy as np
import pytest

from rescue.grid import Pos
from rescue.model import Action, OutcomeKind, ProblemModel, UniformSlip, step_pos
from rescue.plan import (
    Act,
    Branch,
    Goal,
    Loop,
    Plan,
    PlanStats,
    Ref,
    branch_label,
    plan_from_policy,
)
from rescue.planners.andor import and_or_search
from rescue.planners.bfs import UNREACHABLE, bfs_distances
from tests.helpers_phase2 import (
    deck_model,
    deck_start,
    farthest_cell,
    serpentine_mask,
    walk_branches,
    walk_nodes,
)

ARR = OutcomeKind.ARRIVED
SLIP = OutcomeKind.SLIPPED
SKL = OutcomeKind.SKID_L
SKR = OutcomeKind.SKID_R
N, E, S, W = Action.N, Action.E, Action.S, Action.W

TREE_KEYS = {"state", "kind", "action", "edge", "text", "children"}
#: Side of the serpentine corridor used for deep plans (plan depth > 400 Act levels).
DEEP_SIDE = 30
#: Minimum Act depth that makes a plan "deep" (well past the default recursion limit / 2).
DEEP_MIN_DEPTH = 400


# --- hand-built trees ------------------------------------------------------------------------
def deck_shape_plan() -> Plan:
    """The genuine p > 0 plan on an open 2x2 grid, goal (1,1): the deck's root shape."""
    goal = (1, 1)
    at_01 = Act(
        (0, 1),
        S,
        (
            Branch(goal, (ARR,), Goal(goal)),
            Branch((0, 1), (SLIP, SKL), Loop((0, 1))),
            Branch((0, 0), (SKR,), Loop((0, 0))),
        ),
    )
    at_10 = Act(
        (1, 0),
        E,
        (
            Branch(goal, (ARR,), Goal(goal)),
            Branch((1, 0), (SLIP, SKR), Loop((1, 0))),
            Branch((0, 0), (SKL,), Loop((0, 0))),
        ),
    )
    root = Act(
        (0, 0),
        E,
        (
            Branch((0, 1), (ARR,), at_01),
            Branch((0, 0), (SLIP, SKL), Loop((0, 0))),
            Branch((1, 0), (SKR,), at_10),
        ),
    )
    return Plan(root=root, start=(0, 0), goal=goal)


def three_by_three_plan() -> Plan:
    """A realistic 3x3 plan (goal (2,2), start (1,1), action E) with one deep Ref."""
    goal = (2, 2)
    at_12 = Act(
        (1, 2),
        S,
        (
            Branch(goal, (ARR,), Goal(goal)),
            Branch((1, 2), (SLIP, SKL), Loop((1, 2))),
            Branch((1, 1), (SKR,), Loop((1, 1))),
        ),
    )
    at_02 = Act(
        (0, 2),
        S,
        (
            Branch((1, 2), (ARR,), Ref((1, 2))),
            Branch((0, 2), (SLIP, SKL), Loop((0, 2))),
            Branch((0, 1), (SKR,), Loop((0, 1))),
        ),
    )
    at_01 = Act(
        (0, 1),
        E,
        (
            Branch((0, 2), (ARR,), at_02),
            Branch((0, 1), (SLIP, SKL), Loop((0, 1))),
            Branch((1, 1), (SKR,), Loop((1, 1))),
        ),
    )
    at_21 = Act(
        (2, 1),
        E,
        (
            Branch(goal, (ARR,), Goal(goal)),
            Branch((2, 1), (SLIP, SKR), Loop((2, 1))),
            Branch((1, 1), (SKL,), Loop((1, 1))),
        ),
    )
    root = Act(
        (1, 1),
        E,
        (
            Branch((1, 2), (ARR,), at_12),
            Branch((1, 1), (SLIP,), Loop((1, 1))),
            Branch((0, 1), (SKL,), at_01),
            Branch((2, 1), (SKR,), at_21),
        ),
    )
    return Plan(root=root, start=(1, 1), goal=goal)


def act_depth(tree: dict[str, Any]) -> int:
    """Max number of ``kind == "act"`` dicts on a root-to-leaf path of a ``to_tree`` dict."""
    best = 0
    stack = [(tree, 0)]
    while stack:
        node, above = stack.pop()
        here = above + (1 if node["kind"] == "act" else 0)
        best = max(best, here)
        stack.extend((child, here) for child in node["children"])
    return best


def walk_tree(tree: dict[str, Any]) -> list[dict[str, Any]]:
    """Every dict in a ``to_tree`` result (pre-order)."""
    out: list[dict[str, Any]] = []
    stack = [tree]
    while stack:
        node = stack.pop()
        out.append(node)
        stack.extend(reversed(node["children"]))
    return out


# --- construction ----------------------------------------------------------------------------
def test_nodes_are_frozen_and_hashable() -> None:
    goal = Goal((1, 1))
    loop = Loop((0, 0))
    ref = Ref((0, 1))
    br = Branch((1, 1), (ARR,), goal)
    act = Act((0, 0), E, (br,))
    for node in (goal, loop, ref, br, act):
        assert dataclasses.is_dataclass(node)
        hash(node)
    with pytest.raises(dataclasses.FrozenInstanceError):
        goal.state = (0, 0)  # type: ignore[misc]
    with pytest.raises(dataclasses.FrozenInstanceError):
        act.action = N  # type: ignore[misc]
    assert act.state == (0, 0)
    assert act.action is E
    assert act.branches == (br,)
    assert br.labels == (ARR,)
    assert br.node is goal
    assert Loop((0, 0)) == loop
    assert Ref((0, 1)) == ref
    assert Goal((1, 1)) != Loop((1, 1))


def test_inner_nodes_compare_structurally() -> None:
    # Two independently built copies of the same tree are equal and hash alike.
    a, b = three_by_three_plan().root, three_by_three_plan().root
    assert a is not b
    assert a == b
    assert hash(a) == hash(b)
    assert a.branches[0] == b.branches[0]
    assert hash(a.branches[0]) == hash(b.branches[0])
    assert len({a, b}) == 1
    # Any difference anywhere in the subtree breaks equality.
    assert a != deck_shape_plan().root
    assert isinstance(a, Act)
    changed = Act(a.state, a.action, (*a.branches[:-1], Branch((2, 1), (SKR,), Ref((2, 1)))))
    assert a != changed
    assert Act(a.state, N, a.branches) != a
    assert a.branches[1] != Branch((1, 1), (SLIP, SKL), Loop((1, 1)))  # labels differ
    assert a != Loop(a.state)
    assert Act((0, 0), E, ()) == Act((0, 0), E, ())


def test_inner_node_repr_is_one_level() -> None:
    root = three_by_three_plan().root
    assert repr(root) == "Act(state=(1, 1), action=E, branches=<4 branches>)"
    assert repr(root.branches[1]) == (
        "Branch(state=(1, 1), labels=('slipped',), node=Loop(state=(1, 1)))"
    )
    assert repr(root.branches[0]) == (
        "Branch(state=(1, 2), labels=('arrived',), "
        "node=Act(state=(1, 2), action=S, branches=<3 branches>))"
    )


def test_nodes_and_plans_pickle_and_copy() -> None:
    plan = three_by_three_plan()
    for clone in (pickle.loads(pickle.dumps(plan)), copy.deepcopy(plan), copy.copy(plan)):
        assert clone == plan
        assert clone.to_policy() == plan.to_policy()
        assert clone.pretty() == plan.pretty()
    root = plan.root
    assert pickle.loads(pickle.dumps(root)) == root
    assert copy.deepcopy(root) == root
    assert pickle.loads(pickle.dumps(root.branches[0])) == root.branches[0]


def test_plan_is_frozen() -> None:
    plan = deck_shape_plan()
    assert plan.start == (0, 0)
    assert plan.goal == (1, 1)
    with pytest.raises(dataclasses.FrozenInstanceError):
        plan.start = (1, 1)  # type: ignore[misc]


# --- to_policy / envelope --------------------------------------------------------------------
def test_to_policy_collects_every_act() -> None:
    assert deck_shape_plan().to_policy() == {(0, 0): E, (0, 1): S, (1, 0): E}
    assert three_by_three_plan().to_policy() == {
        (1, 1): E,
        (1, 2): S,
        (0, 1): E,
        (0, 2): S,
        (2, 1): E,
    }


def test_goal_root_policy_and_envelope() -> None:
    plan = Plan(root=Goal((2, 3)), start=(2, 3), goal=(2, 3))
    assert plan.to_policy() == {}
    assert plan.envelope() == frozenset({(2, 3)})


def test_envelope_contains_every_state_including_goal() -> None:
    assert deck_shape_plan().envelope() == frozenset({(0, 0), (0, 1), (1, 0), (1, 1)})
    env = three_by_three_plan().envelope()
    assert isinstance(env, frozenset)
    assert env == frozenset({(1, 1), (1, 2), (2, 2), (0, 1), (0, 2), (2, 1)})


# --- pretty ----------------------------------------------------------------------------------
def test_pretty_deck_shape() -> None:
    assert (
        deck_shape_plan().pretty() == "E → {arrived: S-plan, slipped: retry E, skid→(2,1): E-plan}"
    )


def test_pretty_three_by_three() -> None:
    assert three_by_three_plan().pretty() == (
        "E → {arrived: S-plan, slipped: retry E, skid→(1,2): E-plan, skid→(3,2): E-plan}"
    )


def test_pretty_goal_root() -> None:
    assert Plan(root=Goal((0, 0)), start=(0, 0), goal=(0, 0)).pretty() == "goal"


def test_pretty_goal_target_blocked_and_loop_to_other() -> None:
    # Hand-built (geometry not meaningful): exercises the remaining target/label kinds.
    root = Act(
        (1, 0),
        N,
        (
            Branch((0, 0), (ARR,), Goal((0, 0))),
            Branch((1, 0), (SKL,), Loop((1, 0))),
            Branch((1, 1), (SKR,), Loop((3, 4))),
        ),
    )
    plan = Plan(root=root, start=(1, 0), goal=(0, 0))
    assert plan.pretty() == "N → {arrived: goal, blocked: retry N, skid→(2,2): ↺(4,5)}"


def test_pretty_ref_uses_referenced_states_action() -> None:
    # The Ref'd state (1,0) is solved elsewhere with N; the root acts with E.
    at_10 = Act((1, 0), N, (Branch((0, 0), (ARR,), Goal((0, 0))),))
    at_01 = Act(
        (0, 1),
        S,
        (Branch((1, 1), (ARR,), Goal((1, 1))), Branch((1, 0), (SKR,), at_10)),
    )
    root = Act((0, 0), E, (Branch((0, 1), (ARR,), at_01), Branch((1, 0), (SKR,), Ref((1, 0)))))
    plan = Plan(root=root, start=(0, 0), goal=(1, 1))
    assert plan.pretty() == "E → {arrived: S-plan, skid→(2,1): N-plan}"


def test_pretty_p0_blocked_stay_branch() -> None:
    # p = 0: a move into a wall has the single outcome "arrived" that stays put → "blocked".
    root = Act((0, 0), N, (Branch((0, 0), (ARR,), Loop((0, 0))),))
    assert Plan(root=root, start=(0, 0), goal=(1, 1)).pretty() == "N → {blocked: retry N}"


def test_pretty_node_targets() -> None:
    plan = three_by_three_plan()
    root = plan.root
    assert isinstance(root, Act)
    at_12 = root.branches[0].node
    assert isinstance(at_12, Act)
    at_02 = root.branches[2].node.branches[0].node  # type: ignore[union-attr]
    assert isinstance(at_02, Act)
    # Act: its own action; Goal: "goal".
    assert plan.pretty_node(root, None) == "E-plan"
    assert plan.pretty_node(at_12, root.state) == "S-plan"
    assert plan.pretty_node(at_12.branches[0].node, at_12.state) == "goal"
    # Loop to the parent: "retry {parent's action}"; to another ancestor: "↺(r,c)".
    assert plan.pretty_node(root.branches[1].node, root.state) == "retry E"
    assert plan.pretty_node(at_12.branches[1].node, at_12.state) == "retry S"
    assert plan.pretty_node(at_12.branches[2].node, at_12.state) == "↺(2,2)"
    # Ref: the referenced state's action, wherever that Act node sits in the plan.
    assert plan.pretty_node(at_02.branches[0].node, at_02.state) == "S-plan"
    # Targets that do not resolve to an action fall back to the 1-indexed cell.
    assert plan.pretty_node(Ref((5, 5)), at_02.state) == "→(6,6)"
    assert plan.pretty_node(Loop((5, 5)), (5, 5)) == "↺(6,6)"
    for node in walk_nodes(plan.root):
        assert isinstance(plan.pretty_node(node, plan.start), str)


# --- branch_label ----------------------------------------------------------------------------
@pytest.mark.parametrize(
    ("parent", "branch", "expected"),
    [
        ((0, 0), Branch((0, 1), (ARR,), Goal((0, 1))), "arrived"),
        ((0, 0), Branch((0, 0), (SLIP,), Loop((0, 0))), "slipped"),
        ((0, 0), Branch((0, 0), (SLIP, SKL), Loop((0, 0))), "slipped"),
        ((0, 0), Branch((0, 0), (ARR, SLIP, SKR), Loop((0, 0))), "slipped"),
        ((0, 0), Branch((0, 0), (ARR,), Loop((0, 0))), "blocked"),
        ((0, 0), Branch((0, 0), (SKL, SKR), Loop((0, 0))), "blocked"),
        ((0, 0), Branch((1, 0), (SKR,), Ref((1, 0))), "skid→(2,1)"),
        ((2, 3), Branch((1, 3), (SKL,), Loop((1, 3))), "skid→(2,4)"),
        ((2, 3), Branch((2, 4), (ARR, SKL), Goal((2, 4))), "arrived"),
    ],
)
def test_branch_label(parent: Pos, branch: Branch, expected: str) -> None:
    assert branch_label(parent, branch) == expected


# --- to_tree ---------------------------------------------------------------------------------
def test_to_tree_keys_and_root() -> None:
    tree = three_by_three_plan().to_tree(10)
    for d in walk_tree(tree):
        # §3.10 keys plus "label", which §3.6 keeps as an alias of "edge".
        assert set(d) == TREE_KEYS | {"label"}
        assert d["label"] == d["edge"]
        assert isinstance(d["children"], list)
        assert isinstance(d["text"], str)
    assert tree["state"] == (1, 1)
    assert tree["kind"] == "act"
    assert tree["action"] == "E"
    assert tree["edge"] is None
    assert tree["text"] == "E @ (2,2)"
    assert [c["edge"] for c in tree["children"]] == [
        "arrived",
        "slipped",
        "skid→(1,2)",
        "skid→(3,2)",
    ]


def test_to_tree_leaf_texts() -> None:
    tree = three_by_three_plan().to_tree(10)
    arrived, slipped, skid_l, _skid_r = tree["children"]
    assert arrived["kind"] == "act"
    assert arrived["text"] == "S @ (2,3)"
    assert slipped["kind"] == "loop"
    assert slipped["text"] == "↺ retry E"
    goal_leaf, self_loop, other_loop = arrived["children"]
    assert goal_leaf["kind"] == "goal"
    assert goal_leaf["text"] == "goal (3,3)"
    assert goal_leaf["state"] == (2, 2)
    assert self_loop["kind"] == "loop"
    assert self_loop["text"] == "↺ retry S"
    assert other_loop["kind"] == "loop"
    assert other_loop["text"] == "↺ (2,2)"
    assert other_loop["state"] == (1, 1)
    # skid-L → (0,1) E → (0,2) S → arrived (1,2): a Ref to the S-plan at (2,3).
    at_02 = skid_l["children"][0]
    ref = at_02["children"][0]
    assert ref["kind"] == "ref"
    assert ref["state"] == (1, 2)
    assert ref["edge"] == "arrived"
    assert ref["text"] == "→ S-plan @ (2,3)"
    for leaf in (goal_leaf, self_loop, other_loop, ref):
        assert leaf["children"] == []


def test_to_tree_full_depth_has_no_truncation() -> None:
    plan = three_by_three_plan()
    full = plan.to_tree(100)
    nodes = walk_tree(full)
    assert all(d["kind"] != "truncated" for d in nodes)
    assert len(nodes) == plan.stats().nodes
    assert act_depth(full) == plan.stats().depth


def test_to_tree_truncation() -> None:
    # Contract (docs/plan.md §3.10 clarification): the root Act is at depth 1, so
    # to_tree(d) keeps exactly min(d, stats().depth) Act levels.
    plan = three_by_three_plan()
    full_depth = plan.stats().depth
    policy = plan.to_policy()
    sizes = []
    for depth in range(full_depth + 2):
        tree = plan.to_tree(depth)
        got = act_depth(tree)
        assert got == min(depth, full_depth)
        truncated = [d for d in walk_tree(tree) if d["kind"] == "truncated"]
        if got < full_depth:
            assert truncated
        else:
            assert not truncated
        for d in truncated:
            assert d["children"] == []
            assert d["state"] in policy
            assert d["action"] == str(policy[d["state"]])
        sizes.append(len(walk_tree(tree)))
    assert sizes == sorted(sizes)
    assert sizes[-1] == plan.stats().nodes


def test_to_tree_depth_zero_truncates_root() -> None:
    tree = three_by_three_plan().to_tree(0)
    assert tree["kind"] == "truncated"
    assert tree["state"] == (1, 1)
    assert tree["action"] == "E"
    assert tree["edge"] is None
    assert tree["children"] == []
    one = three_by_three_plan().to_tree(1)
    assert one["kind"] == "act"
    assert {c["kind"] for c in one["children"]} == {"truncated", "loop"}


# --- stats -----------------------------------------------------------------------------------
def test_stats_three_by_three() -> None:
    stats = three_by_three_plan().stats()
    assert isinstance(stats, PlanStats)
    assert stats == PlanStats(nodes=17, act_nodes=5, depth=3, loops=9, refs=1, envelope_size=6)


def test_stats_deck_shape_and_goal_root() -> None:
    assert deck_shape_plan().stats() == PlanStats(
        nodes=10, act_nodes=3, depth=2, loops=5, refs=0, envelope_size=4
    )
    goal_stats = Plan(root=Goal((0, 0)), start=(0, 0), goal=(0, 0)).stats()
    assert goal_stats.nodes == 1
    assert goal_stats.act_nodes == 0
    assert goal_stats.depth == 0
    assert goal_stats.loops == 0
    assert goal_stats.refs == 0
    assert goal_stats.envelope_size == 1


# --- plan_from_policy ------------------------------------------------------------------------
def greedy_policy(model: ProblemModel) -> dict[Pos, Action]:
    """Distance-descending action at every free non-goal cell that can reach the goal."""
    dist = bfs_distances(model.passable, model.goal)
    policy: dict[Pos, Action] = {}
    for s in model.states():
        if s == model.goal or dist[s] == UNREACHABLE:
            continue
        for a in (N, E, S, W):
            t = step_pos(s, a)
            if model.is_free(t) and dist[t] == dist[s] - 1:
                policy[s] = a
                break
    return policy


def reachable_under(policy: dict[Pos, Action], model: ProblemModel, start: Pos) -> set[Pos]:
    """Non-goal states reachable from ``start`` when following ``policy``."""
    seen = {start}
    queue = deque([start])
    while queue:
        s = queue.popleft()
        if model.is_goal(s):
            continue
        for o in model.results(s, policy[s]):
            if o.state not in seen:
                seen.add(o.state)
                queue.append(o.state)
    return {s for s in seen if not model.is_goal(s)}


def _check_unfolded(plan: Plan, model: ProblemModel) -> None:
    policy = plan.to_policy()
    for node in walk_nodes(plan.root):
        assert isinstance(node, Act | Goal | Loop | Ref)
        if isinstance(node, Goal):
            assert node.state == model.goal
        if isinstance(node, Loop | Ref):
            assert node.state in policy
        if isinstance(node, Act):
            got = {(br.state, br.labels) for br in node.branches}
            want = {(o.state, o.labels) for o in model.results(node.state, node.action)}
            assert got == want
            for br in node.branches:
                assert br.node.state == br.state


@pytest.mark.parametrize("p", [0.0, 0.1, 0.2, 0.4])
def test_plan_from_policy_round_trip_deck(p: float) -> None:
    model = deck_model(p)
    start = deck_start()
    policy = greedy_policy(model)
    plan = plan_from_policy(policy, model, start)
    reach = reachable_under(policy, model, start)
    assert plan.start == start
    assert plan.goal == model.goal
    assert isinstance(plan.root, Act)
    assert plan.root.action is policy[start]
    assert plan.to_policy() == {s: policy[s] for s in reach}
    _check_unfolded(plan, model)


def test_plan_from_policy_drops_unreachable_entries() -> None:
    passable = np.ones((4, 5), dtype=bool)
    model = ProblemModel(passable, (3, 0), UniformSlip(0.0))
    policy = greedy_policy(model)
    policy[(3, 4)] = N  # junk entry nowhere near the plan
    plan = plan_from_policy(policy, model, (0, 0))
    out = plan.to_policy()
    assert (3, 4) not in out
    assert out == {s: policy[s] for s in reachable_under(policy, model, (0, 0))}


def test_plan_from_policy_missing_state_raises() -> None:
    model = deck_model(0.2)
    start = deck_start()
    policy = greedy_policy(model)
    reach = reachable_under(policy, model, start)
    victim = next(s for s in sorted(reach) if s != start)
    del policy[victim]
    with pytest.raises(ValueError):
        plan_from_policy(policy, model, start)


def test_plan_from_policy_missing_start_raises() -> None:
    model = deck_model(0.2)
    start = deck_start()
    policy = greedy_policy(model)
    del policy[start]
    with pytest.raises(ValueError):
        plan_from_policy(policy, model, start)


def test_plan_from_policy_start_is_goal() -> None:
    model = deck_model(0.2)
    plan = plan_from_policy({}, model, model.goal)
    assert isinstance(plan.root, Goal)
    assert plan.to_policy() == {}
    assert plan.pretty() == "goal"


# --- deep plans --------------------------------------------------------------------------------
@pytest.mark.parametrize("p", [0.0, 0.2])
def test_deep_plan_has_no_recursion_error(p: float) -> None:
    # A serpentine corridor gives a plan hundreds of Act levels deep. Every plan operation,
    # and the identity methods of the inner nodes, must work at the default recursion limit.
    limit = sys.getrecursionlimit()
    mask = serpentine_mask(DEEP_SIDE)
    start = (0, 0)
    goal = farthest_cell(mask, start)
    model = ProblemModel(mask, goal, UniformSlip(p))
    plan = and_or_search(model, start).plan
    again = and_or_search(ProblemModel(mask, goal, UniformSlip(p)), start).plan
    assert plan is not None and again is not None
    assert sys.getrecursionlimit() == limit  # the search restored it
    stats = plan.stats()
    assert stats.depth > DEEP_MIN_DEPTH
    assert stats.depth > limit // 3  # deep enough to break any recursive traversal
    # Plan-level identity and views.
    assert plan == again
    assert hash(plan) == hash(again)
    assert repr(plan).startswith("Plan(start=(1,1)")
    policy = plan.to_policy()
    assert len(policy) == stats.act_nodes
    assert plan.envelope() >= set(policy)
    assert isinstance(plan.pretty(), str)
    tree = plan.to_tree(10**4)
    assert len(walk_tree(tree)) == stats.nodes
    assert act_depth(tree) == stats.depth
    assert len(list(walk_nodes(plan.root))) == stats.nodes
    assert len(list(walk_branches(plan.root))) == stats.nodes - 1
    # Node-level identity (regression: the generated dataclass methods recursed).
    assert plan.root == again.root
    assert hash(plan.root) == hash(again.root)
    assert len(repr(plan.root)) < 100
    assert isinstance(plan.root, Act)
    assert plan.root.branches == again.root.branches  # type: ignore[union-attr]
    # Copies and pickles use the flat encoding.
    assert pickle.loads(pickle.dumps(plan)) == plan
    assert copy.deepcopy(plan) == plan
    assert pickle.loads(pickle.dumps(plan.root)) == plan.root
    # Unfolding the policy again (the fallback path) gives the same tree.
    unfolded = plan_from_policy(policy, model, start)
    assert unfolded.to_policy() == policy
    assert unfolded.stats().depth == stats.depth
    assert sys.getrecursionlimit() == limit
