"""Tests for rescue.config (docs/plan.md section 3.1)."""

from __future__ import annotations

import dataclasses
from pathlib import Path

import pytest

from rescue import config
from rescue.config import (
    MAX_SLIP_P,
    STEP_BUDGET_FACTOR,
    AftershockConfig,
    AgentConfig,
    PlannerConfig,
    WorldConfig,
    aftershock_config_from_dict,
    agent_config_from_dict,
    load_yaml,
    planner_config_from_dict,
    world_config_from_dict,
)


def test_constants_match_contract() -> None:
    assert config.SLIP_STAY_FRACTION == 0.75
    assert config.SKID_FRACTION == 0.25
    assert config.SLIP_STAY_FRACTION + config.SKID_FRACTION == 1.0
    assert config.DEFAULT_SLIP_P == 0.2
    assert config.MAX_SLIP_P == 0.4
    assert config.DEFAULT_SENSOR_RADIUS == 1
    assert config.STEP_BUDGET_FACTOR == 4
    assert config.DEFAULT_TERRAIN_HIGH_P == 0.4
    assert config.MAPGEN_MAX_ATTEMPTS == 200
    assert config.DEFAULT_RUBBLE_DENSITY == 0.25
    assert config.RUBBLE_DENSITY_RANGE == (0.15, 0.35)
    assert config.PROB_EPS == 1e-12


def test_world_config_defaults() -> None:
    wc = WorldConfig()
    assert wc.slip_p == config.DEFAULT_SLIP_P
    assert wc.sensor_radius == config.DEFAULT_SENSOR_RADIUS
    assert wc.terrain_high_p == config.DEFAULT_TERRAIN_HIGH_P
    assert wc.aftershock == AftershockConfig()
    assert wc.budget_factor == STEP_BUDGET_FACTOR


@pytest.mark.parametrize(("rows", "cols"), [(4, 4), (10, 15), (1, 1), (30, 30)])
def test_step_budget_is_4rc(rows: int, cols: int) -> None:
    assert WorldConfig().step_budget(rows, cols) == 4 * rows * cols
    assert WorldConfig(budget_factor=2).step_budget(rows, cols) == 2 * rows * cols


@pytest.mark.parametrize("p", [0.0, 0.1, 0.2, MAX_SLIP_P])
def test_world_config_accepts_valid_slip(p: float) -> None:
    assert WorldConfig(slip_p=p).slip_p == p


@pytest.mark.parametrize("p", [MAX_SLIP_P + 1e-9, 0.5, 1.0, -0.01])
def test_world_config_rejects_bad_slip(p: float) -> None:
    with pytest.raises(ValueError):
        WorldConfig(slip_p=p)


def test_world_config_rejects_bad_terrain_p() -> None:
    with pytest.raises(ValueError):
        WorldConfig(terrain_high_p=MAX_SLIP_P + 0.1)


def test_world_config_sensor_radius() -> None:
    assert WorldConfig(sensor_radius=0).sensor_radius == 0
    assert WorldConfig(sensor_radius=2).sensor_radius == 2
    with pytest.raises(ValueError):
        WorldConfig(sensor_radius=-1)


def test_configs_are_frozen() -> None:
    for obj, attr, value in [
        (WorldConfig(), "slip_p", 0.1),
        (AftershockConfig(), "q", 0.5),
        (PlannerConfig(), "validate", True),
        (AgentConfig(), "replan_mode", "lazy"),
    ]:
        with pytest.raises(dataclasses.FrozenInstanceError):
            setattr(obj, attr, value)


def test_aftershock_config() -> None:
    assert AftershockConfig() == AftershockConfig(q=0.0, k=1)
    assert not AftershockConfig().enabled
    assert AftershockConfig(q=0.5, k=2).enabled
    assert not AftershockConfig(q=0.5, k=0).enabled
    for bad in ({"q": -0.1}, {"q": 1.1}, {"k": -1}):
        with pytest.raises(ValueError):
            AftershockConfig(**bad)


def test_planner_config() -> None:
    pc = PlannerConfig()
    assert pc.depth_limit is None
    assert pc.tie_break == "diagonal"
    assert pc.validate is False
    assert pc.fallback_to_reference is True
    assert PlannerConfig(tie_break="fixed").tie_break == "fixed"
    with pytest.raises(ValueError):
        PlannerConfig(tie_break="random")  # type: ignore[arg-type]
    with pytest.raises(ValueError):
        PlannerConfig(depth_limit=-1)


def test_agent_config() -> None:
    assert AgentConfig().replan_mode == "eager"
    assert AgentConfig(replan_mode="lazy").replan_mode == "lazy"
    with pytest.raises(ValueError):
        AgentConfig(replan_mode="sometimes")  # type: ignore[arg-type]


def test_world_config_from_dict_nested_aftershock() -> None:
    wc = world_config_from_dict(
        {"slip_p": 0.3, "sensor_radius": 2, "aftershock": {"q": 0.1, "k": 3}}
    )
    assert wc.slip_p == 0.3
    assert wc.sensor_radius == 2
    assert isinstance(wc.aftershock, AftershockConfig)
    assert wc.aftershock == AftershockConfig(q=0.1, k=3)


def test_from_dict_defaults() -> None:
    assert world_config_from_dict({}) == WorldConfig()
    assert world_config_from_dict(None) == WorldConfig()
    assert planner_config_from_dict({}) == PlannerConfig()
    assert agent_config_from_dict({}) == AgentConfig()
    assert aftershock_config_from_dict({}) == AftershockConfig()


def test_from_dict_rejects_unknown_keys() -> None:
    with pytest.raises(ValueError):
        world_config_from_dict({"slip": 0.2})
    with pytest.raises(ValueError):
        world_config_from_dict({"aftershock": {"q": 0.1, "count": 2}})
    with pytest.raises(ValueError):
        planner_config_from_dict({"depth": 3})
    with pytest.raises(ValueError):
        agent_config_from_dict({"mode": "lazy"})
    with pytest.raises(ValueError):
        aftershock_config_from_dict({"prob": 0.1})


def test_from_dict_validates_values() -> None:
    with pytest.raises(ValueError):
        world_config_from_dict({"slip_p": 0.9})
    with pytest.raises(ValueError):
        agent_config_from_dict({"replan_mode": "never"})
    with pytest.raises(ValueError):
        planner_config_from_dict({"tie_break": "coin"})


def test_planner_and_agent_from_dict() -> None:
    pc = planner_config_from_dict({"depth_limit": 7, "tie_break": "fixed", "validate": True})
    assert pc == PlannerConfig(depth_limit=7, tie_break="fixed", validate=True)
    assert agent_config_from_dict({"replan_mode": "lazy"}) == AgentConfig(replan_mode="lazy")


def test_load_yaml(tmp_path: Path) -> None:
    path = tmp_path / "cfg.yaml"
    path.write_text(
        "world:\n  slip_p: 0.3\n  aftershock:\n    q: 0.2\n    k: 2\nagent:\n  replan_mode: lazy\n",
        encoding="utf-8",
    )
    data = load_yaml(path)
    assert data == {
        "world": {"slip_p": 0.3, "aftershock": {"q": 0.2, "k": 2}},
        "agent": {"replan_mode": "lazy"},
    }
    assert load_yaml(str(path)) == data
    wc = world_config_from_dict(data["world"])
    assert wc.aftershock == AftershockConfig(q=0.2, k=2)


def test_load_yaml_empty_and_non_mapping(tmp_path: Path) -> None:
    empty = tmp_path / "empty.yaml"
    empty.write_text("", encoding="utf-8")
    assert load_yaml(empty) == {}
    listy = tmp_path / "list.yaml"
    listy.write_text("- 1\n- 2\n", encoding="utf-8")
    with pytest.raises(ValueError):
        load_yaml(listy)
