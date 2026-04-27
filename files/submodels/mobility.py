# submodels/mobility.py
"""
Mobility submodel — Option C hybrid commute (ODD §7.1).

Morning tick : susceptible / exposed / asymptomatic agents may commute to work.
Evening tick : agents at work return home.

Commute is instantaneous (teleport); transmission during commute is handled
separately in transmission.py using transport multipliers.
"""

from __future__ import annotations
import numpy as np

from agents.individual import Individual, EpiState
from agents.health_ministry import HealthMinistry, AlertState
from environment.grid import Grid
from config.parameters import POLICY_COMMUTE_REDUCTION


# ── Policy-level commute reduction map ────────────────────────────────────────

_ALERT_REDUCTION: dict[AlertState, float] = {
    AlertState.NO_ALERT: POLICY_COMMUTE_REDUCTION["no_alert"],
    AlertState.ALERT_1:  POLICY_COMMUTE_REDUCTION["alert_1"],
    AlertState.ALERT_2:  POLICY_COMMUTE_REDUCTION["alert_2"],
    AlertState.ALERT_3:  POLICY_COMMUTE_REDUCTION["alert_3"],
}


def _will_commute(
    agent: Individual,
    alert_state: AlertState,
    rng: np.random.Generator,
) -> bool:
    """
    Decide whether an agent commutes to work this morning (ODD §7.1).

    Rules
    -----
    - Dead agents never commute.
    - Symptomatic agents commute with probability 0.1 (irrespective of policy).
    - Healthy / asymptomatic agents may skip commute based on:
        base_reduction * agent.sensibility_plan
    """
    if not agent.is_alive:
        return False

    if agent.epi_state == EpiState.INFECTIOUS_SYMPTOMATIC:
        return rng.random() < 0.10   # 10 % of symptomatic still go to work

    # Policy-driven reduction scaled by individual sensibility
    base_reduction = _ALERT_REDUCTION[alert_state]
    effective_prob_skip = base_reduction * agent.sensibility_plan

    return rng.random() >= effective_prob_skip


def morning_commute(
    individuals: list[Individual],
    grid: Grid,
    health_ministry: HealthMinistry,
    rng: np.random.Generator,
) -> list[Individual]:
    """
    Move eligible agents from home → work cell (ODD §3, §7.1).

    Returns the list of agents who successfully commuted (used by
    transmission submodel for commute-phase exposure events).
    """
    commuters: list[Individual] = []

    for agent in individuals:
        if agent.at_work:
            continue   # already at work (shouldn't happen at morning tick)

        if _will_commute(agent, health_ministry.alert_state, rng):
            success = grid.move_agent(agent.agent_id, *agent.work_position)
            if success:
                agent.position = agent.work_position
                agent.at_work  = True
                commuters.append(agent)

    return commuters


def evening_commute(
    individuals: list[Individual],
    grid: Grid,
) -> list[Individual]:
    """
    Move all agents currently at work back to their home cell (ODD §3).

    Returns the list of agents who returned home.
    """
    returnees: list[Individual] = []

    for agent in individuals:
        if agent.at_work and agent.is_alive:
            success = grid.move_agent(agent.agent_id, *agent.home_position)
            if success:
                agent.position = agent.home_position
                agent.at_work  = False
                returnees.append(agent)

    return returnees
