# submodels/transmission.py
"""
Transmission submodel (ODD §7.2, §4.5).

Transmission can occur in three contexts:
  1. At work / home  — spatial contact within Moore radius r=1
  2. During commute  — abstract event using transport multipliers (Option C)

The probability of a susceptible agent s becoming exposed by
an infectious neighbour i is:

    p = beta * seasonal_modifier * transport_multiplier * compliance_factor

where compliance_factor = (1 - sensibility * alert_reduction) for s.
"""

from __future__ import annotations
import numpy as np

from agents.individual import Individual, EpiState
from agents.health_ministry import HealthMinistry, AlertState
from environment.grid import Grid
from config.parameters import (
    BASE_TRANSMISSION_RATE,
    CONTACT_RADIUS,
    TRANSPORT_PREFS,
    POLICY_COMMUTE_REDUCTION,
)


# ── Helper maps ───────────────────────────────────────────────────────────────

_TRANSPORT_MULTIPLIER: dict[str, float] = {
    mode: TRANSPORT_PREFS[mode]["transmission_multiplier"]
    for mode in TRANSPORT_PREFS
}

_ALERT_REDUCTION: dict[AlertState, float] = {
    AlertState.NO_ALERT: POLICY_COMMUTE_REDUCTION["no_alert"],
    AlertState.ALERT_1:  POLICY_COMMUTE_REDUCTION["alert_1"],
    AlertState.ALERT_2:  POLICY_COMMUTE_REDUCTION["alert_2"],
    AlertState.ALERT_3:  POLICY_COMMUTE_REDUCTION["alert_3"],
}


def _compliance_factor(agent: Individual, alert_state: AlertState) -> float:
    """
    Protective compliance of a susceptible agent.
    Returns a multiplier in (0, 1] — lower = more protected.
    """
    reduction = _ALERT_REDUCTION[alert_state] * agent.sensibility_plan
    return max(0.0, 1.0 - reduction)


def _transmission_prob(
    beta: float,
    seasonal_mod: float,
    transport_mult: float,
    compliance: float,
) -> float:
    """Clamp final probability to [0, 1]."""
    return min(1.0, beta * seasonal_mod * transport_mult * compliance)


# ── Spatial transmission (home / work) ────────────────────────────────────────

def spatial_transmission(
    individuals: list[Individual],
    grid: Grid,
    health_ministry: HealthMinistry,
    seasonal_mod: float,
    rng: np.random.Generator,
    beta: float = BASE_TRANSMISSION_RATE,
) -> int:
    """
    For every susceptible agent, check contagious neighbours within radius r=1.
    Each contagious neighbour makes an independent Bernoulli trial.

    Returns
    -------
    int  number of new exposures this call
    """
    new_exposures = 0
    alert_state   = health_ministry.alert_state

    # Build a lookup: agent_id → Individual for fast access
    agent_lookup: dict[int, Individual] = {a.agent_id: a for a in individuals}

    for agent in individuals:
        if agent.epi_state != EpiState.SUSCEPTIBLE:
            continue

        x, y = agent.position
        neighbor_ids = grid.get_agents_in_radius(x, y, CONTACT_RADIUS)

        # Count unique contagious contacts
        for nid in neighbor_ids:
            if nid == agent.agent_id:
                continue
            neighbor = agent_lookup.get(nid)
            if neighbor is None or not neighbor.is_contagious:
                continue

            compliance = _compliance_factor(agent, alert_state)
            prob       = _transmission_prob(beta, seasonal_mod, 1.0, compliance)

            if rng.random() < prob:
                agent.transition_to(EpiState.EXPOSED)
                new_exposures += 1
                break   # agent can only be exposed once per tick

    return new_exposures


# ── Commute transmission (Option C — abstract event) ─────────────────────────

def commute_transmission(
    commuters: list[Individual],
    health_ministry: HealthMinistry,
    seasonal_mod: float,
    rng: np.random.Generator,
    beta: float = BASE_TRANSMISSION_RATE,
) -> int:
    """
    Abstract commute-phase exposure events (ODD §7.2, Option C).

    For each susceptible commuter, a single Bernoulli trial is drawn
    using their transport multiplier as a proxy for crowding level.

    The probability scales with the fraction of currently infectious
    commuters (approximated from the commuter list itself).

    Returns
    -------
    int  number of new exposures during commute
    """
    if not commuters:
        return 0

    alert_state = health_ministry.alert_state

    # Fraction of contagious among commuters
    n_contagious     = sum(1 for c in commuters if c.is_contagious)
    contagious_ratio = n_contagious / len(commuters) if commuters else 0.0

    new_exposures = 0

    for agent in commuters:
        if agent.epi_state != EpiState.SUSCEPTIBLE:
            continue

        transport_mult = _TRANSPORT_MULTIPLIER[agent.transport_pref.value]
        compliance     = _compliance_factor(agent, alert_state)
        prob           = _transmission_prob(
            beta, seasonal_mod, transport_mult * contagious_ratio, compliance
        )

        if rng.random() < prob:
            agent.transition_to(EpiState.EXPOSED)
            new_exposures += 1

    return new_exposures
