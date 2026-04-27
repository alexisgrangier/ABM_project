# submodels/medical.py
"""
Medical consultation and reporting submodel (ODD §7.4).

On the first morning of being symptomatic, an agent may decide to
visit a doctor. Doctors always report to the Health Ministry.
This updates the confirmed case count, which drives the policy response.
"""

from __future__ import annotations
import numpy as np

from agents.individual import Individual, EpiState
from agents.health_ministry import HealthMinistry


def apply_medical_consultation(
    individuals: list[Individual],
    health_ministry: HealthMinistry,
    rng: np.random.Generator,
) -> int:
    """
    Process doctor visits for newly symptomatic agents (ODD §7.4).

    An agent visits a doctor if:
      - currently INFECTIOUS_SYMPTOMATIC
      - has not yet seen a doctor this infection cycle
      - has doctor == True (drawn at initialization)

    Returns
    -------
    int  number of new confirmed cases reported today
    """
    new_confirmed = 0

    for agent in individuals:
        if agent.epi_state != EpiState.INFECTIOUS_SYMPTOMATIC:
            continue
        if agent.has_seen_doctor:
            continue
        if not agent.doctor:
            continue

        # First tick of symptoms → doctor visit decision (always True here
        # because `doctor` flag already encoded the probability at init)
        agent.has_seen_doctor = True
        new_confirmed += 1

    return new_confirmed


def daily_report_to_ministry(
    health_ministry: HealthMinistry,
    new_confirmed: int,
) -> None:
    """
    Submit today's confirmed cases to the Health Ministry (ODD §7.4, §7.5).
    Called once per day (every 2 ticks).
    """
    health_ministry.receive_daily_report(new_confirmed)
    health_ministry.update_policy()
