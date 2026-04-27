# submodels/disease_progression.py
"""
Disease progression submodel (ODD §7.3).

Stochastic SEIRD state transitions driven by agent tick counters
and age-specific fatality rates.

Transition schedule (1 tick = 12 h):
  EXPOSED              → INFECTIOUS_ASYMPTOMATIC | INFECTIOUS_SYMPTOMATIC
                         after INCUBATION_PERIOD_TICKS
  INFECTIOUS_*         → RECOVERED | DEAD
                         after INFECTIOUS_PERIOD_TICKS
  RECOVERED            → SUSCEPTIBLE
                         after immunity_ticks reach 0
"""

from __future__ import annotations
import numpy as np

from agents.individual import Individual, EpiState, AgeGroup
from config.parameters import (
    INCUBATION_PERIOD_TICKS,
    INFECTIOUS_PERIOD_TICKS,
    ASYMPTOMATIC_FRACTION,
    RECOVERY_IMMUNITY_DAYS,
    TICKS_PER_DAY,
    AGE_GROUPS,
)

# Pre-compute immunity duration in ticks
_IMMUNITY_TICKS = RECOVERY_IMMUNITY_DAYS * TICKS_PER_DAY

# Age-group fatality map
_FATALITY_RATE: dict[AgeGroup, float] = {
    AgeGroup[g.upper()]: AGE_GROUPS[g]["fatality_rate"]
    for g in AGE_GROUPS
}


def apply_disease_progression(
    individuals: list[Individual],
    rng: np.random.Generator,
) -> dict[str, int]:
    """
    Advance every agent's disease state by one tick (ODD §7.3).

    Returns
    -------
    dict with keys 'new_infectious', 'new_recovered', 'new_dead'
    """
    counts = {"new_infectious": 0, "new_recovered": 0, "new_dead": 0}

    for agent in individuals:
        state = agent.epi_state

        # ── EXPOSED → INFECTIOUS ──────────────────────────────────────────
        if state == EpiState.EXPOSED:
            if agent.ticks_in_state >= INCUBATION_PERIOD_TICKS:
                if rng.random() < ASYMPTOMATIC_FRACTION:
                    agent.transition_to(EpiState.INFECTIOUS_ASYMPTOMATIC)
                else:
                    agent.transition_to(EpiState.INFECTIOUS_SYMPTOMATIC)
                counts["new_infectious"] += 1

        # ── INFECTIOUS → RECOVERED | DEAD ────────────────────────────────
        elif state in (EpiState.INFECTIOUS_ASYMPTOMATIC, EpiState.INFECTIOUS_SYMPTOMATIC):
            if agent.ticks_in_state >= INFECTIOUS_PERIOD_TICKS:
                fatality_rate = _FATALITY_RATE[agent.age_group]
                if rng.random() < fatality_rate:
                    agent.transition_to(EpiState.DEAD)
                    counts["new_dead"] += 1
                else:
                    agent.transition_to(EpiState.RECOVERED)
                    agent.immunity_ticks = _IMMUNITY_TICKS
                    counts["new_recovered"] += 1

        # ── RECOVERED → SUSCEPTIBLE (waning immunity) ─────────────────────
        elif state == EpiState.RECOVERED:
            if agent.immunity_ticks <= 0:
                agent.transition_to(EpiState.SUSCEPTIBLE)

        # Advance internal tick counter for all alive agents
        if agent.is_alive:
            agent.tick()

    return counts
