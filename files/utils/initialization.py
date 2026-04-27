# utils/initialization.py
"""
Initialization utilities for the SEIRD ABM.
Extracted from 01_model_initialization.ipynb (ODD §5).
"""

from __future__ import annotations
import numpy as np
from collections import Counter

from agents.individual import Individual, EpiState, TransportMode, AgeGroup, ResidenceZone
from agents.health_ministry import HealthMinistry, AlertState
from environment.grid import Grid, ZoneType
from config.parameters import (
    POPULATION_SIZE, GRID_WIDTH, GRID_HEIGHT, MAX_AGENTS_PER_CELL,
    RESIDENCE_FRACTION_DENSE_PERIPHERY, WORK_ZONE,
    AGE_GROUPS, TRANSPORT_PREFS, SENSIBILITY_PRIOR, DOCTOR_VISIT_PROB,
    URBAN_CORE, INITIAL_EXPOSED_FRACTION, INITIAL_INFECTED_FRACTION,
    ASYMPTOMATIC_FRACTION, POLICY_THRESHOLD_ALERT_1, POLICY_THRESHOLD_ALERT_2,
    POLICY_THRESHOLD_ALERT_3, MIN_POLICY_DURATION_DAYS, TICKS_PER_DAY,
)


# ── Attribute helpers ──────────────────────────────────────────────────────────

def assign_residence(rng: np.random.Generator) -> ResidenceZone:
    """Draw residential zone from population fractions (ODD §5)."""
    if rng.random() < RESIDENCE_FRACTION_DENSE_PERIPHERY:
        return ResidenceZone.DENSE_PERIPHERY
    return ResidenceZone.SPARSE_PERIPHERY


def assign_age_group(rng: np.random.Generator) -> AgeGroup:
    """Multinomial draw over age groups using configured fractions."""
    groups = list(AGE_GROUPS.keys())
    probs  = [AGE_GROUPS[g]["fraction"] for g in groups]
    idx    = rng.choice(len(groups), p=probs)
    return AgeGroup[groups[idx].upper()]


def assign_transport(rng: np.random.Generator) -> TransportMode:
    """Multinomial draw over transport modes."""
    modes = list(TRANSPORT_PREFS.keys())
    probs = [TRANSPORT_PREFS[m]["fraction"] for m in modes]
    idx   = rng.choice(len(modes), p=probs)
    return TransportMode(modes[idx])


def assign_sensibility(residence: ResidenceZone, rng: np.random.Generator) -> float:
    """Beta draw; dense-periphery residents have lower mean sensibility (ODD §4.7)."""
    prior = SENSIBILITY_PRIOR[residence.value]
    return float(rng.beta(prior["alpha"], prior["beta"]))


def assign_work_cell(rng: np.random.Generator) -> tuple[int, int]:
    """Random cell inside the urban core."""
    x = int(rng.integers(URBAN_CORE["x_start"], URBAN_CORE["x_end"]))
    y = int(rng.integers(URBAN_CORE["y_start"], URBAN_CORE["y_end"]))
    return (x, y)


# ── Grid factory ──────────────────────────────────────────────────────────────

def build_grid() -> Grid:
    """Create and return an initialized 100×100 grid."""
    return Grid(
        width=GRID_WIDTH,
        height=GRID_HEIGHT,
        max_per_cell=MAX_AGENTS_PER_CELL,
    )


# ── Agent factory ─────────────────────────────────────────────────────────────

ZONE_MAP = {
    ResidenceZone.DENSE_PERIPHERY: ZoneType.DENSE_PERIPHERY,
    ResidenceZone.SPARSE_PERIPHERY: ZoneType.SPARSE_PERIPHERY,
}


def build_agents(
    grid: Grid,
    rng:  np.random.Generator,
    population_size: int = POPULATION_SIZE,
) -> list[Individual]:
    """
    Create and place all Individual agents on the grid (ODD §5).

    Returns
    -------
    list[Individual]
        All created agents, already placed at their home cells.
    """
    individuals: list[Individual] = []
    placement_failures = 0

    for agent_id in range(population_size):
        residence    = assign_residence(rng)
        age_group    = assign_age_group(rng)
        transport    = assign_transport(rng)
        sensibility  = assign_sensibility(residence, rng)
        doctor       = bool(rng.random() < DOCTOR_VISIT_PROB)
        work_pos     = assign_work_cell(rng)

        # Place agent in home zone; fall back to sparse if dense is full
        try:
            home_pos = grid.random_cell_in_zone(ZONE_MAP[residence], rng)
        except RuntimeError:
            placement_failures += 1
            home_pos = grid.random_cell_in_zone(ZoneType.SPARSE_PERIPHERY, rng)

        agent = Individual(
            agent_id      = agent_id,
            residence     = residence,
            work_place    = WORK_ZONE,
            age_group     = age_group,
            transport_pref= transport,
            sensibility_plan = sensibility,
            doctor        = doctor,
            home_position = home_pos,
            position      = home_pos,
            work_position = work_pos,
        )
        grid.place_agent(agent_id, *home_pos)
        individuals.append(agent)

    if placement_failures:
        print(f"[init] ⚠ {placement_failures} fallback placements (dense periphery saturated)")

    return individuals


# ── Disease seeding ───────────────────────────────────────────────────────────

def seed_epidemic(
    individuals: list[Individual],
    rng: np.random.Generator,
    exposed_fraction: float  = INITIAL_EXPOSED_FRACTION,
    infected_fraction: float = INITIAL_INFECTED_FRACTION,
    asymptomatic_fraction: float = ASYMPTOMATIC_FRACTION,
) -> None:
    """
    Assign initial epidemic states to a random subset of agents (ODD §5).
    All other agents remain SUSCEPTIBLE.
    """
    n = len(individuals)
    n_exposed  = max(1, int(n * exposed_fraction))
    n_infected = max(1, int(n * infected_fraction))
    n_seed     = n_exposed + n_infected

    seed_ids     = rng.choice(n, size=n_seed, replace=False)
    exposed_ids  = seed_ids[:n_exposed]
    infected_ids = seed_ids[n_exposed:]

    for aid in exposed_ids:
        individuals[aid].transition_to(EpiState.EXPOSED)

    for aid in infected_ids:
        if rng.random() < asymptomatic_fraction:
            individuals[aid].transition_to(EpiState.INFECTIOUS_ASYMPTOMATIC)
        else:
            individuals[aid].transition_to(EpiState.INFECTIOUS_SYMPTOMATIC)


# ── Health Ministry factory ───────────────────────────────────────────────────

def build_health_ministry(population_size: int = POPULATION_SIZE) -> HealthMinistry:
    """Instantiate Health Ministry with ODD-specified thresholds."""
    return HealthMinistry(
        alert_state              = AlertState.NO_ALERT,
        min_policy_duration_ticks= MIN_POLICY_DURATION_DAYS * TICKS_PER_DAY,
        threshold_alert_1        = POLICY_THRESHOLD_ALERT_1,
        threshold_alert_2        = POLICY_THRESHOLD_ALERT_2,
        threshold_alert_3        = POLICY_THRESHOLD_ALERT_3,
        population_size          = population_size,
    )


# ── Sanity checks ─────────────────────────────────────────────────────────────

def run_sanity_checks(
    individuals: list[Individual],
    grid: Grid,
    health_ministry: HealthMinistry,
) -> None:
    """Assert basic invariants after initialization."""
    assert len(individuals) == len(grid.agent_positions), \
        "Not all agents placed on grid"
    assert all(a.epi_state != EpiState.RECOVERED for a in individuals), \
        "No agent should start recovered"
    assert health_ministry.alert_state == AlertState.NO_ALERT, \
        "Ministry should start with no active policy"
    assert all(
        grid.occupancy[y, x] <= grid.max_per_cell
        for (x, y) in grid.agent_positions.values()
    ), "Cell capacity exceeded"
    print("[init] ✓ All sanity checks passed.")
