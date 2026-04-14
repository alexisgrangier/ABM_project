# agents/individual.py
"""
Individual agent class for the SEIRD ABM.
State variables and attributes are drawn from the ODD Protocol (Section 2.1).
"""

from __future__ import annotations
import numpy as np
from enum import Enum, auto
from dataclasses import dataclass, field
from typing import Optional


class EpiState(Enum):
    SUSCEPTIBLE             = auto()
    EXPOSED                 = auto()   # incubating; contagious
    INFECTIOUS_ASYMPTOMATIC = auto()   # contagious, unaware
    INFECTIOUS_SYMPTOMATIC  = auto()   # contagious, aware
    RECOVERED               = auto()   # temporarily immune
    DEAD                    = auto()


class TransportMode(Enum):
    PUBLIC_TRANSIT = "public_transit"
    CAR            = "car"
    WALKING        = "walking"


class AgeGroup(Enum):
    CHILD  = "child"
    YOUNG  = "young"
    ADULT  = "adult"
    SENIOR = "senior"


class ResidenceZone(Enum):
    DENSE_PERIPHERY  = "dense_periphery"
    SPARSE_PERIPHERY = "sparse_periphery"


@dataclass
class Individual:
    """
    Represents a single resident agent.

    Attributes (ODD §2.1)
    ─────────────────────
    agent_id          : unique identifier
    residence         : residential zone (fixed at init)
    work_place        : work zone (always urban_core, fixed at init)
    age_group         : affects fatality rate
    transport_pref    : preferred commute mode
    sensibility_plan  : responsiveness to public-health campaigns [0, 1]
    doctor            : whether agent visits a doctor when symptomatic

    Dynamic state
    ─────────────
    epi_state         : current SEIRD state
    position          : (x, y) cell on the 100×100 grid
    home_position     : fixed home cell
    ticks_in_state    : ticks spent in the current epi_state
    immunity_ticks    : ticks of remaining post-recovery immunity
    has_seen_doctor   : True once per infection cycle (resets on recovery)
    at_work           : True when the agent is at the work location
    """

    # Fixed attributes
    agent_id:         int
    residence:        ResidenceZone
    work_place:       str                  = "urban_core"
    age_group:        AgeGroup             = AgeGroup.ADULT
    transport_pref:   TransportMode        = TransportMode.PUBLIC_TRANSIT
    sensibility_plan: float                = 0.5   # ∈ [0, 1]
    doctor:           bool                 = False

    # Spatial state
    home_position:    tuple[int, int]      = field(default_factory=lambda: (0, 0))
    position:         tuple[int, int]      = field(default_factory=lambda: (0, 0))
    work_position:    tuple[int, int]      = field(default_factory=lambda: (50, 50))
    at_work:          bool                 = False

    # Epidemiological state
    epi_state:        EpiState             = EpiState.SUSCEPTIBLE
    ticks_in_state:   int                  = 0
    immunity_ticks:   int                  = 0
    has_seen_doctor:  bool                 = False

    # ── convenience properties ───────────────────────────────────────────────

    @property
    def is_infectious(self) -> bool:
        return self.epi_state in (
            EpiState.INFECTIOUS_ASYMPTOMATIC,
            EpiState.INFECTIOUS_SYMPTOMATIC,
        )

    @property
    def is_contagious(self) -> bool:
        """Exposed agents are also contagious (pre-symptomatic transmission)."""
        return self.epi_state in (
            EpiState.EXPOSED,
            EpiState.INFECTIOUS_ASYMPTOMATIC,
            EpiState.INFECTIOUS_SYMPTOMATIC,
        )

    @property
    def is_alive(self) -> bool:
        return self.epi_state != EpiState.DEAD

    @property
    def can_commute(self) -> bool:
        return self.is_alive and self.epi_state not in (EpiState.DEAD,)

    # ── state transition helper ──────────────────────────────────────────────

    def transition_to(self, new_state: EpiState) -> None:
        self.epi_state      = new_state
        self.ticks_in_state = 0
        if new_state == EpiState.SUSCEPTIBLE:
            self.has_seen_doctor = False

    def tick(self) -> None:
        """Advance internal counters by one tick."""
        self.ticks_in_state += 1
        if self.immunity_ticks > 0:
            self.immunity_ticks -= 1

    # ── repr ─────────────────────────────────────────────────────────────────

    def __repr__(self) -> str:
        return (
            f"Individual(id={self.agent_id}, "
            f"state={self.epi_state.name}, "
            f"zone={self.residence.value}, "
            f"age={self.age_group.value})"
        )
