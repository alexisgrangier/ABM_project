# environment/grid.py
"""
100 × 100 spatial grid with zone partitions.
ODD Protocol §2.3: Urban core (40×40), Dense periphery (15×20), Sparse periphery (rest).
"""

from __future__ import annotations
import numpy as np
from enum import Enum
from typing import List, Tuple, Optional


class ZoneType(Enum):
    URBAN_CORE       = "urban_core"
    DENSE_PERIPHERY  = "dense_periphery"
    SPARSE_PERIPHERY = "sparse_periphery"


class Grid:
    """
    Spatial environment for the SEIRD ABM.

    Attributes
    ──────────
    width, height       : grid dimensions (100 × 100)
    max_per_cell        : maximum agents per cell (5)
    zone_map            : (height × width) array of ZoneType values
    occupancy           : (height × width) int array – current agent counts
    agent_positions     : dict mapping agent_id → (x, y)
    cell_agents         : dict mapping (x, y) → list[agent_id]
    """

    def __init__(
        self,
        width:        int = 100,
        height:       int = 100,
        max_per_cell: int = 5,
    ) -> None:
        self.width        = width
        self.height       = height
        self.max_per_cell = max_per_cell

        self.zone_map: np.ndarray = np.full(
            (height, width), ZoneType.SPARSE_PERIPHERY, dtype=object
        )
        self.occupancy: np.ndarray         = np.zeros((height, width), dtype=int)
        self.agent_positions: dict         = {}   # agent_id → (x, y)
        self.cell_agents: dict             = {}   # (x, y)  → [agent_id, ...]

        self._assign_zones()

    # ── Zone assignment ───────────────────────────────────────────────────────

    def _assign_zones(self) -> None:
        """
        Urban core   : central 40 × 40 block  (cols 30-69, rows 30-69)
        Dense periph : top-left 15 × 20 block  (cols 0-14,  rows 0-19)
        Sparse periph: everything else
        """
        # Urban core
        self.zone_map[30:70, 30:70] = ZoneType.URBAN_CORE
        # Dense periphery (overwrites any overlap – there is none by construction)
        self.zone_map[0:20, 0:15]   = ZoneType.DENSE_PERIPHERY

    def zone_at(self, x: int, y: int) -> ZoneType:
        return self.zone_map[y, x]

    # ── Cell helpers ──────────────────────────────────────────────────────────

    def is_within_bounds(self, x: int, y: int) -> bool:
        return 0 <= x < self.width and 0 <= y < self.height

    def is_cell_available(self, x: int, y: int) -> bool:
        return self.occupancy[y, x] < self.max_per_cell

    def get_neighbors(
        self, x: int, y: int, radius: int = 1
    ) -> List[Tuple[int, int]]:
        """Return all valid (x, y) cells within Moore neighbourhood of given radius."""
        neighbors = []
        for dx in range(-radius, radius + 1):
            for dy in range(-radius, radius + 1):
                nx, ny = x + dx, y + dy
                if (dx, dy) != (0, 0) and self.is_within_bounds(nx, ny):
                    neighbors.append((nx, ny))
        return neighbors

    def get_agents_in_radius(
        self, x: int, y: int, radius: int = 1
    ) -> List[int]:
        """Return agent IDs in the cell (x,y) and its Moore neighbourhood."""
        cells = [(x, y)] + self.get_neighbors(x, y, radius)
        agents = []
        for cell in cells:
            agents.extend(self.cell_agents.get(cell, []))
        return agents

    # ── Agent placement ───────────────────────────────────────────────────────

    def place_agent(self, agent_id: int, x: int, y: int) -> bool:
        """Place an agent at (x, y). Returns False if the cell is full."""
        if not self.is_within_bounds(x, y):
            return False
        if not self.is_cell_available(x, y):
            return False
        self.agent_positions[agent_id] = (x, y)
        self.cell_agents.setdefault((x, y), []).append(agent_id)
        self.occupancy[y, x] += 1
        return True

    def move_agent(self, agent_id: int, new_x: int, new_y: int) -> bool:
        """Move an agent to (new_x, new_y). Returns False if destination is full."""
        if agent_id not in self.agent_positions:
            return False
        if not self.is_cell_available(new_x, new_y):
            return False
        old_x, old_y = self.agent_positions[agent_id]
        # Remove from old cell
        self.cell_agents[(old_x, old_y)].remove(agent_id)
        self.occupancy[old_y, old_x] -= 1
        # Place in new cell
        return self.place_agent(agent_id, new_x, new_y)

    def remove_agent(self, agent_id: int) -> None:
        if agent_id in self.agent_positions:
            x, y = self.agent_positions.pop(agent_id)
            self.cell_agents[(x, y)].remove(agent_id)
            self.occupancy[y, x] -= 1

    # ── Sampling helpers ──────────────────────────────────────────────────────

    def random_cell_in_zone(
        self, zone: ZoneType, rng: np.random.Generator
    ) -> Tuple[int, int]:
        """Sample a random available cell in the given zone."""
        ys, xs = np.where(self.zone_map == zone)
        candidates = list(zip(xs.tolist(), ys.tolist()))
        rng.shuffle(candidates)
        for x, y in candidates:
            if self.is_cell_available(x, y):
                return (x, y)
        raise RuntimeError(f"No available cells in zone {zone.value}")

    # ── Summary ───────────────────────────────────────────────────────────────

    def zone_cell_counts(self) -> dict:
        counts = {}
        for zone in ZoneType:
            counts[zone.value] = int(np.sum(self.zone_map == zone))
        return counts

    def __repr__(self) -> str:
        return (
            f"Grid({self.width}×{self.height}, "
            f"agents_placed={len(self.agent_positions)}, "
            f"max_per_cell={self.max_per_cell})"
        )
