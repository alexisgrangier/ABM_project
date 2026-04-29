# config/parameters.py
"""
All fixed parameters for the SEIRD ABM.
Drawn from the ODD Protocol specification.
"""

# ── Simulation Scale ────────────────────────────────────────────────────────
POPULATION_SIZE   = 5_000
GRID_WIDTH        = 100
GRID_HEIGHT       = 100
MAX_AGENTS_PER_CELL = 2
TICKS_PER_DAY     = 2          # 1 tick = 12 hours
SIMULATION_DAYS   = 365
TOTAL_TICKS       = TICKS_PER_DAY * SIMULATION_DAYS  # 730

# ── Spatial Zones ───────────────────────────────────────────────────────────
URBAN_CORE = {
    "x_start": 30, "x_end": 70,   # central 40 columns
    "y_start": 30, "y_end": 70,   # central 40 rows
}
DENSE_PERIPHERY = {
    "x_start": 0, "x_end": 15,    # 15 columns
    "y_start": 0, "y_end": 20,    # 20 rows  →  15×20 cells
}
# Sparse periphery = everything else

# ── Population Distribution ─────────────────────────────────────────────────
# Fractions of the residential population per zone
RESIDENCE_FRACTION_DENSE_PERIPHERY  = 0.35
RESIDENCE_FRACTION_SPARSE_PERIPHERY = 0.65
# All agents work in the urban core
WORK_ZONE = "urban_core"

# ── Age Groups & Fatality Rates ─────────────────────────────────────────────
AGE_GROUPS = {
    "child":  {"fraction": 0.15, "fatality_rate": 0.001},
    "young":  {"fraction": 0.30, "fatality_rate": 0.003},
    "adult":  {"fraction": 0.35, "fatality_rate": 0.010},
    "senior": {"fraction": 0.20, "fatality_rate": 0.050},
}

# ── Transport Preferences ───────────────────────────────────────────────────
TRANSPORT_PREFS = {
    "public_transit": {"fraction": 0.50, "transmission_multiplier": 2.0},
    "car":            {"fraction": 0.30, "transmission_multiplier": 0.5},
    "walking":        {"fraction": 0.20, "transmission_multiplier": 1.2},
}

# ── Sensibility to Public Health Campaigns ──────────────────────────────────
# Drawn at initialization from a residence-level Beta distribution
SENSIBILITY_PRIOR = {
    "dense_periphery":  {"alpha": 1.5, "beta": 4.0},  # lower responsiveness
    "sparse_periphery": {"alpha": 3.0, "beta": 2.0},  # higher responsiveness
}

# ── Doctor Visit Probability ────────────────────────────────────────────────
DOCTOR_VISIT_PROB = 0.60   # probability a symptomatic agent visits a doctor

# ── Epidemiological Parameters ──────────────────────────────────────────────
BASE_TRANSMISSION_RATE   = 0.015   # β per contact per tick (summer baseline)
CONTACT_RADIUS           = 1      # Moore neighbourhood radius (cells)
INCUBATION_PERIOD_TICKS  = 4      # ~2 days (4 × 12 h ticks)
ASYMPTOMATIC_FRACTION    = 0.40   # fraction of infectious who never show symptoms
INFECTIOUS_PERIOD_TICKS  = 10     # ~5 days
RECOVERY_IMMUNITY_DAYS   = 180    # days of post-infection immunity
INITIAL_EXPOSED_FRACTION  = 0.002   # ~10 agent
INITIAL_INFECTED_FRACTION = 0.0002   # ~1 agent
# ── Seasonal Transmission Modifier ─────────────────────────────────────────
# sin-based modifier: peaks in Winter (tick 0 = Jan 1)
SEASONAL_AMPLITUDE = 0.30   # ±30% around baseline

# ── Health Ministry Policy Thresholds ──────────────────────────────────────
# 7-day rolling prevalence (%) triggers alert escalation
POLICY_THRESHOLD_ALERT_1 = 0.02   # 2%  → low-coercive measures
POLICY_THRESHOLD_ALERT_2 = 0.05   # 5%  → medium-coercive measures
POLICY_THRESHOLD_ALERT_3 = 0.10   # 10% → high-coercive measures
MIN_POLICY_DURATION_DAYS  = 14    # plans cannot be downgraded before 2 weeks

# ── Policy Effects on Mobility ──────────────────────────────────────────────
POLICY_COMMUTE_REDUCTION = {
    "no_alert": 0.00,
    "alert_1":  0.20,   # 20% of compliant agents skip commute
    "alert_2":  0.40,
    "alert_3":  0.60,
}

# ── Observation Schedule ────────────────────────────────────────────────────
OBSERVATION_INTERVAL_TICKS = 2    # daily aggregate update
REPORTING_WINDOW_DAYS      = 7    # rolling window for ministry surveillance
