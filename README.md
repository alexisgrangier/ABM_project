# SEIRD Agent-Based Model
## Bayesian Agent-Based Model of Infectious Disease Transmission in an Urban Center

### Overview
This ABM simulates the spread of an infectious agent through a heterogeneous urban environment 
structured around three spatially distinct zones. It explores how individual preferences and 
government signaling shape transmission rates and death tolls.

### Project Structure
```
seird_abm/
├── agents/                  # Agent class definitions
│   ├── individual.py        # Individual agent (SEIRD states + attributes)
│   └── health_ministry.py   # Health Ministry agent (policy states)
├── environment/             # Spatial grid and zone definitions
│   ├── grid.py              # 100×100 grid and zone partitions
│   └── zones.py             # Urban core, dense periphery, sparse periphery
├── submodels/               # Core simulation submodels
│   ├── mobility.py          # Commuting and movement logic
│   ├── transmission.py      # Disease transmission mechanics
│   ├── disease_progression.py  # SEIRD state transitions
│   ├── medical.py           # Medical consultation & reporting
│   └── seasonality.py       # Seasonal transmission modifier
├── policy/                  # Government policy logic
│   └── health_policy.py     # Alert state transitions and thresholds
├── config/                  # Model parameters
│   └── parameters.py        # All fixed parameters
├── utils/                   # Utilities
│   ├── data_collector.py    # Observation and output collection
│   └── visualization.py     # Plotting helpers
├── data/
│   ├── raw/                 # Unused in this model (no real-time streams)
│   ├── processed/           # Cleaned data for analysis
│   └── outputs/             # Simulation run outputs
├── notebooks/               # Jupyter notebooks
│   └── 01_model_initialization.ipynb  # Main initialization notebook
├── tests/                   # Unit tests
└── docs/                    # Documentation
    └── ODD_protocol.docx    # Original ODD protocol document
```

### Key Parameters
- **Population:** N = 5,000 agents
- **Grid:** 100 × 100 cells (max 5 agents/cell)
- **Time:** 730 ticks (1 tick = 12 hrs, 365 days)
- **Zones:** Urban core (40×40), Dense periphery (15×20), Sparse periphery (remainder)
- **Disease model:** SEIRD (Susceptible, Exposed, Infectious asymptomatic, Infectious symptomatic, Recovered, Dead)
