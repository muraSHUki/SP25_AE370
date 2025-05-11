# Aerospace Sound Wave Simulation Project

This repository contains a Python-based simulation of 2D acoustic wave propagation using the finite-difference time-domain (FDTD) method. The project explores wave behavior in enclosed environments such as rectangular boxes and a complex polygonal room, including interactions with interior obstacles and multiple speech-like sources.

## Folder Structure

```
Project_2_AeroSpaceSoundWave/
├── Group_930am_AE370_Project_2.pdf       # Paper presenting project details
├── README.md                             # Project overview and usage
├── .gitignore                            # Git exclusions
│
├── sound_model/                          # Core FDTD solver and source functions
│   ├── solver.py                         --- # Time-marching wave update step
│   ├── sources.py                        --- # Pulse and speech waveform sources
│   └── utils.py                          --- # Helpers for ticks and frame logic
│
├── room_geometry/                        # Polygonal room geometry & masking
│   └── geometry.py                       --- # Room layout, pillars, plotting
│
├── simulations/                          # Executable scripts
│   ├── pulse_in_box.py                   --- # Gaussian pulse in simple box
│   ├── speech_in_box.py                  --- # Burst source in simple box
│   ├── pulse_in_room.py                  --- # Gaussian pulse in polygonal room
│   ├── speech_in_room.py                 --- # Single speech burst in room
│   ├── conversation_in_room.py           --- # Two-source interaction
│   └── convergence_test.py               --- # Error for FDTD solver
│
├── results/                              # Saved figures and animations
│   ├── pulse_in_box/
│   ├── speech_in_box/
│   ├── pulse_in_room/
│   ├── speech_in_room/
│   ├── conversation_in_room/
│   └── convergence_test/
```

## Features

- 2D explicit FDTD solver for acoustic wave propagation
- Flexible room geometry with masked interior pillars
- Additive burst-based speech-like source modeling
- Consistent pressure snapshots and GIF animations
- Frame and tick utilities for clean plotting
- Modular architecture for reuse across simulations

## Running

Each file in `simulations/` can be run directly:

```bash
python simulations/pulse_in_room.py
```

Results (figures and animations) are saved under `results/` with matching subfolders.

## Requirements

- Python 3.8+
- numpy
- matplotlib
- shapely

To install dependencies:

```bash
pip install -r requirements.txt
```

## Reproducibility

- All simulations use fixed CFL condition (0.4) for stability.
- Room boundaries are hard-walled (Dirichlet condition: p = 0).
- Interior masking is handled with geometric containment via `shapely`.
- Animations are saved as `.gif` using `matplotlib.animation`.

---
