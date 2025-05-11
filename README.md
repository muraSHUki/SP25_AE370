# AE370 Projects

This repository contains the course projects conducted in **AE370 - Aerospace Numerical Methods** taught by [Professor Andres Goza](https://aerospace.illinois.edu/directory/profile/agoza) during the Spring semester of 2025 at the University of Illinois at Urbana-Champaign. 
This course, as the name suggests, provides a hands-on introduction to core numerical methods essential for solving real-world aerospace engineering problems. 
Emphasis is placed on both the theoretical foundations and computational implementation of techniques used across structural mechanics, aerodynamics, and flight dynamics.

The course culminates in open-ended projects where students design and implement complete numerical pipelines to analyze complex phenomena.

---

## Repository Structure

```
SP25_AE370/
├── Project_1_WingFlutter/                  # Project 1: Aeroelastic Wing Flutter
│   ├── Group_930am_AE370_Project 1.pdf     --- # A copy of the research paper
│   ├── flutter_model.py                    --- # Shared solver logic (RK4, RK45, dynamics)
│   ├── simulations/                        --- # All Python scripts to run simulations
│   ├── results/                            --- # Output plots and animation files
│   ├── README.md                           --- # Project-specific documentation
│   ├── .gitignore  
│   └── requirements.txt                        
├── Project_2_AeroSpaceSoundWave/           # Project 2: Sound Wave Propagation
│ ├── Group_930am_AE370_Project_2.pdf       --- # A copy of the research paper
│ ├── sound_model/                          --- # FDTD solver, sources, utilities
│ ├── room_geometry/                        --- # Room/pillar layout and masking
│ ├── simulations/                          --- # Wave propagation experiments
│ ├── results/                              --- # Snapshots and animations
│ ├── README.md                             --- # Full project documentation
│ ├── .gitignore
│ └── requirements.txt
```

---

## Project 1: Wing Flutter Simulation

Simulates pitch-plunge flutter dynamics of a 2-DOF aeroelastic wing. Includes:
- Fixed and adaptive Runge-Kutta solvers
- Convergence study
- Flutter onset detection
- Full animation of motion

See `Project_1_WingFlutter/README.md` for detailed project explanation.

---

## Project 2: Aerospace Sound Wave Propagation

Implements a 2D finite-difference time-domain (FDTD) solver to simulate acoustic wave propagation in a rectangular box and an irregular polygonal room. Highlights include:
- Second-order FDTD method for the scalar wave equation
- Gaussian pulse and burst-style speech sources
- Complex room masking with internal pillars
- Animated pressure field visualization
- Convergence test confirming second-order spatial accuracy

See `Project_2_AeroSpaceSoundWave/README.md` for full description and running instructions.

---

## Requirements

Each project maintains its own `requirements.txt`. Common packages include:

- `numpy`
- `matplotlib`
- `shapely` (for polygonal masking in Project 2)
- `scipy` (for interpolation during error analysis)

To install dependencies for a project:

```bash
pip install -r requirements.txt
```

### Authors
- Golemis, Shaun - golemis2@illinois.edu
- Kemp, John - jwkemp2@illinois.edu
- Ochs, Ben - bochs2@illinois.edu
- Peters, Karsten - kjp7@illinois.edu
- Veranga, Joshua - veranga2@illinois.edu (repository owner)

---

Designed for modularity and reproducibility.
