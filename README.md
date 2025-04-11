# AE370 Projects

This repository contains the course projects conducted in **AE370 - Aerospace Numerical Methods** taught by [Professor Andres Goza](https://aerospace.illinois.edu/directory/profile/agoza) during the Spring semester of 2025 at the Univeristy of Illinois at Urbana-Champaign. 
This course, as the name suggests, provides a hands-on introduction to core numerical methods essential for solving real-world aerospace engineering problems. 
Emphasis is placed on both the theoretical foundations and computational implementation of techniques used across structural mechanics, aerodynamics, and flight dynamics.

The course culminates in open-ended projects where students design and implement complete numerical pipelines to analyze complex phenomena.

## Repository Structure

```
SP25_AE370/
├── Project_1_WingFlutter/         # Project 1: Aeroelastic Wing Flutter
│   ├── <Paper placeholder>.pdf    # A copy of the research paper
│   ├── flutter_model.py           # Shared solver logic (RK4, RK45, dynamics)
│   ├── simulations/               # All Python scripts to run simulations
│   ├── results/                   # Output plots and animation files
│   ├── README.md                  # Project-specific documentation
│   └── .gitignore                 # Ignore bytecode and results
└── (Project_2_<name>/)            # Future project placeholder
```

## Project 1: Wing Flutter Simulation

Simulates pitch-plunge flutter dynamics of a 2-DOF aeroelastic wing. Includes:
- Fixed and adaptive Runge-Kutta solvers
- Convergence study
- Flutter onset detection
- Full animation of motion

See `Project_1_WingFlutter/README.md` for detailed project explanation.

### Authors
- Golemis, Shaun - golemis2@illinois.edu
- Kemp, John - jwkemp2@illinois.edu
- Ochs, Ben - bochs2@illinois.edu
- Peters, Karsten - kjp@illinois.edu
- Veranga, Joshua - veranga2@illinois.edu (repository owner)

## Requirements

Each project manages its own requirements (typically `numpy`, `matplotlib`). Run individual scripts from their `simulations/` folder.

---

Designed for modularity and reproducibility.
