###### SETUP ########################################################################################
import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from sound_model.FDTD_solver import fdtd_step
from sound_model.sources import synthetic_syllable_wave
from animation_maker.animator import animate_wave, snapshot_grid_plot

import numpy as np
import matplotlib.pyplot as plt

####################################################################################################

###### DOMAIN PARAMETERS ############################################################################
# --- Spatial domain
Lx, Ly = 1.0, 1.0
Nx, Ny = 201, 201
dx = Lx / (Nx - 1)
dy = Ly / (Ny - 1)

# --- Time domain
c = 343.0
CFL = 0.4
dt = CFL * min(dx, dy) / c
T = 0.035
Nt = int(T / dt)

# --- Spatial grid
x = np.linspace(0, Lx, Nx)
y = np.linspace(0, Ly, Ny)
X, Y = np.meshgrid(x, y, indexing='ij')

# --- Source location
x0, y0 = 0.5, 0.5
i0 = int(x0 / dx)
j0 = int(y0 / dy)

# --- Color scale
vmin = -0.01
vmax = 0.01

####################################################################################################

###### INITIALIZATION ###############################################################################
u_nm1 = np.zeros((Nx, Ny))
u_n   = np.zeros((Nx, Ny))
u_np1 = np.zeros((Nx, Ny))

frames = []
snapshots = []
snapshot_times = np.linspace(0, T - 1e-5, 9)
snapshot_indices = [int(t / dt) for t in snapshot_times]

####################################################################################################

###### TIME STEPPING LOOP ###########################################################################
for n in range(Nt):
    t = n * dt
    u_np1 = fdtd_step(u_nm1, u_n, c, dt, dx, dy,
                      source_func=synthetic_syllable_wave,
                      source_loc=(i0, j0),
                      t=t)

    # Dirichlet boundary conditions (reflective)
    u_np1[0, :] = u_np1[-1, :] = 0.0
    u_np1[:, 0] = u_np1[:, -1] = 0.0

    if n in snapshot_indices:
        snapshots.append(u_np1.copy())
    if n % 2 == 0:
        frames.append(u_np1.copy())

    u_nm1, u_n = u_n, u_np1.copy()

####################################################################################################

###### VISUALIZATION ###############################################################################
out_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'results', 'speech_in_box'))
os.makedirs(out_dir, exist_ok=True)

snapshot_grid_plot(snapshots, snapshot_times, X, Y,
                   vmin=vmin, vmax=vmax,
                   out_path=os.path.join(out_dir, 'snapshot_grid.png'))

animate_wave(frames, X, Y, dt,
             vmin=vmin, vmax=vmax,
             out_path=os.path.join(out_dir, 'animation.gif'),
             title_prefix="Speech Wave in Box\n")
