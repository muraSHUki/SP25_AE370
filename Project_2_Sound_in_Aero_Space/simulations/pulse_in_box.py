###### SETUP ########################################################################################
import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from animation_maker.animator import animate_wave, snapshot_grid_plot

import numpy as np
import matplotlib.pyplot as plt

####################################################################################################

###### DOMAIN PARAMETERS ############################################################################
# --- Spatial domain
Lx, Ly = 1.0, 1.0               # box size in meters
Nx, Ny = 201, 201               # grid resolution
dx = Lx / (Nx - 1)
dy = Ly / (Ny - 1)

# --- Time domain
c = 343.0                       # wave speed in m/s
CFL = 0.4
dt = CFL * min(dx, dy) / c
T = 0.003
Nt = int(T / dt)

# --- Spatial grid
x = np.linspace(0, Lx, Nx)
y = np.linspace(0, Ly, Ny)
X, Y = np.meshgrid(x, y, indexing='ij')

# --- Initial pulse parameters
sigma = 0.02
x0, y0 = 0.5, 0.5
p0 = np.exp(-((X - x0)**2 + (Y - y0)**2) / (2 * sigma**2))

# --- Color scale bounds
vmin = -0.0750
vmax = 0.0750

####################################################################################################

###### INITIALIZATION ###############################################################################
u_nm1 = p0.copy()               # u^{n-1}
u_n   = p0.copy()               # u^{n}
u_np1 = np.zeros_like(u_n)      # u^{n+1}

frames = []                     # frames for animation
snapshots = []                  # evenly spaced snapshots
snapshot_times = np.linspace(0, T - 1e-5, 9)
snapshot_indices = [int(t / dt) for t in snapshot_times]

####################################################################################################

###### TIME STEPPING LOOP ###########################################################################
for n in range(Nt):
    u_np1[1:-1, 1:-1] = (
        2 * u_n[1:-1, 1:-1] - u_nm1[1:-1, 1:-1] +
        (c * dt / dx)**2 * (
            u_n[2:, 1:-1] + u_n[:-2, 1:-1] +
            u_n[1:-1, 2:] + u_n[1:-1, :-2] -
            4 * u_n[1:-1, 1:-1]
        )
    )

    # Reflective boundary conditions (Dirichlet = 0)
    u_np1[0, :] = u_np1[-1, :] = 0.0
    u_np1[:, 0] = u_np1[:, -1] = 0.0

    if n in snapshot_indices:
        snapshots.append(u_np1.copy())
    if n % 2 == 0:
        frames.append(u_np1.copy())

    u_nm1, u_n = u_n, u_np1.copy()

####################################################################################################

###### VISUALIZATION ###############################################################################
# --- Ensure output folder exists using absolute path
out_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'results', 'pulse_in_box'))
os.makedirs(out_dir, exist_ok=True)

# --- Save static snapshot plot
snapshot_grid_plot(snapshots, snapshot_times, X, Y,
                   vmin=vmin, vmax=vmax,
                   out_path=os.path.join(out_dir, 'snapshot_grid.png'))

# --- Save animation
animate_wave(frames, X, Y, dt,
             vmin=vmin, vmax=vmax,
             out_path=os.path.join(out_dir, 'animation.gif'),
             title_prefix="Pulse in Box\n")
