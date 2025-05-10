###### SETUP ########################################################################################
import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from sound_model.FDTD_solver import fdtd_step
from sound_model.sources import gaussian_pulse
from sound_model.utils import apply_mask
from room_geometry.aero_space_geometry import generate_room_mask
from animation_maker.animator import animate_wave, snapshot_grid_plot

import numpy as np
import matplotlib.pyplot as plt

####################################################################################################

###### DOMAIN PARAMETERS ############################################################################
# --- Spatial domain
Lx, Ly = 15.0, 5.0
Nx, Ny = 301, 101
dx = Lx / (Nx - 1)
dy = Ly / (Ny - 1)

# --- Time domain
c = 343.0
CFL = 0.4
dt = CFL * min(dx, dy) / c
T = 0.03
Nt = int(T / dt)

# --- Source location
src_x, src_y = 12.0, 2.5
i0 = int(src_x / dx)
j0 = int(src_y / dy)

# --- Spatial grid
x = np.linspace(0, Lx, Nx)
y = np.linspace(0, Ly, Ny)
X, Y = np.meshgrid(x, y, indexing='ij')

# --- Room mask
mask = generate_room_mask(X, Y)

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
                      source_func=gaussian_pulse, source_loc=(i0, j0), t=t,
                      mask=mask)

    # Save visual data
    if n in snapshot_indices:
        snapshots.append(u_np1.copy())
    if n % 2 == 0:
        frames.append(u_np1.copy())

    u_nm1, u_n = u_n, u_np1.copy()

####################################################################################################

###### VISUALIZATION ###############################################################################
snapshot_grid_plot(snapshots, snapshot_times, X, Y,
                   out_path='../results/pulse_in_room/snapshot_grid.png')

animate_wave(frames, X, Y, dt,
             out_path='../results/pulse_in_room/animation.gif',
             title_prefix="Pulse in Room\n")
