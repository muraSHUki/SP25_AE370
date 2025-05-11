import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
os.makedirs("../results/speech_in_room", exist_ok=True)

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import animation
from sound_model.FDTD_solver import fdtd_update
from sound_model.sources import speech_burst
from sound_model.utils import get_tick_labels
from room_geometry.aero_space_geometry import generate_domain_mask_fast, plot_room_and_pillars

# === Simulation Parameters ===
Lx, Ly = 15.0, 5.0
Nx, Ny = 601, 201
c = 343.0
T = 0.050
CFL = 0.4

dx = Lx / (Nx - 1)
dy = Ly / (Ny - 1)
dt = CFL * min(dx, dy) / c
Nt = int(T / dt)

x = np.linspace(0, Lx, Nx)
y = np.linspace(0, Ly, Ny)
X, Y = np.meshgrid(x, y, indexing='ij')
domain_mask = generate_domain_mask_fast(X, Y)

# === Speech-Like Source ===
x0, y0 = 12.0, 2.5
i_src = np.argmin(np.abs(x - x0))
j_src = np.argmin(np.abs(y - y0))

burst_params = [
    {"t0": 0.005, "sigma": 0.002, "f": 300},
    {"t0": 0.010, "sigma": 0.002, "f": 200},
    {"t0": 0.015, "sigma": 0.0025, "f": 250}
]

# === Initial Fields ===
p_nm1 = np.zeros((Nx, Ny))
p_n = np.zeros((Nx, Ny))

snapshots = []
frames = []

snapshot_times = np.linspace(0, T - 3e-5, 15)
snapshot_indices = [int(t / dt) for t in snapshot_times]

# === Time-Stepping Loop ===
for n in range(Nt):
    t = n * dt
    p_np1 = fdtd_update(p_nm1, p_n, dx, dt, c, domain_mask)

    val = speech_burst(t, burst_params)
    p_np1[i_src, j_src] += val
    p_nm1[i_src, j_src] += val

    if n in snapshot_indices:
        snapshots.append(p_np1.copy())
    if n % 2 == 0:
        frames.append(p_np1.copy())

    p_nm1, p_n = p_n, p_np1

# === Plotting Parameters ===
vmin, vmax = -0.060, 0.060
tick_vals, tick_labels = get_tick_labels(vmin, vmax)
levels = np.linspace(vmin, vmax, 100)

# === Snapshot Plot ===
fig, axes = plt.subplots(5, 3, figsize=(16, 14))
fig.subplots_adjust(right=0.9)
cbar_ax = fig.add_axes([1.00, 0.15, 0.02, 0.7])

for ax, snap, t in zip(axes.flat, snapshots, snapshot_times):
    ctf = ax.contourf(X, Y, snap, levels=levels, cmap='viridis', vmin=vmin, vmax=vmax)
    plot_room_and_pillars(ax)
    ax.set_title(f"t = {t*1000:.2f} ms", pad=8)
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_aspect('equal')

for i in range(len(snapshots), 15):
    fig.delaxes(axes.flat[i])

cbar = fig.colorbar(ctf, cax=cbar_ax)
cbar.set_label("Pressure")
cbar.set_ticks(tick_vals)
cbar.set_ticklabels(tick_labels)

plt.tight_layout()
plt.savefig("../results/speech_in_room/snapshots.png", dpi=300, bbox_inches='tight')
plt.show()

# === Animation ===
fig_anim, ax_anim = plt.subplots(figsize=(10, 4))
initial = ax_anim.contourf(X, Y, frames[0], levels=levels, cmap='viridis', vmin=vmin, vmax=vmax)
plot_room_and_pillars(ax_anim)
cbar = fig_anim.colorbar(initial, ax=ax_anim, label="Pressure")
cbar.set_ticks(tick_vals)
cbar.set_ticklabels(tick_labels)

def update_plot(i):
    ax_anim.clear()
    contour = ax_anim.contourf(X, Y, frames[i], levels=levels, cmap='viridis', vmin=vmin, vmax=vmax)
    plot_room_and_pillars(ax_anim)
    ax_anim.set_title(f"Speech Wave in Room\nTime = {i * 2 * dt * 1000:.2f} ms")
    ax_anim.set_xlabel("x")
    ax_anim.set_ylabel("y")
    ax_anim.set_aspect('equal')
    return contour.collections

ani = animation.FuncAnimation(fig_anim, update_plot, frames=len(frames), interval=1000 / 30)
ani.save("../results/speech_in_room/animation.gif", writer="pillow", fps=30)
