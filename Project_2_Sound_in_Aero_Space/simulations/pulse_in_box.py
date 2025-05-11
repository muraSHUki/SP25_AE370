import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
os.makedirs("../results/pulse_in_box", exist_ok=True)

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import animation
from sound_model.FDTD_solver import fdtd_update
from sound_model.sources import gaussian_pulse
from sound_model.utils import get_tick_labels

# === Simulation Parameters ===
Lx, Ly = 1.0, 1.0
Nx, Ny = 201, 201
c = 343.0
T = 0.003
CFL = 0.4

dx = Lx / (Nx - 1)
dy = Ly / (Ny - 1)
dt = CFL * min(dx, dy) / c
Nt = int(T / dt)

x = np.linspace(0, Lx, Nx)
y = np.linspace(0, Ly, Ny)
X, Y = np.meshgrid(x, y, indexing='ij')

# === Initial Gaussian Pulse ===
x0, y0 = 0.5, 0.5
sigma = 0.02
p0 = gaussian_pulse(X, Y, x0, y0, sigma)

p_nm1 = p0.copy()
p_n = p0.copy()

snapshots = []
frames = []

snapshot_times = np.linspace(0, T - 1e-5, 9)
snapshot_indices = [int(t / dt) for t in snapshot_times]

# === Time-Stepping Loop ===
for n in range(Nt):
    p_np1 = fdtd_update(p_nm1, p_n, dx, dt, c)
    
    if n in snapshot_indices:
        snapshots.append(p_np1.copy())
    if n % 2 == 0:
        frames.append(p_np1.copy())

    p_nm1, p_n = p_n, p_np1

# === Plotting Parameters ===
vmin, vmax = -0.075, 0.075
tick_vals, tick_labels = get_tick_labels(vmin, vmax)
levels = np.linspace(vmin, vmax, 100)

# === Snapshot Plot ===
fig, axes = plt.subplots(3, 3, figsize=(12, 10))
cbar_ax = fig.add_axes([1.00, 0.15, 0.02, 0.7])

for ax, snap, t in zip(axes.flat, snapshots, snapshot_times):
    ctf = ax.contourf(X, Y, snap, levels=levels, cmap='viridis', vmin=vmin, vmax=vmax)
    ax.set_title(f"t = {t*1000:.2f} ms", pad=8)
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_aspect('equal')

# Remove empty axes
for i in range(len(snapshots), 9):
    fig.delaxes(axes.flat[i])

cbar = fig.colorbar(ctf, cax=cbar_ax)
cbar.set_label("Pressure")
cbar.set_ticks(tick_vals)
cbar.set_ticklabels(tick_labels)

plt.tight_layout()
plt.savefig("../results/pulse_in_box/snapshots.png", dpi=300, bbox_inches='tight')
plt.show()

# === Animation ===
fig_anim, ax_anim = plt.subplots(figsize=(6, 5))
initial = ax_anim.contourf(X, Y, frames[0], levels=levels, cmap='viridis', vmin=vmin, vmax=vmax)

def update_plot(i):
    ax_anim.clear()
    contour = ax_anim.contourf(X, Y, frames[i], levels=levels, cmap='viridis', vmin=vmin, vmax=vmax)
    ax_anim.set_title(f"Pressure Pulse in Box\nTime = {i * 2 * dt * 1000:.2f} ms")
    ax_anim.set_xlabel("x")
    ax_anim.set_ylabel("y")
    ax_anim.set_aspect('equal')
    return contour.collections

cbar = fig_anim.colorbar(initial, ax=ax_anim, label="Pressure")
cbar.set_ticks(tick_vals)
cbar.set_ticklabels(tick_labels)

ani = animation.FuncAnimation(fig_anim, update_plot, frames=len(frames), interval=50)
ani.save("../results/pulse_in_box/animation.gif", writer="pillow", fps=30)
