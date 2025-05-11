import numpy as np
import matplotlib.pyplot as plt
from matplotlib import animation
from sound_model.FDTD_solver import fdtd_update
from sound_model.sources import gaussian_pulse
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

# === Initial Gaussian Pulse ===
x0, y0 = 12.0, 2.5
sigma = 0.3
p0 = gaussian_pulse(X, Y, x0, y0, sigma)
p0[~domain_mask] = 0

p_nm1 = p0.copy()
p_n = p0.copy()

frames = []
timestamps = []

# === Frame Sampling (fixed duration, 30 fps) ===
frame_count = 600
frame_interval = max(1, Nt // frame_count)

# === Time-Stepping Loop ===
for n in range(Nt):
    p_np1 = fdtd_update(p_nm1, p_n, dx, dt, c, domain_mask)

    if n % frame_interval == 0 and len(frames) < frame_count:
        frames.append(p_np1.copy())
        timestamps.append(n * dt)

    p_nm1, p_n = p_n, p_np1

# === Snapshot Selection ===
snapshot_indices = np.linspace(0, len(frames) - 1, 15, dtype=int)
snapshot_times = [timestamps[i] for i in snapshot_indices]
snapshot_frames = [frames[i] for i in snapshot_indices]

# === Plotting Parameters ===
vmin, vmax = -0.130, 0.130
tick_vals, tick_labels = get_tick_labels(vmin, vmax)
levels = np.linspace(vmin, vmax, 100)

# === Snapshot Plot ===
fig, axes = plt.subplots(5, 3, figsize=(16, 14))
cbar_ax = fig.add_axes([1.00, 0.15, 0.02, 0.7])

for ax, snap, t in zip(axes.flat, snapshot_frames, snapshot_times):
    ctf = ax.contourf(X, Y, snap, levels=levels, cmap='viridis', vmin=vmin, vmax=vmax)
    plot_room_and_pillars(ax)
    ax.set_title(f"t = {t*1000:.2f} ms", pad=8)
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_aspect('equal')

for i in range(len(snapshot_frames), 15):
    fig.delaxes(axes.flat[i])

cbar = fig.colorbar(ctf, cax=cbar_ax)
cbar.set_label("Pressure")
cbar.set_ticks(tick_vals)
cbar.set_ticklabels(tick_labels)

plt.tight_layout()
plt.savefig("../results/pulse_in_room/snapshots.png", dpi=300, bbox_inches='tight')
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
    ax_anim.set_title(f"2D Wave Propagation\nTime: {timestamps[i]:.3f} s")
    ax_anim.set_xlabel("x")
    ax_anim.set_ylabel("y")
    ax_anim.set_aspect('equal')
    return contour.collections

ani = animation.FuncAnimation(fig_anim, update_plot, frames=len(frames), interval=1000 / 30)
ani.save("../results/pulse_in_room/animation.gif", writer="pillow", fps=30)
