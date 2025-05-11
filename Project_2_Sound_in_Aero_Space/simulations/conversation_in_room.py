###### SETUP ########################################################################################
import sys                                                                                          #
import os                                                                                           #
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))                  #
os.makedirs("../results/conversation_in_room", exist_ok=True)                                       #       
                                                                                                    #
import numpy as np                                                                                  #
import matplotlib.pyplot as plt                                                                     #
from matplotlib import animation                                                                    #
from sound_model.FDTD_solver import fdtd_update                                                     #
from sound_model.sources import speech_burst                                                        #
from sound_model.utils import get_tick_labels                                                       #
from room_geometry.aero_space_geometry import generate_domain_mask_fast, plot_room_and_pillars      #
#####################################################################################################



### Simulation Parameters ###########################################################################
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
#####################################################################################################



### Two Speech-Like Sources #########################################################################
# --- Speech Wave Parameters ---
burst_1 = [
    {"t0": 0.005, "sigma": 0.002, "f": 300},
    {"t0": 0.010, "sigma": 0.002, "f": 200},
    {"t0": 0.015, "sigma": 0.0025, "f": 250}
]
burst_2 = [
    {"t0": 0.027, "sigma": 0.002, "f": 300},
    {"t0": 0.032, "sigma": 0.002, "f": 200},
    {"t0": 0.037, "sigma": 0.0025, "f": 250}
]

x1, y1 = 12.0, 2.0
x2, y2 = 12.0, 3.0
i_src1 = np.argmin(np.abs(x - x1))
j_src1 = np.argmin(np.abs(y - y1))
i_src2 = np.argmin(np.abs(x - x2))
j_src2 = np.argmin(np.abs(y - y2))
#####################################################################################################



### Initial Fields ##################################################################################
p_nm1 = np.zeros((Nx, Ny))
p_n = np.zeros((Nx, Ny))

snapshots = []
frames = []
# --- Snapshot Parameters ---
num_snapshots = 18
snapshot_times = np.linspace(0, T - 3e-5, num_snapshots)
snapshot_indices = [int(t / dt) for t in snapshot_times]
#####################################################################################################



### Time-Stepping Loop ##############################################################################
for n in range(Nt):
    t = n * dt
    p_np1 = fdtd_update(p_nm1, p_n, dx, dt, c, domain_mask)

    val1 = speech_burst(t, burst_1)
    val2 = speech_burst(t, burst_2)
    p_np1[i_src1, j_src1] += val1
    p_nm1[i_src1, j_src1] += val1
    p_np1[i_src2, j_src2] += val2
    p_nm1[i_src2, j_src2] += val2

    if n in snapshot_indices:
        snapshots.append(p_np1.copy())
    if n % 2 == 0:
        frames.append(p_np1.copy())

    p_nm1, p_n = p_n, p_np1
#####################################################################################################



### Plotting Parameters #############################################################################
# --- Contour Value Parameters ---
vmin, vmax = -0.105, 0.105      # Adjust accordingly to desired colorbar values
tick_vals, tick_labels = get_tick_labels(vmin, vmax)
levels = np.linspace(vmin, vmax, 100)
#####################################################################################################



### Snapshot Plot ###################################################################################
# --- Plot Typeface Parameters ---
plt.rcParams.update({
    "font.size": 10,
    "axes.titlesize": 10,
    "axes.labelsize": 10,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "font.family": "serif",
})

# --- Plot Layout ---
fig, axes = plt.subplots(num_snapshots/3, 3, figsize=(6.5, 9))  # Full-page layout for 15 plots
cbar_ax = fig.add_axes([0.93, 0.15, 0.015, 0.7])  # Clean colorbar position

for idx, (ax, snap, t) in enumerate(zip(axes.flat, snapshots, snapshot_times)):
    ctf = ax.contourf(X, Y, snap, levels=levels, cmap='viridis', vmin=vmin, vmax=vmax)
    plot_room_and_pillars(ax)
    ax.set_title(f"t = {t*1000:.2f} ms", pad=4)
    ax.set_aspect('equal')

    # Label y-axis on first column
    if idx % 3 == 0:
        ax.set_ylabel("y")
        ax.set_yticks([0, 1, 2, 3, 4, 5])
    else:
        ax.set_yticklabels([])

    # Label x-axis on bottom row
    if idx // 3 == 4:  # Last row
        ax.set_xlabel("x")
        ax.set_xticks([0, 3, 6, 9, 12, 15])
    else:
        ax.set_xticklabels([])

# --- Remove Empty Plots ---
for i in range(len(snapshots), num_snapshots):
    fig.delaxes(axes.flat[i])

# --- Colorbar ---
cbar = fig.colorbar(ctf, cax=cbar_ax)
cbar.set_label("Pressure")
cbar.set_ticks(tick_vals)
cbar.set_ticklabels(tick_labels)
cbar.ax.tick_params(labelsize=9)

# --- Manual Layout Control ---
fig.subplots_adjust(left=0.06, right=0.91, bottom=0.06, top=0.94, wspace=0.1, hspace=0.25)

# --- Save Figure ---
plt.savefig("../results/conversation_in_room/snapshots.png", dpi=300, bbox_inches='tight')
#####################################################################################################



### Animation #######################################################################################
# --- Animation Parameters ---
fig_anim, ax_anim = plt.subplots(figsize=(10, 4))
initial = ax_anim.contourf(X, Y, frames[0], levels=levels, cmap='viridis', vmin=vmin, vmax=vmax)
plot_room_and_pillars(ax_anim)
cbar = fig_anim.colorbar(initial, ax=ax_anim, label="Pressure")
cbar.set_ticks(tick_vals)
cbar.set_ticklabels(tick_labels)

# --- Animation Function ---
def update_plot(i):
    ax_anim.clear()
    contour = ax_anim.contourf(X, Y, frames[i], levels=levels, cmap='viridis', vmin=vmin, vmax=vmax)
    plot_room_and_pillars(ax_anim)
    ax_anim.set_title(f"Two Speech Waves\nTime = {i * 2 * dt * 1000:.2f} ms")
    ax_anim.set_xlabel("x")
    ax_anim.set_ylabel("y")
    ax_anim.set_aspect('equal')
    return contour.collections

# --- Animation Save ---
ani = animation.FuncAnimation(fig_anim, update_plot, frames=len(frames), interval=1000 / 30)
ani.save("../results/conversation_in_room/animation.gif", writer="pillow", fps=30)
#####################################################################################################



### Single Frame Plot ###############################################################################
# --- Get Desired Snapshot ---
desired_time_ms = 45.0      # Adjust accordingly for desired time
desired_index = np.argmin(np.abs(snapshot_times * 1000 - desired_time_ms))
desired_snapshot = snapshots[desired_index]
actual_time_ms = snapshot_times[desired_index] * 1000

# --- Plot Layout ---
fig_single, ax_single = plt.subplots(figsize=(6.5, 4))
ctf = ax_single.contourf(X, Y, desired_snapshot, levels=levels, cmap='viridis', vmin=vmin, vmax=vmax)
plot_room_and_pillars(ax_single)
ax_single.set_xlabel("x")
ax_single.set_ylabel("y")
ax_single.set_aspect('equal')

# --- Colorbar ---
cbar = fig_single.colorbar(ctf, ax=ax_single)
cbar.set_label("Pressure")
cbar.set_ticks(tick_vals)
cbar.set_ticklabels(tick_labels)

# --- Save Figure ---
plt.tight_layout()
plt.savefig(f"../results/conversation_in_room/snapshot_{int(actual_time_ms)}ms.png", dpi=300)
#####################################################################################################
