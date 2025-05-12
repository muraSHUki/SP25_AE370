###### SETUP ########################################################################################
import sys                                                                                          #
import os                                                                                           #
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))                  #
os.makedirs("../results/pulse_in_box", exist_ok=True)                                               #
                                                                                                    #
import numpy as np                                                                                  #
import matplotlib.pyplot as plt                                                                     #
from matplotlib import animation                                                                    #
from sound_model.FDTD_solver import fdtd_update                                                     #
from sound_model.sources import gaussian_pulse                                                      #
from sound_model.utils import get_tick_labels                                                       #
#####################################################################################################



### Simulation Parameters ###########################################################################
# --- Grid Parameters ---
Lx, Ly = 1, 1       # size of box
Nx, Ny = 101, 101   # number of grid points 

# --- Simulation Parameters ---
c = 343.0           # speed of sound
T = 0.010           # simulation time
CFL = 0.4           # CFL number

dx = Lx / (Nx - 1)
dy = Ly / (Ny - 1)
dt = CFL * min(dx, dy) / c
Nt = int(T / dt)

x = np.linspace(0, Lx, Nx)
y = np.linspace(0, Ly, Ny)
X, Y = np.meshgrid(x, y, indexing='ij')
#####################################################################################################



### Source Definition ###############################################################################
# --- Source Parameters ---
x0, y0 = 0.5, 0.5
sigma = 0.02
p0 = gaussian_pulse(X, Y, x0, y0, sigma)

p_nm1 = p0.copy()
p_n = p0.copy()

snapshots = []
frames = []

num_snapshots = 15
snapshot_times = np.linspace(1e-5, T - 1e-4, num_snapshots)
snapshot_indices = [int(t / dt) for t in snapshot_times]

# --- Time-Stepping Loop ---
for n in range(Nt):
    p_np1 = fdtd_update(p_nm1, p_n, dx, dt, c)
    
    if n in snapshot_indices:
        snapshots.append(p_np1.copy())
    if n % 2 == 0:
        frames.append(p_np1.copy())

    p_nm1, p_n = p_n, p_np1
#####################################################################################################



### Plotting Parameters #############################################################################
# --- Contour Value Parameters ---
vmin, vmax = -0.15, 0.15      # Adjust accordingly to desired colorbar values
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
fig, axes = plt.subplots(5, 3, figsize=(6.5, 9))  # Full-page layout
cbar_ax = fig.add_axes([0.92, 0.15, 0.015, 0.7])   # Slim, aligned colorbar

for idx, (ax, snap, t) in enumerate(zip(axes.flat, snapshots, snapshot_times)):
    ctf = ax.contourf(X, Y, snap, levels=levels, cmap='viridis', vmin=vmin, vmax=vmax)
    ax.set_title(f"t = {t*1000:.2f} ms", pad=4)
    ax.set_aspect('equal')

    # y-axis for first column
    if idx % 3 == 0:
        ax.set_ylabel("y")
        ax.set_yticks([0, 0.5, 1])
    else:
        ax.set_yticklabels([])

    # x-axis for bottom row
    if idx // 3 == 4:
        ax.set_xlabel("x")
        ax.set_xticks([0, 0.5, 1])
    else:
        ax.set_xticklabels([])

# --- Remove empty plots ---
for i in range(len(snapshots), num_snapshots):
    fig.delaxes(axes.flat[i])

# --- Colorbar ---
cbar = fig.colorbar(ctf, cax=cbar_ax)
cbar.set_label("Pressure", fontsize=10)
cbar.ax.tick_params(labelsize=9)
cbar.set_ticks(tick_vals)
cbar.set_ticklabels(tick_labels)

# --- Adjust layout ---
fig.subplots_adjust(left=0.06, right=0.90, bottom=0.06, top=0.95, wspace=0.12, hspace=0.25)

# --- Save Figure ---
plt.savefig("../results/pulse_in_box/snapshots.png", dpi=300, bbox_inches='tight')
#####################################################################################################



### Animation #######################################################################################
# --- Animation Parameters ---
fig_anim, ax_anim = plt.subplots(figsize=(6, 5))
initial = ax_anim.contourf(X, Y, frames[0], levels=levels, cmap='viridis', vmin=vmin, vmax=vmax)

# --- Animation Function ---
def update_plot(i):
    ax_anim.clear()
    contour = ax_anim.contourf(X, Y, frames[i], levels=levels, cmap='viridis', vmin=vmin, vmax=vmax)
    ax_anim.set_title(f"Pressure Pulse in Box\nTime = {i * 2 * dt * 1000:.2f} ms")
    ax_anim.set_xlabel("x")
    ax_anim.set_ylabel("y")
    ax_anim.set_aspect('equal')
    return contour.collections

# --- Animation Control ---
cbar = fig_anim.colorbar(initial, ax=ax_anim, label="Pressure")
cbar.set_ticks(tick_vals)
cbar.set_ticklabels(tick_labels)

# --- Animation Save ---
ani = animation.FuncAnimation(fig_anim, update_plot, frames=len(frames), interval=50)
ani.save("../results/pulse_in_box/animation.gif", writer="pillow", fps=30)
#####################################################################################################