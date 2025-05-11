###### SETUP ########################################################################################
import sys                                                                                          #
import os                                                                                           #
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))                  #
os.makedirs("../results/pulse_in_room", exist_ok=True)                                              #
                                                                                                    #
import numpy as np                                                                                  #
import matplotlib.pyplot as plt                                                                     #
from matplotlib import animation                                                                    #
from sound_model.FDTD_solver import fdtd_update                                                     #
from sound_model.sources import gaussian_pulse                                                      #
from sound_model.utils import get_tick_labels                                                       #
from room_geometry.aero_space_geometry import generate_domain_mask_fast, plot_room_and_pillars      #
#####################################################################################################



### Simulation Parameters ###########################################################################
# --- Grid and Simulation Parameters ---
Lx, Ly = 15.0, 5.0
Nx, Ny = 1501, 501
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



### Source Definition ###############################################################################
# --- Gaussian Pulse ---
x0, y0 = 12.0, 2.5
sigma = 0.3
p0 = gaussian_pulse(X, Y, x0, y0, sigma)
p0[~domain_mask] = 0

p_nm1 = p0.copy()
p_n = p0.copy()

frames = []
timestamps = []

# --- Frame Sampling Control ---
frame_count = 600
frame_interval = max(1, Nt // frame_count)
#####################################################################################################



### Time-Stepping Loop ##############################################################################
for n in range(Nt):
    p_np1 = fdtd_update(p_nm1, p_n, dx, dt, c, domain_mask)

    if n % frame_interval == 0 and len(frames) < frame_count:
        frames.append(p_np1.copy())
        timestamps.append(n * dt)

    p_nm1, p_n = p_n, p_np1
#####################################################################################################



### Snapshot Selection ##############################################################################
snapshot_indices = np.linspace(1e-5, len(frames) - 1, 18, dtype=int)
snapshot_times = [timestamps[i] for i in snapshot_indices]
snapshot_frames = [frames[i] for i in snapshot_indices]
#####################################################################################################



### Plotting Parameters #############################################################################
vmin, vmax = -0.300, 0.300
tick_vals, tick_labels = get_tick_labels(vmin, vmax)
levels = np.linspace(vmin, vmax, 100)

plt.rcParams.update({
    "font.size": 10,
    "axes.titlesize": 10,
    "axes.labelsize": 10,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "font.family": "serif",
})
#####################################################################################################



### Snapshot Plot ###################################################################################
fig, axes = plt.subplots(6, 3, figsize=(6.5, 9))                                                    
cbar_ax = fig.add_axes([0.93, 0.15, 0.015, 0.7])                                                    

for idx, (ax, snap, t) in enumerate(zip(axes.flat, snapshot_frames, snapshot_times)):              
    ctf = ax.contourf(X, Y, snap, levels=levels, cmap='viridis', vmin=vmin, vmax=vmax)             
    plot_room_and_pillars(ax)                                                                      
    ax.set_title(f"t = {t*1000:.2f} ms", pad=4)                                                     
    ax.set_aspect('equal')                                                                         

    if idx % 3 == 0:                                                                                
        ax.set_ylabel("y")                                                                         
        ax.set_yticks([0, 1, 2, 3, 4, 5])                                                           
    else:                                                                                           
        ax.set_yticklabels([])                                                                      

    if idx // 3 == 5:                                                                               
        ax.set_xlabel("x")                                                                         
        ax.set_xticks([0, 3, 6, 9, 12, 15])                                                         
    else:                                                                                           
        ax.set_xticklabels([])                                                                      

for i in range(len(snapshot_frames), 18):                                                           
    fig.delaxes(axes.flat[i])                                                                       

cbar = fig.colorbar(ctf, cax=cbar_ax)                                                               
cbar.set_label("Pressure", fontsize=10)                                                             
cbar.ax.tick_params(labelsize=9)                                                                    
cbar.set_ticks(tick_vals)                                                                           
cbar.set_ticklabels(tick_labels)                                                                    

fig.subplots_adjust(left=0.06, right=0.91, bottom=0.06, top=0.94, wspace=0.1, hspace=0.25)           
plt.savefig("../results/pulse_in_room/snapshots.png", dpi=300, bbox_inches='tight')                
#####################################################################################################



### Animation #######################################################################################
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
    ax_anim.set_title(f"2D Wave Propagation\nTime: {timestamps[i] * 1000:.3f} ms")                  
    ax_anim.set_xlabel("x")                                                                         
    ax_anim.set_ylabel("y")                                                                         
    ax_anim.set_aspect('equal')                                                                     
    return contour.collections                                                                      

ani = animation.FuncAnimation(fig_anim, update_plot, frames=len(frames), interval=1000 / 30)        
ani.save("../results/pulse_in_room/animation.gif", writer="pillow", fps=30)                         
#####################################################################################################
