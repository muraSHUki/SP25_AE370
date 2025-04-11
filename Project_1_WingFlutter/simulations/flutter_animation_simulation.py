import sys                                                                                         #
import os                                                                                          #
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))                 #
                                                                                                   #
from flutter_model import wing_flutter_rhs, rk4_solver                                             #
import numpy as np                                                                                 #
import matplotlib.pyplot as plt                                                                    #
import matplotlib.animation as animation                                                           #
from matplotlib.animation import PillowWriter                                                      #
####################################################################################################



###### SIMULATION PARAMETERS #######################################################################
# --- Parameters
params = {
    'rho': 1.225, 
    'U': 60.0, 
    'b': 0.5, 
    'm': 100.0,
    'I_alpha': 20.0, 
    'S_alpha': 30.0,
    'k_h': 5e4, 
    'k_theta': 2000.0,
    'CL_alpha': 2*np.pi, 
    'CM_alpha': -0.1 * 2*np.pi
}

# --- Initial Conditions
y0 = [0.01, 0.0, 0.05, 0.0]

# --- 
t_vals, y_vals = rk4_solver(wing_flutter_rhs, (0, 5), y0, h=0.01, params=params)
h_vals = y_vals[:, 0]
theta_vals = y_vals[:, 2]

# --- Airfoil Geometry
chord = 1.0
x_airfoil = np.array([0, chord])
y_base = np.array([0, 0])



###### ANIMATION ###################################################################################
# --- Plot Setup
fig, ax = plt.subplots(figsize=(8, 4))
ax.set_xlim(-0.2, 1.2)
ax.set_ylim(-0.5, 0.5)
ax.set_aspect('equal')
ax.grid(True)
line, = ax.plot([], [], 'k-', lw=3)
time_text = ax.text(0.02, 0.9, '', transform=ax.transAxes)

# --- Frame Update
def update(frame):
    h = h_vals[frame]
    theta = theta_vals[frame]
    rot_matrix = np.array([[np.cos(theta), -np.sin(theta)],
                           [np.sin(theta),  np.cos(theta)]])
    coords = np.vstack((x_airfoil, y_base)).T - [0.5, 0]
    rotated = coords @ rot_matrix.T + [0.5, h]
    line.set_data(rotated[:, 0], rotated[:, 1])
    time_text.set_text(f"t = {t_vals[frame]:.2f} s")
    return line, time_text

# --- Create and Save Animation
ani = animation.FuncAnimation(fig, update, frames=len(t_vals), interval=30)
ani.save("../results/flutter_animation.gif", writer=PillowWriter(fps=30))




###### SPECIFIC FRAMES #############################################################################
# --- Times to capture
capture_times = [0.0, 0.5, 1.0, 1.5, 2.0, 2.5]
chord = 1.0
x_airfoil = np.array([0, chord])
y_base = np.array([0, 0])

# --- Create Figure
fig, axes = plt.subplots(2, 3, figsize=(12, 6))
axes = axes.flatten()

for i, t_target in enumerate(capture_times):
    idx = np.argmin(np.abs(t_vals - t_target))
    h = h_vals[idx]
    theta = theta_vals[idx]

    rot_matrix = np.array([[np.cos(theta), -np.sin(theta)],
                            [np.sin(theta),  np.cos(theta)]])
    coords = np.vstack((x_airfoil, y_base)).T - [0.5, 0]
    rotated = coords @ rot_matrix.T + [0.5, h]

    ax = axes[i]
    ax.plot(rotated[:, 0], rotated[:, 1], 'k-', lw=2)
    ax.set_xlim(-0.2, 1.2)
    ax.set_ylim(-0.5, 0.5)
    ax.set_aspect('equal')
    ax.set_title(f"t = {t_target:.1f} s")
    ax.grid(True)

# --- Create and Save Image
plt.tight_layout()
plt.savefig("../results/6_frame_flutter.png", dpi=300)
plt.show()