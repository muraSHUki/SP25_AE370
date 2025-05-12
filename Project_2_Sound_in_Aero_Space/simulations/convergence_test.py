import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import numpy as np
import matplotlib.pyplot as plt
from sound_model.FDTD_solver import fdtd_update
from sound_model.sources import gaussian_pulse

# === Parameters ===
c = 343.0
Lx, Ly = 1.0, 1.0
sigma = 0.02
x0, y0 = 0.5, 0.5
T = 0.001  # very short to avoid reflections
CFL = 0.4

# === Grid sizes to test ===
res_list = [51, 101, 201, 401]
errors = []
dx_vals = []

# === Reference grid (finest) ===
Nx_ref = 801
Ny_ref = 801
dx_ref = Lx / (Nx_ref - 1)
dy_ref = Ly / (Ny_ref - 1)
dt_ref = CFL * min(dx_ref, dy_ref) / c
Nt_ref = int(T / dt_ref)

x_ref = np.linspace(0, Lx, Nx_ref)
y_ref = np.linspace(0, Ly, Ny_ref)
X_ref, Y_ref = np.meshgrid(x_ref, y_ref, indexing='ij')
p0_ref = gaussian_pulse(X_ref, Y_ref, x0, y0, sigma)
p_nm1 = p0_ref.copy()
p_n = p0_ref.copy()

for _ in range(Nt_ref):
    p_np1 = fdtd_update(p_nm1, p_n, dx_ref, dt_ref, c)
    p_nm1, p_n = p_n, p_np1

p_ref_final = p_np1.copy()

# === Convergence runs ===
for N in res_list:
    Nx, Ny = N, N
    dx = Lx / (Nx - 1)
    dy = Ly / (Ny - 1)
    dt = CFL * min(dx, dy) / c
    Nt = int(T / dt)
    dx_vals.append(dx)

    x = np.linspace(0, Lx, Nx)
    y = np.linspace(0, Ly, Ny)
    X, Y = np.meshgrid(x, y, indexing='ij')
    p0 = gaussian_pulse(X, Y, x0, y0, sigma)
    p_nm1 = p0.copy()
    p_n = p0.copy()

    for _ in range(Nt):
        p_np1 = fdtd_update(p_nm1, p_n, dx, dt, c)
        p_nm1, p_n = p_n, p_np1

    # Interpolate reference solution to current grid
    from scipy.interpolate import RegularGridInterpolator
    interp = RegularGridInterpolator((x_ref, y_ref), p_ref_final)
    coords = np.stack([X.ravel(), Y.ravel()], axis=-1)
    p_ref_interp = interp(coords).reshape(Nx, Ny)

    # Compute L2 error
    error = np.sqrt(np.mean((p_np1 - p_ref_interp) ** 2))
    errors.append(error)

# === Plot Convergence ===
plt.figure(figsize=(6, 5))
plt.loglog(dx_vals, errors, 'o-', label='FDTD error')
plt.loglog(dx_vals, [errors[0]*(dx/dx_vals[0])**2 for dx in dx_vals], 'k--', label='Slope 2')
plt.xlabel("Δx (grid spacing)")
plt.ylabel("L2 Error")
plt.title("FDTD Convergence Test")
plt.legend()
plt.grid(True, which="both", ls="--")
os.makedirs("../results/convergence_test", exist_ok=True)
plt.savefig("../results/convergence_test/convergence_plot.png", dpi=300)