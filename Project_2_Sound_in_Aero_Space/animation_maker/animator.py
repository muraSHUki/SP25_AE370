import numpy as np
import matplotlib.pyplot as plt
from matplotlib import animation

###### WAVEFIELD ANIMATION #########################################################################

def animate_wave(frames, X, Y, dt, vmin=None, vmax=None, interval=50, out_path=None, title_prefix=""):
    """
    Animate a sequence of wave field frames.

    Parameters
    ----------
    frames : list of 2D ndarrays
        Time-evolved field snapshots.
    X, Y : 2D ndarrays
        Meshgrid for plotting.
    dt : float
        Time step size.
    vmin, vmax : float or None
        Color scale limits. If None, use min/max from frames.
    interval : int
        Delay between frames in milliseconds.
    out_path : str or None
        If specified, save the animation to this path as .gif.
    title_prefix : str
        Title prefix for each frame.

    Returns
    -------
    ani : matplotlib.animation.FuncAnimation
        Animation object.
    """
    if vmin is None or vmax is None:
        all_vals = np.concatenate([f.ravel() for f in frames])
        vmin, vmax = np.percentile(all_vals, [1, 99])

    fig, ax = plt.subplots(figsize=(6, 5))
    ctf = ax.contourf(X, Y, frames[0], levels=100, cmap='viridis', vmin=vmin, vmax=vmax)
    cbar = fig.colorbar(ctf, ax=ax, label="Pressure")

    def update(i):
        ax.clear()
        contour = ax.contourf(X, Y, frames[i], levels=100, cmap='viridis', vmin=vmin, vmax=vmax)
        ax.set_title(f"{title_prefix}t = {i * dt * 1000:.2f} ms")
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        ax.set_aspect('equal')
        return contour.collections

    ani = animation.FuncAnimation(fig, update, frames=len(frames), interval=interval)

    if out_path:
        ani.save(out_path, writer="pillow", fps=1000 // interval)

    return ani


###### SNAPSHOT GRID PLOTS #########################################################################

def snapshot_grid_plot(snapshots, snapshot_times, X, Y, vmin=None, vmax=None, out_path=None):
    """
    Plot a grid of wavefield snapshots.

    Parameters
    ----------
    snapshots : list of 2D ndarrays
        Field snapshots to plot.
    snapshot_times : list of floats
        Corresponding times for each snapshot (in seconds).
    X, Y : 2D ndarrays
        Meshgrid for plotting.
    vmin, vmax : float or None
        Fixed color range. If None, use min/max from snapshots.
    out_path : str or None
        If specified, save the figure.
    """
    if vmin is None or vmax is None:
        all_vals = np.concatenate([f.ravel() for f in snapshots])
        vmin, vmax = np.percentile(all_vals, [1, 99])

    fig, axes = plt.subplots(3, 3, figsize=(12, 10))
    cbar_ax = fig.add_axes([1.00, 0.15, 0.02, 0.7])

    for ax, snap, t in zip(axes.flat, snapshots, snapshot_times):
        ctf = ax.contourf(X, Y, snap, levels=100, cmap='viridis', vmin=vmin, vmax=vmax)
        ax.set_title(f"t = {t*1000:.2f} ms")
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        ax.set_aspect('equal')

    for i in range(len(snapshots), 9):
        fig.delaxes(axes.flat[i])  # Hide unused subplots

        # Add colorbar linked to the last contour
    sm = plt.cm.ScalarMappable(cmap='viridis')
    sm.set_array([])  # dummy array for colorbar
    sm.set_clim(vmin, vmax)
    cbar = fig.colorbar(sm, cax=cbar_ax)
    cbar.set_label("Pressure")

    plt.subplots_adjust(right=0.93)  # avoid overlap with cbar
    plt.tight_layout()
    
    if out_path:
        plt.savefig(out_path, dpi=300, bbox_inches='tight')
    plt.show()