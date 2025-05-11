import numpy as np

def get_tick_labels(vmin, vmax, count=9):
    """
    Returns tick locations and formatted string labels for a colorbar.

    Parameters:
        vmin, vmax : float
            Min and max values for the data range
        count : int
            Number of ticks (default is 9 for symmetric layout)

    Returns:
        tick_vals : np.ndarray
        tick_labels : list of str
    """
    tick_vals = np.linspace(vmin, vmax, count)
    tick_labels = [f"{v:+.4f}" for v in tick_vals]
    return tick_vals, tick_labels

def frame_sampler(Nt, fps=30, duration=2.0):
    """
    Returns a list of time step indices to sample evenly spaced frames for animation.

    Parameters:
        Nt : int
            Total number of simulation steps
        fps : int
            Frames per second of output animation
        duration : float
            Desired duration of animation in seconds

    Returns:
        frame_indices : list of int
    """
    total_frames = int(fps * duration)
    return np.linspace(0, Nt - 1, total_frames, dtype=int)
