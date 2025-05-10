import numpy as np

###### BASIC SOURCE FUNCTIONS ######################################################################

def gaussian_pulse(t, t0=1.5, spread=0.5, amplitude=1.0):
    """
    Time-dependent Gaussian pulse.

    Parameters
    ----------
    t : float
        Current time (seconds).
    t0 : float
        Time center of the pulse.
    spread : float
        Controls pulse width.
    amplitude : float
        Pulse peak amplitude.

    Returns
    -------
    float
        Pulse amplitude at time t.
    """
    return amplitude * np.exp(-((t - t0) / spread) ** 2)

def two_source_pulse(t, loc1, loc2, t0=1.5, spread=0.5, amplitude=1.0):
    """
    Apply the same Gaussian pulse from two spatial locations.

    Parameters
    ----------
    t : float
        Current time.
    loc1, loc2 : tuple of ints
        Grid indices of the two sources.
    t0, spread, amplitude : float
        Pulse shape parameters.

    Returns
    -------
    dict
        Mapping of (i, j) locations to pulse amplitudes at time t.
    """
    pulse_val = gaussian_pulse(t, t0, spread, amplitude)
    return {loc1: pulse_val, loc2: pulse_val}


###### SYNTHETIC SPEECH SOURCES #####################################################################

def synthetic_syllable_wave(t, syllables=None):
    """
    Constructs a synthetic waveform mimicking 'heh - lo - there' using sine bursts.

    Parameters
    ----------
    t : float
        Current time.
    syllables : list of dicts or None
        If None, default to three pre-defined syllables.
        Each dict should contain:
            - 'start': start time
            - 'duration': length of syllable
            - 'freq': frequency in Hz (or lambda τ for time-varying pitch)
            - 'amp': (optional) amplitude

    Returns
    -------
    float
        Total waveform amplitude at time t.
    """
    if syllables is None:
        syllables = [
            {'start': 0.003, 'duration': 0.2, 'freq': 250, 'amp': 1.0},                   # "heh"
            {'start': 0.007, 'duration': 0.4, 'freq': lambda τ: 350 + 150 * τ},          # "lo"
            {'start': 0.013, 'duration': 0.5, 'freq': lambda τ: 500 - 200 * τ},          # "there"
        ]

    total = 0.0
    for s in syllables:
        start = s['start']
        dur = s['duration']
        amp = s.get('amp', 1.0)

        if start <= t <= start + dur:
            τ = (t - start) / dur                           # normalized time within syllable
            window = np.sin(np.pi * τ)                      # smooth window for fade-in/out
            freq = s['freq'](τ) if callable(s['freq']) else s['freq']
            total += amp * window * np.sin(2 * np.pi * freq * (t - start))

    return total

def dual_speaker_speech_wave(t, loc1, loc2):
    """
    Apply the synthetic 'heh - lo - there' waveform from two speakers simultaneously.

    Parameters
    ----------
    t : float
        Current time.
    loc1, loc2 : tuple of ints
        Source locations in the grid.

    Returns
    -------
    dict
        Mapping of each speaker location to the shared waveform value at time t.
    """
    val = synthetic_syllable_wave(t)
    return {loc1: val, loc2: val}
