import numpy as np
import slab
from slab.sound import Sound


def elevation_to_center_hz(elevation, elev_min, elev_max, f_low=400.0, f_high=6300.0):
    """Map elevation to Gaussian center on a log-frequency axis (Parise: low elev → low freq)."""
    elev = np.clip(elevation, elev_min, elev_max)
    t = (elev - elev_min) / (elev_max - elev_min)
    log_fc = np.log2(f_low) + t * (np.log2(f_high) - np.log2(f_low))
    return 2.0 ** log_fc

def gaussian_log_gain(freqs_hz, center_hz, sigma_octaves, peak_gain_db):
    """
    Gaussian emphasis in log2-frequency.
    peak_gain_db: boost at center relative to skirts (before RMS normalization).
    """
    freqs_hz = np.asarray(freqs_hz, dtype=float)
    center_hz = max(center_hz, 1.0)
    log_f = np.log2(np.maximum(freqs_hz, 1.0))
    log_fc = np.log2(center_hz)
    gain_db = peak_gain_db * np.exp(-0.5 * ((log_f - log_fc) / sigma_octaves) ** 2)
    return 10.0 ** (gain_db / 20.0)

def apply_scene_eq(x, sr, elevation, elev_min=0.0, elev_max=70.0,
                   f_low=400.0, f_high=6300.0, sigma_octaves=1.2,
                   peak_gain_db=6.0):
    """
    Shape spectrum with elevation-dependent Gaussian emphasis; preserve RMS power.
    """
    x = np.asarray(x, dtype=float)
    if x.ndim == 2:
        x = x.mean(axis=1)
    rms_orig = np.sqrt(np.mean(x ** 2))
    if rms_orig < 1e-12:
        return x
    n = len(x)
    freqs = np.fft.rfftfreq(n, d=1.0 / sr)
    spectrum = np.fft.rfft(x)
    center_hz = elevation_to_center_hz(elevation, elev_min, elev_max, f_low, f_high)
    gain = gaussian_log_gain(freqs, center_hz, sigma_octaves, peak_gain_db)
    y = np.fft.irfft(spectrum * gain, n=n)
    rms_new = np.sqrt(np.mean(y ** 2))
    if rms_new > 1e-12:
        y *= rms_orig / rms_new
    return y

def shape_training_sound(sound: Sound, elevation, **eq_kwargs):
    """Load mono slab.Sound, apply scene EQ, return new slab.Sound."""
    x = np.asarray(sound.data)
    if x.ndim == 2:
        x_mono = x.mean(axis=1)
    else:
        x_mono = x
    y = apply_scene_eq(x_mono, int(sound.samplerate), elevation, **eq_kwargs)
    return slab.Sound(y, samplerate=sound.samplerate)
