"""
For each naturalsounds165 stimulus, draw several elevations in [0, 60]°, with
selection probability biased toward lower elevation for low spectral centroid and
higher elevation for high centroid (inverse of the Parise-style spectrum–elevation mapping).
"""

import os

import librosa
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import slab

plt.rcParams["svg.fonttype"] = "none"

DIR = os.getcwd()
RESULTS_DIR = f"{DIR}/data/results"
PLOT_DIR = f"{DIR}/data/plots"
TRAINING_STIM_DIR = f"{DIR}/data/stimuli/naturalsounds165"
CENTROID_CACHE = f"{RESULTS_DIR}/data/naturalsounds165_spectral_centroids.csv"
MANIFEST_PATH = f"{RESULTS_DIR}/data/naturalsounds165_probability_bias_manifest.csv"

ELEV_MIN = 0.0
ELEV_MAX = 60.0
N_LOCATIONS_PER_SOUND = 10
RNG_SEED = 42
# Gaussian width on centroid axis in the old sound-selection formulation; converted to ° on elevation
SIGMA_HZ = 1600.0

os.makedirs(RESULTS_DIR, exist_ok=True)

def spectral_centroid(x, sr):
    S = np.abs(librosa.stft(x, n_fft=2048, hop_length=512))
    return float(np.mean(librosa.feature.spectral_centroid(S=S, sr=sr)[0]))


def load_mono(path):
    sound = slab.Sound(path)
    x = np.asarray(sound.data).squeeze()
    if x.ndim == 2:
        x = x.mean(axis=1)
    return x, int(sound.samplerate)


def build_centroid_table(stim_paths):
    rows = []
    for path in stim_paths:
        x, sr = load_mono(path)
        rows.append({"stimulus": path, "spectral_centroid_hz": spectral_centroid(x, sr)})
    return pd.DataFrame(rows).set_index("stimulus")


def centroid_to_target_elevation(centroid_hz, cent_min, cent_max, elev_min, elev_max):
    centroid_hz = np.clip(centroid_hz, cent_min, cent_max)
    t = (centroid_hz - cent_min) / (cent_max - cent_min)
    return elev_min + t * (elev_max - elev_min)


def hz_sigma_to_elev_sigma(sigma_hz, cent_min, cent_max, elev_min, elev_max):
    span_cent = cent_max - cent_min
    span_elev = elev_max - elev_min
    if span_cent <= 0:
        return float(sigma_hz)
    return float(sigma_hz * span_elev / span_cent)


def selection_probabilities(values, target, sigma):
    z = (np.asarray(values, dtype=float) - target) / sigma
    weights = np.exp(-0.5 * z**2)
    total = weights.sum()
    if total <= 0:
        return np.full(len(weights), 1.0 / len(weights))
    return weights / total


def compute_elevation_distribution_for_centroid(
    centroid_hz,
    # n_locations,
    cent_min,
    cent_max,
    elev_min=ELEV_MIN,
    elev_max=ELEV_MAX,
    sigma_elev=None,
    elevation_grid=None,
):
    if elevation_grid is None:
        elevation_grid = np.linspace(elev_min, elev_max, int(elev_max - elev_min) + 1)
    target_elev = centroid_to_target_elevation(
        centroid_hz, cent_min, cent_max, elev_min, elev_max
    )
    if sigma_elev is None:
        sigma_elev = hz_sigma_to_elev_sigma(
            SIGMA_HZ, cent_min, cent_max, elev_min, elev_max
        )
    probs = selection_probabilities(elevation_grid, target_elev, sigma_elev)
    # chosen = rng.choice(elevation_grid, size=n_locations, p=probs)
    # return chosen, float(target_elev)
    return probs
