"""
For each elevation in [0, 60]°, pick one naturalsounds165 stimulus at random, with
selection probability biased toward lower spectral centroid at low elevation and
higher spectral centroid at high elevation (Parise-style spectrum–elevation mapping).
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
TRAINING_STIM_DIR = f"{DIR}/data/raw/stimuli/naturalsounds165"
OUTPUT_DIR = f"{DIR}/stimuli/naturalsounds165_probability_bias"
CENTROID_CACHE = f"{RESULTS_DIR}/naturalsounds165_spectral_centroids.csv"
MANIFEST_PATH = f"{RESULTS_DIR}/naturalsounds165_probability_bias_manifest.csv"

ELEV_MIN = 0.0
ELEV_MAX = 60.0
ELEVATIONS = np.arange(0, 61, 10)  # 0, 10, …, 60
RNG_SEED = 42
# Gaussian width on centroid axis (Hz); smaller → sharper elevation–centroid coupling
SIGMA_HZ = 1800.0

os.makedirs(OUTPUT_DIR, exist_ok=True)
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


def build_centroid_table(stim_dir):
    stim_files = sorted(f for f in os.listdir(stim_dir) if f.startswith("stim") and f.endswith(".wav"))
    rows = []
    for fname in stim_files:
        path = os.path.join(stim_dir, fname)
        x, sr = load_mono(path)
        rows.append({"stimulus": fname, "spectral_centroid_hz": spectral_centroid(x, sr)})
    return pd.DataFrame(rows)


def elevation_to_target_centroid(elevation, elev_min, elev_max, cent_min, cent_max):
    elev = np.clip(elevation, elev_min, elev_max)
    t = (elev - elev_min) / (elev_max - elev_min)
    return cent_min + t * (cent_max - cent_min)


def selection_probabilities(centroids_hz, target_hz, sigma_hz):
    z = (np.asarray(centroids_hz, dtype=float) - target_hz) / sigma_hz
    weights = np.exp(-0.5 * z ** 2)
    total = weights.sum()
    if total <= 0:
        return np.full(len(weights), 1.0 / len(weights))
    return weights / total


def pick_sound_for_elevation(centroid_df, elevation, rng, sigma_hz=SIGMA_HZ):
    centroids = centroid_df["spectral_centroid_hz"].to_numpy()
    c_min, c_max = centroids.min(), centroids.max()
    target = elevation_to_target_centroid(elevation, ELEV_MIN, ELEV_MAX, c_min, c_max)
    probs = selection_probabilities(centroids, target, sigma_hz)
    idx = rng.choice(len(centroid_df), p=probs)
    row = centroid_df.iloc[idx]
    return {
        "elevation": float(elevation),
        "stimulus": row["stimulus"],
        "spectral_centroid_hz": float(row["spectral_centroid_hz"]),
        "target_centroid_hz": float(target),
    }


if os.path.isfile(CENTROID_CACHE):
    centroid_df = pd.read_csv(CENTROID_CACHE)
else:
    centroid_df = build_centroid_table(TRAINING_STIM_DIR)
    centroid_df.to_csv(CENTROID_CACHE, index=False)

rng = np.random.default_rng(RNG_SEED)
manifest_rows = []

for repeat_idx in range(10):
    for elevation in ELEVATIONS:
        pick = pick_sound_for_elevation(centroid_df, elevation, rng)
        pick = pick.copy()
        pick["repeat_index"] = repeat_idx + 1
        manifest_rows.append(pick)
        print(
            f"rep {repeat_idx+1}, {elevation:3.0f}° → {pick['stimulus']} "
            f"(centroid {pick['spectral_centroid_hz']:.0f} Hz, "
            f"target {pick['target_centroid_hz']:.0f} Hz)"
        )

manifest_df = pd.DataFrame(manifest_rows)
manifest_df.to_csv(MANIFEST_PATH, index=False)
print(f"\nWrote {len(manifest_df)} files to {OUTPUT_DIR}")
print(f"Manifest: {MANIFEST_PATH}")

# --- diagnostic: selected centroid vs elevation ---
fig, ax = plt.subplots(figsize=(6, 4))
ax.scatter(
    manifest_df["elevation"],
    manifest_df["spectral_centroid_hz"],
    s=80,
    c=manifest_df["elevation"],
    cmap="inferno",
    edgecolors="k",
    linewidths=0.6,
    zorder=3,
)
cent_all = centroid_df["spectral_centroid_hz"]
t_line = np.linspace(ELEV_MIN, ELEV_MAX, 100)
target_line = elevation_to_target_centroid(
    t_line, ELEV_MIN, ELEV_MAX, cent_all.min(), cent_all.max()
)
ax.plot(t_line, target_line, "k--", linewidth=1.2, label="Target centroid vs elevation")
ax.set_xlim(ELEV_MIN - 2, ELEV_MAX + 2)
ax.set_xlabel("Elevation (°)")
ax.set_ylabel("Spectral centroid of selected sound (Hz)")
ax.set_title("Probability-biased sound selection (one draw per elevation)")
ax.legend(frameon=False)
ax.grid(True, alpha=0.25)
fig.tight_layout()
fig.savefig(f"{PLOT_DIR}/probability_bias_selection.svg", dpi=300)
fig.savefig(f"{PLOT_DIR}/probability_bias_selection.png", dpi=300)
plt.close(fig)
