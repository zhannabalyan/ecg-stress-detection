"""
hrv_features.py

Heart Rate Variability (HRV) feature extraction and combined
Motif+HRV anomaly scoring for the supplementary HRV experiment.

HRV captures changes that waveform shape alone may not fully express,
particularly the rigidity of the heartbeat interval during sympathetic
nervous system activation under stress.

Features computed:

mean_rr : float   Average RR interval (ms)       ↓ under stress
sdnn    : float   Std dev of RR intervals (ms)    ↓ under stress
rmssd   : float   Root mean square of successive
                  RR differences (ms)             ↓ under stress
hr      : float   Mean heart rate (bpm)           ↑ under stress
"""

import numpy as np
import pandas as pd
import stumpy
from scipy.signal import find_peaks
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import f1_score

from preprocessing import FS

# HRV hyperparameters 

HRV_WIN_SEC  = 15   # Sliding window length (s) 
HRV_STEP_SEC = 5    # Step size (s)

# Features and their expected direction under stress:
#   +1 → increases under stress,  -1 → decreases under stress
FEATURE_COLS        = ["motif_dist", "rmssd"]
EXPECTED_DIRECTIONS = {"motif_dist": +1, "rmssd": -1}


# HRV feature extraction 

def extract_hrv_features(
    ecg_segment: np.ndarray,
    fs:          int = FS,
    window_sec:  int = HRV_WIN_SEC,
    step_sec:    int = HRV_STEP_SEC,
) -> pd.DataFrame:
    """
    Slide a window over ecg_segment, detect R-peaks in each window,
    and compute four time-domain HRV features.

    R-peak detection parameters:
        height   = 0.5  : threshold for z-scored ECG
        distance = 0.4 s: enforces physiological minimum RR interval

    Windows with fewer than 4 detected beats are skipped because too few
    RR intervals would produce unreliable statistics.

    Parameters:
    
    ecg_segment : np.ndarray
        Preprocessed ECG signal (1-D).
    fs : int
        Sampling frequency in Hz.
    window_sec : int
        Analysis window length in seconds.
    step_sec : int
        Step size between consecutive windows in seconds.

    Returns:
   
    pd.DataFrame
        Columns: [window_center_s, mean_rr, sdnn, rmssd, hr]
        One row per valid window.
    """
    win  = int(window_sec * fs)
    step = int(step_sec   * fs)
    rows = []

    for start in range(0, len(ecg_segment) - win, step):
        chunk = ecg_segment[start : start + win]
        peaks, _ = find_peaks(chunk, height=0.5, distance=int(0.4 * fs))
        if len(peaks) < 4:
            continue
        rr = np.diff(peaks) / fs * 1000   
        rows.append({
            "window_center_s": (start + win / 2) / fs,
            "mean_rr":         np.mean(rr),
            "sdnn":            np.std(rr),
            "rmssd":           np.sqrt(np.mean(np.diff(rr) ** 2)),
            "hr":              60000 / np.mean(rr),
        })

    return pd.DataFrame(rows)


# Combined Motif+HRV feature extraction 

def compute_combined_features(
    ecg_segment: np.ndarray,
    motifs:      list,
    m:           int = 1050,
    fs:          int = FS,
    window_sec:  int = HRV_WIN_SEC,
    step_sec:    int = HRV_STEP_SEC,
) -> pd.DataFrame:
    """
    For each sliding window, compute motif distance and HRV features
    together, returning a single unified DataFrame.

    Motif distance is the mean minimum z-normalised Euclidean distance
    from every sub-window inside the window to any learned motif.

    Parameters:
    
    ecg_segment : np.ndarray
        Preprocessed ECG signal (1-D).
    motifs : list of np.ndarray
        Learned baseline motifs.
    m : int
        Motif window length in samples.
    fs : int
        Sampling frequency in Hz.
    window_sec : int
        Window length in seconds.
    step_sec : int
        Step size in seconds.

    Returns:
    
    pd.DataFrame
        Columns: [window_center_s, motif_dist, mean_rr, sdnn, rmssd, hr]
    """
    win  = int(window_sec * fs)
    step = int(step_sec   * fs)
    rows = []

    for start in range(0, len(ecg_segment) - win, step):
        chunk = ecg_segment[start : start + win]

        # Motif distance
        all_dists = [stumpy.mass(motif, chunk) for motif in motifs]
        mean_dist = np.nanmean(np.min(all_dists, axis=0))

        # HRV
        peaks, _ = find_peaks(chunk, height=0.5, distance=int(0.4 * fs))
        if len(peaks) < 4:
            continue
        rr = np.diff(peaks) / fs * 1000
        rows.append({
            "window_center_s": (start + win / 2) / fs,
            "motif_dist":      mean_dist,
            "mean_rr":         np.mean(rr),
            "sdnn":            np.std(rr),
            "rmssd":           np.sqrt(np.mean(np.diff(rr) ** 2)),
            "hr":              60000 / np.mean(rr),
        })

    return pd.DataFrame(rows)


# ── Combined scoring ──────────────────────────────────────────────────────────

def build_combined_score(
    tr_bl_feat:  pd.DataFrame,
    val_bl_feat: pd.DataFrame,
    val_st_feat: pd.DataFrame,
    te_bl_feat:  pd.DataFrame,
    te_st_feat:  pd.DataFrame,
) -> tuple:
    """
    Fit a StandardScaler on training baseline features, then build a
    balanced anomaly score by averaging motif_dist (+1) and rmssd (-1)
    after sign-alignment so that higher score always means more stressed.

    Parameters
    ----------
    tr_bl_feat  : training baseline feature DataFrame  (for scaler fitting)
    val_bl_feat : validation baseline feature DataFrame
    val_st_feat : validation stress feature DataFrame
    te_bl_feat  : test baseline feature DataFrame
    te_st_feat  : test stress feature DataFrame

    Returns
    -------
    val_bl_scores, val_st_scores, te_bl_scores, te_st_scores : np.ndarray
    scaler       : fitted StandardScaler
    directions   : dict of {feature: sign}
    """
    scaler  = StandardScaler()
    scaler.fit(tr_bl_feat[FEATURE_COLS])
    dir_vec = np.array([EXPECTED_DIRECTIONS[f] for f in FEATURE_COLS])

    def _score(feat_df):
        scaled   = scaler.transform(feat_df[FEATURE_COLS])
        directed = scaled * dir_vec
        return (directed[:, 0] + directed[:, 1]) / 2   # motif + rmssd

    return (
        _score(val_bl_feat),
        _score(val_st_feat),
        _score(te_bl_feat),
        _score(te_st_feat),
        scaler,
        EXPECTED_DIRECTIONS,
    )


def find_best_threshold_1d(
    val_bl_scores: np.ndarray,
    val_st_scores: np.ndarray,
    pct_range=range(50, 100),
) -> tuple:
    """
    Grid search for the best percentile threshold on 1-D score arrays.
    Identical logic to motif_analysis.find_best_threshold() but operates
    on combined scores rather than raw distance profiles.

    Returns
    -------
    best_pct : int
    best_thr : float
    best_f1  : float
    """
    y_true = np.concatenate([np.zeros(len(val_bl_scores)),
                              np.ones(len(val_st_scores))])
    all_sc = np.concatenate([val_bl_scores, val_st_scores])
    best_f1, best_pct, best_thr = 0, 90, np.percentile(val_bl_scores, 90)

    for pct in pct_range:
        thr   = np.percentile(val_bl_scores, pct)
        y_pred = (all_sc > thr).astype(int)
        f1    = f1_score(y_true, y_pred, zero_division=0)
        if f1 > best_f1:
            best_f1, best_pct, best_thr = f1, pct, thr

    return best_pct, best_thr, best_f1
