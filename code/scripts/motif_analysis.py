"""
motif_analysis.py

Matrix Profile motif learning and anomaly scoring functions.

Workflow:
    1. get_motifs()              : learn K recurring patterns from baseline
    2. compute_distance_profile() : score any ECG segment against those motifs
    3. find_best_threshold()     : tune detection threshold on validation set
"""

import numpy as np
import pandas as pd
import stumpy
from sklearn.metrics import f1_score

from preprocessing import FS

# Default hyperparameters

M_FIXED = 1050   # Window size: 1050 samples = 1.5 s at 700 Hz
                 # Captures one complete P-QRS-T cardiac cycle.
K_FIXED = 5      # Number of motifs to extract from the baseline.


# Motif learning 

def get_motifs(
    calib_signal: np.ndarray,
    m: int = M_FIXED,
    k: int = K_FIXED,
) -> list:
    """
    Compute the Matrix Profile and extract k non-overlapping motifs, which
    the most repetitive heartbeat subsequences in the signal.

    Motifs are sorted by ascending Matrix Profile value (lowest = most
    repetitive). An exclusion zone of m samples prevents overlapping
    selections.

    Parameters:
    
    calib_signal : np.ndarray
        Training baseline ECG (1-D).
    m : int
        Subsequence window length in samples.
    k : int
        Number of motifs to extract.

    Returns:
    
    list of np.ndarray
        List of k motif arrays, each of length m.
    """
    mp          = stumpy.stump(calib_signal, m)
    sorted_idxs = np.argsort(mp[:, 0])
    chosen      = []

    for idx in sorted_idxs:
        if len(chosen) >= k:
            break
        if all(abs(idx - s) > m for s in chosen):
            chosen.append(idx)

    return [calib_signal[i : i + m] for i in chosen]


# Anomaly scoring 

def compute_distance_profile(
    ecg_segment:      np.ndarray,
    motifs:           list,
    smoothing_window: int = FS * 3,
) -> np.ndarray:
    """
    Score every sliding window in ecg_segment by computing the minimum
    z-normalised Euclidean distance to any of the learned motifs.

    A 3-second moving-average smoothing is applied to reduce transient
    noise. High distance means that window looks unlike normal baseline, so it indicates stress.

    Parameters:
    
    ecg_segment : np.ndarray
        ECG array to score (1-D).
    motifs : list of np.ndarray
        Learned baseline motifs from get_motifs().
    smoothing_window : int
        Rolling average window in samples (default: 3 s).

    Returns:
    
    np.ndarray
        Anomaly score array, same length as ecg_segment.
    """
    all_dists = [stumpy.mass(motif, ecg_segment) for motif in motifs]
    min_dist  = np.min(all_dists, axis=0)
    padded    = np.pad(
        min_dist,
        (0, len(ecg_segment) - len(min_dist)),
        constant_values=np.nan,
    )
    smoothed = (
        pd.Series(padded)
        .rolling(window=smoothing_window, min_periods=1)
        .mean()
        .values
    )
    return smoothed


#  Threshold tuning 

def find_best_threshold(
    val_bl_dist: np.ndarray,
    val_st_dist: np.ndarray,
    pct_range=range(50, 100),
) -> tuple:
    """
    Grid search over percentiles 50–99 on the validation set to find the
    threshold that maximises F1 score.

    The threshold is derived from baseline distances only, which means stress labels
    are used only to evaluate F1, not to fit the threshold value.

    Parameters:
   
    val_bl_dist : np.ndarray
        Anomaly scores on the validation baseline.
    val_st_dist : np.ndarray
        Anomaly scores on the validation stress.
    pct_range : range
        Percentile grid to search.

    Returns:
    
    results_df : pd.DataFrame
        Columns: [percentile, threshold, f1]
    best_pct : int
        Percentile that maximised validation F1.
    best_thr : float
        Corresponding threshold value.
    best_f1 : float
        Validation F1 at best_pct.
    """
    y_true   = np.concatenate([np.zeros(len(val_bl_dist)),
                                np.ones(len(val_st_dist))])
    all_dist = np.concatenate([val_bl_dist, val_st_dist])
    valid    = ~np.isnan(all_dist)

    best_f1  = 0
    best_pct = 90
    best_thr = np.nanpercentile(val_bl_dist, 90)
    rows     = []

    for pct in pct_range:
        thr   = np.nanpercentile(val_bl_dist, pct)
        y_pred = (all_dist > thr).astype(int)
        f1    = f1_score(y_true[valid], y_pred[valid], zero_division=0)
        rows.append({"percentile": pct, "threshold": thr, "f1": f1})
        if f1 > best_f1:
            best_f1, best_pct, best_thr = f1, pct, thr

    return pd.DataFrame(rows), best_pct, best_thr, best_f1
