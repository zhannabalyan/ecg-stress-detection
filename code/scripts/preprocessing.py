"""
preprocessing.py

Data loading and preprocessing functions for the WESAD ECG dataset.

WESAD labels:
    1 = Baseline (resting state)
    2 = Stress (TSST protocol)

Sensor: RespiBAN chest ECG at 700 Hz.
"""

import os
import pickle
import numpy as np
from scipy import signal as scipy_signal

# Global constants 

FS        = 700    # Sampling frequency (Hz)
TRAIN_SEC = 300    # Training split: first 5 min of baseline for motif learning
VAL_SEC   = 180    # Validation split: next 3 min for threshold tuning
SUBJECTS  = [f"S{i}" for i in range(2, 18) if i != 12]  # S12 not collected(missing)


#  Data loading 

def load_subject(subject_id: str, data_path: str) -> tuple:
    """
    Load one WESAD subject's chest ECG and return preprocessed
    (baseline_ecg, stress_ecg) as 1-D float arrays at 700 Hz.

    Parameters:
   
    subject_id : str
        Subject identifier, e.g. 'S2'.
    data_path : str
        Absolute path to the directory containing .pkl files.

    Returns:
    
    baseline_ecg : np.ndarray
        Preprocessed resting ECG (label == 1).
    stress_ecg : np.ndarray
        Preprocessed stress ECG (label == 2).
    """
    path_a = os.path.join(data_path, f"{subject_id}.pkl")
    path_b = os.path.join(data_path, subject_id, f"{subject_id}.pkl")

    if os.path.exists(path_a):
        path = path_a
    elif os.path.exists(path_b):
        path = path_b
    else:
        raise FileNotFoundError(
            f"Cannot find {subject_id}.pkl\n"
            f"  Tried: {path_a}\n"
            f"  Tried: {path_b}"
        )

    with open(path, "rb") as f:
        data = pickle.load(f, encoding="latin1")

    ecg_raw = data["signal"]["chest"]["ECG"].flatten()
    labels  = data["label"].flatten()

    baseline_ecg = preprocess_ecg(ecg_raw[labels == 1])
    stress_ecg   = preprocess_ecg(ecg_raw[labels == 2])
    return baseline_ecg, stress_ecg


def preprocess_ecg(sig: np.ndarray, fs: int = FS) -> np.ndarray:
    """
    Preprocess a raw ECG segment.

    Step 1 : Bandpass filter (0.5–45 Hz, 3rd-order Butterworth):
        Removes baseline wander and high-frequency noise while preserving
        P-QRS-T waveform morphology.

    Step 2 : Z-score normalisation:
        Eliminates inter-subject amplitude differences so motif distances
        are comparable across subjects.

    Parameters:
    
    sig : np.ndarray
        Raw ECG signal.
    fs : int
        Sampling frequency in Hz.

    Returns:
    
    np.ndarray
        Filtered and z-score normalised ECG.
    """
    nyq  = 0.5 * fs
    b, a = scipy_signal.butter(3, [0.5 / nyq, 45.0 / nyq], btype="band")
    filt = scipy_signal.filtfilt(b, a, sig)
    return (filt - filt.mean()) / filt.std()


# Train / Validation / Test split 

def split_subject_data(
    baseline_ecg: np.ndarray,
    stress_ecg:   np.ndarray,
    train_sec: int = TRAIN_SEC,
    val_sec:   int = VAL_SEC,
    fs:        int = FS,
) -> tuple:
    """
    Strict train / validation / test split.
    No sample appears in more than one partition.

    Parameters:
   
    baseline_ecg : np.ndarray
        Full preprocessed baseline ECG.
    stress_ecg : np.ndarray
        Full preprocessed stress ECG.
    train_sec : int
        Duration of the training split in seconds.
    val_sec : int
        Duration of each validation split in seconds.
    fs : int
        Sampling frequency in Hz.

    Returns:
    
    train_baseline : np.ndarray   for motif learning
    val_baseline   : np.ndarray   for threshold tuning
    val_stress     : np.ndarray   for threshold tuning
    test_baseline  : np.ndarray   for final evaluation
    test_stress    : np.ndarray   for final evaluation
    """
    train_n = train_sec * fs
    val_n   = val_sec   * fs
    return (
        baseline_ecg[:train_n],
        baseline_ecg[train_n : train_n + val_n],
        stress_ecg[:val_n],
        baseline_ecg[train_n + val_n:],
        stress_ecg[val_n:],
    )
