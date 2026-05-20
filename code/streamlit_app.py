"""
streamlit_app.py

Interactive dashboard for the ECG Stress Detection.

Run from the code/ directory:
    streamlit run streamlit_app.py
"""

import os
import pickle
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import streamlit as st
import stumpy
from scipy import signal as scipy_signal
from scipy.signal import find_peaks
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, confusion_matrix, ConfusionMatrixDisplay,
)

# Page config
st.set_page_config(
    page_title="ECG Stress Detection Dashboard",
    page_icon="🫀",
    layout="wide",
)

# Constants
DATA_PATH = os.path.normpath(os.path.join(os.path.dirname(__file__), "..", "data"))
SUBJECTS  = [f"S{i}" for i in range(2, 18) if i != 12]
FS        = 700
M_FIXED   = 1050
K_FIXED   = 5
TRAIN_SEC = 300
VAL_SEC   = 180
HRV_WIN_SEC  = 15
HRV_STEP_SEC = 5

# Pipeline functions

@st.cache_data(show_spinner=False)
def load_subject(subject_id):
    path_a = os.path.join(DATA_PATH, f"{subject_id}.pkl")
    path_b = os.path.join(DATA_PATH, subject_id, f"{subject_id}.pkl")
    path   = path_a if os.path.exists(path_a) else path_b
    with open(path, "rb") as f:
        data = pickle.load(f, encoding="latin1")
    ecg_raw = data["signal"]["chest"]["ECG"].flatten()
    labels  = data["label"].flatten()
    return _preprocess(ecg_raw[labels == 1]), _preprocess(ecg_raw[labels == 2])


def _preprocess(sig):
    nyq  = 0.5 * FS
    b, a = scipy_signal.butter(3, [0.5 / nyq, 45.0 / nyq], btype="band")
    filt = scipy_signal.filtfilt(b, a, sig)
    return (filt - filt.mean()) / filt.std()


def split_data(bl, st):
    tn, vn = TRAIN_SEC * FS, VAL_SEC * FS
    return (
        bl[:tn],
        bl[tn:tn+vn], st[:vn],
        bl[tn+vn:],   st[vn:],
    )


@st.cache_data(show_spinner=False)
def get_motifs(calib, m=M_FIXED, k=K_FIXED):
    mp = stumpy.stump(calib, m)
    idxs, chosen = np.argsort(mp[:, 0]), []
    for idx in idxs:
        if len(chosen) >= k:
            break
        if all(abs(idx - s) > m for s in chosen):
            chosen.append(idx)
    return [calib[i:i+m] for i in chosen]


@st.cache_data(show_spinner=False)
def distance_profile(seg, _motifs_key, motifs, smooth=FS*3):

    dists   = [stumpy.mass(mo, seg) for mo in motifs]
    min_d   = np.min(dists, axis=0)
    padded  = np.pad(min_d, (0, len(seg)-len(min_d)), constant_values=np.nan)
    return (pd.Series(padded)
              .rolling(window=smooth, min_periods=1)
              .mean().values)


def best_threshold(vbl_d, vst_d):
    y_true = np.concatenate([np.zeros(len(vbl_d)), np.ones(len(vst_d))])
    all_d  = np.concatenate([vbl_d, vst_d])
    valid  = ~np.isnan(all_d)
    best_f1, best_pct, best_thr = 0, 90, np.nanpercentile(vbl_d, 90)
    for pct in range(50, 100):
        thr  = np.nanpercentile(vbl_d, pct)
        yp   = (all_d > thr).astype(int)
        f1   = f1_score(y_true[valid], yp[valid], zero_division=0)
        if f1 > best_f1:
            best_f1, best_pct, best_thr = f1, pct, thr
    return best_pct, best_thr, best_f1


def extract_hrv(seg):
    win, step = HRV_WIN_SEC*FS, HRV_STEP_SEC*FS
    rows = []
    for s in range(0, len(seg)-win, step):
        chunk = seg[s:s+win]
        peaks, _ = find_peaks(chunk, height=0.5, distance=int(0.4*FS))
        if len(peaks) < 4:
            continue
        rr = np.diff(peaks) / FS * 1000
        rows.append({
            "t":     (s + win/2) / FS,
            "rmssd": np.sqrt(np.mean(np.diff(rr)**2)),
            "hr":    60000 / np.mean(rr),
        })
    return pd.DataFrame(rows)


# Per-subject pipeline

@st.cache_data(show_spinner=True)
def run_subject(sub):
    bl, st = load_subject(sub)
    tr_bl, val_bl, val_st, te_bl, te_st = split_data(bl, st)
    motifs = get_motifs(tr_bl)
    motif_key = sub  # cache key

    vbl_d = distance_profile(val_bl, motif_key+"_vbl", motifs)
    vst_d = distance_profile(val_st, motif_key+"_vst", motifs)
    pct, thr, val_f1 = best_threshold(vbl_d, vst_d)

    tbl_d = distance_profile(te_bl, motif_key+"_tbl", motifs)
    tst_d = distance_profile(te_st, motif_key+"_tst", motifs)
    n     = min(len(tbl_d), len(tst_d))
    tbl_d, tst_d = tbl_d[:n], tst_d[:n]

    yt = np.concatenate([np.zeros(n), np.ones(n)])
    yp_raw = np.concatenate([(tbl_d > thr).astype(int),
                              (tst_d > thr).astype(int)])
    valid = ~np.isnan(yp_raw)
    yt_v, yp_v = yt[valid], yp_raw[valid].astype(int)

    metrics = dict(
        acc  = accuracy_score(yt_v, yp_v),
        prec = precision_score(yt_v, yp_v, zero_division=0),
        rec  = recall_score(yt_v, yp_v, zero_division=0),
        f1   = f1_score(yt_v, yp_v, zero_division=0),
        cm   = confusion_matrix(yt_v, yp_v),
        sep  = np.nanmean(tst_d) / np.nanmean(tbl_d),
        val_f1 = val_f1,
    )

    hrv_bl = extract_hrv(te_bl)
    hrv_st = extract_hrv(te_st)

    worst_idx = int(np.nanargmax(tst_d))
    worst_seg = te_st[worst_idx:worst_idx+M_FIXED]

    return dict(
        motifs=motifs, thr=thr, pct=pct,
        tbl_d=tbl_d, tst_d=tst_d,
        te_bl=te_bl, te_st=te_st,
        metrics=metrics,
        hrv_bl=hrv_bl, hrv_st=hrv_st,
        worst_seg=worst_seg, worst_dist=tst_d[worst_idx],
        bl_mean=np.nanmean(tbl_d), st_mean=np.nanmean(tst_d),
    )


# Plotting helpers

BLUE, RED, GREEN = "#4A90D9", "#E05C5C", "#2E8B57"
DARK_RED, DARK_GREEN = "#8B0000", "#1A5C35"


def fig_motifs(motifs, sub):
    fig, axes = plt.subplots(1, K_FIXED, figsize=(16, 3), sharey=True)
    for i, mo in enumerate(motifs):
        axes[i].plot(mo, color=DARK_GREEN, lw=1.8)
        axes[i].set_title(f"Motif {i+1}", fontsize=9)
        axes[i].set_xlabel("Samples")
        axes[i].grid(alpha=0.3)
    axes[0].set_ylabel("Normalised Amplitude")
    fig.suptitle(f"{sub} — 5 Learned Baseline Motifs  (m={M_FIXED}, 1.50 s/window)", fontsize=12)
    plt.tight_layout()
    return fig


def fig_motifs_overlay(motifs, sub):
    fig, ax = plt.subplots(figsize=(7, 4))
    for i, mo in enumerate(motifs):
        ax.plot(mo, lw=1.5, alpha=0.75, label=f"Motif {i+1}")
    ax.set_title(f"{sub} — All 5 Motifs Overlaid")
    ax.set_xlabel("Samples"); ax.set_ylabel("Normalised Amplitude")
    ax.legend(fontsize=8); ax.grid(alpha=0.3)
    plt.tight_layout()
    return fig


def fig_anomaly_timeline(tbl_d, tst_d, thr, pct, sub):
    fig, ax = plt.subplots(figsize=(14, 4))
    t_bl = np.arange(len(tbl_d)) / FS
    t_st = np.arange(len(tst_d)) / FS
    ax.plot(t_bl, tbl_d, color=BLUE,   alpha=0.7, lw=0.7, label="Baseline")
    ax.plot(t_st, tst_d, color=RED,    alpha=0.7, lw=0.7, label="Stress")
    ax.axhline(thr, color="black", linestyle="--",
               label=f"Threshold ({pct}th pct = {thr:.3f})")
    ax.set_xlabel("Time (s)"); ax.set_ylabel("Distance to Nearest Motif")
    ax.set_title(f"{sub} — Anomaly Score Over Time")
    ax.legend(); ax.grid(alpha=0.3)
    plt.tight_layout()
    return fig


def fig_distributions(tbl_d, tst_d, thr, sub):
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.hist(tbl_d[~np.isnan(tbl_d)], bins=80, alpha=0.65, color=BLUE,  label="Baseline")
    ax.hist(tst_d[~np.isnan(tst_d)], bins=80, alpha=0.65, color=RED,   label="Stress")
    ax.axvline(thr, color="black", linestyle="--", label="Threshold")
    ax.set_xlabel("Distance"); ax.set_ylabel("Count")
    ax.set_title(f"{sub} — Distance Distributions")
    ax.legend(); ax.grid(alpha=0.3)
    plt.tight_layout()
    return fig


def fig_confusion(cm, sub):
    fig, ax = plt.subplots(figsize=(4, 4))
    ConfusionMatrixDisplay(cm, display_labels=["Normal", "Stress"]).plot(
        ax=ax, colorbar=False, cmap="Blues")
    ax.set_title(f"{sub} — Confusion Matrix")
    plt.tight_layout()
    return fig


def fig_shape_comparison(motif, worst_seg, worst_dist, sub):
    fig, axes = plt.subplots(1, 3, figsize=(16, 4))
    axes[0].plot(motif,     color=DARK_GREEN, lw=2); axes[0].set_title("Normal Motif")
    axes[1].plot(worst_seg, color=DARK_RED,   lw=2)
    axes[1].set_title(f"Most Anomalous Stress Segment\n(dist = {worst_dist:.3f})")
    axes[2].plot(motif,     color=DARK_GREEN, lw=2, alpha=0.85, label="Normal motif")
    axes[2].plot(worst_seg, color=DARK_RED,   lw=2, alpha=0.85, label="Stress segment")
    axes[2].set_title("Overlay Comparison"); axes[2].legend(fontsize=8)
    for ax in axes:
        ax.set_xlabel("Samples"); ax.grid(alpha=0.3)
    axes[0].set_ylabel("Normalised Amplitude")
    fig.suptitle(f"{sub} — Normal Motif vs. Most Anomalous Stress Segment", fontsize=12)
    plt.tight_layout()
    return fig


def fig_realtime(te_bl, te_st, motifs, motif_key, thr, pct, sub):
    TRANS = 20; TOTAL = 60
    stitch = np.concatenate([te_bl[-(TRANS*FS):], te_st[:((TOTAL-TRANS)*FS)]])
    t      = np.arange(len(stitch)) / FS
    dist   = distance_profile(stitch, motif_key+"_rt", motifs)

    fig, axes = plt.subplots(2, 1, figsize=(14, 7), sharex=True,
                             gridspec_kw={"height_ratios": [1, 1.2]})
    axes[0].plot(t, stitch, color="black", lw=0.8)
    axes[0].axvspan(0, TRANS, color=BLUE, alpha=0.12, label="Relaxed State")
    axes[0].axvspan(TRANS, TOTAL, color=RED, alpha=0.12, label="Stress Task")
    axes[0].set_title(f"{sub} — ECG Signal Transition", fontsize=12, weight="bold")
    axes[0].set_ylabel("Normalised Amplitude"); axes[0].grid(alpha=0.25, linestyle="--")
    axes[0].legend(loc="upper left")

    axes[1].plot(t, dist, color=DARK_RED, lw=1.2, label="Anomaly Score")
    axes[1].axvspan(0, TRANS, color=BLUE, alpha=0.12)
    axes[1].axvspan(TRANS, TOTAL, color=RED, alpha=0.12)
    axes[1].axhline(thr, color="black", linestyle="-.",
                    label=f"Threshold ({pct}th pct = {thr:.2f})")
    axes[1].axvline(TRANS, color="black", linestyle="--", lw=1.5)
    axes[1].set_xlim(TRANS-10, TRANS+10)
    axes[1].set_title("Real-Time Anomaly Detection", fontsize=12, weight="bold")
    axes[1].set_xlabel("Time (s)")
    axes[1].set_ylabel("Distance to Learned Motifs")
    axes[1].grid(alpha=0.25, linestyle="--"); axes[1].legend(loc="lower right")
    plt.tight_layout()
    return fig


def fig_hrv(hrv_bl, hrv_st, sub):
    fig, axes = plt.subplots(1, 2, figsize=(14, 4))

    axes[0].plot(hrv_bl["t"], hrv_bl["rmssd"], color=BLUE,  lw=1.5, label="Baseline", alpha=0.85)
    axes[0].plot(hrv_st["t"], hrv_st["rmssd"], color=RED,   lw=1.5, label="Stress",   alpha=0.85)
    axes[0].set_title(f"{sub} — RMSSD (↓ under stress)")
    axes[0].set_xlabel("Time (s)"); axes[0].set_ylabel("RMSSD (ms)")
    axes[0].legend(); axes[0].grid(alpha=0.3)

    axes[1].plot(hrv_bl["t"], hrv_bl["hr"], color=BLUE,  lw=1.5, label="Baseline", alpha=0.85)
    axes[1].plot(hrv_st["t"], hrv_st["hr"], color=RED,   lw=1.5, label="Stress",   alpha=0.85)
    axes[1].set_title(f"{sub} — Heart Rate (↑ under stress)")
    axes[1].set_xlabel("Time (s)"); axes[1].set_ylabel("HR (bpm)")
    axes[1].legend(); axes[1].grid(alpha=0.3)
    plt.tight_layout()
    return fig


# Sidebar

with st.sidebar:
    st.markdown("## 🫀 ECG Stress Detection")
    st.markdown("**Capstone Project Dashboard**")
    st.markdown("---")

    # Checking which subjects are available
    available = []
    for s in SUBJECTS:
        pa = os.path.join(DATA_PATH, f"{s}.pkl")
        pb = os.path.join(DATA_PATH, s, f"{s}.pkl")
        if os.path.exists(pa) or os.path.exists(pb):
            available.append(s)

    if not available:
        st.error(
            "No WESAD .pkl files found in `data/`.\n\n"
            "Please place S2.pkl … S17.pkl (excluding S12) in the data/ folder."
        )
        st.stop()

    sub = st.selectbox("Select Subject", available)
    st.markdown("---")
    st.markdown(
        "**Pipeline overview**\n\n"
        "1. Load & preprocess ECG\n"
        "2. Learn baseline motifs\n"
        "3. Tune detection threshold\n"
        "4. Evaluate on held-out test set\n"
    )
    st.markdown("---")
    st.caption("WESAD dataset · 700 Hz · m=1050 (1.5 s) · K=5 motifs")


# Main header

st.title(f" Stress Detection Dashboard — {sub}")
st.markdown(
    "Unsupervised motif-based anomaly detection from chest ECG signals "
    "(WESAD dataset, Matrix Profile framework)."
)

# Load & run ─

with st.spinner(f"Running full pipeline for {sub}… (first load may take ~2 min)"):
    res = run_subject(sub)

m = res["metrics"]

# Metric cards

c1, c2, c3, c4, c5 = st.columns(5)
c1.metric("F1 Score",       f"{m['f1']:.3f}")
c2.metric("Accuracy",       f"{m['acc']:.3f}")
c3.metric("Precision",      f"{m['prec']:.3f}")
c4.metric("Recall",         f"{m['rec']:.3f}")
c5.metric("Separation Ratio", f"{m['sep']:.3f}×")

st.markdown("---")

# Tabs

tab1, tab2, tab3, tab4 = st.tabs([
    " Baseline Motifs",
    " Anomaly Detection",
    " Shape Analysis",
    " HRV Features",
])

# Tab 1: Motifs

with tab1:
    st.subheader("Learned Baseline Motifs")
    st.markdown(
        "The Matrix Profile extracts the **5 most repetitive subsequences** from the training "
        "baseline. These represent the subject's normal cardiac signature. "
        "A window size of **1050 samples (1.5 s)** captures one complete P-QRS-T cycle."
    )
    st.pyplot(fig_motifs(res["motifs"], sub))
    st.markdown("#### Motif Overlay")
    st.markdown(
        "If the motifs overlap closely, the subject has a highly stable resting heart rhythm — "
        "ideal for anomaly detection."
    )
    st.pyplot(fig_motifs_overlay(res["motifs"], sub))


# Tab 2: Anomaly Detection

with tab2:
    st.subheader("Anomaly Score & Threshold")
    col_l, col_r = st.columns([2, 1])
    with col_l:
        st.markdown(
            f"Threshold tuned on validation set: **{res['pct']}th percentile = {res['thr']:.4f}**  \n"
            f"Validation F1: **{m['val_f1']:.3f}**  \n"
            f"Mean baseline distance: **{res['bl_mean']:.4f}**  \n"
            f"Mean stress distance: **{res['st_mean']:.4f}**"
        )
    with col_r:
        st.markdown(
            f"| Metric | Value |\n|---|---|\n"
            f"| Accuracy | {m['acc']:.4f} |\n"
            f"| Precision | {m['prec']:.4f} |\n"
            f"| Recall | {m['rec']:.4f} |\n"
            f"| F1 | {m['f1']:.4f} |"
        )

    st.pyplot(fig_anomaly_timeline(res["tbl_d"], res["tst_d"], res["thr"], res["pct"], sub))

    col_dist, col_cm = st.columns([3, 2])
    with col_dist:
        st.pyplot(fig_distributions(res["tbl_d"], res["tst_d"], res["thr"], sub))
    with col_cm:
        st.pyplot(fig_confusion(m["cm"], sub))

    st.subheader("Real-Time Detection")
    st.markdown(
        "A 60-second stitched window showing the ECG signal and anomaly score "
        "**10 s before and 40 s after** the stress task begins."
    )
    st.pyplot(fig_realtime(
        res["te_bl"], res["te_st"],
        res["motifs"], sub,
        res["thr"], res["pct"], sub
    ))


# Tab 3: Shape Analysis

with tab3:
    st.subheader("Normal Motif vs. Most Anomalous Stress Segment")
    st.markdown(
        "The most anomalous segment is the 1.5-second window from the stress test "
        "with the **highest distance** from any learned motif. "
        "Under acute stress, multiple heartbeats compress into the same window that "
        "normally holds one — producing a visually striking shape difference."
    )
    st.pyplot(fig_shape_comparison(
        res["motifs"][0],
        res["worst_seg"],
        res["worst_dist"],
        sub,
    ))


# Tab 4: HRV Features 

with tab4:
    st.subheader("Heart Rate Variability Features")
    st.markdown(
        "HRV quantifies fluctuations in RR intervals. Under stress, the sympathetic "
        "nervous system dominates: heart rate **rises** and variability (RMSSD) **falls**."
    )

    hrv_bl, hrv_st = res["hrv_bl"], res["hrv_st"]
    if hrv_bl.empty or hrv_st.empty:
        st.warning("Not enough data to compute HRV features for this subject.")
    else:
        col_a, col_b, col_c, col_d = st.columns(4)
        col_a.metric("Baseline RMSSD", f"{hrv_bl['rmssd'].mean():.1f} ms")
        col_b.metric("Stress RMSSD",   f"{hrv_st['rmssd'].mean():.1f} ms",
                     delta=f"{hrv_st['rmssd'].mean()-hrv_bl['rmssd'].mean():.1f} ms")
        col_c.metric("Baseline HR",    f"{hrv_bl['hr'].mean():.1f} bpm")
        col_d.metric("Stress HR",      f"{hrv_st['hr'].mean():.1f} bpm",
                     delta=f"{hrv_st['hr'].mean()-hrv_bl['hr'].mean():.1f} bpm")

        st.pyplot(fig_hrv(hrv_bl, hrv_st, sub))

        rmssd_dir = "↓ (expected ✓)" if hrv_st["rmssd"].mean() < hrv_bl["rmssd"].mean() else "↑ (unexpected)"
        hr_dir    = "↑ (expected ✓)" if hrv_st["hr"].mean()    > hrv_bl["hr"].mean()    else "↓ (unexpected)"
        st.caption(f"RMSSD direction: {rmssd_dir}   |   HR direction: {hr_dir}")

st.markdown("---")
st.caption(
    "Capstone Project · ECG Stress Detection · WESAD Dataset · "
    "Matrix Profile (stumpy) · Python 3.8+"
)
