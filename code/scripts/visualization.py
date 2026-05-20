"""
visualization.py

All plotting and figure generation functions for the project.

Every function returns a Matplotlib figure and optionally saves it to disk.

Functions:
plot_ecg_comparison       : baseline vs. stress ECG (5-second window)
plot_motifs               : 5 learned baseline motifs 
plot_motifs_overlay       : all 5 motifs overlaid together
plot_threshold_curve      : F1 vs. threshold percentile
plot_anomaly_timeline     : anomaly score over test set
plot_distance_distributions : histogram of baseline vs. stress distances
plot_confusion_matrix     : confusion matrix heatmap
plot_full_dashboard       : combined 3×3 evaluation dashboard 
plot_realtime_detection   : stitched ECG + anomaly score at transition
plot_shape_comparison     : normal motif vs. most anomalous stress segment
plot_motif_comparison_pair : best vs. worst subject motif comparison
plot_f1_bar               : per-subject F1 bar chart
plot_performance_heatmap  : per-subject × per-metric heatmap
plot_hrv_features         : RMSSD and HR baseline vs. stress
plot_hrv_score            : combined Motif+HRV score timeline
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

from preprocessing  import FS
from motif_analysis import M_FIXED

#  Style constants 

BLUE       = "steelblue"
RED        = "salmon"
DARK_GREEN = "darkgreen"
DARK_RED   = "darkred"
BLACK      = "black"
FIG_DPI    = 150


def _save(fig, save_dir: str, name: str):
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
        path = os.path.join(save_dir, f"{name}.pdf")
        fig.savefig(path, format="pdf", bbox_inches="tight")
        print(f"  Saved → {path}")


#  ECG preprocessing check

def plot_ecg_comparison(tr_bl, st_ecg, sub: str,
                        save_dir: str = None) -> plt.Figure:
    """Plotting 5-second baseline vs. stress ECG windows side by side."""
    fig, axes = plt.subplots(2, 1, figsize=(15, 5), sharex=True)
    t = np.arange(FS * 5) / FS
    axes[0].plot(t, tr_bl[:FS*5], color=BLUE, lw=1.0)
    axes[0].set_ylabel("Normalised Amplitude")
    axes[0].set_title("Cleaned Baseline ECG"); axes[0].grid(alpha=0.3)
    axes[1].plot(t, st_ecg[:FS*5], color=RED, lw=1.0)
    axes[1].set_ylabel("Normalised Amplitude")
    axes[1].set_title(f"{sub} — Cleaned Stress ECG")
    axes[1].set_xlabel("Time (s)"); axes[1].grid(alpha=0.3)
    plt.tight_layout()
    _save(fig, save_dir, f"{sub}_ecg_comparison")
    return fig


# Motif visualisations

def plot_motifs(motifs: list, sub: str, save_dir: str = None) -> plt.Figure:
    """Plotting an individual panel for each of the K learned baseline motifs."""
    k = len(motifs)
    fig, axes = plt.subplots(1, k, figsize=(18, 3), sharey=True)
    for i, mo in enumerate(motifs):
        axes[i].plot(mo, color=DARK_GREEN, lw=1.8)
        axes[i].set_title(f"Motif {i+1}", fontsize=9)
        axes[i].set_xlabel("Samples"); axes[i].grid(alpha=0.3)
    axes[0].set_ylabel("Normalised Amplitude")
    fig.suptitle(
        f"{sub} — {k} Learned Baseline Motifs  "
        f"(m={M_FIXED}, {M_FIXED/FS:.2f}s per window)", fontsize=13)
    plt.tight_layout()
    _save(fig, save_dir, f"{sub}_motifs")
    return fig


def plot_motifs_overlay(motifs: list, sub: str,
                        save_dir: str = None) -> plt.Figure:
    """All motifs overlaid together."""
    fig, ax = plt.subplots(figsize=(6, 4))
    for i, mo in enumerate(motifs):
        ax.plot(mo, lw=1.5, alpha=0.7, label=f"Motif {i+1}")
    ax.set_title(f"{sub} — All {len(motifs)} Motifs Comparison", fontsize=13)
    ax.set_xlabel("Samples"); ax.set_ylabel("Normalised Amplitude")
    ax.grid(alpha=0.3); plt.tight_layout()
    _save(fig, save_dir, f"{sub}_motifs_overlay")
    return fig


#  Threshold calibration 

def plot_threshold_curve(thr_df: pd.DataFrame, best_pct: int,
                         best_f1: float, sub: str,
                         save_dir: str = None) -> plt.Figure:
    """F1 score vs. threshold percentile."""
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(thr_df["percentile"], thr_df["f1"], color=BLUE, lw=2)
    ax.axvline(best_pct, color="red", linestyle="--",
               label=f"Best = {best_pct}th pct  (Val F1 = {best_f1:.3f})")
    ax.set_xlabel("Threshold Percentile"); ax.set_ylabel("F1 Score")
    ax.set_title("F1 vs Threshold Percentile")
    ax.legend(); ax.grid(alpha=0.3); plt.tight_layout()
    _save(fig, save_dir, f"{sub}_threshold_curve")
    return fig


#  Test-set evaluation

def plot_anomaly_timeline(tbl_d, tst_d, thr: float, pct: int,
                          sub: str, save_dir: str = None) -> plt.Figure:
    """Anomaly score over the full test set."""
    fig, ax = plt.subplots(figsize=(18, 4))
    t_bl = np.arange(len(tbl_d)) / FS
    t_st = np.arange(len(tst_d)) / FS
    ax.plot(t_bl, tbl_d, color=BLUE,  alpha=0.7, lw=0.6, label="Baseline")
    ax.plot(t_st, tst_d, color=RED,   alpha=0.7, lw=0.6, label="Stress")
    ax.axhline(thr, color=BLACK, linestyle="--",
               label=f"Threshold ({pct}th pct = {thr:.3f})")
    ax.set_xlabel("Time (s)"); ax.set_ylabel("Distance to Nearest Motif")
    ax.set_title(f"{sub} — Anomaly Score Over Time")
    ax.legend(); ax.grid(alpha=0.3); plt.tight_layout()
    _save(fig, save_dir, f"{sub}_anomaly_timeline")
    return fig


def plot_distance_distributions(tbl_d, tst_d, thr: float,
                                 sub: str, save_dir: str = None) -> plt.Figure:
    """Histogram of baseline vs. stress distances with threshold line."""
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.hist(tbl_d[~np.isnan(tbl_d)], bins=80, alpha=0.65,
            color=BLUE, label="Baseline")
    ax.hist(tst_d[~np.isnan(tst_d)], bins=80, alpha=0.65,
            color=RED,  label="Stress")
    ax.axvline(thr, color=BLACK, linestyle="--", label="Threshold")
    ax.set_xlabel("Distance"); ax.set_ylabel("Count")
    ax.set_title(f"{sub} — Distance Distributions")
    ax.legend(); ax.grid(alpha=0.3); plt.tight_layout()
    _save(fig, save_dir, f"{sub}_distributions")
    return fig


def plot_confusion_matrix(yt, yp, sub: str,
                          save_dir: str = None) -> plt.Figure:
    """Confusion matrix heatmap."""
    from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
    fig, ax = plt.subplots(figsize=(4, 4))
    cm = confusion_matrix(yt, yp)
    ConfusionMatrixDisplay(cm, display_labels=["Normal", "Stress"]).plot(
        ax=ax, colorbar=False, cmap="Blues")
    ax.set_title(f"{sub} — Confusion Matrix"); plt.tight_layout()
    _save(fig, save_dir, f"{sub}_confusion_matrix")
    return fig


def plot_full_dashboard(motifs, tbl_d, tst_d, thr, pct,
                        yt, yp, te_st, sub: str,
                        save_dir: str = None) -> plt.Figure:
    """3×3 dashboard: timeline, distributions, confusion matrix, shapes."""
    fig = plt.figure(figsize=(18, 14))
    gs  = gridspec.GridSpec(3, 3, figure=fig, hspace=0.45, wspace=0.35)

    ax_dist = fig.add_subplot(gs[0, :])
    t_bl = np.arange(len(tbl_d)) / FS
    t_st = np.arange(len(tst_d)) / FS
    ax_dist.plot(t_bl, tbl_d, color=BLUE, alpha=0.7, lw=0.6, label="Baseline")
    ax_dist.plot(t_st, tst_d, color=RED,  alpha=0.7, lw=0.6, label="Stress")
    ax_dist.axhline(thr, color=BLACK, linestyle="--",
                    label=f"Threshold ({pct}th pct = {thr:.3f})")
    ax_dist.set_xlabel("Time"); ax_dist.set_ylabel("Distance to Nearest Motif")
    ax_dist.set_title("Anomaly Score Over Time")
    ax_dist.legend(); ax_dist.grid(alpha=0.3)

    ax_hist = fig.add_subplot(gs[1, :2])
    ax_hist.hist(tbl_d[~np.isnan(tbl_d)], bins=80, alpha=0.65,
                 color=BLUE, label="Baseline")
    ax_hist.hist(tst_d[~np.isnan(tst_d)], bins=80, alpha=0.65,
                 color=RED,  label="Stress")
    ax_hist.axvline(thr, color=BLACK, linestyle="--", label="Threshold")
    ax_hist.set_xlabel("Distance"); ax_hist.set_ylabel("Count")
    ax_hist.set_title("Distance Distributions")
    ax_hist.legend(); ax_hist.grid(alpha=0.3)

    ax_cm = fig.add_subplot(gs[1, 2])
    cm = confusion_matrix(yt, yp)
    ConfusionMatrixDisplay(cm, display_labels=["Normal", "Stress"]).plot(
        ax=ax_cm, colorbar=False, cmap="Blues")
    ax_cm.set_title("Confusion Matrix")

    best_motif = motifs[0]
    worst_idx  = int(np.nanargmax(tst_d))
    worst_seg  = te_st[worst_idx : worst_idx + M_FIXED]

    ax_nm = fig.add_subplot(gs[2, 0])
    ax_nm.plot(best_motif, color=DARK_GREEN, lw=2)
    ax_nm.set_title("Normal Motif", fontsize=9)
    ax_nm.set_xlabel("Samples"); ax_nm.set_ylabel("Normal Amplitude")
    ax_nm.grid(alpha=0.3)

    ax_st = fig.add_subplot(gs[2, 1])
    ax_st.plot(worst_seg, color=DARK_RED, lw=2)
    ax_st.set_title(
        f"Most Anomalous Stress Segment\n(dist = {tst_d[worst_idx]:.3f})",
        fontsize=9)
    ax_st.set_xlabel("Samples"); ax_st.grid(alpha=0.3)

    ax_ov = fig.add_subplot(gs[2, 2])
    ax_ov.plot(best_motif, color=DARK_GREEN, lw=2, alpha=0.85,
               label="Normal motif")
    ax_ov.plot(worst_seg,  color=DARK_RED,   lw=2, alpha=0.85,
               label="Stress segment")
    ax_ov.set_title("Shape Comparison", fontsize=9)
    ax_ov.set_xlabel("Samples"); ax_ov.legend(fontsize=8); ax_ov.grid(alpha=0.3)

    fig.suptitle(f"{sub}: Full Test Evaluation", fontsize=15, y=1.01)
    plt.tight_layout()
    _save(fig, save_dir, f"{sub}_full_dashboard")
    return fig


# Real-time stress detection 

def plot_realtime_detection(te_bl, te_st, motifs, thr: float, pct: int,
                            sub: str, save_dir: str = None) -> plt.Figure:
    """
    Combine the last 20 seconds of baseline with 40 seconds of stress and show
    both the raw ECG and the anomaly score at the transition.
    """
    from motif_analysis import compute_distance_profile
    TRANS = 20; TOTAL = 60
    stitch = np.concatenate([te_bl[-(TRANS*FS):],
                              te_st[:((TOTAL-TRANS)*FS)]])
    t      = np.arange(len(stitch)) / FS
    dist   = compute_distance_profile(stitch, motifs)

    fig, axes = plt.subplots(2, 1, figsize=(16, 7), sharex=True,
                             gridspec_kw={"height_ratios": [1, 1.2]})
    axes[0].plot(t, stitch, color=BLACK, lw=0.8, alpha=0.9)
    axes[0].axvspan(0, TRANS, color=BLUE, alpha=0.15, label="Relaxed State")
    axes[0].axvspan(TRANS, TOTAL, color=RED, alpha=0.15, label="Stress Task")
    axes[0].set_title(f"{sub} — ECG Signal Transition",
                      fontsize=13, weight="bold")
    axes[0].set_ylabel("Normalised Amplitude")
    axes[0].grid(alpha=0.25, linestyle="--"); axes[0].legend(loc="upper left")

    axes[1].plot(t, dist, color=DARK_RED, lw=1.2, label="Anomaly Score")
    axes[1].axvspan(0, TRANS, color=BLUE, alpha=0.15)
    axes[1].axvspan(TRANS, TOTAL, color=RED, alpha=0.15)
    axes[1].axhline(thr, color=BLACK, linestyle="-.", lw=1.5,
                    label=f"Threshold ({pct}th pct = {thr:.2f})")
    axes[1].axvline(TRANS, color=BLACK, linestyle="--", lw=1.5)
    axes[1].set_xlim(TRANS - 10, TRANS + 10)
    axes[1].set_title("Real-Time Anomaly Detection", fontsize=13, weight="bold")
    axes[1].set_xlabel("Time (s)")
    axes[1].set_ylabel("Distance to Learned Motifs")
    axes[1].grid(alpha=0.25, linestyle="--"); axes[1].legend(loc="lower right")
    plt.tight_layout()
    _save(fig, save_dir, f"{sub}_realtime_detection")
    return fig


# Shape comparison 

def plot_shape_comparison(motifs_a, tst_d_a, te_st_a, sub_a: str,
                          motifs_b, tst_d_b, te_st_b, sub_b: str,
                          save_dir: str = None) -> plt.Figure:
    """Side-by-side motif vs. most anomalous segment for two subjects."""
    fig, axes = plt.subplots(1, 2, figsize=(16, 4))
    for ax, motifs, tst_d, te_st, sub in zip(
            axes,
            [motifs_a, motifs_b],
            [tst_d_a,  tst_d_b],
            [te_st_a,  te_st_b],
            [sub_a,    sub_b]):
        worst_idx = int(np.nanargmax(tst_d))
        worst_seg = te_st[worst_idx : worst_idx + M_FIXED]
        ax.plot(motifs[0], color=DARK_GREEN, lw=2.0, alpha=0.85,
                label="Normal motif")
        ax.plot(worst_seg,  color=DARK_RED,   lw=2.0, alpha=0.85,
                label="Most anomalous stress")
        ax.set_title(
            f"{sub} — Motif vs. Most Anomalous Segment\n"
            f"Distance = {tst_d[worst_idx]:.3f}  "
            f"(mean baseline = {np.nanmean(tst_d):.3f})", fontsize=11)
        ax.set_xlabel("Samples"); ax.legend(fontsize=9); ax.grid(alpha=0.3)
    axes[0].set_ylabel("Normalised Amplitude")
    fig.suptitle("ECG Shape Comparison: Normal Motif vs. Most Anomalous "
                 "Stress Segment", fontsize=12, y=1.03)
    plt.tight_layout()
    _save(fig, save_dir, "shapes_comparison")
    return fig


# Cross-subject summaries

def plot_f1_bar(df: pd.DataFrame, save_dir: str = None) -> plt.Figure:
    """Per-subject F1 bar chart: personal threshold vs. common rule."""
    fig, ax = plt.subplots(figsize=(14, 5))
    x = np.arange(len(df))
    ax.bar(x - 0.2, df["Test_F1_Best"],   0.38, label="Personal threshold",
           color=BLUE, alpha=0.88)
    ax.bar(x + 0.2, df["Test_F1_Common"], 0.38, label="Common rule",
           color=RED,  alpha=0.88)
    ax.axhline(df["Test_F1_Best"].mean(),   color=BLUE, linestyle="--", alpha=0.8,
               label=f"Mean personal F1 = {df['Test_F1_Best'].mean():.3f}")
    ax.axhline(df["Test_F1_Common"].mean(), color=RED,  linestyle="--",
               label=f"Mean common F1   = {df['Test_F1_Common'].mean():.3f}")
    ax.set_xticks(x); ax.set_xticklabels(df["Subject"])
    ax.set_ylabel("Test F1 Score"); ax.set_ylim(0, 1.05)
    ax.set_title("Per-Subject Test F1 — Personal Threshold vs Common Rule\n"
                 "(Personal tuning consistently outperforms a fixed rule)")
    ax.legend(fontsize=9); ax.grid(alpha=0.3, axis="y")
    plt.tight_layout()
    _save(fig, save_dir, "f1_bar_chart")
    return fig


def plot_performance_heatmap(df: pd.DataFrame,
                             save_dir: str = None) -> plt.Figure:
    """Per-subject × per-metric normalised heatmap."""
    cols   = ["Test_F1_Best", "Test_Accuracy", "Test_Precision", "Test_Recall"]
    labels = ["F1", "Accuracy", "Precision", "Recall"]
    raw    = df[cols].values

    
    norm = np.zeros_like(raw)
    for j in range(raw.shape[1]):
        mn, mx = raw[:, j].min(), raw[:, j].max()
        norm[:, j] = (raw[:, j] - mn) / (mx - mn + 1e-9)

    fig, ax = plt.subplots(figsize=(10, 4))
    im = ax.imshow(norm.T, aspect="auto", cmap="RdBu", vmin=0, vmax=1)
    ax.set_xticks(range(len(df))); ax.set_xticklabels(df["Subject"], rotation=45)
    ax.set_yticks(range(len(labels))); ax.set_yticklabels(labels)
    for si in range(len(df)):
        for mi in range(len(labels)):
            ax.text(si, mi, f"{raw[si, mi]:.2f}", ha="center", va="center",
                    fontsize=8,
                    color="black" if 0.2 < norm[si, mi] < 0.8 else "white")
    plt.colorbar(im, ax=ax, label="Relative score (per-metric normalised)")
    ax.set_title("Per-Subject Performance Heatmap\n"
                 "(blue = relatively high, red = relatively low)")
    plt.tight_layout()
    _save(fig, save_dir, "performance_heatmap")
    return fig


# HRV plots 

def plot_hrv_features(hrv_bl: pd.DataFrame, hrv_st: pd.DataFrame,
                      sub: str, save_dir: str = None) -> plt.Figure:
    """RMSSD and HR timeline: baseline vs. stress."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 4))
    for ax, feat, ylabel, title in zip(
            axes,
            ["rmssd", "hr"],
            ["RMSSD (ms)", "Heart Rate (bpm)"],
            [f"{sub} — RMSSD (↓ under stress)",
             f"{sub} — Heart Rate (↑ under stress)"]):
        ax.plot(hrv_bl["window_center_s"], hrv_bl[feat],
                color=BLUE, lw=1.5, label="Baseline", alpha=0.85)
        ax.plot(hrv_st["window_center_s"], hrv_st[feat],
                color=RED,  lw=1.5, label="Stress",   alpha=0.85)
        ax.set_xlabel("Time (s)"); ax.set_ylabel(ylabel)
        ax.set_title(title); ax.legend(); ax.grid(alpha=0.3)
    plt.tight_layout()
    _save(fig, save_dir, f"{sub}_hrv_features")
    return fig


def plot_hrv_score(t_bl, tbl_sc, t_st, tst_sc,
                   thr: float, pct: int, sub: str,
                   save_dir: str = None) -> plt.Figure:
    """Combined Motif+HRV anomaly score timeline."""
    fig, ax = plt.subplots(figsize=(14, 5))
    ax.plot(t_bl, tbl_sc, color=BLUE, lw=1.5, label="Baseline", alpha=0.85)
    ax.plot(t_st, tst_sc, color=RED,  lw=1.5, label="Stress",   alpha=0.85)
    ax.axhline(thr, color=BLACK, linestyle="--",
               label=f"Threshold ({pct}th pct = {thr:.3f})")
    ax.set_xlabel("Time (s)"); ax.set_ylabel("Combined Anomaly Score")
    ax.set_title(f"{sub} — Combined Motif+HRV Score: Baseline vs Stress")
    ax.legend(); ax.grid(alpha=0.3); plt.tight_layout()
    _save(fig, save_dir, f"{sub}_hrv_combined_score")
    return fig
