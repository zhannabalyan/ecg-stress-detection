"""
run_pipeline.py

One-command entry point for the scripts-based pipeline.

Two modes:

1. Full reproduction : runs the complete pipeline from scratch:
       python run_pipeline.py

2. Results-only : regenerates all CSVs from
   already-computed data without re-running the heavy Matrix Profile:
       python run_pipeline.py --results-only


All CSV results are saved to: paper/results/


"""

import os
import sys
import time
import argparse
import numpy as np
import pandas as pd

#  Path setup

SCRIPT_DIR   = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.normpath(os.path.join(SCRIPT_DIR, "..", ".."))
DATA_PATH    = os.path.join(PROJECT_ROOT, "data")
VIS_DIR      = os.path.join(PROJECT_ROOT, "visualizations")
RESULTS_DIR  = os.path.join(PROJECT_ROOT, "paper", "results")

sys.path.insert(0, SCRIPT_DIR)

from preprocessing  import load_subject, split_subject_data, SUBJECTS, FS
from motif_analysis import (
    get_motifs, compute_distance_profile, find_best_threshold,
    M_FIXED, K_FIXED,
)
from hrv_features   import extract_hrv_features
from evaluation     import run_motif_pipeline, run_hrv_pipeline
from visualization  import (
    plot_ecg_comparison, plot_motifs, plot_motifs_overlay,
    plot_threshold_curve, plot_full_dashboard,
    plot_realtime_detection, plot_shape_comparison,
    plot_f1_bar, plot_performance_heatmap,
    plot_hrv_features,
)
from sklearn.metrics import f1_score


#  Helpers

def check_data() -> bool:
    """Verify that at least one WESAD subject file is present."""
    if not os.path.isdir(DATA_PATH):
        print(f"\n[ERROR] Data directory not found: {DATA_PATH}")
        return False
    pks = [f for f in os.listdir(DATA_PATH) if f.endswith(".pkl")]
    if not pks:
        print(
            f"\n[ERROR] No .pkl files found in {DATA_PATH}\n"
            "Please download WESAD and place S2.pkl ... S17.pkl in data/.\n"
            "Download: https://uni-siegen.sciebo.de/s/HGdUkoNlW1Ub0Gx\n"
        )
        return False
    print(f"  Found {len(pks)} subject file(s): {sorted(pks)}")
    return True


def check_results() -> bool:
    """Verify that pre-computed CSV results exist for results-only mode."""
    motif_csv = os.path.join(RESULTS_DIR, "motif_results.csv")
    hrv_csv   = os.path.join(RESULTS_DIR, "hrv_results.csv")
    missing   = [f for f in [motif_csv, hrv_csv] if not os.path.exists(f)]
    if missing:
        print(
            "\n[ERROR] Pre-computed results not found:\n"
            + "\n".join(f"  {f}" for f in missing)
            + "\n\nPlease run the full pipeline first:\n"
            "  python run_pipeline.py\n"
        )
        return False
    print(f"  Found: motif_results.csv")
    print(f"  Found: hrv_results.csv")
    return True


def generate_subject_figures(sub: str, save_dir: str):
    """Generate all per-subject diagnostic figures into save_dir."""
    print(f"\n  Generating figures for {sub}...")
    try:
        import matplotlib.pyplot as plt

        bl, st = load_subject(sub, DATA_PATH)
        tr_bl, val_bl, val_st, te_bl, te_st = split_subject_data(bl, st)
        motifs = get_motifs(tr_bl)

        vbl_d = compute_distance_profile(val_bl, motifs)
        vst_d = compute_distance_profile(val_st, motifs)
        thr_df, pct, thr, val_f1 = find_best_threshold(vbl_d, vst_d)

        tbl_d = compute_distance_profile(te_bl, motifs)
        tst_d = compute_distance_profile(te_st, motifs)
        n     = min(len(tbl_d), len(tst_d))
        tbl_d, tst_d = tbl_d[:n], tst_d[:n]

        yt     = np.concatenate([np.zeros(n), np.ones(n)])
        yp_raw = np.concatenate([(tbl_d > thr).astype(int),
                                  (tst_d > thr).astype(int)])
        valid  = ~np.isnan(yp_raw)
        yt_v, yp_v = yt[valid], yp_raw[valid].astype(int)

        plot_ecg_comparison(tr_bl, st, sub, save_dir=save_dir);   plt.close("all")
        plot_motifs(motifs, sub, save_dir=save_dir);               plt.close("all")
        plot_motifs_overlay(motifs, sub, save_dir=save_dir);       plt.close("all")
        plot_threshold_curve(thr_df, pct, val_f1, sub,
                             save_dir=save_dir);                   plt.close("all")
        plot_full_dashboard(motifs, tbl_d, tst_d, thr, pct,
                            yt_v, yp_v, te_st, sub,
                            save_dir=save_dir);                    plt.close("all")
        plot_realtime_detection(te_bl, te_st, motifs, thr, pct,
                                sub, save_dir=save_dir);           plt.close("all")

        hrv_bl = extract_hrv_features(te_bl)
        hrv_st = extract_hrv_features(te_st)
        if not hrv_bl.empty and not hrv_st.empty:
            plot_hrv_features(hrv_bl, hrv_st, sub,
                              save_dir=save_dir);                  plt.close("all")

    except Exception as e:
        print(f"    [warning] Could not generate figures for {sub}: {e}")


def generate_summary_figures(df_motif: pd.DataFrame, save_dir: str):
    """Generate cross-subject summary figures from a results dataframe."""
    import matplotlib.pyplot as plt
    print("\n  Generating cross-subject summary figures...")

    plot_f1_bar(df_motif, save_dir=save_dir);              plt.close("all")
    plot_performance_heatmap(df_motif, save_dir=save_dir); plt.close("all")

    try:
        bl_b, st_b = load_subject("S2", DATA_PATH)
        bl_w, st_w = load_subject("S8", DATA_PATH)
        _, _, _, te_bl_b, te_st_b = split_subject_data(bl_b, st_b)
        _, _, _, te_bl_w, te_st_w = split_subject_data(bl_w, st_w)
        motifs_b = get_motifs(bl_b[:300 * FS])
        motifs_w = get_motifs(bl_w[:300 * FS])
        tst_d_b  = compute_distance_profile(te_st_b, motifs_b)
        tst_d_w  = compute_distance_profile(te_st_w, motifs_w)
        plot_shape_comparison(
            motifs_b, tst_d_b, te_st_b, "S2",
            motifs_w, tst_d_w, te_st_w, "S8",
            save_dir=save_dir)
        plt.close("all")
    except Exception as e:
        print(f"  [warning] Shape comparison skipped: {e}")


#  Modes

def run_full_pipeline():
    """Full mode: run everything from scratch."""
    print("\n  MODE: Full Pipeline Reproduction")
    print("  Estimated time: 60-90 minutes\n")

    if not check_data():
        sys.exit(1)

    os.makedirs(VIS_DIR, exist_ok=True)
    os.makedirs(RESULTS_DIR, exist_ok=True)

    print("\n Running Motif-only pipeline across all subjects...")
    df_motif = run_motif_pipeline(DATA_PATH, RESULTS_DIR)

    print("\n Running Motif+HRV pipeline across all subjects...")
    df_hrv = run_hrv_pipeline(DATA_PATH, RESULTS_DIR)

    generate_summary_figures(df_motif, VIS_DIR)

    print("\n Generating per-subject figures...")
    available = [s for s in SUBJECTS
                 if os.path.exists(os.path.join(DATA_PATH, f"{s}.pkl"))]
    for sub in available:
        generate_subject_figures(sub, save_dir=VIS_DIR)

    return df_motif, df_hrv


def run_results_only():
    """
    Results-only mode: load pre-computed CSVs and print them to the terminal.
    """
    print("\n  MODE: Results-Only (Quick View)")
    print("  Estimated time: < 5 seconds\n")

    if not check_results():
        sys.exit(1)

    # Load pre-computed results
    df_motif = pd.read_csv(os.path.join(RESULTS_DIR, "motif_results.csv"))
    df_hrv = pd.read_csv(os.path.join(RESULTS_DIR, "hrv_results.csv"))

    print("\n" + "=" * 65)
    print("  MOTIF-ONLY PIPELINE FINAL RESULTS")
    print("=" * 65)
    print(df_motif.to_string(index=False))

    print("\n" + "=" * 65)
    print("  MOTIF + HRV PIPELINE FINAL RESULTS")
    print("=" * 65)
    print(df_hrv.to_string(index=False))

    print(f"\n  Motif-only mean F1 : {df_motif['Test_F1_Best'].mean():.4f}")
    print(f"  Motif+HRV  mean F1 : {df_hrv['HRV_F1'].mean():.4f}")

    print("\n  [NOTE] Visualizations were skipped in Quick View mode.")
    print("  To regenerate all figures and distance profiles from scratch, run:")
    print("      python run_pipeline.py")

    return df_motif, df_hrv


# Main

def main():
    parser = argparse.ArgumentParser(
        description="ECG Stress Detection — Pipeline Reproduction",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  Full pipeline from scratch :
      python run_pipeline.py

  Regenerate outputs only from saved results :
      python run_pipeline.py --results-only
        """
    )
    parser.add_argument(
        "--results-only",
        action="store_true",
        help="Skip pipeline computation and regenerate figures from saved CSVs only."
    )
    args = parser.parse_args()

    t0 = time.time()

    print("\n" + "="*65)
    print("  Interpretable Stress Detection from ECG Signals")
    print("  Capstone Project — Pipeline Reproduction")
    print("="*65)
    print(f"\n  Project root   : {PROJECT_ROOT}")
    print(f"  Data path      : {DATA_PATH}")
    print(f"  Visualizations : {VIS_DIR}")
    print(f"  Results        : {RESULTS_DIR}")
    print(f"\n  NOTE: paper/figures/ is not touched by this script.")
    print(f"        It contains the original Jupyter notebook figures")
    print(f"        used in the submitted paper.")

    if args.results_only:
        df_motif, df_hrv = run_results_only()
    else:
        df_motif, df_hrv = run_full_pipeline()

    elapsed = (time.time() - t0) / 60
    print("\n" + "="*65)
    print(f"  Finished in {elapsed:.1f} min")
    print(f"  Visualizations -> {VIS_DIR}")
    print(f"  Results        -> {RESULTS_DIR}")
    print(f"\n  Motif-only mean F1 : {df_motif['Test_F1_Best'].mean():.4f}")
    if not df_hrv.empty:
        print(f"  Motif+HRV  mean F1 : {df_hrv['HRV_F1'].mean():.4f}")
    print()


if __name__ == "__main__":
    main()
