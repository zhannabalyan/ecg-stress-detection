"""
evaluation.py

Train / Validation / Test pipeline across all 15 subjects.

Running both the Motif-only and Motif+HRV approaches, collecting per-subject
metrics, and saving all the results to CSV files in the output directory.

Usage:

Called automatically by run_pipeline.py, or run directly:
    python evaluation.py
"""

import os
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
)

from preprocessing  import load_subject, split_subject_data, SUBJECTS, FS
from motif_analysis import (
    get_motifs, compute_distance_profile, find_best_threshold,
    M_FIXED, K_FIXED,
)
from hrv_features import (
    compute_combined_features, build_combined_score,
    find_best_threshold_1d,
)


# Motif-only pipeline

def run_motif_pipeline(data_path: str, output_dir: str) -> pd.DataFrame:
    """
    Run the full Motif-only stress detection pipeline for all subjects.

    Parameters:

    data_path : str
        Path to the directory containing WESAD .pkl files.
    output_dir : str
        Directory where results CSV will be saved.

    Returns:

    pd.DataFrame
        Per-subject results with columns:
        Subject, Validation_percentile, Validation_F1, Test_F1_Best,
        Test_F1_Common, Test_Accuracy, Test_Precision, Test_Recall,
        Mean_Baseline, Mean_Stress, Separation_ratio
    """
    results = []

    header = (f"{'Subject':<10} {'Val F1':>7} {'Val pct':>8} "
              f"{'Test F1':>9} {'Common F1':>10} "
              f"{'Acc':>7} {'Prec':>7} {'Rec':>7} {'Ratio':>7}")
    print("\nMotif-only Pipeline")
    print("=" * len(header))
    print(header)
    print("─" * len(header))

    for sub in SUBJECTS:
        try:
            bl, st = load_subject(sub, data_path)

            if (len(bl) < (300 + 180 + 60) * FS or
                    len(st) < (180 + 60) * FS):
                print(f"{sub:<10} skipped — insufficient data")
                continue

            tr_bl, val_bl, val_st, te_bl, te_st = split_subject_data(bl, st)
            motifs = get_motifs(tr_bl)

            # Validation
            vbl_d = compute_distance_profile(val_bl, motifs)
            vst_d = compute_distance_profile(val_st, motifs)
            _, best_pct, best_thr, val_f1 = find_best_threshold(vbl_d, vst_d)

            # Common baseline rule (mean + 2 SD)
            common_thr = np.nanmean(vbl_d) + 2.0 * np.nanstd(vbl_d)

            # Test
            tbl_d = compute_distance_profile(te_bl, motifs)
            tst_d = compute_distance_profile(te_st, motifs)
            n     = min(len(tbl_d), len(tst_d))
            tbl_d, tst_d = tbl_d[:n], tst_d[:n]

            y_true = np.concatenate([np.zeros(n), np.ones(n)])

            # Best threshold metrics
            yp_best = np.concatenate([(tbl_d > best_thr).astype(int),
                                       (tst_d > best_thr).astype(int)])
            valid = ~np.isnan(yp_best)
            yt, yp = y_true[valid], yp_best[valid].astype(int)

            test_f1  = f1_score(yt, yp, zero_division=0)
            test_acc = accuracy_score(yt, yp)
            test_pre = precision_score(yt, yp, zero_division=0)
            test_rec = recall_score(yt, yp, zero_division=0)

            # Common threshold F1
            yp_com = np.concatenate([(tbl_d > common_thr).astype(int),
                                      (tst_d > common_thr).astype(int)])
            vc     = ~np.isnan(yp_com)
            f1_com = f1_score(y_true[vc], yp_com[vc].astype(int),
                              zero_division=0)

            sep = np.nanmean(tst_d) / np.nanmean(tbl_d)

            results.append({
                "Subject":              sub,
                "Validation_percentile": best_pct,
                "Validation_F1":        round(val_f1,  4),
                "Test_F1_Best":         round(test_f1, 4),
                "Test_F1_Common":       round(f1_com,  4),
                "Test_Accuracy":        round(test_acc, 4),
                "Test_Precision":       round(test_pre, 4),
                "Test_Recall":          round(test_rec, 4),
                "Mean_Baseline":        round(np.nanmean(tbl_d), 4),
                "Mean_Stress":          round(np.nanmean(tst_d), 4),
                "Separation_ratio":     round(sep, 3),
            })

            print(f"{sub:<10} {val_f1:>7.3f} {best_pct:>8} "
                  f"{test_f1:>9.3f} {f1_com:>10.3f} "
                  f"{test_acc:>7.3f} {test_pre:>7.3f} {test_rec:>7.3f} "
                  f"{sep:>7.3f}x")

        except FileNotFoundError as e:
            print(f"{sub:<10} FILE NOT FOUND — {e}")
        except Exception as e:
            print(f"{sub:<10} ERROR — {e}")

    df = pd.DataFrame(results)


    print("─" * len(header))
    mean = df.mean(numeric_only=True)
    print(f"{'Mean':<10} {mean['Validation_F1']:>7.3f} {'—':>8} "
          f"{mean['Test_F1_Best']:>9.3f} {mean['Test_F1_Common']:>10.3f} "
          f"{mean['Test_Accuracy']:>7.3f} {mean['Test_Precision']:>7.3f} "
          f"{mean['Test_Recall']:>7.3f} {mean['Separation_ratio']:>7.3f}x\n")

    os.makedirs(output_dir, exist_ok=True)
    out_path = os.path.join(output_dir, "motif_results.csv")
    df.to_csv(out_path, index=False)
    print(f"Saved → {out_path}")
    return df


#  Motif+HRV pipeline

def run_hrv_pipeline(data_path: str, output_dir: str) -> pd.DataFrame:
    """
    Run the full Motif+HRV stress detection pipeline for all subjects.

    Parameters:

    data_path : str
        Path to the directory containing WESAD .pkl files.
    output_dir : str
        Directory where results CSV will be saved.

    Returns:

    pd.DataFrame
        Per-subject results comparing Motif-only vs. Motif+HRV F1.
    """
    results = []

    header = (f"{'Subject':<10} "
              f"{'Motif F1':>9} {'Motif Acc':>10} "
              f"{'HRV F1':>8} {'HRV Acc':>9} {'Delta F1':>10}")
    print("\nMotif+HRV Pipeline")
    print("=" * len(header))
    print(header)
    print("─" * len(header))

    for sub in SUBJECTS:
        try:
            bl, st = load_subject(sub, data_path)
            if (len(bl) < (300 + 180 + 60) * FS or
                    len(st) < (180 + 60) * FS):
                continue

            tr_bl, val_bl, val_st, te_bl, te_st = split_subject_data(bl, st)
            motifs = get_motifs(tr_bl)

            # Motif-only 
            vbl_d = compute_distance_profile(val_bl, motifs)
            vst_d = compute_distance_profile(val_st, motifs)
            _, mo_pct, mo_thr, _ = find_best_threshold(vbl_d, vst_d)

            tbl_d = compute_distance_profile(te_bl, motifs)
            tst_d = compute_distance_profile(te_st, motifs)
            n     = min(len(tbl_d), len(tst_d))
            yt    = np.concatenate([np.zeros(n), np.ones(n)])
            yp_mo = np.concatenate([(tbl_d[:n] > mo_thr).astype(int),
                                     (tst_d[:n] > mo_thr).astype(int)])
            v     = ~np.isnan(yp_mo)
            mo_f1  = f1_score(yt[v], yp_mo[v].astype(int), zero_division=0)
            mo_acc = accuracy_score(yt[v], yp_mo[v].astype(int))
            mo_pre = precision_score(yt[v], yp_mo[v].astype(int), zero_division=0)
            mo_rec = recall_score(yt[v], yp_mo[v].astype(int), zero_division=0)

            # Motif+HRV
            trbf  = compute_combined_features(tr_bl, motifs)
            vbl_f = compute_combined_features(val_bl, motifs)
            vst_f = compute_combined_features(val_st, motifs)
            tbl_f = compute_combined_features(te_bl, motifs)
            tst_f = compute_combined_features(te_st, motifs)

            if any(len(x) < 3 for x in [trbf, vbl_f, vst_f, tbl_f, tst_f]):
                print(f"{sub:<10} skipped (HRV) — too few windows")
                continue

            vbl_sc, vst_sc, tbl_sc, tst_sc, _, _ = build_combined_score(
                trbf, vbl_f, vst_f, tbl_f, tst_f)

            hrv_pct, hrv_thr, _ = find_best_threshold_1d(vbl_sc, vst_sc)
            hn  = min(len(tbl_sc), len(tst_sc))
            yt_h = np.concatenate([np.zeros(hn), np.ones(hn)])
            yp_h = np.concatenate([(tbl_sc[:hn] > hrv_thr).astype(int),
                                    (tst_sc[:hn] > hrv_thr).astype(int)])

            hrv_f1  = f1_score(yt_h, yp_h, zero_division=0)
            hrv_acc = accuracy_score(yt_h, yp_h)
            hrv_pre = precision_score(yt_h, yp_h, zero_division=0)
            hrv_rec = recall_score(yt_h, yp_h, zero_division=0)
            delta   = hrv_f1 - mo_f1

            results.append({
                "Subject":        sub,
                "Motif_F1":       round(mo_f1,   4),
                "Motif_Accuracy": round(mo_acc,   4),
                "Motif_Precision":round(mo_pre,   4),
                "Motif_Recall":   round(mo_rec,   4),
                "HRV_F1":         round(hrv_f1,  4),
                "HRV_Accuracy":   round(hrv_acc,  4),
                "HRV_Precision":  round(hrv_pre,  4),
                "HRV_Recall":     round(hrv_rec,  4),
                "F1_Delta":       round(delta,    4),
            })

            print(f"{sub:<10} {mo_f1:>9.3f} {mo_acc:>10.3f} "
                  f"{hrv_f1:>8.3f} {hrv_acc:>9.3f} {delta:>+10.4f}")

        except FileNotFoundError as e:
            print(f"{sub:<10} FILE NOT FOUND — {e}")
        except Exception as e:
            print(f"{sub:<10} ERROR — {e}")

    df = pd.DataFrame(results)
    print("─" * len(header))
    mean = df.mean(numeric_only=True)
    print(f"{'Mean':<10} {mean['Motif_F1']:>9.3f} {mean['Motif_Accuracy']:>10.3f} "
          f"{mean['HRV_F1']:>8.3f} {mean['HRV_Accuracy']:>9.3f} "
          f"{mean['F1_Delta']:>+10.4f}\n")

    os.makedirs(output_dir, exist_ok=True)
    out_path = os.path.join(output_dir, "hrv_results.csv")
    df.to_csv(out_path, index=False)
    print(f"Saved → {out_path}")
    return df
