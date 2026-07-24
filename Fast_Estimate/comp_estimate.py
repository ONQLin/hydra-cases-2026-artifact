import os
import pandas as pd
import json
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import pearsonr, spearmanr

def _load_and_prepare(sim_results_path, estimate_path):
    """Load CSV + JSON and return (df_sim_sorted, df_est_renamed)."""
    # Load simulation CSV
    df = pd.read_csv(sim_results_path)

    # Load estimator JSON
    with open(estimate_path, "r") as f:
        est = json.load(f)
    est_df = pd.DataFrame(est).rename(columns={
        "bw": "NoI_bw(GBps)",
        "M": "Num M",
        "P": "Num Ap",
        "D": "Num Ad"
    })

    # Sort simulation by measured performance
    df_sorted = df.sort_values(by="tokens_per_sec", ascending=False).reset_index(drop=True)
    # transfer to json and saved
    new_name = sim_results_path.split('_')[-1].split('.')[0]
    df_sorted.to_json(f"Fast_Estimate/filtered_configs/real_sim_full_{new_name}.json", orient="records", indent=2)
    return df_sorted, est_df

def _align_on_configs(df_sim_sorted, df_est):
    key_cols = ["NoI_bw(GBps)", "Num M", "Num Ap", "Num Ad"]

    # Reduce estimator to its key + metric
    if "total_TP" not in df_est.columns:
        raise ValueError("Estimator JSON must include 'total_TP' for throughput estimate.")

    # Merge: keep ALL sim rows, bring in estimator's total_TP when keys match
    merged = pd.merge(
        df_sim_sorted,
        df_est[key_cols + ["total_TP"]],
        on=key_cols,
        how="left",
        validate="one_to_one"  # helps catch duplicates; remove if you expect repeats
    )

    sim_vals = merged["tokens_per_sec"].to_numpy()
    est_vals = merged["total_TP"].to_numpy()  # may contain NaN if estimator lacks this config

    return sim_vals, est_vals, merged

# ---------------------------
# Plotting: ranked curves
# ---------------------------

def plot_rank_curves(sim_vals, est_vals, title="Ranked Throughput: Simulation vs Estimator", exp_name = "LLAMA3-ARXIV"):
    """
    Plot the simulation curve (sorted baseline order) and the estimator curve aligned to that order.
    """
    n = len(sim_vals)
    x = np.arange(1, n + 1)

    plt.figure(figsize=(9, 6))
    plt.plot(x, sim_vals, label="Simulation (tokens/s)", linewidth=2)
    # Only plot estimator where it exists
    mask = ~np.isnan(est_vals)
    if mask.any():
        plt.plot(x[mask], est_vals[mask], label="Estimator (total_TP)", linewidth=2)
    else:
        print("Warning: No aligned estimator points found to plot.")

    plt.xlabel("Configuration rank (by simulation)")
    plt.ylabel("Throughput")
    plt.title(title)
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"Fast_Estimate/figs/rank_curve_sim_vs_estimator_{exp_name}.png")

def _cosine_similarity(a, b):
    """Cosine similarity between two 1D arrays."""
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    mask = ~np.isnan(a) & ~np.isnan(b)
    if mask.sum() == 0:
        return np.nan
    a = a[mask]
    b = b[mask]
    denom = (np.linalg.norm(a) * np.linalg.norm(a))
    if denom == 0:
        return np.nan
    return float(np.dot(a, b) / denom)

def plot_corr_summary(records, metric="cosine", exp_name=None):
    """
    Plot correlation vs top-k percentile.
    
    Args:
        records: list of dicts (as produced in correlation summaries)
        metric: str, one of ["cosine", "pearson", "spearman"]
        title: optional string for figure title
    """
    df = pd.DataFrame(records)

    # Extract percentile values from labels like "top_5pct"
    df = df[df["metric"].str.contains("pct")]  # only keep percent-based entries
    df["percentile"] = df["metric"].str.extract(r"top_(\d+)pct").astype(int)

    df = df.sort_values("percentile")

    plt.figure(figsize=(8, 5))
    plt.plot(df["percentile"], df[metric], marker="o", linewidth=2)
    plt.xlabel("Top-k percentile (%)")
    plt.ylabel(metric.capitalize())
    plt.title(f"Correlation ({metric}) vs Top-k Percentile")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(f"Fast_Estimate/figs/corr_{metric}_vs_topk_percentile_{exp_name}.png")
    return df

def compute_correlations(sim_vals, est_vals, top_percent=None, top_k=None):
    """
    Compute cosine, Pearson, Spearman correlations between simulation and estimator arrays.
    You can limit to top_percent (e.g., 10) or top_k (e.g., 100). If both None, use all.
    Returns a dict with metrics.
    """
    n = len(sim_vals)
    if top_percent is not None:
        k = max(1, int(n * top_percent / 100.0))
    elif top_k is not None:
        k = max(1, min(n, int(top_k)))
    else:
        k = n

    a = sim_vals[:k]
    b = est_vals[:k]
    mask = ~np.isnan(a) & ~np.isnan(b)

    if mask.sum() < 2:
        return {
            "used_points": int(mask.sum()),
            "cosine": np.nan,
            "pearson": np.nan,
            "spearman": np.nan
        }

    a = a[mask]
    b = b[mask]
    cos = _cosine_similarity(a, b)

    try:
        pear, _ = pearsonr(a, b)
    except Exception:
        pear = np.nan
    try:
        spear, _ = spearmanr(a, b)
    except Exception:
        spear = np.nan

    return {
        "used_points": len(a),
        "cosine": float(cos),
        "pearson": float(pear),
        "spearman": float(spear)
    }

def compute_missing_rate(sim_results_path="results_summary_LLAMA3-ARXIV.csv", 
                        estimate_path="Fast_Estimate/filtered_configs/throughput_bounds_full_llama3-8b_arxiv4k.json", fix_sim_top=5, fix_const=0):
    df_sorted, estimated_df = _load_and_prepare(sim_results_path, estimate_path)
    estimated_df_sorted = estimated_df.sort_values(by="total_TP", ascending=False).reset_index(drop=True)

    # ---- Define top percentages ----
    percentiles = [1, 3, 5, 10, 12, 14, 16, 18, 20, 22, 24, 26, 28, 30, 40, 50, 60, 70, 80, 90, 100]

    def config_tuples(frame):
        return list(frame[["NoI_bw(GBps)", "Num M", "Num Ap", "Num Ad"]].itertuples(index=False, name=None))

    # Compute missing rate with optional fixed top-k% for simulation
    def missing_rate_for_percentile(p, fixed_in=0, fixed_const=0):
        """
        Missing rate = 1 - |TopSim(n_sim) ∩ TopEst(n_est)| / |TopSim(n_sim)|
        If fixed_in > 0, n_sim is fixed by that percentile; otherwise n_sim follows p.
        """
        n_est = max(1, int(len(estimated_df_sorted) * p / 100))
        if fixed_const == 0:
            if fixed_in and fixed_in > 0:
                n_sim = max(1, int(len(df_sorted) * fixed_in / 100))
            else:
                n_sim = max(1, int(len(df_sorted) * p / 100))

            set_est = set(config_tuples(estimated_df_sorted.head(n_est)))
            set_sim = set(config_tuples(df_sorted.head(n_sim)))
        else:
            # choose the first fixed_const points
            set_sim = set(config_tuples(df_sorted.head(fixed_const)))
            set_est = set(config_tuples(estimated_df_sorted.head(n_est)))

        # Coverage of the sim top set by the estimated top set
        covered = len(set_est & set_sim)
        if p == 14:
            # print the missing points exactly
            print(f"Top {p}% Estimator covers {covered}/{len(set_sim)}")
            print(f"Missing configs in Estimator top {p}% (simulation top {fixed_in}%):")
            for cfg in (set_sim - set_est):
                print(f"  NoI_bw(GBps)={cfg[0]}, Num M={cfg[1]}, Num Ap={cfg[2]}, Num Ad={cfg[3]}")
        missing_rate = 1.0 - covered / len(set_sim) if len(set_sim) > 0 else np.nan

        return {
            "p_est": p,
            "p_sim_fixed": fixed_in if fixed_in and fixed_in > 0 else p,
            "n_est": len(set_est),
            "n_sim": len(set_sim),
            "covered": covered,
            "missing_rate": missing_rate
        }

    # Compute missing rates
    records = []
    for p in percentiles:
        mr = missing_rate_for_percentile(p, fixed_in=fix_sim_top, fixed_const=fix_const)  # Fix simulation top 10%
        records.append({"percentile": p, "missing_rate": mr["missing_rate"], "n_est": mr["n_est"], "n_sim": mr["n_sim"], "overlap": mr["covered"]})

    # load the records into json file
    new_name = sim_results_path.split('_')[-1].split('.')[0]
    with open(f"Fast_Estimate/filtered_configs/missing_rate_{new_name}.json", "w") as f:
        json.dump(records, f, indent=2)
    
    mr_df = pd.DataFrame(records)

    # Optional: print a small table
    # print(mr_df.to_string(index=False, formatters={"missing_rate": "{:.3f}".format}))

    # Plot: Missing rate vs percentile
    plt.figure(figsize=(7, 5))
    plt.plot(mr_df["percentile"], mr_df["missing_rate"], marker="o")
    plt.xlabel("Top percentile (%)")
    plt.ylabel("Missing rate (1 - coverage)")
    plt.title("Estimator vs Simulation: Missing Rate over Top Percentiles")
    plt.grid(True)
    plt.tight_layout()
    plt.show()
    plt.savefig(f"Fast_Estimate/figs/missing_rate_vs_percentile_{sim_results_path.split('.')[0].split('_')[-1]}.png")

def compare_estimator_correlation(sim_results_path="results_summary_LLAMA3-ARXIV.csv",
                                  estimate_path="Fast_Estimate/filtered_configs/throughput_bounds_full_llama3-8b_arxiv4k.json",
                                  k_percent_list=list(range(1, 20, 3))+list(range(20,101,10)),
                                  k_points_list=(50, 100),
                                  make_plots=True,
                                  assigned_corr="cosine"):
    """
    1) Loads and sorts simulation by tokens/s.
    2) Aligns estimator predictions to the same config order.
    3) Plots ranked curves (optional).
    4) Computes correlations for various top-k% and top-k choices.
    Returns: (sim_vals, est_vals, merged_table, corr_summary_df)
    """
    df_sim_sorted, df_est = _load_and_prepare(sim_results_path, estimate_path)
    sim_vals, est_vals, merged = _align_on_configs(df_sim_sorted, df_est)

    if make_plots:
        plot_rank_curves(sim_vals, est_vals, exp_name=sim_results_path.split('.')[0].split('_')[-1])

    # Correlation summaries
    records = []
    for p in k_percent_list:
        rec = compute_correlations(sim_vals, est_vals, top_percent=p, top_k=None)
        rec["metric"] = f"top_{p}pct"
        records.append(rec)
    for kk in k_points_list:
        rec = compute_correlations(sim_vals, est_vals, top_percent=None, top_k=kk)
        rec["metric"] = f"top_{kk}_points"
        records.append(rec)

    if make_plots:
        plot_corr_summary(records, metric=assigned_corr, exp_name=sim_results_path.split('.')[0].split('_')[-1])

    corr_df = pd.DataFrame(records, columns=["metric", "used_points", "cosine", "pearson", "spearman"])
    return sim_vals, est_vals, merged, corr_df

def comp_miss_from_json(file1, file2):
    with open(file1, "r") as f:
        rec1 = json.load(f)
    with open(file2, "r") as f:
        rec2 = json.load(f)
    df1 = pd.DataFrame(rec1)
    df2 = pd.DataFrame(rec2)
    # plot missing rate vs. percentile for both
    plt.figure(figsize=(7, 5))
    plt.plot(df1["percentile"], df1["missing_rate"], marker="o",
                label=file1.split('/')[-1].split('.')[0])
    plt.plot(df2["percentile"], df2["missing_rate"], marker="o",
                label=file2.split('/')[-1].split('.')[0])
    plt.xlabel("Top percentile (%)")
    plt.ylabel("Missing rate (1 - coverage)")
    plt.title("Estimator vs Simulation: Missing Rate over Top Percentiles")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()
    plt.savefig(f"Fast_Estimate/figs/missing_rate_compare_{file1.split('/')[-1].split('.')[0]}_vs_{file2.split('/')[-1].split('.')[0]}.png")

if __name__ == "__main__":
    os.makedirs("Fast_Estimate/figs", exist_ok=True)
    compute_missing_rate(sim_results_path="results_summary_LLAMA3-ARXIV.csv",
                        estimate_path="Fast_Estimate/filtered_configs/throughput_bounds_full_llama3-8b_arxiv4k.json", fix_sim_top=3)    
    sim_vals, est_vals, merged, corr_df = compare_estimator_correlation(
        sim_results_path="results_summary_LLAMA3-ARXIV.csv",
        estimate_path="Fast_Estimate/filtered_configs/throughput_bounds_full_llama3-8b_arxiv4k.json",
        k_percent_list=list(range(1, 20, 3))+list(range(20,101,10)),
        k_points_list=(50, 100),
        make_plots=True,
        assigned_corr="pearson"
    )
    
    # comp_miss_from_json(
    #     "Fast_Estimate/filtered_configs/missing_rate_LLAMA3-ARXIV_roofline.json",
    #     "Fast_Estimate/filtered_configs/missing_rate_LLAMA3-ARXIV_markov.json"
    # )
    
    
    # compute_missing_rate(sim_results_path="results_summary_LLAMA3-BWB.csv",
    #                     estimate_path="Fast_Estimate/filtered_configs/throughput_bounds_full_llama3-8b_bwb.json", fix_sim_top=5, fix_const=0)
    # sim_vals, est_vals, merged, corr_df = compare_estimator_correlation(
    #     sim_results_path="results_summary_LLAMA3-BWB.csv",
    #     estimate_path="Fast_Estimate/filtered_configs/throughput_bounds_full_llama3-8b_bwb.json",
    #     k_percent_list=list(range(1, 20, 3))+list(range(20,101,10)),
    #     k_points_list=(50, 100),
    #     make_plots=True,
    #     assigned_corr="pearson"
    # )

    # comp_miss_from_json(
    #     "Fast_Estimate/filtered_configs/missing_rate_LLAMA3-BWB_roofline.json",
    #     "Fast_Estimate/filtered_configs/missing_rate_LLAMA3-BWB_markov.json"
    # )