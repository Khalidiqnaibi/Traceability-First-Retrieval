"""
run_evaluation.py
=================
Publication-grade evaluation script comparing a Trust-weighted Retrieval (TFR)
pipeline against a Standard (RRF) baseline on biomedical RAG audit logs.

Usage:
    python run_evaluation.py \
        --log   pipeline_audit_log.csv \
        --queries queries.json \
        --out_dir results/

Outputs:
    results/metrics_summary.csv      – per-query metric table
    results/ablation_summary.csv     – stratified ablation table
    results/fig1_dumbbell.png        – slope/dumbbell plot
    results/fig2_grouped_bar.png     – nDCG@3 ablation bar chart
    results/stats_report.txt         – significance test report
"""

from __future__ import annotations

import ast
import io
import json
import os
import sys
import argparse
import warnings
from pathlib import Path
from typing import Any

def _force_utf8_stdout() -> None:
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")   # type: ignore[attr-defined]
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")   # type: ignore[attr-defined]
    except AttributeError:
        sys.stdout = io.TextIOWrapper(                                # type: ignore[assignment]
            sys.stdout.buffer, encoding="utf-8", errors="replace"
        )
        sys.stderr = io.TextIOWrapper(                                # type: ignore[assignment]
            sys.stderr.buffer, encoding="utf-8", errors="replace"
        )

_force_utf8_stdout()

import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import seaborn as sns
from scipy import stats

matplotlib.rcParams.update({
    "figure.dpi": 300,
    "savefig.dpi": 300,
    "font.family": "DejaVu Sans",
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.titlesize": 13,
    "axes.labelsize": 11,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "legend.fontsize": 9,
})
warnings.filterwarnings("ignore", category=RuntimeWarning)

# Constants
TIER_GAIN: dict[str, int] = {"Q1": 4, "Q2": 3, "Q3": 2, "Q4": 1, "Unranked": 0}
TIER_ORDER: list[str] = ["Q1", "Q2", "Q3", "Q4", "Unranked"]
PIPELINE_LABELS: dict[str, str] = {"TFR": "TFR", "Standard": "Standard", "Standard_RRF": "Standard"}
PALETTE: dict[str, str] = {"Standard": "#5B8DB8", "TFR": "#E07B54"}

# 1. Data Parsing Utilities
def safe_parse_results(raw: Any) -> list[dict]:
    """Safely parse a stringified list of dicts from the 'results' column."""
    if isinstance(raw, list):
        return raw
    if not isinstance(raw, str):
        return []
    try:
        parsed = ast.literal_eval(raw.strip())
        return parsed if isinstance(parsed, list) else []
    except (ValueError, SyntaxError):
        return []


def extract_provenance(item: dict) -> dict:
    """
    Extract evidence_level and journal_tier from a result item.
    Falls back gracefully when keys are absent.
    """
    prov = item.get("provenance") or item  # some logs embed fields at top level
    return {
        "evidence_level": int(prov.get("evidence_level", 5)),   # worst-case default
        "journal_tier":   str(prov.get("journal_tier", "Unranked")),
    }


def load_audit_log(path: str | Path) -> pd.DataFrame:
    """
    Load the audit log CSV, normalise pipeline names, and parse the results column.
    Returns a row-per-query-per-pipeline DataFrame with a parsed 'docs' column.
    """
    df = pd.read_csv(path)

    # Normalise pipeline labels (Standard_RRF → Standard)
    df["pipeline"] = df["pipeline"].map(lambda p: PIPELINE_LABELS.get(str(p).strip(), str(p).strip()))

    df["docs"] = df["results"].apply(safe_parse_results)
    return df


def load_queries_json(path: str | Path) -> dict[str, str]:
    """
    Load queries.json and return {query_text: ablation_dimension}.
    Accepts both a flat dict and a list-of-objects format.
    """
    with open(path, "r", encoding="utf-8") as f:
        raw = json.load(f)

    mapping: dict[str, str] = {}
    if isinstance(raw, dict):
        # Format A: {"query_text": "dimension", ...}
        mapping = {str(k): str(v) for k, v in raw.items()}
    elif isinstance(raw, list):
        for item in raw:
            if isinstance(item, dict):
                q  = item.get("query") or item.get("text") or item.get("question", "")
                dim = (
                    item.get("ablation_dimension")
                    or item.get("dimension")
                    or item.get("category", "unknown")
                )
                if q:
                    mapping[str(q)] = str(dim)
    return mapping

# 2. Metric Computations
def evidence_gain(ev_level: int) -> float:
    """Map evidence level to nDCG gain.  Level 1 → 5, Level 5 → 1."""
    return max(0.0, 6.0 - float(ev_level))


def tier_gain(journal_tier: str) -> float:
    """Map journal tier string to nDCG gain."""
    return float(TIER_GAIN.get(str(journal_tier), 0))


def _dcg(gains: list[float], k: int) -> float:
    """Standard DCG@k using log2 discounting."""
    g = gains[:k]
    return sum(rel / np.log2(idx + 2) for idx, rel in enumerate(g))


def ndcg_at_k(gains: list[float], k: int) -> float:
    """
    Normalised DCG@k.
    Returns 0 if there are no gains or ideal DCG is zero.
    """
    if not gains:
        return 0.0
    ideal = sorted(gains, reverse=True)
    idcg  = _dcg(ideal, k)
    if idcg == 0:
        return 0.0
    return _dcg(gains, k) / idcg


def compute_mrr(docs: list[dict]) -> float:
    """
    MRR for 'authoritative' documents (evidence_level <= 2 AND tier in Q1/Q2).
    Returns 1/rank of the first qualifying document, or 0.0 if none.
    """
    for rank, item in enumerate(docs, start=1):
        p = extract_provenance(item)
        if p["evidence_level"] <= 2 and p["journal_tier"] in {"Q1", "Q2"}:
            return 1.0 / rank
    return 0.0


def compute_query_metrics(docs: list[dict]) -> dict:
    """
    Compute all IR metrics for a single pipeline's result list for one query.
    """
    ev_gains   = [evidence_gain(extract_provenance(d)["evidence_level"]) for d in docs]
    tier_gains = [tier_gain(extract_provenance(d)["journal_tier"])        for d in docs]

    top1_prov = extract_provenance(docs[0]) if docs else {"evidence_level": 5, "journal_tier": "Unranked"}

    return {
        # nDCG – Evidence Level
        "ndcg1_ev":  ndcg_at_k(ev_gains, 1),
        "ndcg3_ev":  ndcg_at_k(ev_gains, 3),
        "ndcg5_ev":  ndcg_at_k(ev_gains, 5),
        # nDCG – Journal Tier
        "ndcg1_tier":  ndcg_at_k(tier_gains, 1),
        "ndcg3_tier":  ndcg_at_k(tier_gains, 3),
        "ndcg5_tier":  ndcg_at_k(tier_gains, 5),
        # Top-1 raw quality
        "top1_ev":    int(top1_prov["evidence_level"]),
        "top1_tier":  str(top1_prov["journal_tier"]),
        # MRR
        "mrr":        compute_mrr(docs),
    }

# 3. Main Evaluation Pipeline
def build_metrics_table(df: pd.DataFrame) -> pd.DataFrame:
    """
    Iterate over every (query, pipeline) pair and compute all metrics.
    Returns a long-format DataFrame with one row per (query, pipeline).
    """
    records = []
    for _, row in df.iterrows():
        m = compute_query_metrics(row["docs"])
        m["query"]    = row["query"]
        m["pipeline"] = row["pipeline"]
        records.append(m)
    return pd.DataFrame(records)


def run_significance_tests(metrics: pd.DataFrame, report_path: Path) -> None:
    """
    Paired significance test on nDCG@3 (Evidence Level) differences: TFR − Standard.
    Decides between Paired t-test and Wilcoxon based on Shapiro-Wilk normality.
    """
    pivot = metrics.pivot_table(
        index="query", columns="pipeline", values="ndcg3_ev", aggfunc="first"
    )
    # Drop queries where either pipeline is missing
    pivot = pivot.dropna(subset=["TFR", "Standard"])
    deltas = (pivot["TFR"] - pivot["Standard"]).values

    if len(deltas) < 3:
        print("\n[WARNING] Too few paired samples for significance testing.")
        return

    # Shapiro-Wilk normality test
    sw_stat, sw_p = stats.shapiro(deltas)

    lines: list[str] = [
        "=" * 60,
        "  Statistical Significance Report - nDCG@3 (Evidence Level)",
        "=" * 60,
        f"  Paired queries (n)       : {len(deltas)}",
        f"  Mean Delta (TFR-Standard): {deltas.mean():+.4f}",
        f"  Std  Delta               : {deltas.std():.4f}",
        f"  95% CI Delta             : [{deltas.mean() - 1.96*deltas.std()/np.sqrt(len(deltas)):+.4f}, "
                                       f"{deltas.mean() + 1.96*deltas.std()/np.sqrt(len(deltas)):+.4f}]",
        "",
        f"  Shapiro-Wilk W           : {sw_stat:.4f}",
        f"  Shapiro-Wilk p           : {sw_p:.4f}",
    ]

    if sw_p > 0.05:
        test_name = "Paired Student's t-test"
        t_stat, t_p = stats.ttest_rel(pivot["TFR"].values, pivot["Standard"].values)
        lines += [
            f"  Distribution             : NORMAL (p = {sw_p:.4f} > 0.05)",
            f"  Selected test            : {test_name}",
            f"  t-statistic              : {t_stat:.4f}",
            f"  p-value (two-tailed)     : {t_p:.4g}",
            f"  Significance             : {'YES [PASS]' if t_p < 0.05 else 'NO  [FAIL]'} (alpha = 0.05)",
        ]
    else:
        test_name = "Wilcoxon Signed-Rank test"
        try:
            w_stat, w_p = stats.wilcoxon(deltas, alternative="two-sided")
        except ValueError:
            w_stat, w_p = float("nan"), float("nan")
        lines += [
            f"  Distribution             : SKEWED (p = {sw_p:.4f} <= 0.05)",
            f"  Selected test            : {test_name}",
            f"  W-statistic              : {w_stat:.4f}",
            f"  p-value (two-tailed)     : {w_p:.4g}",
            f"  Significance             : {'YES [PASS]' if w_p < 0.05 else 'NO  [FAIL]'} (alpha = 0.05)",
        ]

    lines.append("=" * 60)
    report = "\n".join(lines)
    print("\n" + report)
    report_path.write_text(report, encoding="utf-8")
    print(f"\n  [OK] Stats report saved -> {report_path}")


# 4. Stratified Ablation Analysis
def ablation_analysis(metrics: pd.DataFrame, query_dim_map: dict[str, str]) -> pd.DataFrame:
    """
    Merge ablation dimension labels into the metrics table, then compute
    per-dimension mean Top-1 Evidence Level and Mean Top-1 Journal Tier
    for both pipelines.
    """
    # Map tier strings to ordinal for aggregation (lower ordinal = better)
    tier_to_ord = {t: i for i, t in enumerate(TIER_ORDER)}

    metrics = metrics.copy()
    metrics["ablation_dimension"] = metrics["query"].map(query_dim_map).fillna("unknown")
    metrics["top1_tier_ord"]      = metrics["top1_tier"].map(tier_to_ord).fillna(len(TIER_ORDER))

    agg = (
        metrics.groupby(["ablation_dimension", "pipeline"])
        .agg(
            mean_top1_ev   =("top1_ev",      "mean"),
            mean_top1_tier_ord=("top1_tier_ord", "mean"),
            mean_ndcg3_ev  =("ndcg3_ev",     "mean"),
            mean_ndcg3_tier=("ndcg3_tier",   "mean"),
            mean_mrr       =("mrr",          "mean"),
            n_queries      =("query",        "count"),
        )
        .reset_index()
    )

    # Decode ordinal back to nearest tier label for readability
    ord_to_tier = {v: k for k, v in tier_to_ord.items()}
    agg["mean_top1_tier_label"] = agg["mean_top1_tier_ord"].apply(
        lambda x: ord_to_tier.get(round(x), "Unranked")
    )
    return agg

# 5. Visualisations
def fig1_dumbbell(metrics: pd.DataFrame, out_path: Path) -> None:
    """
    Dumbbell / slope plot: shift in average Top-1 Evidence Level AND
    Journal Tier (as ordinal) from Standard → TFR, across all queries.

    Each horizontal pair of dots is one query; lines connect pipelines.
    """
    tier_to_ord = {t: i for i, t in enumerate(TIER_ORDER)}

    m = metrics.copy()
    m["top1_tier_ord"] = m["top1_tier"].map(tier_to_ord).fillna(len(TIER_ORDER))

    # Pivot to wide: one row per query, columns = Standard/TFR values
    def _wide(col: str) -> pd.DataFrame:
        p = m.pivot_table(index="query", columns="pipeline", values=col, aggfunc="first")
        p = p.dropna(subset=[c for c in ["Standard", "TFR"] if c in p.columns])
        return p

    ev_wide   = _wide("top1_ev")
    tier_wide = _wide("top1_tier_ord")

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle(
        "Figure 1 - Top-1 Quality Shift: Standard -> TFR",
        fontsize=14, fontweight="bold", y=1.01
    )

    def _draw_dumbbell(ax, wide: pd.DataFrame, title: str, ylabel: str,
                       invert_y: bool = False, yticklabels: list | None = None) -> None:
        std_col = "Standard" if "Standard" in wide.columns else wide.columns[0]
        tfr_col = "TFR"      if "TFR"      in wide.columns else wide.columns[-1]

        std_vals = wide[std_col].values
        tfr_vals = wide[tfr_col].values
        y_pos    = np.arange(len(wide))

        # Draw connecting lines, colour by direction of improvement
        for yp, sv, tv in zip(y_pos, std_vals, tfr_vals):
            color = "#2CA02C" if tv < sv else ("#D62728" if tv > sv else "#AAAAAA") \
                    if invert_y else \
                    ("#2CA02C" if tv > sv else ("#D62728" if tv < sv else "#AAAAAA"))
            ax.plot([0, 1], [sv, tv], color=color, lw=1.0, alpha=0.6, zorder=1)

        ax.scatter(np.zeros(len(std_vals)), std_vals,
                   color=PALETTE["Standard"], s=40, zorder=3, label="Standard", alpha=0.85)
        ax.scatter(np.ones(len(tfr_vals)),  tfr_vals,
                   color=PALETTE["TFR"],      s=40, zorder=3, label="TFR", alpha=0.85)

        # Means
        ax.scatter([0], [std_vals.mean()], color=PALETTE["Standard"],
                   s=180, marker="D", edgecolors="black", lw=1.5, zorder=5, label=f"Mean (Std)={std_vals.mean():.2f}")
        ax.scatter([1], [tfr_vals.mean()],  color=PALETTE["TFR"],
                   s=180, marker="D", edgecolors="black", lw=1.5, zorder=5, label=f"Mean (TFR)={tfr_vals.mean():.2f}")

        ax.set_xticks([0, 1])
        ax.set_xticklabels(["Standard", "TFR"], fontsize=10, fontweight="bold")
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        if invert_y:
            ax.invert_yaxis()
        if yticklabels:
            ax.set_yticks(range(len(yticklabels)))
            ax.set_yticklabels(yticklabels)
        ax.legend(loc="upper right", fontsize=7, framealpha=0.7)
        ax.set_xlim(-0.25, 1.25)

    _draw_dumbbell(
        axes[0], ev_wide,
        title="Evidence Level (lower = better)",
        ylabel="Top-1 Evidence Level",
        invert_y=True,
    )
    _draw_dumbbell(
        axes[1], tier_wide,
        title="Journal Tier (lower ordinal = better)",
        ylabel="Top-1 Journal Tier",
        invert_y=True,
        yticklabels=TIER_ORDER,
    )

    plt.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    print(f"  [OK] Figure 1 saved -> {out_path}")


def fig2_grouped_bar(ablation: pd.DataFrame, out_path: Path) -> None:
    """
    Grouped bar chart: mean nDCG@3 (Evidence Level) per ablation dimension,
    side-by-side bars for Standard vs TFR.
    """
    dimensions = ablation["ablation_dimension"].unique()
    std_vals, tfr_vals = [], []

    for dim in dimensions:
        sub = ablation[ablation["ablation_dimension"] == dim]
        s_row = sub[sub["pipeline"] == "Standard"]
        t_row = sub[sub["pipeline"] == "TFR"]
        std_vals.append(s_row["mean_ndcg3_ev"].values[0] if len(s_row) else 0.0)
        tfr_vals.append(t_row["mean_ndcg3_ev"].values[0] if len(t_row) else 0.0)

    x      = np.arange(len(dimensions))
    width  = 0.35

    fig, ax = plt.subplots(figsize=(10, 5))
    bars_s = ax.bar(x - width / 2, std_vals, width, label="Standard",
                    color=PALETTE["Standard"], edgecolor="white", linewidth=0.8)
    bars_t = ax.bar(x + width / 2, tfr_vals,  width, label="TFR",
                    color=PALETTE["TFR"],      edgecolor="white", linewidth=0.8)

    # Value labels
    for bar in [*bars_s, *bars_t]:
        h = bar.get_height()
        ax.text(bar.get_x() + bar.get_width() / 2.0, h + 0.005,
                f"{h:.3f}", ha="center", va="bottom", fontsize=8)

    # Delta annotation
    for xi, (sv, tv) in enumerate(zip(std_vals, tfr_vals)):
        delta = tv - sv
        color = "#2CA02C" if delta > 0 else "#D62728"
        ax.annotate(
            f"D{delta:+.3f}",
            xy=(xi, max(sv, tv) + 0.03),
            ha="center", fontsize=8, color=color, fontweight="bold",
        )

    ax.set_xticks(x)
    ax.set_xticklabels([d.replace("_", "\n") for d in dimensions], fontsize=9)
    ax.set_ylabel("Mean nDCG@3 (Evidence Level)", fontsize=10)
    ax.set_title(
        "Figure 2 - nDCG@3 by Ablation Dimension: Standard vs TFR",
        fontsize=13, fontweight="bold"
    )
    ax.set_ylim(0, min(1.1, max(max(std_vals), max(tfr_vals)) + 0.15))
    ax.yaxis.set_major_formatter(mticker.FormatStrFormatter("%.2f"))
    ax.legend(framealpha=0.7)
    ax.grid(axis="y", linestyle="--", alpha=0.4)

    plt.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    print(f"  [OK] Figure 2 saved -> {out_path}")


# 6. CLI Entry Point

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Evaluate TFR vs Standard RAG pipeline from an audit log."
    )
    p.add_argument("--log",     default="./logs/pipeline_audit_log.csv",
                   help="Path to pipeline_audit_log.csv")
    p.add_argument("--queries", default="./data/queries.json",
                   help="Path to queries.json (ablation dimension mapping)")
    p.add_argument("--out_dir", default="./results",
                   help="Directory for output files (created if absent)")
    return p.parse_args()


def main() -> None:
    args = parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # 1. Load & parse
    print(f"\n[1/5] Loading audit log  : {args.log}")
    df = load_audit_log(args.log)
    print(f"      Rows loaded          : {len(df)}")
    print(f"      Unique queries       : {df['query'].nunique()}")
    print(f"      Pipelines detected   : {sorted(df['pipeline'].unique())}")

    # 2. Compute per-query metrics
    print("\n[2/5] Computing IR metrics ...")
    metrics = build_metrics_table(df)

    metrics_path = out_dir / "metrics_summary.csv"
    metrics.to_csv(metrics_path, index=False)
    print(f"      Metrics saved        : {metrics_path}")

    # Quick console preview
    display_cols = ["query", "pipeline", "ndcg1_ev", "ndcg3_ev", "ndcg5_ev",
                    "ndcg3_tier", "top1_ev", "top1_tier", "mrr"]
    present = [c for c in display_cols if c in metrics.columns]
    with pd.option_context("display.max_colwidth", 40, "display.float_format", "{:.3f}".format):
        print("\n" + metrics[present].to_string(index=False))

    # 3. Statistical significance
    print("\n[3/5] Running significance tests ...")
    run_significance_tests(metrics, out_dir / "stats_report.txt")

    # 4. Ablation analysis
    print("\n[4/5] Stratified ablation analysis ...")
    query_dim_map: dict[str, str] = {}
    if Path(args.queries).is_file():
        query_dim_map = load_queries_json(args.queries)
        print(f"      Dimensions found     : {sorted(set(query_dim_map.values()))}")
    else:
        print(f"      [WARNING] {args.queries} not found - ablation dimension set to 'unknown'")

    ablation = ablation_analysis(metrics, query_dim_map)

    ablation_path = out_dir / "ablation_summary.csv"
    ablation.to_csv(ablation_path, index=False)
    print(f"      Ablation table saved : {ablation_path}")
    print("\n--- Ablation Summary " + "-" * 46)
    with pd.option_context("display.float_format", "{:.3f}".format):
        print(
            ablation[[
                "ablation_dimension", "pipeline",
                "mean_top1_ev", "mean_top1_tier_label",
                "mean_ndcg3_ev", "mean_ndcg3_tier", "mean_mrr", "n_queries"
            ]].to_string(index=False)
        )
    print("-" * 67)

    # 5. Figures
    print("\n[5/5] Generating figures ...")
    fig1_dumbbell(metrics,  out_dir / "fig1_dumbbell.png")
    fig2_grouped_bar(ablation, out_dir / "fig2_grouped_bar.png")

    print(f"\n{'='*60}")
    print("  Evaluation complete.")
    print(f"  All outputs written to -> {out_dir.resolve()}")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()