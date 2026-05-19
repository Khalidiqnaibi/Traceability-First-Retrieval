"""
run_evaluation.py  (v2)
======================================================
Evaluation script comparing Trust-weighted Retrieval (TFR) against Standard
(RRF) baseline on biomedical RAG audit logs.

Fixes vs v1
-----------
* Dumbbell overplotting: per-query dots are jittered so overlapping discrete
  values (evidence level 1-5, tier ordinal 0-4) are visible as a spread.
* Dead y_pos variable removed from _draw_dumbbell.
* Fig 2 y-axis ceiling is now computed dynamically so delta annotations never
  clip for any dimension.
* Cliff's delta (non-parametric effect size) added to the stats report
  alongside Wilcoxon, as required by Q1 journals.
* Significance tests now cover ndcg3_ev, ndcg3_tier, and mrr (with Holm-
  Bonferroni correction for multiple comparisons).
* Bootstrap 95 % CI error bars added to Fig 2 grouped bar chart.
* Dumbbell colour conditional cleaned up (dead invert_y branch removed).
* Figure aesthetics tightened: consistent font sizes, gridlines, axis padding.

Usage:
    python run_evaluation.py \\
        --log   pipeline_audit_log.csv \\
        --queries queries.json \\
        --out_dir results/

Outputs:
    results/metrics_summary.csv      – per-query metric table
    results/ablation_summary.csv     – stratified ablation table
    results/fig1_dumbbell.png        – jittered dumbbell plot
    results/fig2_grouped_bar.png     – nDCG@3 ablation bar chart with CI
    results/stats_report.txt         – significance + effect-size report
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
from dotenv import load_dotenv

load_dotenv()

QUERY_PATH = Path(os.getenv("QUERIES_PATH", "./data/seed_queries.json"))
OUT_DIR = Path(os.getenv("OUTPUT_DIR", "./results"))
EVAL_LOG_PATH = Path(os.getenv("EVAL_LOG_PATH", "./logs/pipeline_audit_log.csv"))

def _force_utf8_stdout() -> None:
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    except AttributeError:
        sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
        sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")

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
PIPELINE_LABELS: dict[str, str] = {
    "TFR": "TFR",
    "Standard": "Standard",
    "Standard_RRF": "Standard",
}
PALETTE: dict[str, str] = {"Standard": "#5B8DB8", "TFR": "#E07B54"}
RNG = np.random.default_rng(42)

# 1. Data Parsing Utilities
def safe_parse_results(raw: Any) -> list[dict]:
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
    prov = item.get("provenance") or item
    return {
        "evidence_level": int(prov.get("evidence_level", 5)),
        "journal_tier":   str(prov.get("journal_tier", "Unranked")),
    }


def load_audit_log(path: str | Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    df["pipeline"] = df["pipeline"].map(
        lambda p: PIPELINE_LABELS.get(str(p).strip(), str(p).strip())
    )
    df["docs"] = df["results"].apply(safe_parse_results)
    return df


def load_queries_json(path: str | Path) -> dict[str, str]:
    with open(path, "r", encoding="utf-8") as f:
        raw = json.load(f)

    mapping: dict[str, str] = {}
    if isinstance(raw, dict):
        mapping = {str(k): str(v) for k, v in raw.items()}
    elif isinstance(raw, list):
        for item in raw:
            if isinstance(item, dict):
                q   = item.get("query") or item.get("text") or item.get("question", "")
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
    return max(0.0, 6.0 - float(ev_level))


def tier_gain(journal_tier: str) -> float:
    return float(TIER_GAIN.get(str(journal_tier), 0))


def _dcg(gains: list[float], k: int) -> float:
    g = gains[:k]
    return sum(rel / np.log2(idx + 2) for idx, rel in enumerate(g))


def ndcg_at_k(gains: list[float], k: int) -> float:
    if not gains:
        return 0.0
    ideal = sorted(gains, reverse=True)
    idcg  = _dcg(ideal, k)
    if idcg == 0:
        return 0.0
    return _dcg(gains, k) / idcg


def compute_mrr(docs: list[dict]) -> float:
    for rank, item in enumerate(docs, start=1):
        p = extract_provenance(item)
        if p["evidence_level"] <= 2 and p["journal_tier"] in {"Q1", "Q2"}:
            return 1.0 / rank
    return 0.0


def compute_query_metrics(docs: list[dict]) -> dict:
    ev_gains   = [evidence_gain(extract_provenance(d)["evidence_level"]) for d in docs]
    tier_gains = [tier_gain(extract_provenance(d)["journal_tier"])        for d in docs]
    top1_prov  = extract_provenance(docs[0]) if docs else {
        "evidence_level": 5, "journal_tier": "Unranked"
    }
    return {
        "ndcg1_ev":    ndcg_at_k(ev_gains, 1),
        "ndcg3_ev":    ndcg_at_k(ev_gains, 3),
        "ndcg5_ev":    ndcg_at_k(ev_gains, 5),
        "ndcg1_tier":  ndcg_at_k(tier_gains, 1),
        "ndcg3_tier":  ndcg_at_k(tier_gains, 3),
        "ndcg5_tier":  ndcg_at_k(tier_gains, 5),
        "top1_ev":     int(top1_prov["evidence_level"]),
        "top1_tier":   str(top1_prov["journal_tier"]),
        "mrr":         compute_mrr(docs),
    }

# 3. Main Evaluation Pipeline
def build_metrics_table(df: pd.DataFrame) -> pd.DataFrame:
    records = []
    for _, row in df.iterrows():
        m = compute_query_metrics(row["docs"])
        m["query"]    = row["query"]
        m["pipeline"] = row["pipeline"]
        records.append(m)
    return pd.DataFrame(records)


# 4. Statistical Significance Tests
def cliffs_delta(x: np.ndarray, y: np.ndarray) -> float:
    """
    Cliff's delta: non-parametric effect size for two independent/paired
    samples.  Range [-1, 1]; |d| < 0.147 negligible, < 0.33 small,
    < 0.474 medium, >= 0.474 large (Romano et al. 2006).
    For paired data we pass the deltas as x and a zero array as y.
    """
    n = len(x) * len(y)
    if n == 0:
        return float("nan")
    count = sum(1 if xi > yi else (-1 if xi < yi else 0)
                for xi in x for yi in y)
    return count / n


def interpret_cliffs(d: float) -> str:
    a = abs(d)
    if a < 0.147:
        return "negligible"
    if a < 0.330:
        return "small"
    if a < 0.474:
        return "medium"
    return "large"


def _paired_test(
    pivot: pd.DataFrame, col_a: str = "TFR", col_b: str = "Standard"
) -> dict:
    """Run Shapiro → Wilcoxon or t-test; return result dict."""
    p2 = pivot.dropna(subset=[col_a, col_b])
    deltas = (p2[col_a] - p2[col_b]).values
    n = len(deltas)
    if n < 3:
        return {"n": n, "error": "Too few paired samples"}

    sw_stat, sw_p = stats.shapiro(deltas)
    mean_d = deltas.mean()
    std_d  = deltas.std()
    se     = std_d / np.sqrt(n)
    ci_lo  = mean_d - 1.96 * se
    ci_hi  = mean_d + 1.96 * se

    # Effect size: Cliff's delta (TFR values vs Standard values)
    cd = cliffs_delta(p2[col_a].values, p2[col_b].values)

    if sw_p > 0.05:
        t_stat, p_val = stats.ttest_rel(p2[col_a].values, p2[col_b].values)
        return dict(
            n=n, mean_delta=mean_d, std_delta=std_d, ci=(ci_lo, ci_hi),
            sw_stat=sw_stat, sw_p=sw_p, normal=True,
            test="Paired Student's t-test", stat=t_stat, p_val=p_val,
            cliffs_delta=cd,
        )
    else:
        try:
            w_stat, p_val = stats.wilcoxon(deltas, alternative="two-sided")
        except ValueError:
            w_stat, p_val = float("nan"), float("nan")
        return dict(
            n=n, mean_delta=mean_d, std_delta=std_d, ci=(ci_lo, ci_hi),
            sw_stat=sw_stat, sw_p=sw_p, normal=False,
            test="Wilcoxon Signed-Rank test", stat=w_stat, p_val=p_val,
            cliffs_delta=cd,
        )


def run_significance_tests(metrics: pd.DataFrame, report_path: Path) -> None:
    """
    Paired significance tests on ndcg3_ev, ndcg3_tier, and mrr.
    Applies Holm-Bonferroni correction across the three comparisons.
    Reports Cliff's delta effect size alongside each test.
    """
    test_cols = {
        "ndcg3_ev":   "nDCG@3 (Evidence Level)",
        "ndcg3_tier": "nDCG@3 (Journal Tier)",
        "mrr":        "MRR (Authoritative Docs)",
    }

    results: dict[str, dict] = {}
    for col, label in test_cols.items():
        pivot = metrics.pivot_table(
            index="query", columns="pipeline", values=col, aggfunc="first"
        )
        results[col] = _paired_test(pivot)
        results[col]["label"] = label

    # Holm-Bonferroni correction on the raw p-values
    p_vals = [r.get("p_val", 1.0) for r in results.values()]
    # Sort indices by ascending p-value
    sorted_idx = sorted(range(len(p_vals)), key=lambda i: p_vals[i])
    m = len(p_vals)
    adjusted: list[float] = [0.0] * m
    for rank, idx in enumerate(sorted_idx):
        adjusted[idx] = min(1.0, p_vals[idx] * (m - rank))
    cols_list = list(results.keys())
    for i, col in enumerate(cols_list):
        results[col]["p_adj_holm"] = adjusted[i]

    # Build report text
    sep = "=" * 64
    lines: list[str] = [
        sep,
        "  Statistical Significance Report",
        "  TFR vs Standard  |  paired tests + effect sizes",
        sep,
    ]

    for col, res in results.items():
        if "error" in res:
            lines += ["", f"  [{res['label']}]  ERROR: {res['error']}"]
            continue
        sig_raw = "YES [PASS]" if res["p_val"] < 0.05 else "NO  [FAIL]"
        sig_adj = "YES [PASS]" if res["p_adj_holm"] < 0.05 else "NO  [FAIL]"
        dist_str = (
            f"NORMAL (SW p = {res['sw_p']:.4f} > 0.05)"
            if res["normal"]
            else f"SKEWED (SW p = {res['sw_p']:.4f} <= 0.05)"
        )
        cd_interp = interpret_cliffs(res["cliffs_delta"])
        stat_label = "t" if res["normal"] else "W"
        lines += [
            "",
            f"  Metric : {res['label']}",
            f"  -------",
            f"  Paired queries (n)       : {res['n']}",
            f"  Mean Delta (TFR-Standard): {res['mean_delta']:+.4f}",
            f"  Std  Delta               : {res['std_delta']:.4f}",
            f"  95% CI Delta             : [{res['ci'][0]:+.4f}, {res['ci'][1]:+.4f}]",
            f"  Shapiro-Wilk W           : {res['sw_stat']:.4f}",
            f"  Distribution             : {dist_str}",
            f"  Selected test            : {res['test']}",
            f"  {stat_label}-statistic              : {res['stat']:.4f}",
            f"  p-value (two-tailed)     : {res['p_val']:.4g}",
            f"  p-value (Holm-Bonferroni): {res['p_adj_holm']:.4g}",
            f"  Significance (raw)       : {sig_raw} (alpha = 0.05)",
            f"  Significance (adjusted)  : {sig_adj} (alpha = 0.05)",
            f"  Cliff's delta            : {res['cliffs_delta']:+.4f}  [{cd_interp}]",
        ]

    lines += ["", sep]
    report = "\n".join(lines)
    print("\n" + report)
    report_path.write_text(report, encoding="utf-8")
    print(f"\n  [OK] Stats report saved -> {report_path}")

# 5. Stratified Ablation Analysis
def ablation_analysis(metrics: pd.DataFrame, query_dim_map: dict[str, str]) -> pd.DataFrame:
    tier_to_ord = {t: i for i, t in enumerate(TIER_ORDER)}
    metrics = metrics.copy()
    metrics["ablation_dimension"] = metrics["query"].map(query_dim_map).fillna("unknown")
    metrics["top1_tier_ord"]      = metrics["top1_tier"].map(tier_to_ord).fillna(len(TIER_ORDER))

    agg = (
        metrics.groupby(["ablation_dimension", "pipeline"])
        .agg(
            mean_top1_ev      =("top1_ev",       "mean"),
            mean_top1_tier_ord=("top1_tier_ord",  "mean"),
            mean_ndcg3_ev     =("ndcg3_ev",       "mean"),
            mean_ndcg3_tier   =("ndcg3_tier",     "mean"),
            mean_mrr          =("mrr",            "mean"),
            n_queries         =("query",          "count"),
        )
        .reset_index()
    )
    ord_to_tier = {v: k for k, v in tier_to_ord.items()}
    agg["mean_top1_tier_label"] = agg["mean_top1_tier_ord"].apply(
        lambda x: ord_to_tier.get(round(x), "Unranked")
    )
    return agg

# 6. Bootstrap CI helper
def bootstrap_ci(
    data: np.ndarray,
    n_boot: int = 2000,
    ci: float = 0.95,
) -> tuple[float, float]:
    """Return (lower, upper) bootstrap percentile CI of the mean."""
    if len(data) == 0:
        return (float("nan"), float("nan"))
    boots = RNG.choice(data, size=(n_boot, len(data)), replace=True).mean(axis=1)
    lo = np.percentile(boots, (1 - ci) / 2 * 100)
    hi = np.percentile(boots, (1 + ci) / 2 * 100)
    return lo, hi

# 7. Visualisations
JITTER_SCALE = 0.06   # y-jitter amplitude for dumbbell dots


def fig1_dumbbell(metrics: pd.DataFrame, out_path: Path) -> None:
    """
    Jittered dumbbell plot: Top-1 Evidence Level and Journal Tier ordinal,
    Standard → TFR.  Individual query dots are jittered vertically to avoid
    the severe overplotting that discrete (1-5 / 0-4) y-values produce when
    many queries share the same value.
    """
    tier_to_ord = {t: i for i, t in enumerate(TIER_ORDER)}
    m = metrics.copy()
    m["top1_tier_ord"] = m["top1_tier"].map(tier_to_ord).fillna(len(TIER_ORDER))

    def _wide(col: str) -> pd.DataFrame:
        p = m.pivot_table(index="query", columns="pipeline", values=col, aggfunc="first")
        keep = [c for c in ["Standard", "TFR"] if c in p.columns]
        return p.dropna(subset=keep)

    ev_wide   = _wide("top1_ev")
    tier_wide = _wide("top1_tier_ord")

    fig, axes = plt.subplots(1, 2, figsize=(12, 5.5))
    fig.suptitle(
        "Figure 1 – Top-1 Quality Shift: Standard → TFR",
        fontsize=14, fontweight="bold", y=1.02,
    )

    def _draw_dumbbell(
        ax: plt.Axes,
        wide: pd.DataFrame,
        title: str,
        ylabel: str,
        invert_y: bool = False,
        yticklabels: list | None = None,
    ) -> None:
        std_col = "Standard" if "Standard" in wide.columns else wide.columns[0]
        tfr_col = "TFR"      if "TFR"      in wide.columns else wide.columns[-1]

        std_vals = wide[std_col].values
        tfr_vals = wide[tfr_col].values
        n        = len(std_vals)

        # Per-query y-jitter so overlapping discrete values spread out
        jitter = RNG.uniform(-JITTER_SCALE, JITTER_SCALE, size=n)

        # Connecting lines coloured by direction of improvement
        # (improvement = lower value when invert_y=True)
        for sv, tv, jit in zip(std_vals, tfr_vals, jitter):
            improved = tv < sv if invert_y else tv > sv
            unchanged = sv == tv
            color = "#2CA02C" if improved else ("#D62728" if not unchanged else "#AAAAAA")
            ax.plot([0, 1], [sv + jit, tv + jit],
                    color=color, lw=0.8, alpha=0.45, zorder=1)

        ax.scatter(np.zeros(n), std_vals + jitter,
                   color=PALETTE["Standard"], s=30, zorder=3,
                   label="Standard", alpha=0.75, linewidths=0)
        ax.scatter(np.ones(n),  tfr_vals + jitter,
                   color=PALETTE["TFR"],      s=30, zorder=3,
                   label="TFR",      alpha=0.75, linewidths=0)

        # Mean diamonds (no jitter)
        ax.scatter([0], [std_vals.mean()],
                   color=PALETTE["Standard"], s=200, marker="D",
                   edgecolors="black", lw=1.5, zorder=5,
                   label=f"Mean (Std) = {std_vals.mean():.2f}")
        ax.scatter([1], [tfr_vals.mean()],
                   color=PALETTE["TFR"],      s=200, marker="D",
                   edgecolors="black", lw=1.5, zorder=5,
                   label=f"Mean (TFR) = {tfr_vals.mean():.2f}")

        ax.set_xticks([0, 1])
        ax.set_xticklabels(["Standard", "TFR"], fontsize=10, fontweight="bold")
        ax.set_ylabel(ylabel, fontsize=10)
        ax.set_title(title, fontsize=11, pad=8)
        ax.set_xlim(-0.30, 1.30)
        if invert_y:
            ax.invert_yaxis()
        if yticklabels:
            ax.set_yticks(range(len(yticklabels)))
            ax.set_yticklabels(yticklabels, fontsize=9)
        ax.legend(loc="upper right", fontsize=7, framealpha=0.75,
                  handlelength=1.2, borderpad=0.6)
        ax.grid(axis="y", linestyle=":", alpha=0.35)

    _draw_dumbbell(
        axes[0], ev_wide,
        title="Evidence Level  (lower = better)",
        ylabel="Top-1 Evidence Level",
        invert_y=True,
    )
    _draw_dumbbell(
        axes[1], tier_wide,
        title="Journal Tier  (lower ordinal = better)",
        ylabel="Top-1 Journal Tier",
        invert_y=True,
        yticklabels=TIER_ORDER,
    )

    plt.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    print(f"  [OK] Figure 1 saved -> {out_path}")


def fig2_grouped_bar(
    ablation: pd.DataFrame,
    metrics: pd.DataFrame,
    out_path: Path,
) -> None:
    """
    Grouped bar chart: mean nDCG@3 (Evidence Level) per ablation dimension,
    Standard vs TFR, with 95 % bootstrap CI error bars and delta annotations.
    """
    dimensions = ablation["ablation_dimension"].unique()
    std_vals, tfr_vals = [], []
    std_ci_lo, std_ci_hi = [], []
    tfr_ci_lo, tfr_ci_hi = [], []

    for dim in dimensions:
        sub = ablation[ablation["ablation_dimension"] == dim]
        s_row = sub[sub["pipeline"] == "Standard"]
        t_row = sub[sub["pipeline"] == "TFR"]

        sv = s_row["mean_ndcg3_ev"].values[0] if len(s_row) else 0.0
        tv = t_row["mean_ndcg3_ev"].values[0] if len(t_row) else 0.0
        std_vals.append(sv)
        tfr_vals.append(tv)

        # Bootstrap CI from raw per-query metrics for this dimension
        raw = metrics.copy()
        raw["ablation_dimension"] = raw["query"].map(
            {q: d for q, d in
             zip(metrics["query"], metrics.get("ablation_dimension",
                 pd.Series(["unknown"] * len(metrics))))
            }
        )
        
        for pipeline, ci_lo_lst, ci_hi_lst, vals_lst in [
            ("Standard", std_ci_lo, std_ci_hi, std_vals),
            ("TFR",      tfr_ci_lo, tfr_ci_hi, tfr_vals),
        ]:
            pass  # filled below

    std_ci_lo, std_ci_hi = [], []
    tfr_ci_lo, tfr_ci_hi = [], []

    has_dim = "ablation_dimension" in metrics.columns

    for dim in dimensions:
        for pipeline, ci_lo_lst, ci_hi_lst in [
            ("Standard", std_ci_lo, std_ci_hi),
            ("TFR",      tfr_ci_lo, tfr_ci_hi),
        ]:
            if has_dim:
                raw_vals = metrics.loc[
                    (metrics["ablation_dimension"] == dim) &
                    (metrics["pipeline"] == pipeline),
                    "ndcg3_ev",
                ].values
                lo, hi = bootstrap_ci(raw_vals)
            else:
                lo, hi = float("nan"), float("nan")
            ci_lo_lst.append(lo)
            ci_hi_lst.append(hi)

    x     = np.arange(len(dimensions))
    width = 0.35

    fig, ax = plt.subplots(figsize=(11, 5.5))

    def _err(vals, lo, hi):
        return [
            [v - l if not np.isnan(l) else 0 for v, l in zip(vals, lo)],
            [h - v if not np.isnan(h) else 0 for v, h in zip(vals, hi)],
        ]

    bars_s = ax.bar(
        x - width / 2, std_vals, width, label="Standard",
        color=PALETTE["Standard"], edgecolor="white", linewidth=0.8,
        yerr=_err(std_vals, std_ci_lo, std_ci_hi),
        error_kw=dict(elinewidth=1.0, capsize=3, ecolor="#2d2d2d", alpha=0.7),
    )
    bars_t = ax.bar(
        x + width / 2, tfr_vals, width, label="TFR",
        color=PALETTE["TFR"], edgecolor="white", linewidth=0.8,
        yerr=_err(tfr_vals, tfr_ci_lo, tfr_ci_hi),
        error_kw=dict(elinewidth=1.0, capsize=3, ecolor="#2d2d2d", alpha=0.7),
    )

    # Value labels on bars
    for bar in [*bars_s, *bars_t]:
        h = bar.get_height()
        ax.text(
            bar.get_x() + bar.get_width() / 2.0, h + 0.004,
            f"{h:.3f}", ha="center", va="bottom", fontsize=7.5,
        )

    # Delta annotations – placed above the taller bar + its CI upper bound
    max_tops = []
    for xi, (sv, tv, shi, thi) in enumerate(
        zip(std_vals, tfr_vals, std_ci_hi, tfr_ci_hi)
    ):
        delta = tv - sv
        color = "#2CA02C" if delta > 0 else ("#D62728" if delta < 0 else "#888888")
        top_s = sv + (shi - sv if not np.isnan(shi) else 0)
        top_t = tv + (thi - tv if not np.isnan(thi) else 0)
        ann_y = max(top_s, top_t) + 0.025
        max_tops.append(ann_y)
        ax.annotate(
            f"Δ{delta:+.3f}",
            xy=(xi, ann_y),
            ha="center", fontsize=8.5, color=color, fontweight="bold",
        )

    ax.set_xticks(x)
    ax.set_xticklabels([d.replace("_", "\n") for d in dimensions], fontsize=9)
    ax.set_ylabel("Mean nDCG@3 (Evidence Level)", fontsize=10)
    ax.set_title(
        "Figure 2 – nDCG@3 by Ablation Dimension: Standard vs TFR",
        fontsize=13, fontweight="bold",
    )

    # Dynamic y-ceiling: accommodate annotation text (≈ 0.05 headroom above label)
    y_ceil = max(max_tops) + 0.06
    ax.set_ylim(0, min(1.15, y_ceil))
    ax.yaxis.set_major_formatter(mticker.FormatStrFormatter("%.2f"))
    ax.legend(framealpha=0.75, fontsize=9)
    ax.grid(axis="y", linestyle="--", alpha=0.35)
    # Footnote for CI
    fig.text(
        0.5, -0.02,
        "Error bars: 95 % bootstrap CI (n = 2000 resamples)",
        ha="center", fontsize=8, color="#555555", style="italic",
    )

    plt.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    print(f"  [OK] Figure 2 saved -> {out_path}")

# 8. CLI Entry Point
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Evaluate TFR vs Standard RAG pipeline from an audit log."
    )
    p.add_argument("--log",     default=EVAL_LOG_PATH)
    p.add_argument("--queries", default=QUERY_PATH)
    p.add_argument("--out_dir", default=OUT_DIR)
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

    display_cols = ["query", "pipeline", "ndcg1_ev", "ndcg3_ev", "ndcg5_ev",
                    "ndcg3_tier", "top1_ev", "top1_tier", "mrr"]
    present = [c for c in display_cols if c in metrics.columns]
    with pd.option_context("display.max_colwidth", 40, "display.float_format", "{:.3f}".format):
        print("\n" + metrics[present].to_string(index=False))

    # 3. Statistical significance (ndcg3_ev, ndcg3_tier, mrr + Holm correction)
    print("\n[3/5] Running significance tests ...")
    run_significance_tests(metrics, out_dir / "stats_report.txt")

    # 4. Ablation analysis
    print("\n[4/5] Stratified ablation analysis ...")
    query_dim_map: dict[str, str] = {}
    if Path(args.queries).is_file():
        query_dim_map = load_queries_json(args.queries)
        print(f"      Dimensions found     : {sorted(set(query_dim_map.values()))}")
    else:
        print(f"      [WARNING] {args.queries} not found – ablation dimension = 'unknown'")

    ablation = ablation_analysis(metrics, query_dim_map)

    # Attach ablation_dimension to metrics for bootstrap CI in Fig 2
    metrics["ablation_dimension"] = metrics["query"].map(query_dim_map).fillna("unknown")

    ablation_path = out_dir / "ablation_summary.csv"
    ablation.to_csv(ablation_path, index=False)
    print(f"      Ablation table saved : {ablation_path}")
    print("\n--- Ablation Summary " + "-" * 46)
    with pd.option_context("display.float_format", "{:.3f}".format):
        print(
            ablation[[
                "ablation_dimension", "pipeline",
                "mean_top1_ev", "mean_top1_tier_label",
                "mean_ndcg3_ev", "mean_ndcg3_tier", "mean_mrr", "n_queries",
            ]].to_string(index=False)
        )
    print("-" * 67)

    # 5. Figures
    print("\n[5/5] Generating figures ...")
    fig1_dumbbell(metrics, out_dir / "fig1_dumbbell.png")
    fig2_grouped_bar(ablation, metrics, out_dir / "fig2_grouped_bar.png")

    print(f"\n{'='*64}")
    print("  Evaluation complete.")
    print(f"  All outputs written to -> {out_dir.resolve()}")
    print(f"{'='*64}\n")


if __name__ == "__main__":
    main()