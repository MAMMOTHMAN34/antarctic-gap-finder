"""Stage 5: Detect research gaps.

This is the core idea of the project. For every theme I measure three things:

  * size            how many papers (FEW papers  -> more of a gap)
  * velocity        median citations-per-year of its papers
                    (FAST citations -> the field is paying attention)
  * recent_share    fraction of papers from RECENT_FROM onward
                    (RECENT -> emerging, not a settled topic)

A "gap" is a theme that is small, fast, and recent at the same time. I put the
three signals on a common scale (z-scores) and combine them:

    gap_score =  w_v * z(velocity)
               + w_r * z(recent_share)
               - w_s * z(log size)        # subtract: big themes aren't gaps

The themes with the highest gap_score are candidate under-served frontiers.

Why citations-per-year (not raw citations): a 2023 paper with 20 citations is
accelerating far faster than a 2011 paper with 40. Dividing by age makes a
recent, fast-cited theme stand out instead of being buried by old, big ones.

Run:
    python -m src.gaps
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from src import config


def zscore(s: pd.Series) -> pd.Series:
    """Standardize to mean 0, sd 1 (so different signals are comparable)."""
    return (s - s.mean()) / s.std(ddof=0)


def theme_metrics(df: pd.DataFrame) -> pd.DataFrame:
    """Aggregate per-paper data into per-theme size / velocity / recency."""
    age = (config.REFERENCE_YEAR - df["year"]).clip(lower=1)
    df = df.assign(
        cites_per_year=df["cited_by_count"] / age,
        is_recent=df["year"] >= config.RECENT_FROM,
    )
    g = df.groupby("cluster")
    metrics = pd.DataFrame(
        {
            "size": g.size(),
            "velocity": g["cites_per_year"].median(),
            "recent_share": g["is_recent"].mean(),
            "mean_year": g["year"].mean(),
            "median_citations": g["cited_by_count"].median(),
        }
    )
    return metrics.drop(index=-1, errors="ignore")  # drop the noise cluster


def score_gaps(metrics: pd.DataFrame) -> pd.DataFrame:
    """Combine the standardized signals into a single gap score."""
    z_vel = zscore(metrics["velocity"])
    z_rec = zscore(metrics["recent_share"])
    z_size = zscore(np.log(metrics["size"]))
    metrics = metrics.assign(
        gap_score=(
            config.GAP_W_VELOCITY * z_vel
            + config.GAP_W_RECENCY * z_rec
            - config.GAP_W_SIZE * z_size
        )
    )
    return metrics.sort_values("gap_score", ascending=False)


def plot_gap_map(metrics: pd.DataFrame, themes: pd.DataFrame, path) -> None:
    """Scatter size vs velocity; top gaps sit in the upper-left (few + fast)."""
    fig, ax = plt.subplots(figsize=(11, 8))
    sc = ax.scatter(
        metrics["size"], metrics["velocity"],
        c=metrics["gap_score"], cmap="plasma",
        s=60 + 200 * metrics["recent_share"],   # bigger = more recent
        edgecolors="k", linewidths=0.4,
    )
    ax.set_xscale("log")
    ax.set_xlabel("Theme size  (# papers, log scale)  ← fewer = more of a gap")
    ax.set_ylabel("Citation velocity  (median cites/year)  ↑ faster = hotter")
    ax.set_title("Antarctic research gap map\n"
                 "upper-left = small but fast-cited (emerging) · "
                 "bubble size = recency", fontsize=12)
    fig.colorbar(sc, label="gap score")

    # Label the top gap themes.
    top = metrics.head(5)
    for cl, row in top.iterrows():
        label = themes.loc[themes.cluster == cl, "label"].values
        name = label[0] if len(label) else str(cl)
        ax.annotate(name, (row["size"], row["velocity"]),
                    fontsize=8, xytext=(5, 5), textcoords="offset points")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def main() -> None:
    papers = pd.read_parquet(config.PAPERS_PARQUET)
    clusters = pd.read_parquet(config.CLUSTERS_PARQUET)
    themes = pd.read_csv(config.THEMES_CSV)
    df = papers.merge(clusters[["id", "cluster"]], on="id")

    metrics = theme_metrics(df)
    metrics = score_gaps(metrics)

    # Attach the human-readable labels.
    out = metrics.merge(
        themes[["cluster", "label", "top_terms"]],
        left_index=True, right_on="cluster",
    )
    cols = ["cluster", "label", "size", "velocity", "recent_share",
            "mean_year", "median_citations", "gap_score", "top_terms"]
    out = out[cols].round(3)
    out.to_csv(config.GAPS_CSV, index=False)
    plot_gap_map(metrics, themes, config.GAP_MAP_PNG)

    print(f"Scored {len(out)} themes -> {config.GAPS_CSV}")
    print(f"Gap map -> {config.GAP_MAP_PNG}\n")
    print("TOP GAP CANDIDATES (small + fast + recent):")
    print(f"{'score':>6}  {'size':>5}  {'vel':>5}  {'yr':>6}  label")
    for _, r in out.head(8).iterrows():
        print(f"{r['gap_score']:>6.2f}  {r['size']:>5.0f}  "
              f"{r['velocity']:>5.1f}  {r['mean_year']:>6.0f}  {r['label']}")


if __name__ == "__main__":
    main()
