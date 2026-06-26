"""Interactive version of the research landscape (Plotly).

The static `landscape.png` shows the shape of the field; this builds an HTML map
I can actually explore. Hovering a point shows that paper's title, year,
citations and theme, and I can zoom/pan into any cluster.

Run:
    python -m src.viz_interactive

Output: outputs/landscape.html
"""

from __future__ import annotations

import textwrap

import pandas as pd
import plotly.express as px

from src import config


def short(title: str, width: int = 70) -> str:
    """Wrap long titles so the hover tooltip stays readable."""
    return "<br>".join(textwrap.wrap(str(title), width)) or "—"


def main() -> None:
    papers = pd.read_parquet(config.PAPERS_PARQUET)
    coords = pd.read_parquet(config.CLUSTERS_PARQUET)        # id, cluster, x, y
    themes = pd.read_csv(config.THEMES_CSV)                  # cluster -> label

    df = papers.merge(coords, on="id").merge(
        themes[["cluster", "label"]], on="cluster", how="left"
    )
    # Name the noise cluster and give every clustered point its theme label.
    df["theme"] = df["label"].fillna("unclustered")
    df["title_wrapped"] = df["title"].map(short)

    fig = px.scatter(
        df,
        x="x", y="y",
        color="theme",
        hover_data={
            "title_wrapped": True,
            "year": True,
            "cited_by_count": True,
            "theme": True,
            "x": False, "y": False,
        },
        title="Antarctic research landscape — hover a paper, zoom into a theme",
        opacity=0.7,
    )
    fig.update_traces(marker=dict(size=4))
    fig.update_layout(
        legend_title_text="theme",
        xaxis_title="UMAP-1", yaxis_title="UMAP-2",
        height=800,
    )

    config.OUTPUTS.mkdir(parents=True, exist_ok=True)
    fig.write_html(config.LANDSCAPE_HTML, include_plotlyjs="cdn")
    print(f"Saved interactive map -> {config.LANDSCAPE_HTML}")
    print("Open it in a browser: "
          f"open {config.LANDSCAPE_HTML}")


if __name__ == "__main__":
    main()
