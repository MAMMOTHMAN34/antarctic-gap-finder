"""Stage 3: Reduce and cluster the embeddings into research themes.

SPECTER2 gives us a 768-d vector per paper, but two things are hard in 768-d:
clustering and plotting. So, I use UMAP to project the vectors down, then HDBSCAN to find
clusters.

Two projections:
  * UMAP -> 5-D : a denser space that HDBSCAN clusters more cleanly.
  * UMAP -> 2-D : purely for the map we draw.

HDBSCAN is density-based, which matters for this project: it finds clusters of varying density and labels genuinely sparse points as noise (-1) instead of
forcing them into a theme. Those sparse regions are exactly the gaps I hunt for in Stage 5.

Run:
    python -m src.cluster

Outputs:
    data/clusters.parquet   id, cluster, x, y  (row-aligned with papers)
    outputs/landscape.png    the research-landscape map
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from umap import UMAP
from hdbscan import HDBSCAN

from src import config


def reduce_umap(embeddings: np.ndarray, n_components: int) -> np.ndarray:
    """Project embeddings to `n_components` dims with our standard settings."""
    reducer = UMAP(
        n_components=n_components,
        n_neighbors=config.UMAP_NEIGHBORS,
        min_dist=config.UMAP_MIN_DIST,
        metric=config.UMAP_METRIC,
        random_state=config.RANDOM_STATE,
    )
    return reducer.fit_transform(embeddings)


def cluster_hdbscan(points: np.ndarray) -> np.ndarray:
    """Cluster the reduced points; returns an int label per row (-1 = noise)."""
    clusterer = HDBSCAN(
        min_cluster_size=config.HDBSCAN_MIN_CLUSTER_SIZE,
        min_samples=config.HDBSCAN_MIN_SAMPLES,
        metric="euclidean",          # UMAP output lives in Euclidean space
    )
    return clusterer.fit_predict(points)


def plot_landscape(xy: np.ndarray, labels: np.ndarray, path) -> None:
    """Scatter the 2-D map, colouring clusters and greying out noise points."""
    fig, ax = plt.subplots(figsize=(11, 9))
    noise = labels == -1
    # Noise first, underneath, in light grey.
    ax.scatter(xy[noise, 0], xy[noise, 1], s=3, c="lightgrey",
               alpha=0.5, linewidths=0, label="unclustered")
    # Clustered points coloured by label.
    clustered = ~noise
    ax.scatter(xy[clustered, 0], xy[clustered, 1], s=5, c=labels[clustered],
               cmap="tab20", alpha=0.8, linewidths=0)
    ax.set_title("Antarctic research landscape — SPECTER2 + UMAP + HDBSCAN",
                 fontsize=13)
    ax.set_xlabel("UMAP-1"); ax.set_ylabel("UMAP-2")
    ax.legend(loc="upper right", markerscale=3, frameon=False)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def main() -> None:
    config.OUTPUTS.mkdir(parents=True, exist_ok=True)

    df = pd.read_parquet(config.PAPERS_PARQUET)
    embeddings = np.load(config.EMBEDDINGS_NPY)
    print(f"Loaded {embeddings.shape[0]:,} embeddings of dim {embeddings.shape[1]}")

    print("UMAP -> 5-D for clustering ...")
    reduced = reduce_umap(embeddings, config.UMAP_CLUSTER_DIMS)

    print("HDBSCAN ...")
    labels = cluster_hdbscan(reduced)

    print("UMAP -> 2-D for the map ...")
    xy = reduce_umap(embeddings, 2)

    # Persist labels + 2-D coordinates alongside paper ids.
    out = pd.DataFrame(
        {"id": df["id"], "cluster": labels, "x": xy[:, 0], "y": xy[:, 1]}
    )
    out.to_parquet(config.CLUSTERS_PARQUET, index=False)
    plot_landscape(xy, labels, config.LANDSCAPE_PNG)

    n_clusters = len({c for c in labels if c != -1})
    noise_frac = (labels == -1).mean()
    sizes = pd.Series(labels[labels != -1]).value_counts()
    print(f"\nFound {n_clusters} themes | {noise_frac:.0%} of papers unclustered")
    print(f"Cluster sizes: median {sizes.median():.0f}, "
          f"largest {sizes.max()}, smallest {sizes.min()}")
    print(f"Saved -> {config.CLUSTERS_PARQUET}")
    print(f"Map   -> {config.LANDSCAPE_PNG}")


if __name__ == "__main__":
    main()
