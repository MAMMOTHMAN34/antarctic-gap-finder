"""Stage 4 — Name each theme with c-TF-IDF.

We have 39 clusters but only numbers. c-TF-IDF (class-based TF-IDF, the idea
behind BERTopic's labels) gives each cluster a name *from the papers' own words*:

  1. Treat every cluster as one big document (all its papers concatenated).
  2. Count words per cluster.
  3. Weight each word by how characteristic it is of that cluster vs the rest:
         tf-idf(t, c) = tf(t, c) * log(1 + avg_words_per_class / freq(t))
     Words common to every cluster (e.g. "ice") get a low weight; words that
     concentrate in one cluster score high. The top-weighted words are the label.

This is why we don't just reuse OpenAlex's concept tags: these labels are
derived independently from our own embedding-driven clusters, which makes the
themes ours to defend.

Run:
    python -m src.label

Output: outputs/themes.csv  (cluster, size, label, top_terms)
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction import text as sk_text

from src import config


def class_tfidf(docs: list[str]) -> tuple[np.ndarray, np.ndarray]:
    """Compute the c-TF-IDF matrix for per-cluster documents.

    Returns (matrix of shape [n_clusters, vocab], feature names).
    """
    stop_words = list(sk_text.ENGLISH_STOP_WORDS.union(config.LABEL_EXTRA_STOPWORDS))
    vec = CountVectorizer(
        stop_words=stop_words,
        ngram_range=(1, config.LABEL_NGRAM_MAX),
        min_df=config.LABEL_MIN_DF,
        token_pattern=r"(?u)\b[a-zA-Z][a-zA-Z-]+\b",  # words only, no pure numbers
    )
    counts = vec.fit_transform(docs).toarray().astype(float)  # [n_clusters, vocab]

    # tf: term frequency within each cluster, normalised by cluster length.
    tf = counts / counts.sum(axis=1, keepdims=True)

    # idf: log(1 + average words per class / total frequency of the term).
    avg_words = counts.sum(axis=1).mean()
    freq = counts.sum(axis=0)                 # total per term across clusters
    idf = np.log(1.0 + avg_words / freq)

    return tf * idf, np.array(vec.get_feature_names_out())


def main() -> None:
    df = pd.read_parquet(config.PAPERS_PARQUET)
    clusters = pd.read_parquet(config.CLUSTERS_PARQUET)
    df = df.merge(clusters[["id", "cluster"]], on="id")

    # Drop noise (-1); concatenate each cluster's papers into one document.
    labelled = sorted(c for c in df["cluster"].unique() if c != -1)
    docs = [
        " ".join(
            (df.loc[df.cluster == c, "title"].fillna("") + " " +
             df.loc[df.cluster == c, "abstract"].fillna("")).tolist()
        )
        for c in labelled
    ]

    ctfidf, vocab = class_tfidf(docs)

    rows = []
    for row_i, c in enumerate(labelled):
        top_idx = ctfidf[row_i].argsort()[::-1][: config.LABEL_TOP_TERMS]
        terms = vocab[top_idx]
        rows.append(
            {
                "cluster": c,
                "size": int((df.cluster == c).sum()),
                "label": ", ".join(terms[:4]),     # short human label
                "top_terms": ", ".join(terms),     # full keyword list
            }
        )

    themes = pd.DataFrame(rows).sort_values("size", ascending=False)
    config.OUTPUTS.mkdir(parents=True, exist_ok=True)
    themes.to_csv(config.THEMES_CSV, index=False)

    print(f"Labelled {len(themes)} themes -> {config.THEMES_CSV}\n")
    for _, r in themes.head(15).iterrows():
        print(f"  [{r['size']:>4}]  {r['label']}")


if __name__ == "__main__":
    main()
