"""Stage 1 — Acquire the corpus.

Pull Antarctic research papers from the OpenAlex API and save a tidy table to
`data/papers.parquet`. We keep only the fields the rest of the pipeline needs:

    id, title, abstract, year, cited_by_count, venue, doi, concepts

Run:
    python -m src.fetch_openalex          # from the project root

OpenAlex returns abstracts as an "inverted index" (word -> positions) to dodge
copyright on full text. We rebuild readable text from it; that text is what
SPECTER2 will embed in Stage 2.
"""

from __future__ import annotations

import time
import requests
import pandas as pd
from tqdm import tqdm

from src import config

OPENALEX_WORKS = "https://api.openalex.org/works"


def reconstruct_abstract(inverted_index: dict | None) -> str:
    """Turn OpenAlex's {word: [positions]} index back into a string.

    Example: {"Sea": [0, 5], "ice": [1]} -> "Sea ice ...".
    Returns "" when no abstract is available.
    """
    if not inverted_index:
        return ""
    # Build (position, word) pairs, then sort by position.
    positions: list[tuple[int, str]] = []
    for word, idxs in inverted_index.items():
        for i in idxs:
            positions.append((i, word))
    positions.sort(key=lambda p: p[0])
    return " ".join(word for _, word in positions)


def build_filter() -> str:
    """Compose the OpenAlex `filter` parameter for our Antarctic query."""
    return ",".join(
        [
            f"title_and_abstract.search:{config.SEARCH_TERMS}",
            f"from_publication_date:{config.YEAR_FROM}-01-01",
            f"to_publication_date:{config.YEAR_TO}-12-31",
            "has_abstract:true",          # nothing to embed without one
            "type:article",               # drop datasets, errata, etc.
            "language:en",                # SPECTER2 is English-trained
        ]
    )


def fetch() -> pd.DataFrame:
    """Page through OpenAlex with a cursor until we hit TARGET_PAPERS."""
    params = {
        "filter": build_filter(),
        "per-page": config.PER_PAGE,
        "cursor": "*",                    # "*" starts cursor pagination
        "mailto": config.OPENALEX_MAILTO,
        # Only ask for the fields we use -> smaller, faster responses.
        "select": (
            "id,title,abstract_inverted_index,publication_year,"
            "cited_by_count,primary_location,doi,concepts"
        ),
    }

    rows: list[dict] = []
    pbar = tqdm(total=config.TARGET_PAPERS, unit="paper", desc="OpenAlex")

    while len(rows) < config.TARGET_PAPERS:
        resp = requests.get(OPENALEX_WORKS, params=params, timeout=60)
        resp.raise_for_status()
        payload = resp.json()

        results = payload.get("results", [])
        if not results:
            break  # ran out of papers before reaching the target

        for w in results:
            abstract = reconstruct_abstract(w.get("abstract_inverted_index"))
            if len(abstract) < 100:
                continue  # too short to be a useful signal

            loc = w.get("primary_location") or {}
            source = loc.get("source") or {}
            rows.append(
                {
                    "id": w["id"],
                    "title": w.get("title") or "",
                    "abstract": abstract,
                    "year": w.get("publication_year"),
                    "cited_by_count": w.get("cited_by_count", 0),
                    "venue": source.get("display_name"),
                    "doi": w.get("doi"),
                    # Keep the top concept names OpenAlex assigns; handy as a
                    # sanity check against our own clusters later.
                    "concepts": "; ".join(
                        c["display_name"] for c in (w.get("concepts") or [])[:5]
                    ),
                }
            )
        pbar.update(min(len(results), config.TARGET_PAPERS - pbar.n))

        # Advance the cursor; absence means we've reached the end.
        next_cursor = payload.get("meta", {}).get("next_cursor")
        if not next_cursor:
            break
        params["cursor"] = next_cursor
        time.sleep(0.1)  # be gentle on the API

    pbar.close()
    return pd.DataFrame(rows)


def main() -> None:
    config.DATA.mkdir(parents=True, exist_ok=True)
    df = fetch()

    # Drop duplicate works (same paper can surface under different runs).
    df = df.drop_duplicates(subset="id").reset_index(drop=True)

    df.to_parquet(config.PAPERS_PARQUET, index=False)

    print(f"\nSaved {len(df):,} papers -> {config.PAPERS_PARQUET}")
    print(f"Year range : {df['year'].min()}-{df['year'].max()}")
    print(f"Median citations: {df['cited_by_count'].median():.0f}")
    print(f"Top venues:\n{df['venue'].value_counts().head(5).to_string()}")


if __name__ == "__main__":
    main()
