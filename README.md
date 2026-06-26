# Antarctic Research Gap Finder

## Why this is interesting

Most literature reviews reveal what *has* been studied. On the other hand,
this project aims to find out emerging topics, via embedding the whole corpus and looking for **sparse but accelerating** regions. This intersection of low density and high citation velocity is a quantitative definition of a "research gap."

## Pipeline

| Stage | Script | Output |
|-------|--------|--------|
| 1. Acquire | `src/fetch_openalex.py` | `data/papers.parquet` |
| 2. Embed | _(coming)_ | `data/embeddings.npy` |
| 3. Reduce + cluster | _(coming)_ | cluster labels |
| 4. Label themes | _(coming)_ | theme keywords |
| 5. Gap detection | _(coming)_ | ranked gaps |
| 6. Report | _(coming)_ | 3 research questions + plots |

## Stack

- **Data:** [OpenAlex](https://openalex.org) (open scholarly index, no API key)
- **Embeddings:** [SPECTER2](https://github.com/allenai/SPECTER2); built for
  scientific-document similarity from the citation graph
- **Clustering:** UMAP → HDBSCAN (density-based, finds natural themes *and*
  leaves sparse gaps unclustered)
- **Theme labels:** c-TF-IDF