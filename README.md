# Antarctic Research Gap Finder

Most literature reviews tell you what has already been studied. This project flips that question around: instead of summarising the past, it tries to spot where the field is heading next.

The idea is simple. A topic with very few papers but fast-growing citations is a signal that something is starting to matter to researchers, even though almost nobody has written about it yet. Low paper density plus high citation velocity gives a quantitative definition of a "research gap".

## Why Antarctica

I picked Antarctic research specifically because I'm interested in environmental data, and it's a field that's growing fast but still relatively underexplored with this kind of quantitative approach.

## Beyond Antarctica

The pipeline isn't specific to polar science. Feed it any field's papers and citation data, and it'll point to the same kind of white space, areas that are under-studied relative to how fast attention is growing.

## How it works

I take the full corpus of roughly 12k Antarctic research papers and embed each one using SPECTER2, a model built specifically for scientific-document similarity, trained on the citation graph rather than generic text. These embeddings get reduced and clustered using UMAP and HDBSCAN to map out the actual research themes in the field.

From there, the gap detector looks for clusters that are sparse (few papers) but accelerating (citations growing fast for the papers that do exist).

I aim to find out the top 3 research questions Antarctic data science hasn't answered yet.

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