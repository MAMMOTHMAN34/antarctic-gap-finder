"""Central configuration for the Antarctic Research Gap Finder.

Keeping paths and tunable parameters in one place means every stage of the
pipeline reads from the same source of truth, and the writeup can point at this
file to explain the exact run that produced the results.
"""

from pathlib import Path

# --- Paths ---------------------------------------------------------------
ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"
OUTPUTS = ROOT / "outputs"

PAPERS_PARQUET = DATA / "papers.parquet"
EMBEDDINGS_NPY = DATA / "embeddings.npy"

# --- Stage 2: SPECTER2 embeddings ---------------------------------------
# SPECTER2 = a base encoder + a task adapter. The "proximity" adapter is the
# one trained for document-similarity / retrieval, which is exactly what we
# want for clustering.
SPECTER2_BASE = "allenai/specter2_base"
SPECTER2_ADAPTER = "allenai/specter2"
EMBED_MAX_TOKENS = 512    # SPECTER2's context limit (per paper)
# Token-budget batching: cap each batch by total tokens, not paper count, so
# batches of long abstracts stay small and never OOM the Apple GPU. This is the
# single setting that keeps memory flat; lower it if you ever see an OOM.
EMBED_TOKENS_PER_BATCH = 8000
EMBED_MAX_BATCH = 64      # also cap count, so tiny-abstract batches stay sane

# --- Stage 3: dimensionality reduction + clustering ---------------------
CLUSTERS_PARQUET = DATA / "clusters.parquet"   # id, cluster, x, y per paper
LANDSCAPE_PNG = OUTPUTS / "landscape.png"

# We run UMAP twice: a moderate-dim projection that HDBSCAN clusters on (dense
# spaces cluster better than raw 768-d), and a 2-D projection purely for the map.
UMAP_CLUSTER_DIMS = 5
UMAP_NEIGHBORS = 15        # ~how local vs global the structure is
UMAP_MIN_DIST = 0.0        # 0.0 packs points tightly -> cleaner clusters
UMAP_METRIC = "cosine"     # right metric for transformer embeddings
RANDOM_STATE = 42          # reproducible runs (UMAP becomes single-threaded)

# HDBSCAN finds clusters of varying density and leaves outliers unlabeled (-1).
# With ~12k papers, ~80 keeps themes substantive rather than dozens of tiny ones.
HDBSCAN_MIN_CLUSTER_SIZE = 80
HDBSCAN_MIN_SAMPLES = 10   # higher -> more points treated as noise/outliers

# --- OpenAlex query ------------------------------------------------------
# OpenAlex is a free, open scholarly index (no API key needed). The "polite
# pool" just asks that you identify yourself with an email for faster, more
# reliable service. Put your own email here.
OPENALEX_MAILTO = "raeannetanrt@gmail.com"

# We want Antarctic *science*, so we search title + abstract for Antarctic
# terms and require an abstract (no abstract = nothing to embed later).
# Multiple terms catch papers that say "Southern Ocean" or "Antarctica"
# without the word "Antarctic".
SEARCH_TERMS = "Antarctic OR Antarctica OR \"Southern Ocean\" OR Weddell OR \"Ross Sea\""

# Restrict to a recent-ish window so "citation velocity" is meaningful and the
# corpus reflects current research fronts rather than historical expeditions.
YEAR_FROM = 2010
YEAR_TO = 2024

# Stop once we have at least this many usable papers.
TARGET_PAPERS = 12000

# OpenAlex allows up to 200 results per page via cursor pagination.
PER_PAGE = 200
