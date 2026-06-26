"""Stage 2: Embed each paper with SPECTER2.

SPECTER2 maps a paper's (title + abstract) to a single 768-dimensional vector
such that papers that cite / are cited by each other land near each other in
space. That citation-aware geometry is what makes the later clustering find
real research themes instead of just shared vocabulary.

Run:
    python -m src.embed

Output: `data/embeddings.npy`, shape (n_papers, 768), row-aligned with
`papers.parquet`. Cache it so this step only runs once.

Key SPECTER2 details:
  * Input format is  title + [SEP] + abstract  (the model was trained this way).
  * The paper vector is the [CLS] token, i.e. row 0 of the last hidden state.
  * Load the "proximity" adapter.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import torch
from adapters import AutoAdapterModel
from transformers import AutoTokenizer
from tqdm import tqdm

from src import config


def pick_device() -> str:
    """Prefer Apple-Silicon GPU (MPS), then CUDA, else CPU."""
    if torch.backends.mps.is_available():
        return "mps"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


def load_model(device: str):
    """Load the SPECTER2 base encoder and activate the proximity adapter."""
    tokenizer = AutoTokenizer.from_pretrained(config.SPECTER2_BASE)
    model = AutoAdapterModel.from_pretrained(config.SPECTER2_BASE)
    model.load_adapter(
        config.SPECTER2_ADAPTER,
        source="hf",
        load_as="proximity",
        set_active=True,
    )
    # `set_active=True` alone doesn't always persist into the forward pass
    # (the adapters lib warns "none are activated"), so pin it explicitly.
    model.set_active_adapters("proximity")
    model.to(device).eval()
    return tokenizer, model


@torch.no_grad()
def embed_batch(texts: list[str], tokenizer, model, device: str) -> np.ndarray:
    """Embed one batch of 'title[SEP]abstract' strings -> (batch, 768)."""
    inputs = tokenizer(
        texts,
        padding=True,
        truncation=True,
        max_length=config.EMBED_MAX_TOKENS,
        return_tensors="pt",
    ).to(device)
    output = model(**inputs)
    # [CLS] token = position 0 of the sequence. This is SPECTER2's paper vector.
    cls = output.last_hidden_state[:, 0, :]
    result = cls.cpu().numpy()
    # Drop GPU references each batch. We clear the MPS cache only periodically
    # (in main) — calling empty_cache() every batch forces a GPU sync and is a
    # major slowdown.
    del inputs, output, cls
    return result


def build_inputs(df: pd.DataFrame, sep_token: str) -> list[str]:
    """Join title and abstract the way SPECTER2 expects."""
    return [
        f"{t}{sep_token}{a}"
        for t, a in zip(df["title"].fillna(""), df["abstract"].fillna(""))
    ]


def est_tokens(text: str) -> int:
    """Cheap token-count estimate (~4 chars/token), capped at the model limit.

    Good enough for batching decisions and avoids a second full tokenizer pass.
    """
    return min(len(text) // 4 + 1, config.EMBED_MAX_TOKENS)


def make_batches(indices: list[int], texts: list[str]) -> list[list[int]]:
    """Group row indices into batches with a bounded total token budget.

    `indices` must be sorted by length. We grow a batch until adding the next
    paper would exceed EMBED_TOKENS_PER_BATCH or EMBED_MAX_BATCH papers. Because
    longer papers cost more tokens each, their batches end up smaller — which is
    exactly what keeps GPU memory flat.
    """
    batches: list[list[int]] = []
    cur: list[int] = []
    cur_tokens = 0
    for i in indices:
        t = est_tokens(texts[i])
        # Batch length is padded to its longest member, so the cost of adding
        # this paper is (len(cur)+1) * max_token_len. Approximate with t.
        if cur and (
            cur_tokens + t > config.EMBED_TOKENS_PER_BATCH
            or len(cur) >= config.EMBED_MAX_BATCH
        ):
            batches.append(cur)
            cur, cur_tokens = [], 0
        cur.append(i)
        cur_tokens += t
    if cur:
        batches.append(cur)
    return batches


def main() -> None:
    device = pick_device()
    print(f"Device: {device}")

    df = pd.read_parquet(config.PAPERS_PARQUET)
    print(f"Loaded {len(df):,} papers")

    tokenizer, model = load_model(device)
    texts = build_inputs(df, tokenizer.sep_token)

    # Disk-backed .npy: batches are written straight to disk. If the file already
    # exists (e.g. a previous run stopped early) we open it for update and skip
    # rows that are already filled — so we resume instead of starting over.
    if config.EMBEDDINGS_NPY.exists():
        embeddings = np.lib.format.open_memmap(config.EMBEDDINGS_NPY, mode="r+")
        done = ~(np.asarray(embeddings) == 0).all(axis=1)
        todo = [i for i in range(len(texts)) if not done[i]]
        print(f"Resuming: {int(done.sum()):,} already done, {len(todo):,} to go")
    else:
        embeddings = np.lib.format.open_memmap(
            config.EMBEDDINGS_NPY, mode="w+", dtype=np.float32,
            shape=(len(texts), 768),
        )
        todo = list(range(len(texts)))

    # Sort the remaining papers by length, then pack into token-budget batches.
    todo.sort(key=lambda i: len(texts[i]))
    batches = make_batches(todo, texts)

    for idx in tqdm(batches, desc="SPECTER2", unit="batch"):
        vecs = embed_batch([texts[i] for i in idx], tokenizer, model, device)
        embeddings[idx] = vecs               # scatter back to original rows
        # Clear the MPS pool every batch to stop memory creeping up. With
        # token-budget batches this sync is cheap (short-text batches are big).
        if device == "mps":
            torch.mps.empty_cache()

    embeddings.flush()
    print(f"\nSaved embeddings {embeddings.shape} -> {config.EMBEDDINGS_NPY}")
    assert embeddings.shape[0] == len(df), "row count must match papers.parquet"


if __name__ == "__main__":
    main()
