"""
Core retrieval functions shared by search.py (CLI), app.py (Streamlit UI) and
eval.py (evaluation).

Kept separate from search.py so eval.py can call the exact same ranking code
the CLI uses, instead of a re-implementation that could quietly drift.
"""
from pathlib import Path

import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer

MODEL_NAME = "all-MiniLM-L6-v2"
BASE_DIR = Path(__file__).resolve().parent
EMBEDDINGS_PATH = BASE_DIR / "song_embeddings.npy"
METADATA_PATH = BASE_DIR / "song_metadata.csv"


def load_index(embeddings_path=EMBEDDINGS_PATH, metadata_path=METADATA_PATH):
    for path in (embeddings_path, metadata_path):
        if not Path(path).exists():
            raise FileNotFoundError(f"{path} not found -- run `python embed.py` first to build the index.")
    embeddings = np.load(embeddings_path)
    metadata = pd.read_csv(metadata_path)
    assert len(embeddings) == len(metadata), "embeddings/metadata row count mismatch"
    return embeddings, metadata


def load_model():
    return SentenceTransformer(MODEL_NAME)


def cosine_similarity(embeddings, query_embedding):
    return np.dot(embeddings, query_embedding) / (
        np.linalg.norm(embeddings, axis=1) * np.linalg.norm(query_embedding)
    )


def first_unique(metadata, ordered_idx, top_k):
    """
    Walk `ordered_idx` (best first) and keep the first `top_k` rows whose
    (track_name, track_artist) hasn't been seen yet. The catalog lists many
    songs more than once (single + album release, etc.), which would otherwise
    show up as back-to-back duplicate results.

    Returns positions into `ordered_idx`.
    """
    names = metadata["track_name"].astype(str).str.lower().str.strip()
    artists = metadata["track_artist"].astype(str).str.lower().str.strip()
    seen, keep = set(), []
    for pos, i in enumerate(ordered_idx):
        key = (names.iat[i], artists.iat[i])
        if key in seen:
            continue
        seen.add(key)
        keep.append(pos)
        if len(keep) == top_k:
            break
    return np.array(keep, dtype=int)


def hybrid_rank(metadata, similarities, top_k=10, alpha=0.8, candidate_pool=100):
    """
    Rerank the top `candidate_pool` cosine-similarity matches by a weighted
    blend of semantic similarity and track popularity.

    alpha=1.0 -> pure semantic search. alpha=0.0 -> pure popularity ranking.
    Popularity only reorders among songs that already passed the semantic
    cut (the candidate pool) -- it can never pull in a song the query embedding
    doesn't think is relevant at all. The pool grows to 2 * top_k when more
    results are requested than the pool would hold.

    Returns (ranked_indices, hybrid_scores, similarity_scores_for_those_indices).
    """
    candidate_pool = min(max(candidate_pool, 2 * top_k), len(similarities))
    candidate_idx = np.argsort(similarities)[::-1][:candidate_pool]

    sim_scores = similarities[candidate_idx]
    sim_min, sim_max = sim_scores.min(), sim_scores.max()
    norm_sim = (sim_scores - sim_min) / (sim_max - sim_min + 1e-9)

    pop_scores = metadata.iloc[candidate_idx]["track_popularity"].to_numpy()
    norm_pop = pop_scores / 100.0  # track_popularity is already bounded [0, 100]

    hybrid_scores = alpha * norm_sim + (1 - alpha) * norm_pop

    order = np.argsort(hybrid_scores)[::-1]
    order = order[first_unique(metadata, candidate_idx[order], top_k)]
    ranked_idx = candidate_idx[order]
    return ranked_idx, hybrid_scores[order], sim_scores[order]


def semantic_rank(metadata, similarities, top_k=10):
    """Rank by cosine similarity only. Returns (ranked_indices, similarity_scores)."""
    ordered = np.argsort(similarities)[::-1]
    idx = ordered[first_unique(metadata, ordered, top_k)]
    return idx, similarities[idx]
