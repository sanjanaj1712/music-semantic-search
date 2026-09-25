"""
Core retrieval functions shared by search.py (CLI), app.py (Streamlit UI) and
eval.py (evaluation).

Kept separate from search.py so eval.py can call the exact same ranking code
the CLI uses, instead of a re-implementation that could quietly drift.
"""
import re
from pathlib import Path

import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer

MODEL_NAME = "all-MiniLM-L6-v2"
BASE_DIR = Path(__file__).resolve().parent
EMBEDDINGS_PATH = BASE_DIR / "song_embeddings.npy"
METADATA_PATH = BASE_DIR / "song_metadata.csv"

# Spotify audio features (all on a 0-1 scale) that mood keywords can target.
MOOD_FEATURES = ["energy", "valence", "danceability", "acousticness"]

# Query words that imply a direction on an audio feature. The text embedding
# can't hear a song, so "calm" or "workout" only match songs whose *words* say
# so; these keywords let the ranker use the measured features instead.
# Matched on word prefixes, so "relax" also covers "relaxing".
MOOD_KEYWORDS = {
    "energy": {
        +1: ["energetic", "energy", "workout", "gym", "hype", "pump", "intense", "aggressive", "upbeat",
             "party", "banger", "running", "loud"],
        -1: ["calm", "chill", "relax", "mellow", "soft", "sleep", "study", "dreamy", "quiet", "peaceful",
             "ballad", "slow", "lullaby"],
    },
    "valence": {
        +1: ["happy", "cheerful", "joyful", "feel-good", "feel good", "sunny", "upbeat", "uplifting", "fun"],
        -1: ["sad", "heartbreak", "heartbroken", "breakup", "melancholy", "lonely", "cry", "depressing",
             "gloomy", "dark"],
    },
    "danceability": {
        +1: ["dance", "dancing", "danceable", "club", "groovy", "party", "workout"],
        -1: [],
    },
    "acousticness": {
        +1: ["acoustic", "unplugged"],
        -1: ["electronic", "edm", "synth"],
    },
}


def load_index(embeddings_path=EMBEDDINGS_PATH, metadata_path=METADATA_PATH):
    for path in (embeddings_path, metadata_path):
        if not Path(path).exists():
            raise FileNotFoundError(f"{path} not found -- run `python embed.py` first to build the index.")
    embeddings = np.load(embeddings_path)
    metadata = pd.read_csv(metadata_path)
    assert len(embeddings) == len(metadata), "embeddings/metadata row count mismatch"
    missing = [c for c in MOOD_FEATURES if c not in metadata.columns]
    if missing:
        raise ValueError(f"{metadata_path} is missing {missing} -- re-run `python embed.py` to rebuild the index.")
    return embeddings, metadata


def load_model():
    return SentenceTransformer(MODEL_NAME)


def cosine_similarity(embeddings, query_embedding):
    return np.dot(embeddings, query_embedding) / (
        np.linalg.norm(embeddings, axis=1) * np.linalg.norm(query_embedding)
    )


def mood_targets(query):
    """
    Map mood keywords in `query` to {feature: +1 or -1}, e.g.
    "sad acoustic ballad" -> {"energy": -1, "valence": -1, "acousticness": +1}.
    Features with no keywords, or with keywords that cancel out, are left out.
    """
    query = query.lower()
    targets = {}
    for feature, directions in MOOD_KEYWORDS.items():
        vote = sum(
            sign
            for sign, words in directions.items()
            for word in words
            if re.search(r"\b" + re.escape(word), query)
        )
        if vote:
            targets[feature] = 1 if vote > 0 else -1
    return targets


def mood_scores(metadata, idx, targets):
    """How well each song in `idx` fits the mood `targets`, in [0, 1] (mean over targeted features)."""
    values = metadata.iloc[idx][list(targets)].to_numpy(dtype=float)
    directions = np.array(list(targets.values()))
    return np.where(directions > 0, values, 1 - values).mean(axis=1)


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


def hybrid_rank(
    metadata, similarities, top_k=10, alpha=0.8, candidate_pool=100, mood=None, mood_weight=0.0, mood_pool=200
):
    """
    Rerank the top `candidate_pool` cosine-similarity matches by a weighted
    blend of semantic similarity and track popularity -- and, when `mood`
    targets are given (see mood_targets), how well each song's audio features
    fit that mood:

        score = (1 - mood_weight) * [alpha * sim + (1 - alpha) * popularity] + mood_weight * mood_fit

    alpha=1.0 -> pure semantic search. alpha=0.0 -> pure popularity ranking.
    Popularity only reorders among songs that already passed the semantic
    cut (the candidate pool) -- it can never pull in a song the query embedding
    doesn't think is relevant at all. The pool grows to 2 * top_k when more
    results are requested than the pool would hold, and to `mood_pool` when a
    mood is applied, so audio features have more semantically relevant songs
    to choose from.

    Returns (ranked_indices, hybrid_scores, similarity_scores_for_those_indices).
    """
    use_mood = bool(mood) and mood_weight > 0
    if use_mood:
        candidate_pool = max(candidate_pool, mood_pool)
    candidate_pool = min(max(candidate_pool, 2 * top_k), len(similarities))
    candidate_idx = np.argsort(similarities)[::-1][:candidate_pool]

    sim_scores = similarities[candidate_idx]
    sim_min, sim_max = sim_scores.min(), sim_scores.max()
    norm_sim = (sim_scores - sim_min) / (sim_max - sim_min + 1e-9)

    pop_scores = metadata.iloc[candidate_idx]["track_popularity"].to_numpy()
    norm_pop = pop_scores / 100.0  # track_popularity is already bounded [0, 100]

    hybrid_scores = alpha * norm_sim + (1 - alpha) * norm_pop
    if use_mood:
        hybrid_scores = (1 - mood_weight) * hybrid_scores + mood_weight * mood_scores(metadata, candidate_idx, mood)

    order = np.argsort(hybrid_scores)[::-1]
    order = order[first_unique(metadata, candidate_idx[order], top_k)]
    ranked_idx = candidate_idx[order]
    return ranked_idx, hybrid_scores[order], sim_scores[order]


def semantic_rank(metadata, similarities, top_k=10):
    """Rank by cosine similarity only. Returns (ranked_indices, similarity_scores)."""
    ordered = np.argsort(similarities)[::-1]
    idx = ordered[first_unique(metadata, ordered, top_k)]
    return idx, similarities[idx]
