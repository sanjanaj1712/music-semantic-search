"""
Evaluation harness for the semantic search system.

There are no human relevance judgments for this dataset, so ground truth is
built from proxy labels: a song counts as "relevant" to a query if it matches
a genre/subgenre/audio-feature rule that captures what the query is asking
for (e.g. "aggressive rap with heavy bass" -> playlist_genre == "rap" AND
energy/speechiness above a threshold). This is a weak-label eval, not a
substitute for real human judgments -- see DEFENSE.md for the honest version
of that caveat.

Ground truth is computed from the raw catalog (data/spotify_songs.csv), not
from song_metadata.csv -- the ranker only ever sees song_text and
track_popularity, so it can't "cheat" using the same genre/audio-feature
columns the eval labels are built from.

Usage:
    python eval.py
"""
import numpy as np
import pandas as pd

from retrieval import BASE_DIR, cosine_similarity, hybrid_rank, load_index, load_model, semantic_rank

RAW_DATA_PATH = BASE_DIR / "data" / "spotify_songs.csv"
K = 10


def relevance_labels(df):
    """query -> boolean mask over `df` marking rows considered relevant."""
    return {
        "upbeat energetic edm": (df.playlist_genre == "edm") & (df.energy > 0.8) & (df.danceability > 0.6),
        "sad acoustic ballad": (df.acousticness > 0.7) & (df.valence < 0.35),
        "aggressive rap with heavy bass": (df.playlist_genre == "rap") & (df.energy > 0.6) & (df.speechiness > 0.15),
        "romantic latin pop": (df.playlist_genre == "latin") & (df.playlist_subgenre == "latin pop") & (df.valence > 0.5),
        "chill r&b for late night": (df.playlist_genre == "r&b") & (df.energy < 0.5) & (df.acousticness > 0.3),
        "classic rock anthem": (df.playlist_subgenre == "classic rock") & (df.energy > 0.6),
        "high energy workout music": (df.energy > 0.8) & (df.danceability > 0.7) & (df.tempo > 120),
        "calm and dreamy for studying late at night": (df.energy < 0.4) & (df.acousticness > 0.5) & (df.instrumentalness > 0.01),
        "party reggaeton": (df.playlist_subgenre == "reggaeton") & (df.danceability > 0.7),
        "heartbreak sad pop or r&b song": (df.valence < 0.25) & (df.playlist_genre.isin(["pop", "r&b"])),
    }


def precision_at_k(hits, k):
    return sum(hits[:k]) / k


def recall_at_k(hits, k, n_relevant_total):
    if n_relevant_total == 0:
        return 0.0
    return sum(hits[:k]) / n_relevant_total


def ndcg_at_k(hits, k):
    dcg = sum(rel / np.log2(i + 2) for i, rel in enumerate(hits[:k]))
    ideal = sorted(hits, reverse=True)
    idcg = sum(rel / np.log2(i + 2) for i, rel in enumerate(ideal[:k]))
    return dcg / idcg if idcg > 0 else 0.0


def evaluate(k=K, hybrid=True, alpha=0.8):
    raw = pd.read_csv(RAW_DATA_PATH)
    labels = relevance_labels(raw)

    embeddings, metadata = load_index()
    model = load_model()

    rows = []
    for query, relevant_mask in labels.items():
        relevant_track_ids = set(raw.loc[relevant_mask, "track_id"])
        n_relevant = len(relevant_track_ids)

        query_embedding = model.encode(query)
        similarities = cosine_similarity(embeddings, query_embedding)

        if hybrid:
            idx, _, _ = hybrid_rank(metadata, similarities, top_k=k, alpha=alpha)
        else:
            idx, _ = semantic_rank(metadata, similarities, top_k=k)

        ranked_track_ids = metadata.iloc[idx]["track_id"].tolist()
        hits = [1 if tid in relevant_track_ids else 0 for tid in ranked_track_ids]

        rows.append(
            {
                "query": query,
                "n_relevant": n_relevant,
                f"precision@{k}": precision_at_k(hits, k),
                f"recall@{k}": recall_at_k(hits, k, n_relevant),
                f"ndcg@{k}": ndcg_at_k(hits, k),
            }
        )

    return pd.DataFrame(rows)


def main():
    for hybrid in (False, True):
        label = "hybrid (semantic + popularity, alpha=0.8)" if hybrid else "semantic only (cosine similarity)"
        print(f"\n=== {label} ===")
        results = evaluate(hybrid=hybrid)
        print(results.to_string(index=False))
        print("\nmean:")
        print(results[[f"precision@{K}", f"recall@{K}", f"ndcg@{K}"]].mean().to_string())


if __name__ == "__main__":
    main()
