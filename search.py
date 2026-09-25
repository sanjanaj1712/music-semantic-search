"""
CLI for the semantic music search system.

Usage:
    python search.py "calm and dreamy for studying late at night"
    python search.py "aggressive rap with heavy bass" --top-k 5 --alpha 0.6
    python search.py "classic rock anthem" --no-hybrid
    python search.py "sad songs about missing someone" --mood-weight 0.5
"""
import argparse

from retrieval import MOOD_FEATURES, cosine_similarity, hybrid_rank, load_index, load_model, mood_targets, semantic_rank


def validate(query, top_k, alpha, mood_weight=0.0):
    if not query or not query.strip():
        raise ValueError("query must not be empty")
    if top_k < 1:
        raise ValueError("top_k must be at least 1")
    if not 0.0 <= alpha <= 1.0:
        raise ValueError("alpha must be between 0.0 and 1.0")
    if not 0.0 <= mood_weight <= 1.0:
        raise ValueError("mood_weight must be between 0.0 and 1.0")


def search(query, top_k=10, alpha=0.8, hybrid=True, mood_weight=0.3, index=None, model=None):
    """
    `index` (embeddings, metadata) and `model` can be passed in by callers that
    keep them loaded between queries (e.g. app.py); otherwise they're loaded here.

    hybrid=False ranks by cosine similarity only (no popularity or mood rerank).
    The mood detected in the query is returned in `results.attrs["mood"]`.
    """
    validate(query, top_k, alpha, mood_weight)
    embeddings, metadata = index if index is not None else load_index()
    model = model if model is not None else load_model()
    query_embedding = model.encode(query)
    similarities = cosine_similarity(embeddings, query_embedding)

    mood = mood_targets(query) if hybrid and mood_weight > 0 else {}
    if hybrid:
        idx, scores, sim_scores = hybrid_rank(
            metadata, similarities, top_k=top_k, alpha=alpha, mood=mood, mood_weight=mood_weight
        )
    else:
        idx, sim_scores = semantic_rank(metadata, similarities, top_k=top_k)
        scores = sim_scores

    results = metadata.iloc[idx][["track_id", "track_name", "track_artist", "track_popularity", *MOOD_FEATURES]].copy()
    results["similarity"] = sim_scores
    results["score"] = scores
    results.attrs["mood"] = mood
    return results


def main():
    parser = argparse.ArgumentParser(description="Semantic music search over the Spotify song set")
    parser.add_argument("query", type=str, help="natural language query, e.g. 'sad acoustic breakup song'")
    parser.add_argument("--top-k", type=int, default=10, help="number of results to return")
    parser.add_argument(
        "--alpha",
        type=float,
        default=0.8,
        help="weight on semantic similarity vs. popularity in hybrid ranking "
        "(1.0 = pure semantic, 0.0 = pure popularity). Default 0.8.",
    )
    parser.add_argument(
        "--no-hybrid",
        action="store_true",
        help="rank by cosine similarity only, skip popularity and mood reranking",
    )
    parser.add_argument(
        "--mood-weight",
        type=float,
        default=0.3,
        help="weight on matching mood keywords in the query (e.g. 'calm', 'sad', 'workout') "
        "to songs' audio features. 0 disables. Default 0.3.",
    )
    args = parser.parse_args()

    try:
        results = search(
            args.query, top_k=args.top_k, alpha=args.alpha, hybrid=not args.no_hybrid, mood_weight=args.mood_weight
        )
    except (ValueError, FileNotFoundError) as e:
        parser.error(str(e))

    mood = results.attrs["mood"]
    if args.no_hybrid:
        mode = "cosine similarity only"
    else:
        mode = f"hybrid, alpha={args.alpha}"
        if mood:
            mood_desc = ", ".join(f"{'high' if d > 0 else 'low'} {f}" for f, d in mood.items())
            mode += f"; mood: {mood_desc}, weight={args.mood_weight}"
    print(f'\nTop {len(results)} results for "{args.query}" ({mode})\n')
    for rank, (_, row) in enumerate(results.iterrows(), start=1):
        print(
            f"{rank:2d}. {row['track_name']} — {row['track_artist']} "
            f"(sim={row['similarity']:.3f}, pop={int(row['track_popularity'])}, score={row['score']:.3f})"
        )


if __name__ == "__main__":
    main()
