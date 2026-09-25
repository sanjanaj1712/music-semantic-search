import subprocess
import sys

import numpy as np
import pandas as pd
import pytest

from retrieval import (
    BASE_DIR,
    METADATA_PATH,
    MOOD_FEATURES,
    cosine_similarity,
    first_unique,
    hybrid_rank,
    load_index,
    mood_targets,
    semantic_rank,
)
from search import validate

HAS_INDEX = METADATA_PATH.exists() and (BASE_DIR / "song_embeddings.npy").exists()
needs_index = pytest.mark.skipif(not HAS_INDEX, reason="index not built -- run `python embed.py`")


@pytest.fixture
def catalog():
    n = 300
    rng = np.random.default_rng(0)
    metadata = pd.DataFrame(
        {
            "track_id": [f"id{i}" for i in range(n)],
            "track_name": [f"Song {i}" for i in range(n)],
            "track_artist": [f"Artist {i}" for i in range(n)],
            "track_popularity": rng.integers(0, 101, n),
            **{f: rng.random(n) for f in MOOD_FEATURES},
        }
    )
    similarities = rng.random(n)
    return metadata, similarities


# --- unit tests (synthetic data, no model) ---


def test_cosine_similarity_matches_definition():
    emb = np.array([[1.0, 0.0], [0.0, 2.0], [3.0, 3.0]])
    sims = cosine_similarity(emb, np.array([1.0, 0.0]))
    np.testing.assert_allclose(sims, [1.0, 0.0, 1 / np.sqrt(2)])


def test_semantic_rank_orders_by_similarity(catalog):
    metadata, sims = catalog
    idx, scores = semantic_rank(metadata, sims, top_k=10)
    assert len(idx) == 10
    assert list(idx) == list(np.argsort(sims)[::-1][:10])
    assert np.all(np.diff(scores) <= 0)


def test_hybrid_alpha_one_is_pure_semantic(catalog):
    metadata, sims = catalog
    idx, _, _ = hybrid_rank(metadata, sims, top_k=10, alpha=1.0)
    assert list(idx) == list(np.argsort(sims)[::-1][:10])


def test_hybrid_only_reorders_within_candidate_pool(catalog):
    metadata, sims = catalog
    pool = set(np.argsort(sims)[::-1][:100])
    idx, _, _ = hybrid_rank(metadata, sims, top_k=10, alpha=0.0)
    assert set(idx) <= pool
    pops = metadata.iloc[idx]["track_popularity"].to_numpy()
    assert np.all(np.diff(pops) <= 0)


def test_hybrid_returns_more_than_candidate_pool(catalog):
    metadata, sims = catalog
    idx, _, _ = hybrid_rank(metadata, sims, top_k=150)
    assert len(idx) == 150


def test_top_k_larger_than_catalog(catalog):
    metadata, sims = catalog
    assert len(hybrid_rank(metadata, sims, top_k=10_000)[0]) == len(metadata)
    assert len(semantic_rank(metadata, sims, top_k=10_000)[0]) == len(metadata)


def test_duplicate_songs_are_collapsed(catalog):
    metadata, sims = catalog
    best = np.argsort(sims)[::-1]
    # make the 2nd-best row a re-release of the best one (different case/whitespace)
    metadata.loc[best[1], "track_name"] = metadata.loc[best[0], "track_name"].upper() + " "
    metadata.loc[best[1], "track_artist"] = metadata.loc[best[0], "track_artist"]
    for idx in (semantic_rank(metadata, sims, top_k=10)[0], hybrid_rank(metadata, sims, top_k=10)[0]):
        assert len(idx) == 10
        assert (best[0] in idx) + (best[1] in idx) == 1


@pytest.mark.parametrize(
    "query,expected",
    [
        ("sad acoustic ballad", {"energy": -1, "valence": -1, "acousticness": 1}),
        ("happy energetic songs for a workout", {"energy": 1, "valence": 1, "danceability": 1}),
        ("Relaxing music to STUDY to", {"energy": -1}),
        ("songs for a late-night drive", {}),
        ("taylor swift", {}),
        ("calm but energetic", {}),  # conflicting keywords cancel out
    ],
)
def test_mood_targets(query, expected):
    assert mood_targets(query) == expected


def test_mood_targets_match_word_starts_only():
    # prefix match: "sadly" counts as "sad", but "ambassador" (sad mid-word) does not
    assert mood_targets("sadly") == {"valence": -1}
    assert mood_targets("ambassador") == {}


def test_mood_rerank_prefers_matching_audio_features(catalog):
    metadata, sims = catalog
    plain, _, _ = hybrid_rank(metadata, sims, top_k=10)
    calm, _, _ = hybrid_rank(metadata, sims, top_k=10, mood={"energy": -1}, mood_weight=0.5)
    assert metadata.iloc[calm]["energy"].mean() < metadata.iloc[plain]["energy"].mean()


def test_mood_weight_zero_is_a_no_op(catalog):
    metadata, sims = catalog
    plain, _, _ = hybrid_rank(metadata, sims, top_k=10)
    same, _, _ = hybrid_rank(metadata, sims, top_k=10, mood={"energy": -1}, mood_weight=0.0)
    assert list(plain) == list(same)


def test_mood_uses_larger_candidate_pool(catalog):
    metadata, sims = catalog
    pool = set(np.argsort(sims)[::-1][:200])
    idx, _, _ = hybrid_rank(metadata, sims, top_k=10, alpha=1.0, mood={"energy": -1}, mood_weight=1.0)
    assert set(idx) <= pool
    assert not set(idx) <= set(np.argsort(sims)[::-1][:100])


def test_load_index_rejects_stale_metadata(tmp_path):
    np.save(tmp_path / "e.npy", np.zeros((1, 3), dtype=np.float32))
    pd.DataFrame({"track_id": ["a"], "track_name": ["x"], "track_artist": ["y"], "track_popularity": [1]}).to_csv(
        tmp_path / "m.csv", index=False
    )
    with pytest.raises(ValueError, match="embed.py"):
        load_index(tmp_path / "e.npy", tmp_path / "m.csv")


def test_first_unique_empty():
    metadata = pd.DataFrame({"track_name": [], "track_artist": []})
    assert len(first_unique(metadata, np.array([], dtype=int), 10)) == 0


@pytest.mark.parametrize(
    "query,top_k,alpha",
    [("", 10, 0.8), ("   ", 10, 0.8), ("rock", 0, 0.8), ("rock", -3, 0.8), ("rock", 10, 1.5), ("rock", 10, -0.1)],
)
def test_validate_rejects_bad_input(query, top_k, alpha):
    with pytest.raises(ValueError):
        validate(query, top_k, alpha)


@pytest.mark.parametrize("mood_weight", [-0.1, 1.5])
def test_validate_rejects_bad_mood_weight(mood_weight):
    with pytest.raises(ValueError):
        validate("rock", 10, 0.8, mood_weight)


def test_validate_accepts_good_input():
    validate("rock", 1, 0.0)
    validate("rock", 50, 1.0)


def test_load_index_missing_files_has_helpful_error(tmp_path):
    with pytest.raises(FileNotFoundError, match="embed.py"):
        load_index(tmp_path / "nope.npy", tmp_path / "nope.csv")


# --- CLI error paths (fail before the model loads, so they're fast) ---


@pytest.mark.parametrize(
    "args", [["   "], ["rock", "--top-k", "0"], ["rock", "--alpha", "2"], ["rock", "--mood-weight", "-1"]]
)
def test_cli_rejects_bad_input(args):
    proc = subprocess.run([sys.executable, str(BASE_DIR / "search.py"), *args], capture_output=True, text=True)
    assert proc.returncode == 2
    assert "error:" in proc.stderr


# --- integration tests (real model + index) ---


@pytest.fixture(scope="module")
def loaded():
    from retrieval import load_model

    return load_index(), load_model()


@needs_index
@pytest.mark.parametrize("hybrid", [True, False])
def test_search_end_to_end(loaded, hybrid):
    from search import search

    index, model = loaded
    results = search("songs for a late-night drive", hybrid=hybrid, index=index, model=model)
    assert len(results) == 10
    assert list(results.columns) == [
        "track_id", "track_name", "track_artist", "track_popularity", *MOOD_FEATURES, "similarity", "score"
    ]
    assert not results.duplicated(["track_name", "track_artist"]).any()
    assert results["similarity"].between(-1, 1).all()


@needs_index
def test_search_mood_lowers_energy_for_calm_query(loaded):
    from search import search

    index, model = loaded
    q = "calm and dreamy for studying late at night"
    without = search(q, mood_weight=0.0, index=index, model=model)
    with_mood = search(q, mood_weight=0.5, index=index, model=model)
    assert with_mood.attrs["mood"] == {"energy": -1}
    assert with_mood["energy"].mean() < without["energy"].mean()


@needs_index
def test_cli_runs_from_another_directory(tmp_path):
    proc = subprocess.run(
        [sys.executable, str(BASE_DIR / "search.py"), "happy energetic songs for a workout", "--top-k", "3"],
        capture_output=True, text=True, cwd=tmp_path,
    )
    assert proc.returncode == 0, proc.stderr
    assert "Top 3 results" in proc.stdout


@needs_index
def test_streamlit_app_renders_results():
    from streamlit.testing.v1 import AppTest

    at = AppTest.from_file(str(BASE_DIR / "app.py"), default_timeout=120).run()
    assert not at.exception
    at.text_input(key="query").input("happy energetic songs for a workout").run()
    assert not at.exception
    cards = [m for m in at.markdown if 'class="card"' in m.value]
    assert len(cards) == 10
