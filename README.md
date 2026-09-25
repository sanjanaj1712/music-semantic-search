# 🎧 Semantic Music Search

Search ~18.5k Spotify songs by describing what you want to hear — *"songs for a
late-night drive"*, *"sad acoustic breakup song"*, *"party reggaeton"* — instead
of typing exact titles or artists.

Each song's title, artist, genre and lyrics are encoded into a sentence
embedding; a query is embedded the same way and matched by cosine similarity,
with an optional re-rank that nudges well-known songs up among equally good
matches. It ships with a Streamlit UI, a CLI, and an offline evaluation script.

![Streamlit UI](docs/screenshot.png)

## Features

- **Natural-language search** — describe a mood, activity, theme or genre in plain English.
- **Semantic retrieval** — query and songs are embedded with `all-MiniLM-L6-v2`; results are ranked by cosine similarity (top 10 by default).
- **Hybrid ranking** — optionally blends semantic similarity with Spotify popularity (`alpha` controls the mix).
- **Duplicate collapsing** — the same song released on several albums appears only once in results.
- **Streamlit UI** — search box, example queries, adjustable result count / ranking, result cards with album, year, genre tags and a link to the track on Spotify. Searches are shareable via `?q=...` in the URL.
- **CLI** — `python search.py "your query"` with `--top-k`, `--alpha`, `--no-hybrid`.
- **Offline evaluation** — Precision@10 / Recall@10 / NDCG@10 over a 10-query weak-labeled set, comparing semantic-only vs. hybrid ranking.

## Tech stack

Python 3.10+ · [Sentence Transformers](https://www.sbert.net/) (`all-MiniLM-L6-v2`) · NumPy · Pandas · Streamlit · pytest

## Architecture

```mermaid
flowchart TD
    A[data/spotify_songs.csv<br/>18,454 songs] --> B[embed.py<br/>song_text = title + artist + genre + lyrics<br/>encode with all-MiniLM-L6-v2]
    B --> C[(song_embeddings.npy<br/>18454 x 384)]
    B --> D[(song_metadata.csv<br/>track_id, name, artist, popularity)]
    C --> E[retrieval.py<br/>cosine similarity · hybrid rerank · dedupe]
    D --> E
    E --> F[search.py<br/>CLI + search function]
    F --> G[app.py<br/>Streamlit UI]
    E --> H[eval.py<br/>Precision / Recall / NDCG @10]
```

- **`embed.py`** is a one-time offline step. Row *i* of `song_embeddings.npy` and `song_metadata.csv` refer to the same song.
- **`retrieval.py`** is the shared ranking core, so the CLI, the UI and the evaluation all rank with exactly the same code.
- Search is a **brute-force** dot product over all 18k vectors (milliseconds at this size). There is no approximate-nearest-neighbour index — see [Future improvements](#future-improvements).

### How ranking works

1. Embed the query and compute cosine similarity against every song.
2. **Semantic-only mode:** sort by similarity.
3. **Hybrid mode (default):** take the top 100 songs by similarity (or `2 × top_k` if more results are requested), then re-rank them by
   `score = alpha × norm_similarity + (1 − alpha) × popularity / 100`,
   where similarity is min-max normalized within that candidate pool. Popularity can only reorder songs that already matched semantically — it never pulls in an unrelated song. `alpha = 0.8` by default (1.0 = pure semantic, 0.0 = pure popularity); it was chosen by hand, not tuned.
4. Collapse duplicates (same title + artist, case-insensitive) and return the top *k*.

Cosine similarity is used rather than a raw dot product because the model encodes meaning in the *direction* of a vector; normalizing removes any effect of text length on magnitude.

## Prerequisites

- Python 3.10 or newer (developed and tested on 3.13)
- ~2 GB free disk space for dependencies (PyTorch) and ~110 MB for the data + index
- The dataset: **[Audio features and lyrics of Spotify songs](https://www.kaggle.com/datasets/imuhammad/audio-features-and-lyrics-of-spotify-songs)** (Kaggle, free account required). It is not included in this repo because it contains copyrighted lyrics.

## Installation

```bash
# 1. Clone
git clone https://github.com/sanjanaj1712/music-semantic-search.git
cd music-semantic-search

# 2. Create a virtual environment and install dependencies
python3 -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt

# 3. Add the dataset
#    Download the Kaggle dataset above and place spotify_songs.csv at:
#    data/spotify_songs.csv
mkdir -p data
mv ~/Downloads/spotify_songs.csv data/   # adjust to wherever you saved it

# 4. Build the search index (one-time, ~2 minutes on a laptop CPU)
python embed.py
```

Step 4 downloads the embedding model (~90 MB) on first run and writes
`song_embeddings.npy` and `song_metadata.csv` to the project root.

## Environment variables

No environment variables are required. See [`.env.example`](.env.example):

| Variable   | Required | Purpose |
|------------|----------|---------|
| `HF_TOKEN` | No       | Hugging Face token. Only raises rate limits for the one-time model download and silences the "unauthenticated requests" warning. Set it in your shell (`export HF_TOKEN=...`); the code does not read `.env` files. |

## Running

**Web UI**

```bash
streamlit run app.py
```

Then open http://localhost:8501. Type a query or click an example; open
**Settings** to change the number of results, toggle hybrid ranking, or adjust
`alpha`.

**CLI**

```bash
python search.py "songs for a late-night drive"
python search.py "aggressive rap with heavy bass" --top-k 5 --alpha 0.6
python search.py "classic rock anthem" --no-hybrid
```

**Evaluation**

```bash
python eval.py
```

**Tests**

```bash
pytest
```

The suite has 23 tests: unit tests for ranking, deduplication and input
validation (synthetic data, no model needed), CLI error handling, and
end-to-end tests that run a real search and render the Streamlit app
headlessly. The end-to-end tests are skipped automatically if you haven't run
`embed.py` yet.

## Usage example

```
$ python search.py "songs for a late-night drive"

Top 10 results for "songs for a late-night drive" (hybrid, alpha=0.8)

 1. Midnight City — M83 (sim=0.540, pop=74, score=0.859)
 2. Drive (feat. Ezra Carey) — Myzica (sim=0.552, pop=23, score=0.846)
 3. drive all night — joan (sim=0.542, pop=50, score=0.825)
 4. Awake — TWO LANES (sim=0.536, pop=42, score=0.766)
 5. Night Drive — Gorgon City (sim=0.527, pop=43, score=0.701)
 ...
```

- `sim` — cosine similarity between query and song
- `pop` — Spotify popularity (0–100)
- `score` — the hybrid ranking score (equals `sim` with `--no-hybrid`)

Invalid input is rejected with a clear message, e.g.
`search.py: error: alpha must be between 0.0 and 1.0`.

## Evaluation

There are no human relevance judgments for this dataset, so `eval.py` uses a
**weak-labeled** set of 10 queries. Each query is paired with a rule over the
catalog's genre and audio-feature columns — e.g. *"aggressive rap with heavy
bass"* counts a song as relevant if `playlist_genre == "rap" AND energy > 0.6
AND speechiness > 0.15` (full rules in `eval.py::relevance_labels`). The ranker
never sees those columns, so it can't match the labels directly.

Mean over the 10 queries, K = 10:

| Ranking            | Precision@10 | Recall@10 | NDCG@10 |
|--------------------|:------------:|:---------:|:-------:|
| Semantic only      | 0.190        | 0.0030    | 0.430   |
| Hybrid (alpha=0.8) | 0.180        | 0.0028    | 0.440   |

How to read this:

- Treat these numbers as **directional**. The labels are hand-written proxies, and 10 queries is too few to separate the two rankers with confidence.
- Recall@10 is tiny by construction: many queries have 300–1,400 "relevant" songs.
- Three queries score 0 under both rankings: *classic rock anthem*, *calm and dreamy for studying late at night*, and *heartbreak sad pop or r&b song*. These are real failure cases. The model only sees text (title, artist, genre, lyrics), so mood or energy queries with no matching words have little to latch onto.

## Project structure

```
music-semantic-search/
├── app.py               # Streamlit UI
├── search.py            # CLI + search() used by the UI
├── retrieval.py         # shared core: index loading, similarity, ranking, dedupe
├── embed.py             # builds the embedding index from the dataset
├── eval.py              # offline evaluation (Precision/Recall/NDCG@10)
├── explore.py           # exploratory data-analysis script (not part of the pipeline)
├── tests/
│   └── test_retrieval.py
├── .streamlit/config.toml
├── docs/screenshot.png
├── data/                # (not tracked) spotify_songs.csv goes here
├── requirements.txt
├── pytest.ini
├── .env.example
└── LICENSE
```

## Future improvements

- **Use audio features in ranking.** The dataset already has `energy`, `valence`, `tempo` and `danceability`. Matching a query's implied mood against them is the direct fix for the mood-query failures above.
- **Tune `alpha`** with a sweep over the evaluation set, ideally using queries and labels written by people who didn't build the ranker.
- **Approximate nearest-neighbour index** (FAISS / HNSW) for catalogs in the millions. Brute force is fine at 18k songs.
- **Stronger or music-aware embedding model.** `all-MiniLM-L6-v2` is small, fast and general-purpose, but it knows nothing about music.
- **Personalization** from user feedback or listening history.

## License

Code is released under the [MIT License](LICENSE). The dataset is not included
and is subject to its own terms on Kaggle.
