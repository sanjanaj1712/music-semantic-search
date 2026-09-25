import sys
from pathlib import Path

import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer

BASE_DIR = Path(__file__).resolve().parent
DATA_PATH = BASE_DIR / "data" / "spotify_songs.csv"

if not DATA_PATH.exists():
    sys.exit(f"{DATA_PATH} not found -- download the dataset first (see README).")

songs = pd.read_csv(DATA_PATH)
songs["lyrics"] = songs["lyrics"].fillna("")

songs["song_text"] = (
    songs["track_name"] + " by " + songs["track_artist"]
    + " " + songs["playlist_genre"] + " " + songs["lyrics"]
)

model = SentenceTransformer("all-MiniLM-L6-v2")

embeddings = model.encode(songs["song_text"].tolist(), show_progress_bar=True)

print(embeddings.shape)

np.save(BASE_DIR / "song_embeddings.npy", embeddings)
songs[["track_id", "track_name", "track_artist", "song_text", "track_popularity"]].to_csv(
    BASE_DIR / "song_metadata.csv", index=False
)