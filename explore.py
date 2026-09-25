import pandas as pd

songs = pd.read_csv("data/spotify_songs.csv")

print("First 5 rows:")

print(songs.head())

print("\nDataset shape:")

print(songs.shape)

print("\nMissing values:")

print(songs.isnull().sum())

print("\nUnique track IDs:")

print(songs["track_id"].nunique())

print("\nLanguages:")

print(songs["language"].value_counts())

print("\nFirst song:")

print(songs["track_name"].iloc[0])

print("\nFirst song with artist:")

print(songs["track_name"].iloc[0] + " by " + songs["track_artist"].iloc[0])

songs["lyrics"] = songs["lyrics"].fillna("")

songs["song_text"] = (
    songs["track_name"]
    + " by "
    + songs["track_artist"]
    + " "
    + songs["playlist_genre"]
    + " "
    + songs["lyrics"]
)

print("\nMissing lyrics after cleaning:")

print(songs["lyrics"].isnull().sum())

print("\nFirst song text:")

print(songs["song_text"].iloc[0])

print(songs.columns.tolist())