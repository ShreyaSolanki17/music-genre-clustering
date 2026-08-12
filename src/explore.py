"""Step 1 — data acquisition & exploration.

Loads the Kaggle Spotify Tracks Dataset (114k tracks, downloaded fully
upfront via `kaggle datasets download`, no streaming) and reports what
matters before any preprocessing: missing values, duplicates, feature
distributions, correlations, and genre-label coverage. Genre is explored
here only to know what we'll evaluate against later — it is never used as
a clustering input.
"""
import pandas as pd

AUDIO_FEATURES = [
    "danceability", "energy", "loudness", "speechiness", "acousticness",
    "instrumentalness", "liveness", "valence", "tempo",
]

RAW_PATH = "data/raw/dataset.csv"


def load_raw() -> pd.DataFrame:
    df = pd.read_csv(RAW_PATH, index_col=0)
    return df


def main():
    df = load_raw()
    print(f"shape: {df.shape}")

    print("\n--- missing values ---")
    na = df.isna().sum()
    print(na[na > 0] if na.any() else "none")

    print("\n--- duplicate track_id ---")
    print(df["track_id"].duplicated().sum(), "duplicate track_ids out of", len(df))

    print("\n--- genre labels ---")
    print(df["track_genre"].nunique(), "unique genres")
    print("tracks per genre: min", df["track_genre"].value_counts().min(),
          "max", df["track_genre"].value_counts().max())

    print("\n--- audio feature summary ---")
    print(df[AUDIO_FEATURES].describe().T)

    print("\n--- correlation matrix (audio features) ---")
    print(df[AUDIO_FEATURES].corr().round(2))

    print("\n--- rows with all audio features == 0 (possible bad rows) ---")
    print((df[AUDIO_FEATURES] == 0).all(axis=1).sum())


if __name__ == "__main__":
    main()
