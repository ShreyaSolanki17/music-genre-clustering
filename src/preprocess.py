"""Step 2 — preprocessing & feature scaling.

- Dedup: 21% of rows are the same track re-listed under multiple genres
  (found in Step 1). One row per track_id for clustering; genre tags kept
  as a list per track for honest Step 6 evaluation instead of picking one
  arbitrarily.
- Feature selection: the 9 audio-character features. Dropped `popularity`
  (listenership, not sound), `duration_ms`/`time_signature`/`explicit`
  (structural/metadata, not timbral), `key`/`mode` (categorical/circular,
  not a natural fit for Euclidean distance), and `track_genre` (eval-only).
- log1p on speechiness/instrumentalness/liveness: all bounded [0,1],
  zero-inflated (e.g. 70% of tracks have instrumentalness < 0.01) with a
  long tail (skew > 1.7) — log1p compresses the tail so a handful of
  near-1 outliers don't dominate Euclidean distance after scaling.
  loudness is left alone: its skew is a genuine spread of quiet tracks on
  an already-log dB scale, not zero-inflation.
- StandardScaler fit on the (log-transformed) features.
"""
import os

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

RAW_PATH = "data/raw/dataset.csv"
OUT_PATH = "data/processed/tracks_scaled.csv"

FEATURES = [
    "danceability", "energy", "loudness", "speechiness", "acousticness",
    "instrumentalness", "liveness", "valence", "tempo",
]
LOG_FEATURES = ["speechiness", "instrumentalness", "liveness"]


def load_deduped() -> pd.DataFrame:
    df = pd.read_csv(RAW_PATH, index_col=0)
    genres = df.groupby("track_id")["track_genre"].apply(lambda s: sorted(set(s)))
    df = df.drop_duplicates(subset="track_id", keep="first").set_index("track_id")
    df["track_genres"] = genres
    df["n_genres"] = df["track_genres"].str.len()
    return df.reset_index()


def scale_features(df: pd.DataFrame) -> pd.DataFrame:
    X = df[FEATURES].copy()
    for col in LOG_FEATURES:
        X[col] = np.log1p(X[col])

    scaled = StandardScaler().fit_transform(X)
    return pd.DataFrame(scaled, columns=FEATURES, index=df.index)


def main():
    df = load_deduped()
    print(f"deduped: {len(df)} unique tracks (from 114000 raw rows)")
    print(f"tracks tagged under >1 genre: {(df['n_genres'] > 1).sum()}")

    scaled = scale_features(df)
    print("\n--- scaled feature summary ---")
    print(scaled.describe().T[["mean", "std", "min", "max"]])

    out = pd.concat(
        [df[["track_id", "track_name", "artists", "track_genres", "n_genres"]], scaled],
        axis=1,
    )
    os.makedirs("data/processed", exist_ok=True)
    out.to_csv(OUT_PATH, index=False)
    print(f"\nwrote {OUT_PATH} ({out.shape[0]} rows, {len(FEATURES)} scaled features)")


if __name__ == "__main__":
    main()
