"""Step 6 — evaluate clusters against genre (post-hoc only; genre was
never a clustering input).

Genre ground truth here is noisy: a track can be tagged under several
genres (Step 1/2), and many of the 114 genre labels are near-synonyms
(alt-rock/rock/alternative). So this isn't "grading" the clustering --
it's characterizing what each cluster musically represents, honestly
reporting when that's ambiguous.

Rejoins cluster labels onto the *original* (non-deduped) raw rows, so
each (track, genre) tagging counts once -- a track tagged under 5 genres
contributes 5 rows, which is the natural way to let multi-genre tracks
speak to every genre they're tagged under rather than picking one
arbitrarily.
"""
import pandas as pd
from sklearn.metrics import normalized_mutual_info_score

RAW_PATH = "data/raw/dataset.csv"
CLUSTERED_PATH = "data/processed/tracks_clustered.csv"
RAW_FEATURES = [
    "danceability", "energy", "loudness", "speechiness", "acousticness",
    "instrumentalness", "liveness", "valence", "tempo",
]
TOP_N_GENRES = 5


def main():
    raw = pd.read_csv(RAW_PATH, index_col=0)
    clustered = pd.read_csv(CLUSTERED_PATH)[["track_id", "kmeans_cluster", "dbscan_cluster"]]
    df = raw.merge(clustered, on="track_id", how="inner")
    print(f"joined: {len(df)} (track, genre) rows across {df['track_id'].nunique()} unique tracks")

    # NMI: how much does cluster assignment tell us about genre, and vice versa
    nmi_km = normalized_mutual_info_score(df["kmeans_cluster"], df["track_genre"])
    nmi_db = normalized_mutual_info_score(df["dbscan_cluster"], df["track_genre"])
    print(f"\nNormalized Mutual Info vs track_genre: K-means={nmi_km:.4f}  DBSCAN={nmi_db:.4f}")
    print("(0 = no relationship, 1 = clusters perfectly predict genre)")

    print("\n--- per K-means cluster: size, top genres, feature profile ---")
    rows = []
    global_means = df[RAW_FEATURES].mean()
    for c in sorted(df["kmeans_cluster"].unique()):
        sub = df[df["kmeans_cluster"] == c]
        n_tracks = sub["track_id"].nunique()
        top = sub["track_genre"].value_counts(normalize=True).head(TOP_N_GENRES)
        top_str = ", ".join(f"{g} ({p:.0%})" for g, p in top.items())
        profile = sub[RAW_FEATURES].mean()
        # features that most distinguish this cluster from the global mean, in std units
        diff = ((profile - global_means) / df[RAW_FEATURES].std()).sort_values(key=abs, ascending=False)
        distinguishing = ", ".join(f"{f}={'+' if v > 0 else ''}{v:.1f}sd" for f, v in diff.head(3).items())
        print(f"\ncluster {c}: {n_tracks} tracks, {len(sub)} genre-taggings")
        print(f"  top genres: {top_str}")
        print(f"  distinguishing features: {distinguishing}")
        rows.append({"cluster": c, "n_tracks": n_tracks, "top_genres": top_str,
                      "distinguishing_features": distinguishing})

    pd.DataFrame(rows).to_csv("results/cluster_genre_summary.csv", index=False)
    print("\nwrote results/cluster_genre_summary.csv")

    print("\n--- deep dive: cluster 7 (the small outlier cluster flagged in Steps 3-5) ---")
    sub = df[df["kmeans_cluster"] == 7]
    print(sub["track_genre"].value_counts().head(10))
    print("\nfeature means vs global:")
    print(pd.DataFrame({"cluster_7": sub[RAW_FEATURES].mean(), "global": global_means}))


if __name__ == "__main__":
    main()
