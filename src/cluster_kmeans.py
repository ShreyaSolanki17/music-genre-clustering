"""Step 3 — determine cluster count & run K-means.

Elbow method (inertia) + silhouette score across a range of K, both
computed properly instead of eyeballing an arbitrary K. Silhouette score
is O(n^2) so it's estimated on a fixed random subsample of the 89,741
tracks (silhouette_score's `sample_size` arg) rather than the full set --
full pairwise distances at this n would be ~8B entries.
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

IN_PATH = "data/processed/tracks_scaled.csv"
OUT_PATH = "data/processed/tracks_clustered.csv"
FEATURES = [
    "danceability", "energy", "loudness", "speechiness", "acousticness",
    "instrumentalness", "liveness", "valence", "tempo",
]
K_RANGE = range(2, 16)
SILHOUETTE_SAMPLE = 10000
RANDOM_STATE = 42

# k=8: best silhouette among non-trivial K (0.191) and where the elbow
# curve visibly flattens -- see results/kmeans_k_selection.png. k=2 scores
# higher (0.257) but is a trivial one-axis split, not genre-like structure.
CHOSEN_K = 8


def select_k(X):
    inertias, silhouettes = [], []
    for k in K_RANGE:
        km = KMeans(n_clusters=k, n_init=10, random_state=RANDOM_STATE).fit(X)
        inertias.append(km.inertia_)
        sil = silhouette_score(X, km.labels_, sample_size=SILHOUETTE_SAMPLE, random_state=RANDOM_STATE)
        silhouettes.append(sil)
        print(f"k={k:2d}  inertia={km.inertia_:10.0f}  silhouette={sil:.4f}")

    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    axes[0].plot(list(K_RANGE), inertias, marker="o")
    axes[0].set_xlabel("k")
    axes[0].set_ylabel("inertia")
    axes[0].set_title("Elbow method")

    axes[1].plot(list(K_RANGE), silhouettes, marker="o", color="darkorange")
    axes[1].set_xlabel("k")
    axes[1].set_ylabel("silhouette score")
    axes[1].set_title("Silhouette score")

    fig.tight_layout()
    fig.savefig("results/kmeans_k_selection.png", dpi=150)
    print("\nwrote results/kmeans_k_selection.png")


def main():
    df = pd.read_csv(IN_PATH)
    X = df[FEATURES].values

    select_k(X)

    km = KMeans(n_clusters=CHOSEN_K, n_init=10, random_state=RANDOM_STATE).fit(X)
    df["kmeans_cluster"] = km.labels_
    print(f"\nfinal K-means (k={CHOSEN_K}) cluster sizes:")
    print(df["kmeans_cluster"].value_counts().sort_index())

    df.to_csv(OUT_PATH, index=False)
    print(f"\nwrote {OUT_PATH}")


if __name__ == "__main__":
    main()
