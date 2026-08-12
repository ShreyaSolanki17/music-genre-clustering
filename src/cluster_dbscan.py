"""Step 4 — second clustering algorithm, compared against K-means.

DBSCAN instead of agglomerative: agglomerative's O(n^2) memory for full
pairwise distances doesn't fit at 89,741 rows without subsampling (or a
connectivity graph, which just becomes a fiddlier way to get the same
answer). DBSCAN runs on the full dataset via sklearn's tree-based
neighbor search, and -- more importantly -- it can leave points as noise
instead of being forced to assign every point to a cluster like K-means.
That's a genuinely different question to ask of this data given how low
the K-means silhouette scores were.

min_samples = 2 * n_features (standard rule of thumb, Sander et al.).
eps chosen from the k-distance plot (elbow in the sorted distance to the
min_samples-th nearest neighbor), the DBSCAN analogue of the elbow method.
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.cluster import DBSCAN
from sklearn.metrics import adjusted_rand_score
from sklearn.neighbors import NearestNeighbors

IN_PATH = "data/processed/tracks_clustered.csv"
OUT_PATH = "data/processed/tracks_clustered.csv"
FEATURES = [
    "danceability", "energy", "loudness", "speechiness", "acousticness",
    "instrumentalness", "liveness", "valence", "tempo",
]
MIN_SAMPLES = 2 * len(FEATURES)  # 18

# picked from the k-distance elbow plot, see results/dbscan_k_distance.png --
# curve has no sharp knee (consistent with the earlier silhouette finding of
# no crisp structure), but visibly bends around the 85-90th percentile (~1.0-1.2)
EPS = 1.0


def plot_k_distance(X):
    nn = NearestNeighbors(n_neighbors=MIN_SAMPLES).fit(X)
    dist, _ = nn.kneighbors(X)
    k_dist = np.sort(dist[:, -1])

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(k_dist)
    ax.set_xlabel("points, sorted")
    ax.set_ylabel(f"distance to {MIN_SAMPLES}th nearest neighbor")
    ax.set_title("k-distance plot (elbow = eps candidate)")
    fig.tight_layout()
    fig.savefig("results/dbscan_k_distance.png", dpi=150)
    print("wrote results/dbscan_k_distance.png")


def main():
    df = pd.read_csv(IN_PATH)
    X = df[FEATURES].values

    plot_k_distance(X)

    db = DBSCAN(eps=EPS, min_samples=MIN_SAMPLES).fit(X)
    labels = db.labels_
    n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
    n_noise = (labels == -1).sum()
    print(f"\nDBSCAN (eps={EPS}, min_samples={MIN_SAMPLES}): "
          f"{n_clusters} clusters, {n_noise} noise points ({n_noise / len(df):.1%})")

    df["dbscan_cluster"] = labels
    print("\ncluster sizes:")
    print(df["dbscan_cluster"].value_counts().sort_index())

    ari = adjusted_rand_score(df["kmeans_cluster"], df["dbscan_cluster"])
    print(f"\nAdjusted Rand Index (K-means vs DBSCAN): {ari:.4f}")

    print("\ncross-tab (kmeans_cluster rows x dbscan_cluster cols):")
    print(pd.crosstab(df["kmeans_cluster"], df["dbscan_cluster"]))

    df.to_csv(OUT_PATH, index=False)
    print(f"\nwrote {OUT_PATH}")


if __name__ == "__main__":
    main()
