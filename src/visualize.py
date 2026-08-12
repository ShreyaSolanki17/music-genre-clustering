"""Step 5 — dimensionality reduction & visualization.

PCA first: cheap, and its explained-variance ratio tells us honestly how
much a 2D picture can be trusted before we look at it. t-SNE second: much
better at revealing local neighborhood structure than PCA's linear
projection, standard choice for visualizing this kind of data -- but run
on a subsample (t-SNE is O(n^2)-ish and 89,741 points would take a very
long time) with a fixed seed for reproducibility.
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

IN_PATH = "data/processed/tracks_clustered.csv"
FEATURES = [
    "danceability", "energy", "loudness", "speechiness", "acousticness",
    "instrumentalness", "liveness", "valence", "tempo",
]
TSNE_SAMPLE = 8000
RANDOM_STATE = 42


def scatter(ax, xy, labels, title):
    sc = ax.scatter(xy[:, 0], xy[:, 1], c=labels, cmap="tab10", s=4, alpha=0.5)
    ax.set_title(title)
    ax.set_xticks([])
    ax.set_yticks([])
    return sc


def main():
    df = pd.read_csv(IN_PATH)
    X = df[FEATURES].values

    pca = PCA(n_components=2, random_state=RANDOM_STATE)
    pca_xy = pca.fit_transform(X)
    print(f"PCA explained variance ratio: {pca.explained_variance_ratio_} "
          f"(total {pca.explained_variance_ratio_.sum():.1%})")

    fig, ax = plt.subplots(figsize=(6, 5))
    sc = scatter(ax, pca_xy, df["kmeans_cluster"], "PCA (2D) — colored by K-means cluster")
    fig.colorbar(sc, ax=ax, label="cluster")
    fig.tight_layout()
    fig.savefig("results/pca_kmeans.png", dpi=150)
    print("wrote results/pca_kmeans.png")

    sample = df.sample(n=TSNE_SAMPLE, random_state=RANDOM_STATE)
    X_sample = sample[FEATURES].values
    tsne_xy = TSNE(n_components=2, random_state=RANDOM_STATE, init="pca", perplexity=30).fit_transform(X_sample)

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    sc0 = scatter(axes[0], tsne_xy, sample["kmeans_cluster"], "t-SNE — colored by K-means cluster")
    fig.colorbar(sc0, ax=axes[0], label="cluster")
    dbscan_labels = sample["dbscan_cluster"]
    sc1 = scatter(axes[1], tsne_xy, dbscan_labels, "t-SNE — colored by DBSCAN cluster (-1 = noise)")
    fig.colorbar(sc1, ax=axes[1], label="cluster")
    fig.tight_layout()
    fig.savefig("results/tsne_clusters.png", dpi=150)
    print(f"wrote results/tsne_clusters.png (n={TSNE_SAMPLE} sample)")


if __name__ == "__main__":
    main()
