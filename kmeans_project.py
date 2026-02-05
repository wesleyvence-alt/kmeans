import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs
from sklearn.cluster import KMeans

np.random.seed(42)

X, _ = make_blobs(
    n_samples=300,
    centers=3,
    cluster_std=1.2,
    random_state=42
)

print("Dataset Shape:", X.shape)

class MyKMeans:

    def __init__(self, k=3, max_iters=100, tol=1e-4):
        self.k = k
        self.max_iters = max_iters
        self.tol = tol

    def fit(self, X):

        idx = np.random.choice(len(X), self.k, replace=False)
        centroids = X[idx]

        for _ in range(self.max_iters):

            distances = np.sqrt(((X[:, None] - centroids) ** 2).sum(axis=2))
            labels = np.argmin(distances, axis=1)

            new_centroids = np.array([
                X[labels == i].mean(axis=0) for i in range(self.k)
            ])

            if np.allclose(centroids, new_centroids, atol=self.tol):
                break

            centroids = new_centroids

        self.labels_ = labels
        self.centroids = centroids
        self.wcss_ = sum(
            ((X[labels == i] - centroids[i]) ** 2).sum()
            for i in range(self.k)
        )

k = 3

custom_model = MyKMeans(k)
custom_model.fit(X)

custom_labels = custom_model.labels_
custom_centroids = custom_model.centroids
custom_wcss = custom_model.wcss_

sk_model = KMeans(n_clusters=k, random_state=42, n_init=10)
sk_model.fit(X)

print("\nCustom WCSS :", custom_wcss)
print("Sklearn WCSS:", sk_model.inertia_)

plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.scatter(X[:, 0], X[:, 1], c=custom_labels)
plt.scatter(custom_centroids[:, 0], custom_centroids[:, 1], marker='X', s=200)
plt.title("Custom KMeans")

plt.subplot(1, 2, 2)
plt.scatter(X[:, 0], X[:, 1], c=sk_model.labels_)
plt.scatter(sk_model.cluster_centers_[:, 0], sk_model.cluster_centers_[:, 1], marker='X', s=200)
plt.title("Sklearn KMeans")

plt.tight_layout()
plt.show()
