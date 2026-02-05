import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs

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

sse = []
K_range = range(1, 11)

for k in K_range:
    model = MyKMeans(k=k)
    model.fit(X)
    sse.append(model.wcss_)

print("\nElbow Method SSE Values:")
for k, val in zip(K_range, sse):
    print(f"K = {k} -> SSE = {val:.2f}")

plt.figure()
plt.plot(K_range, sse, marker='o')
plt.xlabel("Number of Clusters (K)")
plt.ylabel("SSE / WCSS")
plt.title("Elbow Method for Optimal K")
plt.show()

optimal_k = 3
final_model = MyKMeans(k=optimal_k)
final_model.fit(X)

plt.figure()
plt.scatter(X[:, 0], X[:, 1], c=final_model.labels_)
plt.scatter(
    final_model.centroids[:, 0],
    final_model.centroids[:, 1],
    marker='X',
    s=200
)
plt.title("Final Clusters using Optimal K = 3")
plt.xlabel("Feature 1")
plt.ylabel("Feature 2")
plt.show()

print("\nFinal WCSS for K = 3:", final_model.wcss_)
