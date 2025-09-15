import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.decomposition import PCA

iris = load_iris()
X = iris.data

def initialize_centroids(X, k):
    indices = np.random.choice(X.shape[0], k, replace=False)
    return X[indices]

def assign_clusters(X, centroids):
    distances = np.linalg.norm(X[:, np.newaxis] - centroids, axis=2)
    return np.argmin(distances, axis=1)

def update_centroids(X, clusters, k):
    return np.array([X[clusters == i].mean(axis=0) for i in range(k)])

def kmeans(X, k, max_iters=100, tol=1e-4):
    centroids = initialize_centroids(X, k)
    for _ in range(max_iters):
        clusters = assign_clusters(X, centroids)
        new_centroids = update_centroids(X, clusters, k)
        if np.allclose(new_centroids, centroids, atol=tol):
            break
        centroids = new_centroids
    return clusters, centroids

k = 3
clusters, centroids = kmeans(X, k)

pca = PCA(n_components=2)
X_pca = pca.fit_transform(X)
centroids_pca = pca.transform(centroids)

plt.figure(figsize=(8, 6))
colors = ['red', 'blue', 'green']

for i in range(k):
    plt.scatter(X_pca[clusters == i, 0], X_pca[clusters == i, 1],
                s=50, c=colors[i], label=f'Cluster {i}')

plt.scatter(centroids_pca[:, 0], centroids_pca[:, 1],
            marker='X', s=200, c='yellow', edgecolor='black', label='Centroids')

plt.xlabel("PCA Component 1")
plt.ylabel("PCA Component 2")
plt.title("K-Means Clustering from Scratch (Iris Dataset)")
plt.legend()
plt.grid(True)
plt.savefig("iris_kmeans_scratch.png")
print("Cluster visualization saved as iris_kmeans_scratch.png")
