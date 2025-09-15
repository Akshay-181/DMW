import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA

iris = load_iris()
X = iris.data

wcss = []
K = range(1, 11)
for k in K:
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
    kmeans.fit(X)
    wcss.append(kmeans.inertia_)

# Plot Elbow curve
plt.figure(figsize=(8, 5))
plt.plot(K, wcss, 'bo-')
plt.xlabel("Number of Clusters (k)")
plt.ylabel("WCSS (Inertia)")
plt.title("Elbow Method for Optimal k")
plt.grid(True)
plt.savefig("elbow_method.png")
print("Elbow method graph saved as elbow_method.png")

kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
clusters = kmeans.fit_predict(X)
centroids = kmeans.cluster_centers_

pca = PCA(n_components=2)
X_pca = pca.fit_transform(X)
centroids_pca = pca.transform(centroids)

plt.figure(figsize=(8, 6))
colors = ['red', 'blue', 'green']

for i in range(3):
    plt.scatter(X_pca[clusters == i, 0], X_pca[clusters == i, 1], 
                s=50, c=colors[i], label=f'Cluster {i}')

plt.scatter(centroids_pca[:, 0], centroids_pca[:, 1],
            marker='X', s=200, c='yellow', edgecolor='black', label='Centroids')

plt.xlabel("PCA Component 1")
plt.ylabel("PCA Component 2")
plt.title("K-Means Clustering on Iris Dataset (k=3)")
plt.legend()
plt.grid(True)
plt.savefig("iris_kmeans_elbow_clusters.png")
print("Cluster visualization saved as iris_kmeans_elbow_clusters.png")
