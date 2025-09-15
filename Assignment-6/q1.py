import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA

iris = load_iris()
X = iris.data

kmeans = KMeans(n_clusters=3, random_state=42)
clusters = kmeans.fit_predict(X)
centroids = kmeans.cluster_centers_

pca = PCA(n_components=2)
X_pca = pca.fit_transform(X)
centroids_pca = pca.transform(centroids)

plt.figure(figsize=(8, 6))
colors = ['red', 'blue', 'green']

for i in range(len(colors)):
    plt.scatter(X_pca[clusters == i, 0], X_pca[clusters == i, 1], 
                s=50, c=colors[i], label=f'Cluster {i}')

plt.scatter(centroids_pca[:, 0], centroids_pca[:, 1], 
            marker='X', s=200, c='yellow', edgecolor='black', label='Centroids')

plt.xlabel("PCA Component 1")
plt.ylabel("PCA Component 2")
plt.title("K-Means Clustering on Iris Dataset (k=3)")
plt.legend()
plt.grid(True)
plt.savefig("iris_kmeans_clusters.png")
print("Cluster visualization saved as iris_kmeans_clusters.png")
