import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.cluster import AgglomerativeClustering
from sklearn.decomposition import PCA

iris = load_iris()
X = iris.data

clustering = AgglomerativeClustering(n_clusters=3)
clusters = clustering.fit_predict(X)

pca = PCA(n_components=2)
X_pca = pca.fit_transform(X)

plt.figure(figsize=(8, 6))
colors = ['red', 'blue', 'green']

for i in range(3):
    plt.scatter(X_pca[clusters == i, 0], X_pca[clusters == i, 1],
                s=50, c=colors[i], label=f'Cluster {i}')

plt.xlabel("PCA Component 1")
plt.ylabel("PCA Component 2")
plt.title("Agglomerative Clustering on Iris Dataset")
plt.legend()
plt.grid(True)
plt.savefig("iris_agglomerative_clusters.png")
print("Cluster visualization saved as iris_agglomerative_clusters.png")
