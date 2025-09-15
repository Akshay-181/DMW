import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

X = np.array([[1,2], [2,3], [3,4], [5,8], [6,9], [7,10]])

kmeans = KMeans(n_clusters=2, random_state=42, n_init=10)
clusters = kmeans.fit_predict(X)
centers = kmeans.cluster_centers_

print("Cluster Assignments:")
for i, point in enumerate(X):
    print(f"Point {point} -> Cluster {clusters[i]}")

print("\nCluster Centers:")
print(centers)

score = silhouette_score(X, clusters)
print(f"\nSilhouette Score: {score:.4f}")

plt.figure(figsize=(6, 5))
colors = ['red', 'blue']

for i in range(len(X)):
    plt.scatter(X[i][0], X[i][1], c=colors[clusters[i]], s=80)
    plt.text(X[i][0]+0.1, X[i][1]+0.1, f"P{i+1}", fontsize=9)

plt.scatter(centers[:,0], centers[:,1], marker='X', s=200, c='green', edgecolor='black', label='Centroids')

plt.xlabel("X1")
plt.ylabel("X2")
plt.title("K-Means Clustering (k=2)")
plt.legend()
plt.grid(True)
plt.savefig("kmeans_2_clusters.png")
print("\nCluster visualization saved as kmeans_2_clusters.png")
