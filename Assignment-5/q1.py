import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

points = np.array([
    [1, 2],  # A
    [3, 9],  # B
    [5, 2],  # C
    [4, 1],  # D
    [3, 6],  # E
    [4, 7]   # F
])

labels = ['A', 'B', 'C', 'D', 'E', 'F']

kmeans = KMeans(n_clusters=2, random_state=0)
kmeans.fit(points)

clusters = kmeans.labels_
centroids = kmeans.cluster_centers_

print("Cluster assignments:")
for label, cluster in zip(labels, clusters):
    print(f"{label} -> Cluster {cluster}")

print("\nCluster centroids:")
print(centroids)

for i in range(len(points)):
    plt.scatter(points[i][0], points[i][1],
                c=('red' if clusters[i] == 0 else 'blue'),
                s=80)
    plt.text(points[i][0] + 0.1, points[i][1] + 0.1, labels[i], fontsize=9)

plt.scatter(centroids[:, 0], centroids[:, 1],
            c='green', marker='X', s=200, label='Centroids')

plt.xlabel('X Coordinate')
plt.ylabel('Y Coordinate')
plt.title('K-Means Clustering (k=2)')
plt.grid(True)
plt.legend()

plt.savefig("kmeans_plot.png")
print("Plot saved as kmeans_plot.png")

