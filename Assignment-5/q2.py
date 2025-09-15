import numpy as np
import matplotlib.pyplot as plt

points = {
    "A1": (2, 10),
    "A2": (2, 5),
    "A3": (8, 4),
    "B1": (5, 8),
    "B2": (7, 5),
    "B3": (6, 4),
    "C1": (1, 2),
    "C2": (4, 9)
}

labels = list(points.keys())
data = np.array(list(points.values()))

centers = np.array([points["A1"], points["B1"], points["C1"]], dtype=float)

def assign_clusters(data, centers):
    clusters = []
    for p in data:
        distances = np.linalg.norm(p - centers, axis=1)
        cluster = np.argmin(distances)
        clusters.append(cluster)
    return np.array(clusters)

def compute_new_centers(data, clusters, k):
    new_centers = []
    for i in range(k):
        cluster_points = data[clusters == i]
        if len(cluster_points) > 0:
            new_centers.append(np.mean(cluster_points, axis=0))
        else:
            new_centers.append([0,0])
    return np.array(new_centers)

clusters_first = assign_clusters(data, centers)
centers_first = compute_new_centers(data, clusters_first, 3)

print("Cluster centers after first round:")
print(centers_first)

centers_final = centers_first.copy()
while True:
    clusters = assign_clusters(data, centers_final)
    new_centers = compute_new_centers(data, clusters, 3)
    if np.allclose(new_centers, centers_final):
        break
    centers_final = new_centers

print("\nFinal cluster centers:")
print(centers_final)

print("\nFinal clusters:")
for lbl, cluster in zip(labels, clusters):
    print(f"{lbl} -> Cluster {cluster}")

colors = ['red', 'blue', 'green']

for i, point in enumerate(data):
    plt.scatter(point[0], point[1], color=colors[clusters[i]], s=100)
    plt.text(point[0]+0.2, point[1]+0.2, labels[i], fontsize=9)

for i, center in enumerate(centers_final):
    plt.scatter(center[0], center[1], color=colors[i], marker='X', s=200, edgecolor='black')

plt.xlabel('X Coordinate')
plt.ylabel('Y Coordinate')
plt.title('K-Means Clustering (Manual Implementation)')
plt.grid(True)
plt.savefig("kmeans_manual_plot.png")
print("\nPlot saved as kmeans_manual_plot.png")
