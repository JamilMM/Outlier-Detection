# K Means Outlier Detection On Make_Blobs DataSet

# Generate a single blob of 100 points
# Identify the five points that are furthest from the centroid
from sklearn.datasets import make_blobs
X, labels = make_blobs(100, centers=1)

# The k means should have a single center for most occassions
from sklearn.cluster import KMeans
kmeans = KMeans(n_clusters=1)
kmeans.fit(X)
KMeans(algorithm='auto', copy_x=True, init='k-means++', max_iter=300,
 n_clusters=1, n_init=10, n_jobs=1, precompute_distances='auto', random_state=None, tol=0.0001, verbose=0)

# Visualize the blobs with a scatter plot to see the centroid
import matplotlib.pyplot as plt
f, ax = plt.subplots(figsize=(8, 5))
ax.set_title("Blob")
ax.scatter(X[:, 0], X[:, 1], label='Points')
ax.scatter(kmeans.cluster_centers_[:, 0],kmeans.cluster_centers_[:, 1], label='Centroid',color='r')
ax.legend()
plt.show()


# Identify five closest points
distances = kmeans.transform(X)
# argsort returns an array of indexes which will sort the array in ascending order
 # so we reverse it via [::-1] and take the top five with [:5]
import numpy as np
sorted_idx = np.argsort(distances.ravel())[::-1][:5]

# Detect plots which are the farthest away

f, ax = plt.subplots(figsize=(7, 5))
ax.set_title("Single Cluster")
ax.scatter(X[:, 0], X[:, 1], label='Points')
ax.scatter(kmeans.cluster_centers_[:, 0],kmeans.cluster_centers_[:, 1],label='Centroid', color='r')
ax.scatter(X[sorted_idx][:, 0], X[sorted_idx][:, 1],label='Extreme Value', edgecolors='g',facecolors='none', s=100)
ax.legend(loc='best')
plt.show()

# Remove detected points if necessary
new_X = np.delete(X, sorted_idx, axis=0)
new_kmeans = KMeans(n_clusters=1)
new_kmeans.fit(new_X)

f, ax = plt.subplots(figsize=(7, 5))
ax.set_title("Extreme Values Removed")
ax.scatter(new_X[:, 0], new_X[:, 1], label='Pruned Points')
ax.scatter(kmeans.cluster_centers_[:, 0],kmeans.cluster_centers_[:, 1], label='Old Centroid',color='r',s=80, alpha=.5)
ax.scatter(new_kmeans.cluster_centers_[:, 0],new_kmeans.cluster_centers_[:, 1], label='New Centroid',color='m', s=80, alpha=.5)
ax.legend(loc='best')
plt.show()






