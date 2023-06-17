import tenseal as ts
import numpy as np

# Function to calculate the distance between two encrypted distributions
def calculate_distance(encrypted_mean1, encrypted_cov1, encrypted_mean2, encrypted_cov2):
    diff = encrypted_mean1 - encrypted_mean2
    inv_cov2 = encrypted_cov2.inverse()
    mahalanobis = diff.dot(inv_cov2).dot(diff.transpose())
    return mahalanobis.norm()

# Function to assign encrypted distributions to the nearest centroid
def assign_clusters(data, centroids):
    assigned_clusters = []
    for dist in data:
        distances = [calculate_distance(dist[0], dist[1], centroid[0], centroid[1]) for centroid in centroids]
        cluster = np.argmin(distances)
        assigned_clusters.append(cluster)
    return assigned_clusters

# Function to update the centroids based on assigned encrypted distributions
def update_centroids(data, assigned_clusters, k):
    centroids = []
    for i in range(k):
        assigned_dists = [data[j] for j, cluster in enumerate(assigned_clusters) if cluster == i]
        means = [dist[0] for dist in assigned_dists]
        covariances = [dist[1] for dist in assigned_dists]
        mean = ts.mean(means)
        covariance = ts.mean(covariances)
        centroids.append((mean, covariance))
    return centroids

# K-means clustering algorithm with encrypted distributions
def kmeans_encrypted(data, k, max_iterations):
    # Initialization
    centroids = data[:k]

    for _ in range(max_iterations):
        # Assign encrypted distributions to the nearest centroid
        assigned_clusters = assign_clusters(data, centroids)

        # Update centroids
        centroids = update_centroids(data, assigned_clusters, k)

    return centroids

# Example usage
# Assume the encrypted distributions are obtained from clients and stored in encrypted_means, encrypted_covariances lists

encrypted_means = [ts.ckks_tensor(mean) for mean in means]
encrypted_covariances = [ts.ckks_tensor(covariance) for covariance in covariances]
coefficients = [0.3, 0.5, 0.2]
k = 3
max_iterations = 10

data = list(zip(encrypted_means, encrypted_covariances, coefficients))

final_centroids = kmeans_encrypted(data, k, max_iterations)

# Decrypt the final centroids and print them
print("Final Centroids:")
for centroid in final_centroids:
    decrypted_mean = centroid[0].decrypt()
    decrypted_covariance = centroid[1].decrypt()
    print(decrypted_mean, decrypted_covariance)
