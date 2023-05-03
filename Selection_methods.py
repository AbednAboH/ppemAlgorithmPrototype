import numpy as np

# define the EM algorithm
def EM_algorithm(data, num_clusters, num_iterations):
    # randomly initialize the means and standard deviations for each cluster
    means = np.random.uniform(low=min(data), high=max(data), size=num_clusters)
    std_devs = np.random.uniform(low=0, high=1, size=num_clusters)
    weights = np.ones(num_clusters) / num_clusters

    for i in range(num_iterations):
        # E-step: compute the posterior probability of each data point belonging to each cluster
        posteriors = np.zeros((len(data), num_clusters))
        for j in range(num_clusters):
            posteriors[:, j] = weights[j] * normal_distribution(data, means[j], std_devs[j])
        posteriors /= posteriors.sum(axis=1, keepdims=True)

        # M-step: update the means, standard deviations, and weights for each cluster
        means = np.sum(posteriors * data.reshape(-1, 1), axis=0) / np.sum(posteriors, axis=0)
        std_devs = np.sqrt(np.sum(posteriors * (data.reshape(-1, 1) - means) ** 2, axis=0) / np.sum(posteriors, axis=0))
        weights = np.sum(posteriors, axis=0) / len(data)

    return means, std_devs, weights

# define the normal distribution function
def normal_distribution(x, mean, std_dev):
    return 1 / (std_dev * np.sqrt(2 * np.pi)) * np.exp(-(x - mean) ** 2 / (2 * std_dev ** 2))

# generate some random data
data = np.concatenate((np.random.normal(loc=5, scale=1, size=100), np.random.normal(loc=10, scale=2, size=100)))

# run the EM algorithm
num_clusters = 2
num_iterations = 100
means, std_devs, weights = EM_algorithm(data, num_clusters, num_iterations)

# print the results
print("Means:", means)
print("Standard deviations:", std_devs)
print("Weights:", weights)

