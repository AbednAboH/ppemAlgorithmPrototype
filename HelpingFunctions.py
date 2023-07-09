import csv
import os

import imageio as imageio
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import multivariate_normal
import glob


def em_algorithm(data, num_clusters, max_iter=1000, eps=1e-4):
    """
    Expectation-Maximization algorithm for Gaussian mixture model.

    :param data: numpy array of shape (num_samples, num_dimensions)
    :param num_clusters: integer, number of clusters
    :param max_iter: integer, maximum number of iterations
    :param eps: float, tolerance for stopping criterion
    :return: tuple of (pi, means, covariances, log_likelihoods)
    """

    # Initialize parameters
    num_samples, num_dimensions = data.shape
    pi = np.ones(num_clusters) / num_clusters
    means = np.random.rand(num_clusters, num_dimensions)
    covariances = np.array([np.eye(num_dimensions)] * num_clusters)
    log_likelihoods = []

    for i in range(max_iter):
        # E-step: compute responsibilities
        responsibilities = np.zeros((num_samples, num_clusters))
        for j in range(num_clusters):
            responsibilities[:, j] = pi[j] * multivariate_normal.pdf(data, means[j], covariances[j])
        responsibilities /= np.sum(responsibilities, axis=1, keepdims=True)

        # M-step: update parameters
        N_k = np.sum(responsibilities, axis=0)
        pi = N_k / num_samples
        for j in range(num_clusters):
            means[j] = np.sum(responsibilities[:, j].reshape(-1, 1) * data, axis=0) / N_k[j]
            covariances[j] = np.zeros((num_dimensions, num_dimensions))
            for n in range(num_samples):
                x = data[n, :] - means[j, :]
                covariances[j] += responsibilities[n, j] * np.outer(x, x)
            covariances[j] /= N_k[j]

        # Compute log-likelihood
        log_likelihood = np.sum(np.log(np.sum(pi[j] * multivariate_normal.pdf(data, means[j], covariances[j])
                                              for j in range(num_clusters))))
        log_likelihoods.append(log_likelihood)
        print(log_likelihoods)
        # Check for convergence
        if i > 0 and np.abs(log_likelihoods[-1] - log_likelihoods[-2]) < eps:
            break

    return pi, means, covariances, log_likelihoods


def twoDimentionsRepresentation(data,means,covariances,numberOfClusters):
    x_min, x_max = data[:, 0].min() - 1, data[:, 0].max() + 1
    y_min, y_max = data[:, 1].min() - 1, data[:, 1].max() + 1
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, 100), np.linspace(y_min, y_max, 100))
    # Compute PDF values for contour plot
    Z = np.zeros((xx.shape[0], xx.shape[1], numberOfClusters))
    for k in range(numberOfClusters):
        Z_k = pi[k] * multivariate_normal.pdf(np.hstack((xx.reshape(-1, 1), yy.reshape(-1, 1))), means[k],
                                              covariances[k])
        Z[:, :, k] = Z_k.reshape(xx.shape)
    # Plot data points and contour plot
    fig, ax = plt.subplots()
    ax.scatter(data[:, 0], data[:, 1], alpha=0.5)
    for k in range(numberOfClusters):
        ax.contour(xx, yy, Z[:, :, k], levels=10, colors=[plt.cm.Set1(k / numberOfClusters)])
    ax.set_xlim([x_min, x_max])
    ax.set_ylim([y_min, y_max])
    plt.show()


def multiDimentionsRepresentation(data, pi, means, covariances, numberOfDimensions):
    if numberOfDimensions < 2 or numberOfDimensions > 4:
        print("Cannot plot more than 4 dimensions or less than 2 dimensions.")
        return
    # Create meshgrid for contour plot
    #gets minmum and max value in each dimention
    ranges = [(data[:, i].min() - 1, data[:, i].max() + 1) for i in range(numberOfDimensions)]
    # np.linespace creates 100 samples from the minmum to the maximum
    # mesh = np.meshgrid(*[np.linspace(r[0], r[1], 100) for r in ranges])
    mesh = np.meshgrid(*[np.linspace(data[:, i].min() - 1, data[:,i].max() + 1, 100) for i in range(numberOfDimensions)])

    # Compute PDF values for contour plot
    Z = np.zeros((mesh[0].shape[0], mesh[0].shape[1], len(pi)))
    for k in range(len(pi)):
        Z_k = pi[k] * multivariate_normal.pdf(np.column_stack(mesh), mean=means[k], cov=covariances[k])
        Z[:, :, k] = Z_k.reshape(mesh[0].shape)
    # Plot data points and contour plot
    fig, axs = plt.subplots(nrows=numberOfDimensions - 1, ncols=numberOfDimensions - 1, figsize=(10, 10))
    for i in range(numberOfDimensions - 1):
        for j in range(i + 1, numberOfDimensions):
            for k in range(len(pi)):
                axs[i, j - 1].contour(mesh[i], mesh[j], Z[:, :, k, ], levels=10, colors=[plt.cm.Set1(k / len(pi))])
            axs[i, j - 1].scatter(data[:, i], data[:, j], alpha=0.5)
            axs[i, j - 1].set_xlabel(f"Dimension {i + 1}")
            axs[i, j - 1].set_ylabel(f"Dimension {j + 1}")
    plt.show()

def twoDimentionalGifCreator(data,means,covariances,numberOfClusters,i,plots,pi,name=None):
    print("two dimentional")
    x_min, x_max = data[:, 0].min() - 1, data[:, 0].max() + 1
    y_min, y_max = data[:, 1].min() - 1, data[:, 1].max() + 1
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, 100), np.linspace(y_min, y_max, 100))
    # Compute PDF values for contour plot
    Z = np.zeros((xx.shape[0], xx.shape[1], numberOfClusters))
    for k in range(numberOfClusters):
        Z_k = pi[k] * multivariate_normal.pdf(np.hstack((xx.reshape(-1, 1), yy.reshape(-1, 1))), means[k],
                                              covariances[k])
        Z[:, :, k] = Z_k.reshape(xx.shape)
    # Plot data points and contour plot
    fig, ax = plt.subplots()
    ax.scatter(data[:, 0], data[:, 1], alpha=0.5)
    for k in range(numberOfClusters):
        ax.contour(xx, yy, Z[:, :, k], levels=10, colors=[plt.cm.Set1(k / numberOfClusters)])
    ax.set_xlim([x_min, x_max])
    ax.set_ylim([y_min, y_max])
    ax.set_title('Frame %d' % i)

    if not os.path.exists('temp'):
        os.makedirs('temp')
    print("temp should be checked")

    if name is None:
        fig.savefig('temp/temp%d.png' % i, dpi=200)
        plt.close(fig)
        plots.append('temp/temp%d.png' % i)
        print("hello")
    else:
        print("ELSE")
        name2=name.split('/')
        # name2.remove(name[-1])
        string=""
        for n in name2:
            string+=n+'/'
        if not os.path.exists(string):
            os.makedirs(string)
        print(string)
        fig.savefig(f'{name}.png', dpi=200)
        plt.close(fig)
        plots.append(f'{name}.png' )

def writeData( pi:np.array, means:np.array, covariances:np.array, log_likelihoods:list, n_input:list,ticks:list,time_line:list,FileName:str):

    # Combine the data into a list of rows
    rows = [
        ['pi'] + pi.tolist(),
        ['means'] + means.tolist(),
        ['covariances'] + [c.tolist() for c in covariances],
        ['log_likelihoods'] + log_likelihoods,
        ['n_input'] + [row.tolist() for row in n_input],
        ['ticks'] + ticks,
        ['time_line'] + time_line
    ]

    # Write the rows to a CSV file
    second=FileName.split("/")
    second.pop(len(second)-1)
    sting=""
    for name in second:
        sting+=name+"/"
    directory = os.path.dirname(sting)
    if not os.path.exists(directory):
        os.makedirs(directory)
    with open(FileName, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerows(rows)
def debug_log(all_mean, all_covariances, fileName):
    # Combine the data into a list of columns
    columns = [['covariances'] + ['means']]
    columns.extend([[c, m] for c, m in zip(all_covariances, all_mean)])

    # Write the columns to a CSV file
    second = fileName.split("/")
    second.pop(len(second) - 1)
    sting = ""
    for name in second:
        sting += name + "/"
    directory = os.path.dirname(sting)
    if not os.path.exists(directory):
        os.makedirs(directory)
    with open(fileName, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerows(zip(*columns))


def plot_log_likelihood_from_csv(FileName, saveFileName,fileName):
    iterations = []
    log_likelihoods = []
    print("hello",FileName)
    # Read the CSV file
    with open(FileName, 'r') as csvfile:
        reader = csv.reader(csvfile)
        for row in reader:
            if row[0] == 'log_likelihoods':
                log_likelihoods = list(map(float, row[1:]))
        newlog=[log_likelihoods[i]-log_likelihoods[i-1] for i in range(len(log_likelihoods)) ]
    iterations=[i for i in range(len(newlog))]
    # Plot the log-likelihood per iteration
    plt.plot(iterations, newlog, marker='o')
    plt.xlabel('Iteration')
    plt.ylabel('Log-Likelihood')
    plt.title('Log-Likelihood per Iteration'+fileName)

    # Save the plot as an image file
    plt.savefig(saveFileName)

    # Close the plot to release resources (optional)
    plt.close()

def plot_log_likelihoods_from_csv_files(file_pattern, output_directory):
    # Find all CSV files matching the pattern
    csv_files = glob.glob(file_pattern)
    # Iterate over the CSV files

    if not os.path.exists(output_directory):
        os.makedirs(output_directory)
    for csv_file in csv_files:

        # Extract the filename without the extension
        filename = csv_file.split('.')[0]
        filename=filename.split('/')
        filename=filename[-1].split('\\')

        print(filename[-1])
        # Generate the output filename for the plot
        output_filename = f"{output_directory}/{filename[-1]}_LogLikelyhood_plot.png"
        # Call the plot_log_likelihood_from_csv function for each CSV file
        plot_log_likelihood_from_csv(csv_file, output_filename,filename[-1])

if __name__ == '__main__':
    # """ Testing functions and algorithms """
    # # Set number of clusters
    # numberOfClusters = 3
    # numberOfDimensions = 3
    #
    # # Generate toy dataset with 3 clusters
    # np.random.seed(42)
    # data = np.vstack((np.random.randn(100, numberOfDimensions), np.random.randn(100, numberOfDimensions) + 5, np.random.randn(100, numberOfDimensions) + 10))
    #
    # # Run EM algorithm
    # pi, means, covariances, log_likelihoods = em_algorithm(data, num_clusters=numberOfClusters)
    #
    # # Create meshgrid for contour plot
    #
    # twoDimentionsRepresentation(data,means[:,1:3],covariances[:,1:3,1:3],numberOfClusters)
    #
    # twoDimentionsRepresentation(data,means[:,:2],covariances[:,:2,:2],numberOfClusters)
    #
    # # Plot data points and contour plot
    # multiDimentionsRepresentation(data, pi, means, covariances, numberOfDimensions)
    plot_log_likelihoods_from_csv_files("Results/PPEM/*.csv","Results/PPEM/LogLikelyhoods")