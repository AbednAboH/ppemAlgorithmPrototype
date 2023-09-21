import csv
import math
import os
import re

import imageio as imageio
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy.stats import multivariate_normal
import glob
import seaborn as sns

from Settings_Parameters import accepted_values


def em_algorithm(data, num_clusters, max_iter=1000, eps=1e-4):
    """
    basic Expectation-Maximization algorithm for Gaussian mixture model works only on 2 dimentions.

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

# create a plot of the algorithms progression

def twoDimentionsRepresentation(data, means, covariances, numberOfClusters):
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


# create a plot with colored points representing the clusters
def colored_plot(data_full, means_full, covariances_full, pi,directory=None):
    begin,end=0,1

    # Calculate the likelihood of each data point under each cluster
    likelihoods = np.zeros((len(data_full), len(covariances_full)))
    for j in range(len(covariances_full)):
        likelihoods[:, j] = pi[j] * multivariate_normal.pdf(data_full, mean=means_full[j], cov=covariances_full[j])

    # Assign labels based on the cluster with the highest likelihood
    cluster_labels = np.argmax(likelihoods, axis=1)

    # Extract points for "false" and "true" based on predicted labels
    false_points = data_full[cluster_labels == 0]
    true_points = data_full[cluster_labels == 1]
    # Create the scatter plot
    plt.scatter(false_points[:, begin], false_points[:, end], c='blue', label='Predicted False')
    plt.scatter(true_points[:, begin], true_points[:, end], c='red', label='Predicted True')


    plt.xlabel('Feature 1')
    plt.ylabel('Feature 2')
    plt.title('Clusters in Colors')
    plt.legend()

    if directory is not None:
        plt.savefig(f'{directory}.png')
    else:
        plt.show()

# create a gif of the algorithms progression
def twoDimentionalGifCreator(data, means, covariances, numberOfClusters, i, plots, pi, name=None,gif=False):
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
    if gif:
        fig.savefig(f'temp/temp{i+1}.png', dpi=200)
        plt.close(fig)
        plots.append(f'temp/temp{i+1}.png')
    else:
        name2 = name.split('/')
        string = ""
        for i, n in enumerate(name2):
            if i != len(name2) - 1:
                string += n + '/'
        if not os.path.exists(string):
            os.makedirs(string)
        fig.savefig(f'{name}.png', dpi=200)
        plt.close(fig)
        plots.append(f'{name}.png')




# write the data from each algorithm to a csv file for future processing
def writeData(pi: np.array, means: np.array, covariances: np.array, log_likelihoods: list, n_input: list, ticks: list,
              time_line: list, FileName: str):
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
    second = FileName.split("/")
    second.pop(len(second) - 1)
    sting = ""
    for name in second:
        sting += name + "/"
    directory = os.path.dirname(sting)
    if not os.path.exists(directory):
        os.makedirs(directory)
    with open(FileName+'.csv', 'w', newline='') as csvfile:

        writer = csv.writer(csvfile)
        writer.writerows(rows)


def pdf(x, mean, covariance):
    mv_normal = multivariate_normal(mean=mean, cov=covariance)
    return mv_normal.pdf(x)

# plotting from a single csv file with option to continue plottin
def plot_log_likelihood_from_csv(FileName, title, linewidth, linestyle, color='red',single=True,output_directory=""):

    log_likelihoods = []
    print("Processing", FileName)
    # Read the CSV file
    with open(FileName, 'r') as csvfile:
        reader = csv.reader(csvfile)
        for row in reader:
            if row[0] == 'log_likelihoods':
                log_likelihoods = list(map(float, row[1:]))
        newlog = [log_likelihoods[i] / 100.0 for i in range(len(log_likelihoods))]
    iterations = [i for i in range(len(newlog))]
    # Plot the log-likelihood per iteration with different line styles and markers
    plt.plot(iterations, newlog, label=title,linewidth=linewidth,linestyle=linestyle,color=color)
    if single:
        # Customize the plot appearance
        plt.xlabel('Iteration')
        plt.ylabel('Log-Likelihood')
        plt.title('Log-Likelihood per Iteration')
        plt.legend()

        # Save the plot as an image file
        output_filename = f"{output_directory}/combined_LogLikelihood_plot.png"
        plt.savefig(output_filename)


# plot a plot from all csv files from single directory
def plot_log_likelihoods_from_csv_files(output_directory,csv_files):
    if not os.path.exists(output_directory):
        os.makedirs(output_directory)
    linewidth=4*len(csv_files)
    line=['-','-.','--',':']
    for i,csv_file in enumerate(csv_files):
        # Extract the filename without the extension
        filename = os.path.splitext(os.path.basename(csv_file))[0]
        # Call the plot_log_likelihood_from_csv function for each CSV file
        plot_log_likelihood_from_csv(csv_file, filename,linewidth/(i+1),line[i%len(line)],single=False)

    # Customize the plot appearance
    plt.xlabel('Iteration')
    plt.ylabel('Log-Likelihood')
    plt.title('Log-Likelihood per Iteration')
    plt.legend()

    # Save the plot as an image file
    output_filename = f"{output_directory}/combined_LogLikelihood_plot.png"
    plt.savefig(output_filename)



def plot_log_likelihoods_from_csv_files_v2(directory1, directory2, output_directory):
    if not os.path.exists(output_directory):
        os.makedirs(output_directory)

    # Find all CSV files in the specified directories
    csv_files1 = glob.glob(os.path.join(directory1, '*.csv'))
    csv_files2 = glob.glob(os.path.join(directory2, '*.csv'))

    # Group CSV files by 'n' and 'k' values
    file_groups = {}
    for csv_file in csv_files1 + csv_files2:
        # Extract 'n' and 'k' values from the filename
        filename = os.path.splitext(os.path.basename(csv_file))[0]
        parts = filename.split('_')
        n, k = None, None
        for part in parts:
            if part.startswith('n'):
                n = part[1:]
            elif part.startswith('k'):
                k = part[1:]

        # Use 'n' and 'k' as the key to group files
        key = (n, k)
        if key not in file_groups:
            file_groups[key] = []
        file_groups[key].append(csv_file)

    # Customize line styles and colors
    line_styles = ['-', '-.', '--', ':']
    colors = ['b', 'g', 'r', 'c', 'm', 'y', 'k']

    for i, (key, group) in enumerate(file_groups.items()):
        combined_filename = f"combined_n{key[0]}_k{key[1]}_LogLikelihood_plot.png"
        plt.figure(figsize=(10, 6))
        linewidth = 2

        for j, csv_file in enumerate(group):
            # Extract the filename without the extension
            filename = os.path.splitext(os.path.basename(csv_file))[0]
            if csv_file in csv_files1:
                plot_log_likelihood_from_csv(csv_file, f"Folder1: {filename}", linewidth,
                                             line_styles[i % len(line_styles)], colors[j % len(colors)],
                                             single=True)
            else:
                plot_log_likelihood_from_csv(csv_file, f"Folder2: {filename}", linewidth,
                                             line_styles[i % len(line_styles)], colors[j % len(colors)],
                                             single=True)
            linewidth -= 0.5

        # Customize the plot appearance
        plt.xlabel('Iteration')
        plt.ylabel('Log-Likelihood')
        plt.title(f'Log-Likelihood per Iteration (n={key[0]}, k={key[1]})')
        plt.legend()

        # Save the plot as an image file
        output_filename = os.path.join(output_directory, combined_filename)
        plt.savefig(output_filename)
        plt.close()

# mini plot function that adds a line to an existing plot or saves a full plot based on the single value
# single =true/false -> saves/continue plotting
def plot_log_likelihood_from_csv_per_nk(FileName, legend_label, linewidth, linestyle, color, single=True):
    # Read the CSV file
    iterations = []
    log_likelihoods = []

    with open(FileName, 'r') as csvfile:
        reader = csv.reader(csvfile)
        for row in reader:
            if row[0] == 'log_likelihoods':
                log_likelihoods = list(map(float, row[1:]))

    iterations = [i for i in range(len(log_likelihoods))]

    # Plot the log-likelihood per iteration
    plt.plot(iterations, log_likelihoods, linewidth=linewidth, linestyle=linestyle, color=color, label=legend_label)

    if single:
        plt.xlabel('Iteration')
        plt.ylabel('Log-Likelihood')

# compare the csv files with the same n and k and save them as a plot
def compare_csv_files_by_nk(folder1_path, folder2_path,outputdir):
    # Find all CSV files in both folders
    csv_files1 = glob.glob(os.path.join(folder1_path, '*.csv'))
    csv_files2 = glob.glob(os.path.join(folder2_path, '*.csv'))

    # Create a dictionary to group CSV files by their 'n', 'k', and 'c' values
    grouped_csv_files = {}

    for csv_file in csv_files1 + csv_files2:
        # Extract 'n', 'k', and 'c' values from the filename
        filename = os.path.splitext(os.path.basename(csv_file))[0]
        parts = filename.split('_')
        if 'n' in parts[1] and 'k' in parts[2]:
            n, k, c = parts[1], parts[2], parts[3]

        else:
            n, k, c = parts[1], parts[1], parts[1]
        if (n, k) not in grouped_csv_files:
            grouped_csv_files[(n, k)] = []
        grouped_csv_files[(n, k)].append(csv_file)
        # Add the CSV file to the group


    # Customize the plot appearance
    line_styles = ['-', '--', '-.', ':']

    for i, (n, k) in enumerate(grouped_csv_files.keys()):
        group = grouped_csv_files[(n, k)]
        plt.figure(figsize=(10, 6))  # Adjust the figure size as needed

        for j, csv_file in enumerate(group):
            # Extract the filename without the extension
            filename = os.path.splitext(os.path.basename(csv_file))[0]

            # Use different line styles for each file within the group
            line_style = line_styles[j % len(line_styles)]

            # Use different colors for each group
            color = plt.cm.viridis(j / len(group))
            linewidth=2*(len(group)-j)
            legend_label = f"{filename} (Folder {j + 1})"
            plot_log_likelihood_from_csv_per_nk(csv_file, legend_label, linewidth=linewidth, linestyle=line_style, color=color,
                                         single=False)

        # Add more details to the plot
        plt.xlabel('Iteration')
        plt.ylabel('Log-Likelihood')
        plt.title(f'Log-Likelihood Comparison for n={n}, k={k}')
        plt.legend(loc='upper right')

        # Save the plot as an image file
        output_filename = f"{n}_k{k}_LogLikelihood_plot.png"
        output_directory=outputdir+'/'+output_filename
        if not os.path.exists(outputdir):
            os.makedirs(outputdir)
        plt.savefig(output_directory)

# compare the data .csv files from both results
def compare_both_algorithms(folder1_path, folder2_path, output_path,accepted_values):
    # Initialize a dictionary to store the data
    data = {}

    # Iterate through files in folder1 and extract relevant data
    extract_data(data, folder1_path,"Folder1")
    identifiers_array=[]
    # Iterate through files in folder2 and extract relevant data (similar to folder1)
    for filename in os.listdir(folder2_path):
        if filename.endswith('.csv'):
            identifier = filename.split('_')[1:]
            identifier = "_".join(identifier)
            identifier=identifier.split('.')[0]
            identifiers_array.append(identifier)
            with open(os.path.join(folder2_path, filename), 'r') as file:
                reader = csv.reader(file)
                if identifier in data:
                    data[identifier]["Folder2"] = {}
                    for row in reader:
                        data[identifier]["Folder2"][row[0]] = row[-1]
                        if row[0] in "log_likelihoods":
                            data[identifier]["Folder2"]["Iterations"] = len(row) - 1

    # Write the data to a CSV file
    identifiers_array.sort(key=lambda x:(int(x.split('_')[0][1:]),int(x.split('_')[1][1:]),int(x.split('_')[2][1:])))
    with open(output_path, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)

        # Write the header row
        header = ["Identifier"]
        for key in data[list(data.keys())[0]]["Folder1"].keys():
            if key in accepted_values:
                header.append(f"{key} EM")
                header.append(f"{key} PPEM")

        writer.writerow(header)

        # Write the data rows
        for id in identifiers_array:
            values=data[id]
            row = [id]
            for key in values["Folder1"].keys():
                if key in accepted_values:
                    try:
                        row.extend([round(float(values["Folder1"][key]),3), round(float(values["Folder2"].get(key, "")),3)])
                    except Exception as e:
                        print("key doesn't exist in Folder 1 ,inserting folder1' values instead")
                        row.extend([round(float(values["Folder1"][key]),3),"N\A"])
            writer.writerow(row)

# extract the data from the files we created earlier
def extract_data(data, folder1_path,key):
    for filename in os.listdir(folder1_path):
        if filename.endswith('.csv'):
            # Parse the filename to extract the identifier (e.g., n1_k2_c5)
            identifier = filename.split('_')[1:]  # Extract the parts after the first underscore
            identifier = "_".join(identifier)  # Join the parts to form the identifier
            identifier = identifier.split('.')[0]
            # Read the CSV file into a dictionary
            with open(os.path.join(folder1_path, filename), 'r') as file:
                reader = csv.reader(file)
                data[identifier] = {key: {}}
                for row in reader:
                    data[identifier][key][row[0]] = row[-1]
                    if row[0] in "log_likelihoods":
                        data[identifier][key]["Iterations"]=len(row)-1


# Usage example

# if __name__ == '__main__':
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


    # csv_files = glob.glob(file_pattern)

    # plot_log_likelihoods_from_csv_files("Results/FEM/Parkingsons/*.csv", "Results/FEM/Parkingsons/LogLikelyhoods")

    # compare_both_algorithms('Results/PPEM/optimized','Results/MultiPartyEM_With_initialization','Results/comparison_data.csv',accepted_values)
    # compare_csv_files_by_nk('Results/PPEM/optimized', 'Results/MultiPartyEM_With_initialization','Results/LikleyhoodComparisions')
    # ,'Results/LikleyhoodComparisions')
