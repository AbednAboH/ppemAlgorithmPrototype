from sklearn.decomposition import PCA

from PrivacyPreserving_EM_Server import PPserver
from Federated_EM_Server import Server
from Settings_Parameters import *
from HelpingFunctions import*

def check_validity(string, start, end):
    x = input(string)
    while int(x) not in range(start, end):
        if int(x) == 0:
            break
        x = input(string)
    return int(x)


def prepareDataWithNormalization(data):

    data = data[1:].to_numpy(dtype=np.float64)
    return (data - np.mean(data, axis=0)) / np.std(data, axis=0)


if __name__ == '__main__':
    number_of_clusters = 2
    data = None
    show_results = True
    max_iter = 300
    clients = 2
    data_set = "Random"
    algorithms={1:[Server,PPserver],2:[PPserver],3:[Server]}
    plotNames={1:[FEM_directory,PPEM_directory],2:[PPEM_directory],3:[FEM_directory]}

    print("Select the function you would like to do :")
    chosen = check_validity(CHOSEN_ALGO, 1, 3)
    if chosen!=0:
        dataset=check_validity(DATASET_SELECTION,1,4)

        if dataset==1:
            clients=int(input(NUM_CLIENTS))
            data=prepareDataWithNormalization(parkingson_data)
            pca = PCA(n_components=2)
            # Fit the PCA model to your data
            pca.fit(data)
            X= pca.transform(data)
            for algo, directory in zip(algorithms[chosen], plotNames[chosen]):
                server = algo(n=len(X), max_iter=max_iter, number_of_clustures=number_of_clusters, plottingTools=True, eps=1e-4, clients=clients,
                                  plot_name=directory+f"Parkingsons/Parkingsons_c{clients}",input=X)
                pi, means, covariances, log_likelihoods, n_input, ticks, time_line = server.solve()

                writeData(pi, means, covariances, log_likelihoods, n_input, ticks, time_line,
                          directory+f"Parkingsons/Parkingsons_c{clients}")

                colored_plot(n_input,means,covariances,pi)

        elif dataset==2:
            n=int(input("Enter the number of data points (100,10000):"))
            max_iter=int(input(ITERATIONS))
            number_of_clusters=int(input(CLUSTERS))
            plotting_tools=True if "1" in input(PLOTTING) else False
            d=input(DIRECTORY)
            for algo, directory in zip(algorithms[chosen], plotNames[chosen]):
                directory=d if "1" in DIR else directory+f"_n{n}_k{number_of_clusters}_c{clients}"
                server = algo(n=n, max_iter=max_iter, number_of_clustures=number_of_clusters, plottingTools=plotting_tools,eps=1e-4
                              , clients=clients,plot_name=directory)
                pi, means, covariances, log_likelihoods, n_input, ticks, time_line = server.solve()

                writeData(pi, means, covariances, log_likelihoods, n_input, ticks, time_line,
                          directory)

                colored_plot(n_input, means, covariances, pi)
        elif dataset==3:
            print(NOTES_ON_DATA)
            progress_selection=input(SELECT_UCL_DATA)
            if "1" in progress_selection:
                max_iter = int(input(ITERATIONS))
                number_of_clusters = int(input(CLUSTERS))
                directory = input(DIRECTORY)
                plotting_tools = True if "1" in input(PLOTTING) else False
                data = pd.read_csv('directory', names=column_names)
                data = prepareDataWithNormalization(data)
                pca = PCA(n_components=2)
                pca.fit(data)
                X = pca.transform(data)
                for algo, directory in zip(algorithms[chosen], plotNames[chosen]):
                    server = algo(n=len(X), max_iter=max_iter, number_of_clustures=number_of_clusters,
                                  plottingTools=plotting_tools, eps=1e-4
                                  , clients=clients, plot_name=directory,input=X)
                    pi, means, covariances, log_likelihoods, n_input, ticks, time_line = server.solve()

                    writeData(pi, means, covariances, log_likelihoods, n_input, ticks, time_line,
                              directory)

                    colored_plot(n_input, means, covariances, pi)
            else:
                pass



