from sklearn.decomposition import PCA

from PPEM_Server import PPserver
from EMserver import Server
from settings import *
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
    # plot_name = f"Results/tests/{}_{chosen}_n{n}_k{k}_c{clients}"
    # dictionary of algorithms with different options to allow automation
    algorithms={1:[Server,PPserver],2:[PPserver],3:[Server]}
    plotNames={1:[FEM_directory,PPEM_directory],2:[PPEM_directory],3:[FEM_directory]}

    print("Select the function you would like to do :")
    chosen = check_validity("1.Compare both algorithms\n2.PPEM\n3.Federated EM", 1, 3)
    if chosen!=0:
        dataset=check_validity("1.use UCL's Parkinson DataBase that consists of 31 participants (23 infected) with 23 features\n"
                       "2.custom parameters with randomized data\n3.enter a dataset with '.data'"
                       "\n4.enter a dataset with '.csv'",1,4)
        for algo,directory in zip(algorithms[chosen],plotNames[chosen]):
            if dataset==1:
                data=prepareDataWithNormalization(data)
                pca = PCA(n_components=2)
                # Fit the PCA model to your data
                pca.fit(data)
                X= pca.transform(data)
                server = algo(n=len(X), max_iter=max_iter, number_of_clustures=number_of_clusters, plottingTools=True, eps=1e-19, clients=clients,
                                  plot_name=directory+f"Parkingsons/Parkingsons_c{clients}",input=X)
                pi, means, covariances, log_likelihoods, n_input, ticks, time_line = server.solve()

                writeData(pi, means, covariances, log_likelihoods, n_input, ticks, time_line,
                          directory+f"Parkingsons/Parkingsons_c{clients}")

                colored_plot(n_input,means,covariances,data[:,15],1,[],pi,directory+f"Parkingsons/Parkingsons_c{clients}_f",15,(0,1))

            elif dataset==2:
                n=input("please enter the number of data points:")

            elif dataset==3:
                pass
            elif dataset==4:
                pass
            else:
                pass


        print("Choose the Privacy preserving EM algorithm (PPEM) or the Federated EM algorithm")
