from HelpingFunctions import writeData
from PPEM_Server import PPserver

# PPEM_n6000_k2_c6
if __name__ == '__main__':

    n=200
    k=2
    clients=2
    server = PPserver(n=n, max_iter=300, number_of_clustures=k, plottingTools=False, eps=1e-4,
                      clients=clients,inputDimentions=7,
                      plot_name=f"Results/tests/PPEM_n{n}_k{k}_c{clients}")
    pi, means, covariances, log_likelihoods, n_input, ticks, time_line = server.solve()
    print(pi, means, covariances)
    writeData(pi, means, covariances, log_likelihoods, n_input, ticks, time_line,
              f"Results/tests/PPEM_n{n}_k{k}_c{clients}.csv")