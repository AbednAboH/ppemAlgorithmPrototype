from HelpingFunctions import writeData
from PPEM import PPserver

n=3800
k=2
clients=2
server = PPserver(n=n, max_iter=1000, number_of_clustures=k, plottingTools=False, eps=0.0001,
                  clients=clients,
                  plot_name=f"Results/PPEM/PPEM_n{n}_k{k}_c{clients}")
pi, means, covariances, log_likelihoods, n_input, ticks, time_line = server.solve()
writeData(pi, means, covariances, log_likelihoods, n_input, ticks, time_line,
          f"Results/PPEM/PPEM_n{n}_k{k}_c{clients}.csv")