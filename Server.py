from HelpingFunctions import writeData
from PPEM_Server import PPserver
from EMserver import Server

n=102
k=2
clients=1
server = Server(n=n, max_iter=400, number_of_clustures=k, plottingTools=False, eps=1e-20,
                  clients=clients,
                  plot_name=f"Results/tests/PPEM_n{n}_k{k}_c{clients}")
pi, means, covariances, log_likelihoods, n_input, ticks, time_line = server.solve()
print(pi, means, covariances)
writeData(pi, means, covariances, log_likelihoods, n_input, ticks, time_line,
          f"Results/tests/PPEM_n{n}_k{k}_c{clients}.csv")