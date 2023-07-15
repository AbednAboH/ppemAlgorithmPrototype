from HelpingFunctions import writeData
from PPEM_Server import PPserver
from EMserver import Server
# PPEM_n6000_k2_c6
n=6000
k=2
clients=12
server = Server(n=n, max_iter=400, number_of_clustures=k, plottingTools=False, eps=1e-4,
                  clients=clients,inputDimentions=2,
                  plot_name=f"Results/tests/PPEM_n{n}_k{k}_c{clients}")
pi, means, covariances, log_likelihoods, n_input, ticks, time_line = server.solve()
print(pi, means, covariances)
writeData(pi, means, covariances, log_likelihoods, n_input, ticks, time_line,
          f"Results/tests/PPEM_n{n}_k{k}_c{clients}.csv")