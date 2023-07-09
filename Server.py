from HelpingFunctions import writeData,debug_log
from PPEM import Server,PPserver

n=100
k=2
clients=2
server = Server(n=n, max_iter=1000, number_of_clustures=k, eps=1e-9,
                  clients=clients,
                  plot_name=f"Results/tests/em_n{n}_k{k}_c{clients}")
cov,means=server.covariances,server.means
pi, means2, covariances, log_likelihoods, n_input, ticks, time_line = server.solve()
writeData(pi, means2, covariances, log_likelihoods, n_input, ticks, time_line,
          f"Results/tests/EM_n{n}_k{k}_c{clients}.csv")
debug_log(server.log_means,server.log_covariances,f"Results/tests/debug_EM_n{n}_k{k}_c{clients}.csv")

common_partitioning=[]
for client in server.clients:
    common_partitioning.append(client.n_inputs)

server = PPserver(n=n, max_iter=1000, number_of_clustures=k, eps=1e-9,
                  clients=clients,input=n_input,
                  plot_name=f"Results/tests/ppem_n{n}_k{k}_c{clients}")
for i,client in enumerate(server.clients):
    client.n_inputs=common_partitioning[i]

server.covariances,server.means=cov,means
pi, means, covariances, log_likelihoods, n_input, ticks, time_line = server.solve()
writeData(pi, means, covariances, log_likelihoods, n_input, ticks, time_line,
          f"Results/tests/PPEM_n{n}_k{k}_c{clients}.csv")
debug_log(server.log_means,server.log_covariances,f"Results/tests/debug_ppEM_n{n}_k{k}_c{clients}.csv")


