from Partial_EM import Partial_EM,Server
import numpy as np
from TenSEAL_encryption_unit import encryption
from HelpingFunctions import writeData
class Partial_PPEM(Partial_EM):
    def __init__(self, n, inputDimentions: int = 2, max_iter: int = 1, number_of_clustures: int = 2, eps: float = 1e-5,
                 epsilonExceleration: bool = True,
                 input: np.array = None, plottingTools: bool = False,plot_name="",encrypt=None):
        """
        Parameters:
            n:number of parameters.
            max_iter:number of iterations
            number_of_clustures :number of clusters to be associated with the data points
            input :input of the PPEM algorithm when we use server client model
            inputDimentions:the dimensions of the input array
            epsilonExceleration:a boolean that enables/disables convergence with epsilons aid
            eps:the value of epsilon that helps with the convergence criteria
            plottingTools:if True the algorithm plots a gif of the EM process of the algorithm

          """
        super(Partial_PPEM, self).__init__(n, inputDimentions, max_iter, number_of_clustures, eps, epsilonExceleration,
                                         input, plottingTools,plot_name)
        # encryption unit for encrypting the data for each client
        self.encryption_unit = encrypt
        self.qisaEncrypted=None
    def update_encryption(self,context):
        self.encryption_unit=context
    def mStep_epsilon(self):
        """ calculate the sum of the responsibilities ,and the uppder side of the means equation meaning q_i,s,a* X_i then return them to the server"""
        q_i_s_a, self.means, self.numberOfSamples=super(Partial_PPEM, self).mStep_epsilon()
        return self.encryption_unit.CKKS_encrypt(q_i_s_a),self.encryption_unit.CKKS_encrypt(self.means)

    def mStep_Covariance(self, means,qisa=None,n=None):
        """calculate the after you get the means from the server calculate the covariance matrix based on the means of all of the data """
        # step 2 when sending to the server !
        self.means =self.encryption_unit.decrypt(means)
        self.qisaEncrypted=self.encryption_unit.decrypt(qisa)
        self.means=self.means/self.qisaEncrypted[:, np.newaxis]
        self.pi=self.qisaEncrypted/self.encryption_unit.decrypt(n)[0]
        self.covariances=super(Partial_PPEM, self).mStep_Covariance(self.means)
        return self.encryption_unit.CKKS_encrypt(self.covariances)

    def m_step_actualCovariances(self,covariances):
        self.covariances=self.encryption_unit.decrypt(covariances)/self.qisaEncrypted[:, np.newaxis,np.newaxis]

class PPserver(Server):
    """
      Server EM Processor

    """

    def __init__(self, n=200, inputDimentions: int = 2, max_iter: int = 100, number_of_clustures: int = 2,
                 eps: float = 0.00001,
                 epsilonExceleration: bool = True,
                 input: np.array = None, plottingTools: bool = False, clients: int = 2,plot_name=""):
        """
        Parameters:
            n :number of parameters.
            max_iter :number of iterations
            number_of_clustures :number of clusters to be associated with the data points
            input :input of the PPEM algorithm when we use server client model
            inputDimentions:the dimensions of the input array
            epsilonExceleration:a boolean that enables/disables convergence with epsilons aid
            eps:the value of epsilon that helps with the convergence criteria
            plottingTools:if True the algorithm plots a gif of the EM process of the algorithm
            clients:The number of clients expected /created

        """

        super(PPserver, self).__init__(n, inputDimentions, max_iter, number_of_clustures, eps, epsilonExceleration, input,
                                     plottingTools,clients, plot_name,Partial_em=Partial_PPEM)

        self.encryptionUnit = encryption

    # update encryption for all clients
        self.encryptor=None
    def algo(self,i=0):
        self.encryptor=self.encryptionUnit()
        for client in self.clients:
            client.update_encryption(self.encryptor)
        super(PPserver, self).algo(i)

    def mStep_epsilon(self):
        """
         M step
         1. apply mstep for each client and get from each client the qisa,and the sum of the upper part of the means equation
         2. calculate sum_q_i_s_a ,q_i_s_a_DOT_Xi (the upper parts of the means and Pi's  equations )
         3. update the Pi for all the data and the means of all the data
         4. send the means back to the clients and get the upper portion of the covariance equation
         5. calculate the new covariance matrices
         6. send the parameters back to the clients and calculate the log likelyhood in each clients enviroment
         7. get the likelyhood function from the clients and check for convergence
        """
        # ____________________________________________________________
        # step 1
        # ____________________________________________________________
        all_qisa, means, covariances, num_samples = [], [], [], []
        oldMeans = self.means.copy()

        # for each client do its own m step
        for client in self.clients:
            qisa, mean = client.mStep_epsilon()
            all_qisa.append(qisa)
            means.append(mean)

        # ____________________________________________________________
        # step 2
        # ____________________________________________________________

        for index,qisa in  enumerate(all_qisa):
            if index==0:
                sum_q_i_s_a=qisa
            else:
                sum_q_i_s_a=sum_q_i_s_a+qisa

        for index,means in  enumerate(means):
            if index==0:
                q_i_s_a_DOT_Xi=means
            else:
                q_i_s_a_DOT_Xi=q_i_s_a_DOT_Xi+means

        # sum_q_i_s_a = np.sum(all_qisa, axis=0)
        # sum_q_i_s_a
        # q_i_s_a_DOT_Xi = np.sum(means, axis=0)
        # q_i_s_a_DOT_Xi_Minus_miu_squared = np.sum(covariances, axis=0)

        # ____________________________________________________________
        # step 1
        # ____________________________________________________________

        # update Pi
        self.pi = sum_q_i_s_a.mul(1/ self.n)

        self.means = q_i_s_a_DOT_Xi

        # ____________________________________________________________
        # step 4
        # ____________________________________________________________

        # todo add epsilon emplementation for faster convergence

        for client in self.clients:
            covariances.append(client.mStep_Covariance(self.means,sum_q_i_s_a,
                                                       self.encryptor.CKKS_encrypt([self.n])))

        q_i_s_a_DOT_Xi_Minus_miu_squared = np.sum(covariances, axis=0)
        # q_(i, s, a)(Xi-miu_s)(Xi-miu_s)^T /sum of q_(i,s,a)
        self.covariances = q_i_s_a_DOT_Xi_Minus_miu_squared

        # ____________________________________________________________
        # step 5
        # ____________________________________________________________

        for client in self.clients:
            client.m_step_actualCovariances(self.covariances)
            client.LogLikelyhood()
        # ____________________________________________________________
        # step 6
        # ____________________________________________________________77

        likelyhood = [client.getLikelyhood() for client in self.clients]
        self.log_likelihoods.append(np.sum(likelyhood))

if __name__ == '__main__':
    for n in range(100, 10000, 100):
        for k in range(2, 4):
            if n % k == 0:
                server = PPserver(n=n, max_iter=1000, number_of_clustures=k, plottingTools=False, eps=0.0001, clients=1,
                                plot_name=f"PPEM_n{n}_k3_c1")
                pi, means, covariances, log_likelihoods, n_input,ticks,time_line = server.solve()
                writeData(pi, means, covariances, log_likelihoods, n_input,ticks,time_line,f"Results/PPEM/PPEM_n{n}_k3_c1")
                for clients in range(2, 10, 4):
                    if n % clients == 0 and n / k > 20:
                        print("\n\n\n", "------" * 10)
                        server = Server(n=n, max_iter=1000, number_of_clustures=k, plottingTools=False, eps=0.0001,
                                        clients=clients,
                                        plot_name=f"Results/PPEM/PPEM_n{n}_k{k}_c{clients}", input=n_input)
                        pi, means, covariances, log_likelihoods, n_input,ticks,time_line=server.solve()
                        writeData(pi, means, covariances, log_likelihoods, n_input,ticks,time_line,f"PPEM_n{n}_k{k}_c{clients}")
