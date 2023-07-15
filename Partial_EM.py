from HelpingFunctions import *
from FastEM import algortithem


class Partial_EM(algortithem):

    def __init__(self, n, inputDimentions: int = 2, max_iter: int = 1, number_of_clustures: int = 2, eps: float = 1e-5,
                 epsilonExceleration: bool = True,
                 input: np.array = None, plottingTools: bool = False, plot_name=""):
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
        super(Partial_EM, self).__init__(n, inputDimentions, max_iter, number_of_clustures, eps, epsilonExceleration,
                                         input, plottingTools, plot_name)

        self._pi, self._means, self._covariances,_,_,_,_=algortithem(n, inputDimentions, max_iter, number_of_clustures, eps, epsilonExceleration,
                                         input, plottingTools,show_time=None).solve()

        # encryption unit for encrypting the data for each client
        self.qisa = None




    def mStep_epsilon(self):
        """ calculate the sum of the responsibilities ,and the uppder side of the means equation meaning q_i,s,a* X_i then return them to the server"""
        a = np.sum(self.responisbilities, axis=0)
        b,c =[],[]
        for j in range(self.k):
            b.append(np.sum(self.responisbilities[:, j].reshape(-1, 1) * self.n_inputs, axis=0))
        # todo check: checked , the same as it should be
            c.append(np.zeros((self.inputDimentions, self.inputDimentions)))
            for n in range(self.numberOfSamples):
                x = self.n_inputs[n, :] - self._means[j, :]
                c[j] += self.responisbilities[n, j] * np.outer(x, x)
        return a,b,c

    def update(self, a_all,b_all,c_all,n):
        for j in range(self.k):
            self._means[j] = b_all[j] / a_all[j]
            self._covariances[j] = c_all[j] / a_all[j]
        self._pi =a_all/ n
