### Our Approach

Our model builds upon the Federated EM approach, which involves a server-client architecture. We have previously demonstrated that the privacy issue in the Federated EM approach arises from the exchange of intermediate updates, specifically $a^t_{ij}$, $b^t_{ij}$, and $C^t_{ij}$, which reveal the private data of each client (node). To address this issue, we propose concealing the data from both the server and clients by encrypting it with Fully Homomorphic Encryption (FHE), specifically CKKS. One of the advantages of using CKKS is that addition and subtraction can be performed on encrypted data without revealing the individual $a^t_{ij}$, $b^t_{ij}$, and $C^t_{ij}$ values of each client (node). We will explain CKKS encryption and then illustrate how we apply it in the PPEM (Privacy-Preserving Expectation-Maximization) that we have designed.

### Proposed PPEM
For iteration $t$, the encryption unit creates a public key and a secret key as a pair ($Public-k_t,secret-k_t)$ which is then distributed to all trusted nodes (clients) except the server which has no access to any key, each node then calculates it's own E-step ( PDF) then each node calculates the partial m-step (intermediate m-step) in which it calculates:
    $a^t_{i,j}=P(x_i,N^t_j)$
    $b^t_{i,j}=P(x_i|N^t_j)X_i$
    $c^t_{i,j}=P(x_i,N^t_j)(x_i-\mu^t_j)(x_i-\mu^t_j)^T$
where $i\in {1,2,3..n}$ nodes ,$j\in {1,2,...k}$  Gaussian distributions and assuming that the nodes are honest nodes that are not corrupted then each client(node) prepares a,b,c as three vectors that are made flat because we might be dealing with more than two dimensions , meaning each Gaussian distribution (Cluster would be made up of multiple dimensions .
lets consider d dimensions for Data X  with k Gaussian mixture models : $a_i\in V^{1xk}  $ where V is a vector space ,$b_i\in M^{kxd}$ ,$c_i \in M^{kxdxd}$ Where M is a matrix space.
our implementation takes $b_i,c_i$ and flattens them to be one single vector ,meaning $b_i'\in V^{kd}, c_i'\in V^{kdd}$ :
    $a_i'=[a_{i1},a_{i2},..a_{ik},n_i]$ 
 where $n_i$ is the number of data points in node i
    $b_i'=[b_{i11},b_{i12},...b_{i1d},b_{i21}..b_{ikd}]$
    $c_i'=[c_{i111},c_{i112},...,c_{i11d},c_{i121},...c_{i1dd},c_{i211}...c_{ikdd}]$
after defining those vectors we continue with the partial m-step , we apply the CKKS encryption on each vector $a_i',b_i',c_i'$ then each node sends them back to the server . The server calculates using property (7) :
    $a^{encrypted}_{total}=\sum^n_{i=1}(a^{encrypted}_i)'$
    $b^{encrypted}_{total}=\sum^n_{i=1}(b^{encrypted}_i)'$
    $c^{encrypted}_{total}=\sum^n_{i=1}(c^{encrypted}_i)'$
The server then sends the results to each client where each client(node) then calculates:
     $a,b,c=decrypt(a_{total}),decrypt(b_{total}),decrypt(c_{total})$
      $n_c=a[k+1]$
        $\beta^{t+1} =\frac{a_j}{n_c}$
       $\mu^{t+1}=\frac{b_j}{a_j}$
        $\Sigma^{t+1}=\frac{c_j}{a_j}$
 Thus finishing the m-step , we repeat this process until convergence.
### Initialization
The initialization process is carried out before the first iteration:

- $\Sigma_i$ is set to a diagonal matrix of 1's.
- $\mu_i$ is randomly generated.
- $\pi_i$ is initialized with a Uniform distribution, ensuring that each cluster (Gaussian distribution) has the same probability as the others. 
### Privacy
In our analysis, we assume that the nodes are honest and not corrupted. As previously demonstrated, the original Federated EM algorithm has a vulnerability where $x_i$ can be exposed and inferred by an attacker during the partial M-step. This means that the following mutual information is exposed to the server maximally:
$I(X_i; A^t_{ij}, B^t_{ij}) = I(X_i; X_i)$
To address this vulnerability, we encrypt both $A^t_{ij}$ and $B^t_{ij}$ using CKKS. Assuming that $enc(Y) represents the CKKS-encrypted version of Y, and the server operates as described, the information that the server holds is:
$I(X; ∑enc(A^t_{i}), ∑enc(B^t_{i}), ∑enc(C^t_{i}))$
Since CKKS encryption is indistinguishable under the Chosen Plaintext Attack (IND-CPA), we can assume that each $enc(A^t_i)$, $enc(B^t_i)$, and $enc(C^t_i)$ are indistinguishable from one another. Therefore, as an Honest but Curious adversary, the server cannot distinguish enc(A^t_i) from enc(B^t_i). Moreover, the server does not have the private key of the nodes (clients), thus cannot decrypt the information to find out the true values of A_i and B_i. This addresses the main vulnerability of the original Federated EM algorithm:
$I(X_i; enc(A^t_{i}), enc(B^t_{i})) = IND-CPA 0$
Furthermore, when considering the combined information held by the server in the form of $enc(A^t_{i})$, $enc(B^t_{i})$, and $enc(C^t_{i})$, they are indistinguishable from each other, enhancing privacy even further.

Regarding privacy on the client's side, as the server sends the encrypted sums A, B, and C, the client(node) decrypts them into a_total, b_total, and c_total, which does not pose any security concerns when dealing with large datasets. This security concern would be significant only when the number of nodes and datasets is significantly smaller, in which case this approach may not be recommended or useful.

If we visualize the system as a graph, the server acts as the root, and all other nodes are leaves. There are no direct connections between leaf nodes; they can only communicate with the server. The usage of shared keys (public and secret) for all nodes does not breach privacy between nodes, as nodes can only communicate with the server. Any potential breach of privacy between leaf nodes would occur during the transition between the server and the client. However, this has been addressed by ensuring an honest but curious server cannot exploit this situation.

To address a passive adversary, such as eavesdropping, on the server, we change the encryption keys at each iteration to prevent the adversary from identifying patterns in the plaintext received by the server.
### Stoppage Criteria
The stopping criteria for our algorithm are based on the difference between log-likelihoods at each iteration, with the parameter $\epsilon$ serving as the decision maker for stopping or continuing the algorithm. Smaller values of $\epsilon$ result in more precise but potentially slower convergence, while larger values may lead to less accurate results. The ideal value for $\epsilon$ depends on the specific task, but it can approach zero for maximum precision. However, the algorithm may still run until reaching the user-defined maximum number of iterations.

The decision for stopping is based on the change in log-likelihoods between iterations, with $\epsilon$ controlling the threshold. The log likelihood is the sum of likelihoods from all clients, and the logarithmic transformation simplifies the aggregation process.


### Correctness
The correctness of our model is determined by the estimated parameters of the Gaussian Mixture Model (GMM) and the degree to which our implementation impacts them. We send the $a_i$, $b_i$, $c_i$ parameters encrypted by CKKS, which introduces some level of approximation and noise. However, if the encryption parameters are chosen correctly, this should not significantly impact the accuracy of the model. Precise encryption parameters are essential, and in the context of GMMs, high-precision parameters are recommended, as observed in tests conducted by the TenSEAL team.