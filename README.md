# Privacy Preserving Expectation maximization (PPEM) On Gaussian Mixture models
    This is an implementation of our proposal of a new protocol that guarantees privacy to a certain degree
    Supervised by Prof.Adi Akvia
    Proposed by Abed Elrahman abo hussien and Michael Rodel

    Some background on the project can be found ![here](Background.md).

    Our proposal can be found ![here](our_approach.md).


# Tests and how to navigate them
    We performed tests on each algorithm with different sets of randomized data .
    each test is represented as 'algorithm_n{number of points}_k{clusters}_c{number of clients}'
    where algorithm is replaced by the algorithm PPEM or FEM.
    n- is the number of data points in the data set
    k- is the number of clusters
    c- is the number of clients

    The tests were done on both algorithms:
       1. each individual drawing of the tests is in the 'Results/algorithm/algorithm_ni_kj_cm' folder with an excel that contains all relevant information of that test
       2. the Log likelihood test that compares both algorithms is in the 'Results/LikleyhoodComparisions' folder
       3. the parkinsons tests each are in both 'Results/algorithm/parkinsons' folder
       4. the excel that compares all data together is in the 'Results' folder named 'comparition data'


    To reproduce our findings please run the TestsScript.py where all the folders would be created from scrach

    To reproduce Parkinsons/randomized data tests you can use the main.py and then chose the appropriate options :

       Select the function you would like to do :
        1.Run on both algorithms
        2.PPEM
        3.Federated EM

       Then:
        1.use UCL's Parkinson DataBase that consists of 31 participants (23 infected) with 23 features
        2.custom parameters with randomized data
        3.enter a dataset with '.data'/'.csv'
# Requirements
   found in the requirements.txt
    imageio==2.15.0
    matplotlib==3.3.4
    numpy==1.19.5
    pandas==2.1.1
    scikit_learn==1.3.1
    scipy==1.5.4
    seaborn==0.12.2
    self==2020.12.3
    tenseal==0.3.6

#

