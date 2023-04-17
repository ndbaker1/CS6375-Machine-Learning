import numpy as np
from Util import *
from CLT_class import CLT


class MIXTURE_CLT():

    def __init__(self):
        self.n_components = 0  # number of components
        self.mixture_probs = []  # mixture probabilities
        self.clt_list = []  # List of Tree Bayesian networks

    def learn(self, dataset, n_components=2, max_iter=50, epsilon=1e-5):
        '''
            Learn Mixtures of Trees using the EM algorithm.
        '''
        self.n_components = n_components
        # For each component and each data point, we have a weight
        weights = np.zeros((n_components, dataset.shape[0]))

        # Randomly initialize the chow-liu trees and the mixture probabilities
        # Your code for random initialization goes here
        self.mixture_probs = np.random.dirichlet(np.ones(self.n_components))

        self.clt_list = []
        for i in range(self.n_components):
            clt = CLT()
            clt.learn(dataset)
            self.clt_list.append(clt)

        current_ll = self.computeLL(dataset)
        for _ in range(max_iter):
            # E-step: Complete the dataset to yield a weighted dataset
            # We store the weights in an array weights[ncomponents,number of points]
            for i in range(self.n_components):
                clt = self.clt_list[i]
                log_likelihoods = clt.computeLL(dataset)
                weights[i] = self.mixture_probs[i] * np.exp(log_likelihoods)

            weights = np.exp(weights - np.max(weights, axis=0))
            # normalize weights
            weights /= np.sum(weights, axis=0)

            # M-step: Update the Chow-Liu Trees and the mixture probabilities
            for i in range(self.n_components):
                self.mixture_probs[i] = np.mean(weights[i])
                self.clt_list[i].update(dataset, weights[i])

            # test for convergence
            ll_new = self.computeLL(dataset)
            if abs(ll_new - current_ll) < epsilon:
                break
            current_ll = ll_new

    def computeLL(self, dataset):
        """
            Compute the log-likelihood score of the dataset
        """
        ll = 0.0

        for i in range(dataset.shape[0]):
            ll += np.log(
                sum(prob * clt.getProb(dataset[i])
                    for prob, clt in zip(self.mixture_probs, self.clt_list)))

        return ll


"""
    After you implement the functions learn and computeLL, you can learn a mixture of trees using
    To learn Chow-Liu trees, you can use
    mix_clt=MIXTURE_CLT()
    ncomponents=10 #number of components
    max_iter=50 #max number of iterations for EM
    epsilon=1e-1 #converge if the difference in the log-likelihods between two iterations is smaller 1e-1
    dataset=Util.load_dataset(path-of-the-file)
    mix_clt.learn(dataset,ncomponents,max_iter,epsilon)
    
    To compute average log likelihood of a dataset w.r.t. the mixture, you can use
    mix_clt.computeLL(dataset)/dataset.shape[0]
"""
