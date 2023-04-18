import numpy as np
from Util import *
from CLT_class import CLT


class RANDOM_FOREST_CLT():

    def __init__(self):
        self.n_components = 0  # number of components
        self.mixture_probs = []  # mixture probabilities
        self.clt_list = []  # List of Tree Bayesian networks

    def learn(self, dataset, n_components=2, r=0, max_iter=50, epsilon=1e-5):
        '''
            Learn Mixtures of Trees using the EM algorithm.
        '''
        self.n_components = n_components
        # For each component and each data point, we have a weight
        weights = np.zeros((n_components, dataset.shape[0]))

        # Randomly initialize the chow-liu trees and the mixture probabilities
        # Your code for random initialization goes here
        self.mixture_probs = np.random.dirichlet(np.ones(self.n_components))

        self.clt_list = [CLT() for _ in range(self.n_components)]
        for clt in self.clt_list:
            clt.learn(dataset)

        current_ll = -np.inf
        for _ in range(max_iter):
            # E-step: Complete the dataset to yield a weighted dataset
            # We store the weights in an array weights[ncomponents,number of points]
            for i in range(dataset.shape[0]):
                for j in range(self.n_components):
                    datapoint_prob = self.clt_list[j].getProb(dataset[i])
                    weights[j, i] = self.mixture_probs[j] * datapoint_prob
                # normalize weights among dataset for each component
                weights[:, i] = Util.normalize(weights[:, i])

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

        for datapoint in dataset:
            prob_sum = np.sum([
                prob * clt.getProb(datapoint)
                for prob, clt in zip(self.mixture_probs, self.clt_list)
            ])
            ll += np.log(prob_sum)

        return ll
