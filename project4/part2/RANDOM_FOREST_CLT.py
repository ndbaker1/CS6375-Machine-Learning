import numpy as np
from Util import *
from CLT_class import CLT


class RANDOM_FOREST_CLT():

    def __init__(self):
        self.mixture_probs = []
        self.bootstrap_sets = []  # bootstrap samples
        self.clt_list = []  # List of Tree Bayesian networks

    def learn(self, dataset, k, r=0.0, max_iter=50, epsilon=1e-5):
        '''
            Learn Mixtures of Trees using the EM algorithm.
        '''
        weights = np.zeros((k, dataset.shape[0]))

        # Randomly initialize the chow-liu trees and the mixture probabilities
        # Your code for random initialization goes here
        self.mixture_probs = [1 / k for _ in range(k)]

        # Input dataset D; two integers k and r
        # /* Dataset D has N examples */
        # For i =  to k do
        #   D_i = Generate "N" samples with replacement (this is called bootstrap) from the dataset D
        #   Use D_i to construct the complete mutual information graph G (complete graph with edges weighted using mutual information)
        #   G'=Delete r edges randomly from the graph (you can use r as % edges if that helps instead of a number r so that r does not depend on the number of features/variables)
        #   Construct a Chow-Liu tree using G'
        # Return a mixture over k chow-liu trees constructed as above. Set the mixture weights appropriately. (read the project description)
        # For each component and each data point, we have a weight

        for i in range(k):
            D = np.array([
                dataset[np.random.randint(low=0, high=dataset.shape[0])]
                for _ in range(dataset.shape[0])
            ])
            clt = CLT()
            clt.learn(D, r)
            self.clt_list.append(clt)
            self.bootstrap_sets.append(D)

        current_ll = -np.inf
        for _ in range(max_iter):
            # E-step: Complete the dataset to yield a weighted dataset
            # We store the weights in an array weights[ncomponents,number of points]
            for bootstrap in self.bootstrap_sets:
                for i in range(bootstrap.shape[0]):
                    for j in range(k):
                        datapoint_prob = self.clt_list[j].getProb(bootstrap[i])
                        weights[j, i] = self.mixture_probs[j] * datapoint_prob
                    # normalize weights among dataset for each component
                    weights[:, i] = Util.normalize(weights[:, i])

            # M-step: Update the Chow-Liu Trees and the mixture probabilities
            for i in range(k):
                self.mixture_probs[i] = np.mean(weights[i])
                self.clt_list[i].update(self.bootstrap_sets[i], weights[i])

            # test for convergence
            ll_new = np.mean([
                self.computeLL(bootstrap) for bootstrap in self.bootstrap_sets
            ])
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
