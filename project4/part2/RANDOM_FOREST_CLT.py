import numpy as np
from Util import *
from CLT_class import CLT


class RANDOM_FOREST_CLT():

    def __init__(self):
        self.mixture_probs = []
        self.clt_list = []  # List of Tree Bayesian networks

    def learn(self, dataset, k, r=0.0):
        '''
            Learn Mixtures of Trees using a Random Forest approach.
        '''
        # evenly weighted mixture coefficients
        self.mixture_probs = [1 / k for _ in range(k)]

        # Input dataset D; two integers k and r
        # - Dataset D has N examples
        #
        # For i =  to k do
        #   D_i = Generate "N" samples with replacement (this is called bootstrap) from the dataset D
        #   Use D_i to construct the complete mutual information graph G (complete graph with edges weighted using mutual information)
        #   G'=Delete r edges randomly from the graph (you can use r as % edges if that helps instead of a number r so that r does not depend on the number of features/variables)
        #   Construct a Chow-Liu tree using G'
        #
        # Return a mixture over k chow-liu trees constructed as above.

        for _ in range(k):
            D = np.array([
                dataset[np.random.randint(low=0, high=dataset.shape[0])]
                for _ in range(dataset.shape[0])
            ])
            clt = CLT()
            clt.learn(D, r)
            self.clt_list.append(clt)

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
