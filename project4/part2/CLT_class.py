"""
Define the Chow_liu Tree class
"""

import sys
import time

import numpy as np
from Util import *
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import minimum_spanning_tree
from scipy.sparse.csgraph import depth_first_order


class CLT:
    '''
    Class Chow-Liu Tree.
    Members:
        nvariables: Number of variables
        xycounts: 
            Sufficient statistics: counts of value assignments to all pairs of variables
            Four dimensional array: first two dimensions are variable indexes
            last two dimensions are value indexes 00,01,10,11
        xcounts:
            Sufficient statistics: counts of value assignments to each variable
            First dimension is variable, second dimension is value index [0][1]
        xyprob:
            xycounts converted to probabilities by normalizing them
        xprob:
            xcounts converted to probabilities by normalizing them
        topo_order:
            Topological ordering over the variables
        parents:
            Parent of each node. Parent[i] gives index of parent of variable indexed by i
            If Parent[i]=-9999 then i is the root node
    '''

    def __init__(self):
        self.nvariables = 0
        self.xycounts = np.ones((1, 1, 2, 2), dtype=int)
        self.xcounts = np.ones((1, 2), dtype=int)
        self.xyprob = np.zeros((1, 1, 2, 2))
        self.xprob = np.zeros((1, 2))
        self.topo_order = []
        self.parents = []

    def learn(self, dataset, r=0.0):
        '''
            Learn the structure of the Chow-Liu Tree using the given dataset
        '''
        self.nvariables = dataset.shape[1]
        self.xycounts = Util.compute_xycounts(dataset) + 1  # laplace correction
        self.xcounts = Util.compute_xcounts(dataset) + 2  # laplace correction
        self.xyprob = Util.normalize2d(self.xycounts)
        self.xprob = Util.normalize1d(self.xcounts)
        # compute mutual information score for all pairs of variables
        # weights are multiplied by -1.0 because we compute the minimum spanning tree
        edgemat = Util.compute_MI_prob(self.xyprob, self.xprob) * (-1.0)
        if r > 0.0:
            for _ in range(int(len(edgemat) * r)):
                (x,y) = (
                    np.random.randint(len(edgemat)),
                    np.random.randint(len(edgemat[0])),
                )
                edgemat[x,y] = 0.0

        edgemat[edgemat == 0.0] = 1e-20  # to avoid the case where the tree is not connected
        # compute the minimum spanning tree
        Tree = minimum_spanning_tree(csr_matrix(edgemat))
        # Convert the spanning tree to a Bayesian network
        self.topo_order, self.parents = depth_first_order(Tree, 0, directed=False)

    def update(self, dataset, weights):
        '''
            Update the Chow-Liu Tree using weighted samples.
            Note that we assume that weight of each sample >0. 
            Important function for performing updates when running EM
        '''
        # Update the Chow-Liu Tree based on a weighted dataset
        # assume that dataset_.shape[0] equals weights.shape[0] because we assume each example has a weight
        if not np.all(weights):
            print('Error: Weight of an example in the dataset is zero')
            sys.exit(-1)
        if weights.shape[0] == dataset.shape[0]:
            # Weighted laplace correction, note that num-examples=dataset.shape[0]
            # If 1-laplace smoothing is applied to num-examples,
            # then laplace smoothing applied to "sum-weights" equals "sum-weights/num-examples"
            smooth = np.sum(weights) / dataset.shape[0]
            self.xycounts = Util.compute_weighted_xycounts(dataset, weights) + smooth
            self.xcounts = Util.compute_weighted_xcounts(dataset, weights) + 2.0 * smooth
        else:
            print('Error: Each example must have a weight')
            sys.exit(-1)
        self.xyprob = Util.normalize2d(self.xycounts)
        self.xprob = Util.normalize1d(self.xcounts)
        edgemat = Util.compute_MI_prob(self.xycounts, self.xcounts) * (-1.0)
        Tree = minimum_spanning_tree(csr_matrix(edgemat))
        self.topo_order, self.parents = depth_first_order(Tree, 0, directed=False)

    def computeLL(self, dataset):
        '''
            Compute the Log-likelihood score of the dataset
        '''
        ll = 0.0
        for i in range(dataset.shape[0]):
            for x in self.topo_order:
                assignx = dataset[i, x]
                # if root sample from marginal
                if self.parents[x] == -9999:
                    ll += np.log(self.xprob[x][assignx])
                else:
                    # sample from p(x|y)
                    y = self.parents[x]
                    assigny = dataset[i, y]
                    ll += np.log(self.xyprob[x, y, assignx, assigny] / self.xprob[y, assigny])
        return ll

    def getProb(self, sample):
        prob = 1.0
        for x in self.topo_order:
            assignx = sample[x]
            # if root sample from marginal
            if self.parents[x] == -9999:
                prob *= self.xprob[x][assignx]
            else:
                # sample from p(x|y)
                y = self.parents[x]
                assigny = sample[y]
                prob *= self.xyprob[x, y, assignx, assigny] / self.xprob[y, assigny]
        return prob


'''
    You can read the dataset using
    dataset=Util.load_dataset(path-of-the-file)
    
    To learn Chow-Liu trees, you can use
    clt=CLT()
    clt.learn(dataset)
    
    To compute average log likelihood of a dataset, you can use
    clt.computeLL(dataset)/dataset.shape[0]
'''
