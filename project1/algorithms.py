import random
from math import log, exp
from abc import ABC, abstractmethod
from collections import defaultdict


class Predictor(ABC):

    @abstractmethod
    def predict(self, test_x):
        pass


class MultinomialNBClassifier(Predictor):

    def __init__(self, train_x, train_y):
        # train input present
        assert len(train_x) > 0

        self.feature_ptrs = range(len(train_x[0]))
        self.prior = dict()
        self.condprob = defaultdict(dict)
        self.classes = set(train_y)

        for c in self.classes:
            class_samples = list(sample for i, sample in enumerate(train_x)
                                 if train_y[i] == c)
            self.prior[c] = len(class_samples) / len(train_y)

            T = defaultdict(int)
            for t in self.feature_ptrs:
                T[t] += sum(x[t] for x in class_samples)
            for t in self.feature_ptrs:
                numer = T[t] + 1
                denom = sum(T[t] + 1 for t in self.feature_ptrs)
                self.condprob[t][c] = numer / denom

    def predict(self, test_x):
        # test input feature size if the same as the trained model
        assert len(test_x[0]) == len(self.condprob)

        # return a prediction for each feature set in the test input
        predictions = list()

        for features in test_x:
            score = dict()
            for c in self.classes:
                score[c] = log(self.prior[c])
                for t in self.feature_ptrs:
                    # only account for the feature if it exists in the document
                    for _ in range(features[t]):
                        score[c] += log(self.condprob[t][c])

            # add highest class to prediction
            class_prediction, _ = max(
                score.items(),
                key=lambda score: score[1],
            )

            predictions.append(class_prediction)

        return predictions


class DiscreteNBClassifier(Predictor):

    def __init__(self, train_x, train_y):
        # train input present
        assert len(train_x) > 0

        self.feature_ptrs = range(len(train_x[0]))
        self.prior = dict()
        self.condprob = defaultdict(dict)
        self.classes = set(train_y)

        for c in self.classes:
            class_samples = list(sample for i, sample in enumerate(train_x)
                                 if train_y[i] == c)
            self.prior[c] = len(class_samples) / len(train_y)

            for t in self.feature_ptrs:
                numer = sum(x[t] for x in class_samples) + 1
                denom = len(class_samples) + 2
                self.condprob[t][c] = numer / denom

    def predict(self, test_x):
        # test input feature size if the same as the trained model
        assert len(test_x[0]) == len(self.condprob)

        # return a prediction for each feature set in the test input
        predictions = list()

        for features in test_x:
            score = dict()
            for c in self.classes:
                score[c] = log(self.prior[c])
                for t in self.feature_ptrs:
                    # only account for the feature if it exists in the document
                    if features[t] == 1:
                        score[c] += log(self.condprob[t][c])
                    else:
                        score[c] += log(1 - self.condprob[t][c])

            # add highest class to prediction
            class_prediction, _ = max(
                score.items(),
                key=lambda score: score[1],
            )

            predictions.append(class_prediction)

        return predictions


class LogisticRegressionClassifier(Predictor):

    def __init__(self, train_x, train_y, iterations=5, learning_rate=0.08):
        # train input present
        assert len(train_x) > 0

        # TODO - learn this somehow using a 70/30 train/validation split
        mcap_factor = 0.01

        self.feature_ptrs = range(len(train_x[0]))

        # initialize random weights
        random_weight = lambda: random.random() - 0.5
        self.weights = [random_weight()]
        for _ in self.feature_ptrs:
            self.weights.append(random_weight())

        # MCAP update iterations
        for _ in range(iterations):
            # compute gradients for all features to use within the
            # weight loop for gradient ascent
            gradients = [
                self.feature_gradient(features, label)
                for features, label in zip(train_x, train_y)
            ]
            for t, _ in enumerate(self.weights):
                if t >= 1:  # ignore bias
                    # sum gradients along a feature axis
                    update_term = sum(grad[t - 1] for grad in gradients)
                    # use MCAP lambda
                    mcap_term = mcap_factor * self.weights[t]

                    self.weights[t] += learning_rate * (update_term -
                                                        mcap_term)

    def feature_gradient(self, X, Y):
        bias, *feature_weights = self.weights

        weighted_features = bias + sum(
            weight * feature for feature, weight in zip(X, feature_weights))

        error = Y - self.sigmoid(weighted_features)

        return [feature * error for feature in X]

    def predict(self, test_x):
        # test input present
        assert len(test_x) > 0

        predictions = list()

        bias, *feature_weights = self.weights

        for features in test_x:
            weighted_features = bias + sum(
                weight * feature
                for feature, weight in zip(features, feature_weights))

            # returns a binary class prediction
            class_prediciton = 1 if weighted_features > 0 else 0
            predictions.append(class_prediciton)

        return predictions

    def sigmoid(self, x):
        # always use the variation which results in small numbers
        return exp(x) / (1 + exp(x)) if x < 0 else 1 / (1 + exp(-x))
