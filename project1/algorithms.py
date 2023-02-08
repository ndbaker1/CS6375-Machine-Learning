import random
from math import log, exp, ceil
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

    def __init__(self, train_x, train_y):
        # train input present
        assert len(train_x) > 0

        mcap_factor = 0.05
        learning_rate = 0.001

        self.feature_ptrs = range(len(train_x[0]))

        # initialize random weights
        random_weight = lambda: (random.random() * 2) - 1
        self.weights = [random_weight()]
        for _ in self.feature_ptrs:
            self.weights.append(random_weight())

        # repeat until convergence
        for t, _ in enumerate(self.weights):
            # MCAP update iterations
            # TODO
            for _ in range(1):
                if t > 1:
                    # error in cases where t = 0?
                    update_term = sum(
                        train_x[k][t - 1] *
                        (train_y[k] - self.sigmoid(self.weights[0] + sum(
                            weight * feature for feature, weight in zip(
                                train_x[k], self.weights[1:]))))
                        for k in range(len(train_x)))
                    mcap_term = mcap_factor * self.weights[t]
                    self.weights[t] += learning_rate * (update_term -
                                                        mcap_term)

    def predict(self, test_x):
        # test input present
        assert len(test_x) > 0

        predictions = list()

        bias, *feature_weights = self.weights

        for features in test_x:
            weighted_features = sum(
                weight * feature
                for feature, weight in zip(features, feature_weights))

            # returns a binary class prediction
            class_prediciton = ceil(bias + weighted_features)
            predictions.append(class_prediciton)

        return predictions

    def sigmoid(self, x):
        y = exp(x)
        return y / (1 + y)
