from math import log, exp
from abc import ABC, abstractmethod
from collections import defaultdict


class Predictor(ABC):

    @abstractmethod
    def predict(self, test_x):
        pass


class NaiveBayesClassifier(Predictor):

    def __init__(self, train_x, train_y):
        # test input present
        assert len(train_x) > 0

        self.feature_ptrs = range(len(train_x[0]))
        self.prior = dict()
        self.condprob = defaultdict(dict)
        self.classes = set(train_y)

        for c in self.classes:
            class_samples = list(sample for i, sample in enumerate(train_x)
                                 if train_y[i] == c)
            self.prior[c] = len(class_samples)

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

        for x in test_x:
            score = dict()
            for c in self.classes:
                score[c] = log(self.prior[c])
                for t in self.feature_ptrs:
                    # only account for the feature if it exists in the document
                    for _ in range(x[t]):
                        score[c] += log(self.condprob[t][c])

            # add highest class to prediction
            class_prediction, _ = max(
                score.items(),
                key=lambda score: score[1],
            )

            predictions.append(class_prediction)

        return predictions


class LogisticRegressionClassifier(Predictor):

    def __init__(self, train_x, train_y):

        # repeat until convergence
        for _ in range(100):
            pass

        pass

    def predict(self, test_x):
        # test input present
        assert len(test_x) > 0

        # w0+n∑i=1wixi>0

        return (0 for _ in test_x)

    def sigmoid(self, x):
        return 1 / (1 + exp(x))
