import random
from math import log, exp
from abc import ABC, abstractmethod
from collections import defaultdict

from sklearn.model_selection import train_test_split


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
                    score[c] += features[t] * log(self.condprob[t][c])

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

    def __init__(
        self,
        train_x,
        train_y,
        iterations=20,
        learning_rate=0.1,
        penalty_factor=None,
    ):
        # train input present
        assert len(train_x) > 0

        if penalty_factor is None:
            # learn λ using a (70/30):(train/validation) split
            performance_params = dict()
            # compute accuracy to the validation set for a set of test parameters
            for λ in (0.0001, 0.001, 0.01, 0.1):
                (t_x, v_x, t_y, v_y) = train_test_split(
                    train_x,
                    train_y,
                    train_size=0.3,
                )

                validation_predictions = LogisticRegressionClassifier(
                    t_x,
                    t_y,
                    iterations,
                    learning_rate,
                    penalty_factor=λ,
                ).predict(v_x)

                correct_predictions = sum(
                    1 for y, Y in zip(validation_predictions, v_y) if y == Y)

                performance_params[λ] = correct_predictions / len(v_y)

            print(performance_params)
            # extract the best performing
            penalty_factor, _ = max(
                performance_params.items(),
                key=lambda score: score[1],
            )

            print("learned", penalty_factor, "as penalty factor.")

        self.feature_ptrs = range(len(train_x[0]))

        # initialize random weights
        random_weight = lambda: random.random() - 0.5
        self.weights = [random_weight()]
        for _ in self.feature_ptrs:
            self.weights.append(random_weight())

        # MCAP update iterations
        for _ in range(iterations):
            # errors for each point in the dataset
            errors = [
                self.weighted_cond_error(features, label)
                for features, label in zip(train_x, train_y)
            ]

            for i, weight in enumerate(self.weights):
                # compute gradient by multiplying errors by the current feature
                if i == 0:
                    gradient = sum(errors)
                else:
                    gradient = sum(features[i - 1] * error
                                   for features, error in zip(train_x, errors))

                # use MCAP lambda to get complexity penalty
                penalty = penalty_factor * weight

                self.weights[i] += learning_rate * (gradient - penalty)

    def weighted_cond_error(self, x, y):
        bias, *feature_weights = self.weights

        weighted_features = bias + sum(
            weight * feature for feature, weight in zip(x, feature_weights))

        error = y - self.sigmoid(weighted_features)

        return error

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
