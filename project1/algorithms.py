from abc import ABC, abstractmethod


class Predictor(ABC):

    @abstractmethod
    def predict(self, test_x):
        pass


class NaiveBayesClassifier(Predictor):

    def __init__(self, train_x, train_y):
        # test input present
        assert len(train_x) > 0

        training_samples = len(train_x)
        training_features = len(train_x[0])

        # probabilities for each feature (multinomial)
        self.feature_p_set = [0.0] * training_features

        # compute the weight that each word has per document
        for feature_vec, class_label in zip(train_x, train_y):
            # find document word count
            total_feature_count = sum(feature_vec)
            # if the class falls into "p", then add the weight.
            # this will innately give us "1-p" as well.
            if class_label == 1:  # 1 = spam, 0 = ham
                for feature_index, feature in enumerate(feature_vec):
                    # normalize by document word count
                    normalized_weight = feature / total_feature_count
                    self.feature_p_set[feature_index] += normalized_weight

        # normalize by training set size
        for feature_index in range(training_features):
            self.feature_p_set[feature_index] /= training_samples

    def predict(self, test_x):
        # test input present
        assert len(test_x) > 0
        # test input feature size if the same as the trained model
        assert len(test_x[0]) == len(self.feature_p_set)

        # return a prediction for each feature set in the test input
        predictions = list()

        for feature_vec in test_x:
            # Which is higher: P(y = spam | x_1 = #, x_2 = #, ...) or P(y = ham | x_1 = #, x_2 = #, ...)
            class_probability = 1
            # use the input dataset to compute likelihood of each class
            for feature_index, feature in enumerate(feature_vec):
                # fetch parameter probability from model
                feature_p = self.feature_p_set[feature_index]
                feature_p_not = 1 - self.feature_p_set[feature_index]
                # compute cumulative probability over next feature parameter
                class_probability *= feature_p if feature > 0 else feature_p_not

            # break the probability at 0.5, because there are only 2 classes
            predictions.append(1 if class_probability >= 0.5 else 0)

        return predictions


class LogisticRegressionClassifier(Predictor):

    def __init__(self, train_x, train_y):
        pass

    def predict(self, test_x):
        # test input present
        assert len(test_x) > 0

        return (0 for _ in test_x)
