import os
import algorithms

from sklearn.linear_model import SGDClassifier
from sklearn.feature_extraction.text import CountVectorizer


def parse_dataset_records(dataset_dir):

    dataset_encoding = "unicode_escape"

    train_set_x, train_set_y = list(), list()
    test_set_x, test_set_y = list(), list()

    for test_file in (x.path for x in os.scandir(dataset_dir + "/test/ham")):
        with open(test_file, encoding=dataset_encoding) as test_data:
            test_set_x.append(test_data.read())
            test_set_y.append(0)

    for test_file in (x.path for x in os.scandir(dataset_dir + "/test/spam")):
        with open(test_file, encoding=dataset_encoding) as test_data:
            test_set_x.append(test_data.read())
            test_set_y.append(1)

    for train_file in (x.path for x in os.scandir(dataset_dir + "/train/ham")):
        with open(train_file, encoding=dataset_encoding) as train_data:
            train_set_x.append(train_data.read())
            train_set_y.append(0)

    for train_file in (x.path
                       for x in os.scandir(dataset_dir + "/train/spam")):
        with open(train_file, encoding=dataset_encoding) as train_data:
            train_set_x.append(train_data.read())
            train_set_y.append(1)

    return (
        (train_set_x, train_set_y),
        (test_set_x, test_set_y),
    )


def test_accuracy(predictor, test_input, test_expected):
    ''' Use Classifier `.predict` and compute metrics with the labelled dataset '''

    predictions = predictor.predict(test_input)
    correct_predictions = [
        pre for pre, act in zip(predictions, test_expected) if pre == act
    ]

    predicted_positive = sum(1 for c in predictions if c == 1)
    actual_positive = sum(1 for c in test_expected if c == 1)
    true_positive = sum(1 for c in correct_predictions if c == 1)

    accuracy = len(correct_predictions) / len(test_input)
    precision = 0 if predicted_positive == 0 else true_positive / predicted_positive
    recall = 0 if actual_positive == 0 else true_positive / actual_positive
    f1_score = 0 if (
        precision +
        recall) == 0 else 2 * (precision * recall) / (precision + recall)

    print("Accuracy: {}, Precision: {}, Recall: {}, F1: {}".format(
        accuracy,
        precision,
        recall,
        f1_score,
    ))


def classifier_runner(classifier_name, classifier_constructor, data_models):
    for model, (train_set, test_set) in data_models.items():
        print("{} <{}>".format(classifier_name, model))
        test_accuracy(
            classifier_constructor(train_set, train_y),
            test_set,
            test_y,
        )


def top_k_vectorizer(vectorizer, train_data, k=100):
    train_vectorized = vectorizer.fit_transform(train_data)
    words_sorted = sorted(((word, train_vectorized.sum(axis=0)[0, idx])
                           for word, idx in vectorizer.vocabulary_.items()),
                          key=lambda x: x[1],
                          reverse=True)

    top_k = (word for word, _ in words_sorted[:k])
    vectorizer.fit(top_k)
    return vectorizer


from sys import argv
# allow custom paths for dataset.
# not required
dataset_parent = "./datasets" if len(argv) <= 1 else argv[1]

# Gather statistics for each dataset based on the classification Algorithm
for dataset_dir in (d.path for d in os.scandir(dataset_parent) if d.is_dir()):

    print("\nDataset: <{}>".format(dataset_dir))

    # get train/test sets from dataset files
    (train_x, train_y), (test_x, test_y) = parse_dataset_records(dataset_dir)

    # Load datasets into model types
    data_models = dict()

    # Compute BagOfWords and Bernoulli Model of Dataset using sklearn CountVectorizer.
    #
    # The Vectorizer is fit to the training data before transforming the
    # test data, that way both train and test features will be the same
    # for running a classifier which takes a fixes set of features.
    #
    # In the case of Bernoulli, the CountVectorizer must be set to 'binary',
    # which will output 1 if a feature is present, regardless of frequency.

    binary_vectorizer = top_k_vectorizer(
        CountVectorizer(stop_words="english", binary=True),
        train_data=train_x,
        k=200,
    )

    data_models["Bernoulli"] = (
        binary_vectorizer.transform(train_x).toarray(),
        binary_vectorizer.transform(test_x).toarray(),
    )

    bow_vectorizer = top_k_vectorizer(
        CountVectorizer(stop_words="english"),
        train_data=train_x,
        k=200,
    )

    data_models["BagOfWords"] = (
        bow_vectorizer.transform(train_x).toarray(),
        bow_vectorizer.transform(test_x).toarray(),
    )

    # Naive Bayes Multinomial
    classifier_runner(
        "Multinomial Naive Bayes",
        algorithms.MultinomialNBClassifier,
        {"BagOfWords": data_models["BagOfWords"]},
    )

    # Naive Bayes Discrete
    classifier_runner(
        "Discrete Naive Bayes",
        algorithms.DiscreteNBClassifier,
        {"Bernoulli": data_models["Bernoulli"]},
    )

    # MCAP Logistic Regression
    classifier_runner(
        "MCAP Logistic Regression",
        algorithms.LogisticRegressionClassifier,
        data_models,
    )

    # SGDClassifier
    classifier_runner(
        "SGD Classifier",
        SGDClassifier().fit,
        data_models,
    )
