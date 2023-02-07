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
    # Use Classifier to predict the results
    predictions = predictor.predict(test_input)
    correct_predictions = filter(
        lambda v: v[0] == v[1],
        zip(predictions, test_expected),
    )

    return sum(1 for _ in correct_predictions) / len(test_input)


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

    binary_vectorizer = CountVectorizer(stop_words="english", binary=True)

    data_models["Bernoulli"] = (
        binary_vectorizer.fit_transform(train_x).toarray(),
        binary_vectorizer.transform(test_x).toarray(),
    )

    count_vectorizer = CountVectorizer(stop_words="english")

    data_models["BagOfWords"] = (
        count_vectorizer.fit_transform(train_x).toarray(),
        count_vectorizer.transform(test_x).toarray(),
    )

    # Naive Bayes
    for (train_set, test_set) in [data_models["BagOfWords"]]:
        print("Naive Bayes <BagOfWords> Accuracy: {}".format(
            test_accuracy(
                algorithms.NaiveBayesClassifier(train_set, train_y),
                test_set,
                test_y,
            )))

    # MCAP Logistic Regression
    for model, (train_set, test_set) in data_models.items():
        print("MCAP Logistic Regression <{}> Accuracy: {}".format(
            model,
            test_accuracy(
                algorithms.LogisticRegressionClassifier(train_set, train_y),
                test_set,
                test_y,
            ),
        ))

    # SGDClassifier
    for model, (train_set, test_set) in data_models.items():
        print("SGD Classifier <{}> Accruacy: {}".format(
            model,
            test_accuracy(
                SGDClassifier().fit(train_set, train_y),
                test_set,
                test_y,
            ),
        ))
