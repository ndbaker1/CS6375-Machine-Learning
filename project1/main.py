import os
import algorithms
import models

import sklearn


def test_accuracy(predictor, test_input, test_expected):

    accuracy = 0

    # Use Classifier to predict the results
    for features, expected in zip(test_input, test_expected):
        actual = predictor.predict(features)
        if actual == expected:
            accuracy += 1

    return accuracy / len(test_input)


def parse_dataset_records(dataset_dir):

    dataset_encoding = "utf-8"

    train_set_x, train_set_y = list(), list()
    test_set_x, test_set_y = list(), list()

    for test_file in (x.path for x in os.scandir(dataset_dir + "/test/ham")):
        with open(test_file, encoding=dataset_encoding) as test_data:
            test_set_x.append(test_data.read())
            test_set_y.append(0)

    for test_file in (x.path for x in os.scandir(dataset_dir + "/test/spam")):
        with open(test_file, encoding=dataset_encoding) as test_data:
            print(test_file)
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
        (test_set_x, train_set_y),
    )


# Gather statistics for each dataset based on the classification Algorithm
for dataset_dir in (d.path for d in os.scandir("./datasets") if d.is_dir()):

    print("dataset: <{}>".format(dataset_dir))

    # get train/test sets from dataset files
    (train_x, train_y), (test_x, test_y) = parse_dataset_records(dataset_dir)

    # Load datasets into model types
    data_models = {
        "Bernoulli": (
            models.bernoulli(train_x),
            models.bernoulli(test_x),
        ),
        "BagOfWords": (
            models.bag_of_words(train_x),
            models.bag_of_words(test_x),
        ),
    }

    print(data_models)
    input("pause..")

    # Naive Bayes
    for model, (train_set, test_set) in [data_models["BagOfWords"]]:
        print("Naive Bayes <BagOfWords> Accuracy: {}".format(
            test_accuracy(
                algorithms.NaiveBayes(train_set, train_y),
                test_set,
                test_y,
            )))

    # MCAP Logistic Regression
    for model, (train_set, test_set) in data_models.items():
        print("MCAP Logistic Regression <{}> Accuracy: {}".format(
            model,
            test_accuracy(
                algorithms.LogisticRegression(train_set, train_x),
                test_set,
                test_y,
            ),
        ))

    # SGDClassifier
    for model, (train_set, test_set) in data_models.items():
        print("SGD Classifier <{}> Accruacy: {}".format(
            model,
            test_accuracy(
                sklearn.linear_model.SGDClassifier().fit(train_set, train_y),
                test_set,
                test_y,
            ),
        ))
