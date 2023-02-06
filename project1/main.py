import os
import algorithms, models


def test_accuracy(predictor, test_set):
    return 1


def parse_dataset_records(dataset_dir):
    train_set_x, train_set_y = list(), list()
    test_set_x, test_set_y = list(), list()

    for test_file in (x.path for x in os.scandir(dataset_dir + "/test/ham")):
        with open(test_file) as f:
            test_set_x.append(f.read())
            test_set_y.append(0)

    for test_file in (x.path for x in os.scandir(dataset_dir + "/test/spam")):
        with open(test_file) as f:
            print(test_file)
            test_set_x.append(f.read())
            test_set_y.append(1)

    for train_file in (x.path for x in os.scandir(dataset_dir + "/train/ham")):
        with open(train_file) as f:
            train_set_x.append(f.read())
            train_set_y.append(0)

    for train_file in (x.path
                       for x in os.scandir(dataset_dir + "/train/spam")):
        with open(train_file) as f:
            train_set_x.append(f.read())
            train_set_y.append(1)

    return (
        (train_set_x, train_set_y),
        (test_set_x, train_set_y),
    )


# Gather statistics for each dataset based on the classification Algorithm
for dataset_dir in (d.path for d in os.scandir("./datasets") if d.is_dir()):

    print("<{}> dataset:".format(dataset_dir))

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
    for model, (train_set, test_set) in [data_models["Bernoulli"]]:
        print("Naive Bayes <Bernoulli> Accuracy: {}".format(
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
        # print("SGD Classifier <{}> Accruacy: {}".format(
        #     model,
        #     test_accuracy(
        #         # TODO - SGDClassifier
        #         algorithms.LogisticRegression(bernoulli_test_set),
        #         test_set,
        #     ),
        # ))
        pass
