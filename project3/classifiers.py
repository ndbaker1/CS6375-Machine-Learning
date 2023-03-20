if __name__ == "__main__":
    from sklearn.tree import DecisionTreeClassifier
    from sklearn.ensemble import BaggingClassifier, RandomForestClassifier, GradientBoostingClassifier
    from sklearn.model_selection import GridSearchCV
    from sklearn.metrics import accuracy_score, f1_score

    from common import *

    from collections import defaultdict
    import os

    DATA_DIR = "./all_data"

    datasets = defaultdict(dict)

    for dataset_path in os.listdir(DATA_DIR):
        group, name = dataset_path.split('_', maxsplit=1)
        X_valid, y_valid = [], []

        with open(DATA_DIR + "/" + dataset_path) as dataset_file:
            for data in dataset_file.read().splitlines():
                *features, classification = list(map(int, data.split(',')))
                X_valid.append(features)
                y_valid.append(classification)

        datasets[name][group] = (X_valid, y_valid)

    # 1. (15 points) Use the sklearn.tree.DecisionTreeClassifier on the 15 datasets.
    # Use the validation set to tune the parameters (see the documentation for parameters; e.g., criterion, splitter, max depth, etc.).
    # After tuning the parameters, mix the training and validation sets, relearn the decision tree
    # using the “best parameter settings found via tuning” and report the accuracy and F1 score on the test set.
    # For each dataset, also report the “best parameter settings found via tuning.”
    search_model = GridSearchCV(
        DecisionTreeClassifier(),
        DECISION_TREE_CLASSIFIER_PARAMS,
    )

    for dataset_key, dataset in datasets.items():
        X_valid, y_valid = dataset["valid"]
        X_train, y_train = dataset["train"]
        X_test, y_test = dataset["test"]

        search_model.fit(X_valid, y_valid)

        model = DecisionTreeClassifier(**search_model.best_params_)
        model.fit([*X_valid, *X_train], [*y_valid, *y_train])

        predictions = model.predict(X_test)

        accuracy = accuracy_score(y_test, predictions)
        f1 = f1_score(y_test, predictions)

        print_table(
            "DecisionTreeClassifier",
            dataset_key,
            search_model.best_params_,
            accuracy,
            f1,
        )

    # 2. (15 points) Repeat the experiment described above using:
    # sklearn.ensemble.BaggingClassifier with “DecisionTreeClassifier” as the base estimator.
    # Again, use the validation set to tune the parameters, mix training and validation after tuning to learn a new classifier and report
    # (a) Best parameter settings after tuning
    # (b) Classification accuracy and F1 score.
    search_model = GridSearchCV(
        BaggingClassifier(),
        BAGGING_CLASSIFIER_PARAMS,
    )

    for dataset_key, dataset in datasets.items():

        X_valid, y_valid = dataset["valid"]
        X_train, y_train = dataset["train"]
        X_test, y_test = dataset["test"]

        search_model.fit(X_valid, y_valid)

        model = BaggingClassifier(**search_model.best_params_, )
        model.fit([*X_valid, *X_train], [*y_valid, *y_train])

        predictions = model.predict(X_test)

        accuracy = accuracy_score(y_test, predictions)
        f1 = f1_score(y_test, predictions)

        print_table(
            "BaggingClassifier",
            dataset_key,
            search_model.best_params_,
            accuracy,
            f1,
        )

    # 3. (15 points) Repeat the experiment described above using:
    # sklearn.ensemble.RandomForestClassifier.
    search_model = GridSearchCV(
        RandomForestClassifier(),
        RANDOM_FOREST_CLASSIFIER_PARAMS,
    )

    for dataset_key, dataset in datasets.items():
        X_valid, y_valid = dataset["valid"]
        X_train, y_train = dataset["train"]
        X_test, y_test = dataset["test"]

        search_model.fit(X_valid, y_valid)

        model = RandomForestClassifier(**search_model.best_params_)
        model.fit([*X_valid, *X_train], [*y_valid, *y_train])

        predictions = model.predict(X_test)

        accuracy = accuracy_score(y_test, predictions)
        f1 = f1_score(y_test, predictions)

        print_table(
            "RandomForestClassifier",
            dataset_key,
            search_model.best_params_,
            accuracy,
            f1,
        )

    # 4. (15 points) Repeat the experiment described above using:
    # sklearn.ensemble.GradientBoostingClassifier.
    search_model = GridSearchCV(
        GradientBoostingClassifier(),
        GRADIENT_BOOSTING_CLASSIFIER_PARAMS,
    )

    for dataset_key, dataset in datasets.items():
        X_valid, y_valid = dataset["valid"]
        X_train, y_train = dataset["train"]
        X_test, y_test = dataset["test"]

        search_model.fit(X_valid, y_valid)

        model = GradientBoostingClassifier(**search_model.best_params_)
        model.fit([*X_valid, *X_train], [*y_valid, *y_train])

        predictions = model.predict(X_test)

        accuracy = accuracy_score(y_test, predictions)
        f1 = f1_score(y_test, predictions)

        print_table(
            "GradientBoostingClassifier",
            dataset_key,
            search_model.best_params_,
            accuracy,
            f1,
        )
