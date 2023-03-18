if __name__ == "__main__":
    from sklearn.tree import DecisionTreeClassifier
    from sklearn.ensemble import BaggingClassifier, RandomForestClassifier, GradientBoostingClassifier
    from sklearn.model_selection import GridSearchCV
    from sklearn.metrics import accuracy_score, f1_score

    from collections import defaultdict
    import os

    DATA_DIR = "./all_data"

    datasets = defaultdict(dict)

    for dataset_path in os.listdir(DATA_DIR):
        group, name = dataset_path.split('_', maxsplit=1)
        X_valid, Y_valid = [], []

        with open(DATA_DIR + "/" + dataset_path) as dataset_file:
            for data in dataset_file.read().splitlines():
                *features, classification = list(map(int, data.split(',')))
                X_valid.append(features)
                Y_valid.append(classification)

        datasets[name][group] = (X_valid, Y_valid)

    # 1. (15 points) Use the sklearn.tree.DecisionTreeClassifier on the 15 datasets.
    # Use the validation set to tune the parameters (see the documentation for parameters; e.g., criterion, splitter, max depth, etc.).
    # After tuning the parameters, mix the training and validation sets, relearn the decision tree
    # using the “best parameter settings found via tuning” and report the accuracy and F1 score on the test set.
    # For each dataset, also report the “best parameter settings found via tuning.”
    search_model = GridSearchCV(
        DecisionTreeClassifier(),
        {
            "criterion": ["gini", "entropy", "log_loss"],
            "splitter": ["best", "random"],
            "max_depth": [5, 20, None],
            "max_features": ["sqrt", "log2", None],
        },
    )

    for dataset_key, dataset in datasets.items():
        print(f"DecisionTreeClassifier {dataset_key}")

        X_valid, Y_valid = dataset["valid"]
        X_train, Y_train = dataset["train"]
        X_test, Y_test = dataset["test"]

        search_model.fit(X_valid, Y_valid)

        print(f"parameters: {search_model.best_params_}")

        model = DecisionTreeClassifier(**search_model.best_params_)
        model.fit([*X_valid, *X_train], [*Y_valid, *Y_train])

        predictions = model.predict(X_test)

        accuracy = accuracy_score(Y_test, predictions)
        f1 = f1_score(Y_test, predictions)

        print(f"accuracy: {accuracy}, f1: {f1}")

    # 2. (15 points) Repeat the experiment described above using:
    # sklearn.ensemble.BaggingClassifier with “DecisionTreeClassifier” as the base estimator.
    # Again, use the validation set to tune the parameters, mix training and validation after tuning to learn a new classifier and report
    # (a) Best parameter settings after tuning
    # (b) Classification accuracy and F1 score.
    search_model = GridSearchCV(
        BaggingClassifier(),
        {
            "n_estimators": [5, 10, 20],
            "max_samples": [10, 0.5, 1.0],
            "max_features": [10, 0.5, 1.0],
        },
    )

    for dataset_key, dataset in datasets.items():
        print(f"BaggingClassifier {dataset_key}")

        X_valid, Y_valid = dataset["valid"]
        X_train, Y_train = dataset["train"]
        X_test, Y_test = dataset["test"]

        search_model.fit(X_valid, Y_valid)

        print(f"parameters: {search_model.best_params_}")

        model = BaggingClassifier(**search_model.best_params_, )
        model.fit([*X_valid, *X_train], [*Y_valid, *Y_train])

        predictions = model.predict(X_test)

        accuracy = accuracy_score(Y_test, predictions)
        f1 = f1_score(Y_test, predictions)

        print(f"accuracy: {accuracy}, f1: {f1}")

    # 3. (15 points) Repeat the experiment described above using:
    # sklearn.ensemble.RandomForestClassifier.
    search_model = GridSearchCV(
        RandomForestClassifier(),
        {
            "n_estimators": [5, 10, 20],
            "criterion": ["gini", "entropy", "log_loss"],
            "max_depth": [5, 20, None],
            "max_features": ["sqrt", "log2", None],
            "min_samples_split": [2, 0.1, 0.5],
        },
    )

    for dataset_key, dataset in datasets.items():
        print(f"RandomForestClassifier {dataset_key}")

        X_valid, Y_valid = dataset["valid"]
        X_train, Y_train = dataset["train"]
        X_test, Y_test = dataset["test"]

        search_model.fit(X_valid, Y_valid)

        print(f"parameters: {search_model.best_params_}")

        model = RandomForestClassifier(**search_model.best_params_)
        model.fit([*X_valid, *X_train], [*Y_valid, *Y_train])

        predictions = model.predict(X_test)

        accuracy = accuracy_score(Y_test, predictions)
        f1 = f1_score(Y_test, predictions)

        print(f"accuracy: {accuracy}, f1: {f1}")

    # 4. (15 points) Repeat the experiment described above using:
    # sklearn.ensemble.GradientBoostingClassifier.
    search_model = GridSearchCV(
        GradientBoostingClassifier(),
        {
            "n_estimators": [5, 10, 20],
            "loss": ['log_loss', 'exponential'],
            "criterion": ['friedman_mse', 'squared_error'],
            "max_features": [1.0, 'sqrt', 'log2'],
            "learning_rate": [0.1, 0.5, 2],
            "subsample": [0.05, 0.2, 1.0],
        },
    )

    for dataset_key, dataset in datasets.items():
        print(f"GradientBoostingClassifier {dataset_key}")

        X_valid, Y_valid = dataset["valid"]
        X_train, Y_train = dataset["train"]
        X_test, Y_test = dataset["test"]

        search_model.fit(X_valid, Y_valid)

        print(f"parameters: {search_model.best_params_}")

        model = GradientBoostingClassifier(**search_model.best_params_)
        model.fit([*X_valid, *X_train], [*Y_valid, *Y_train])

        predictions = model.predict(X_test)

        accuracy = accuracy_score(Y_test, predictions)
        f1 = f1_score(Y_test, predictions)

        print(f"accuracy: {accuracy}, f1: {f1}")
