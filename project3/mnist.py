if __name__ == "__main__":
    from sklearn.tree import DecisionTreeClassifier
    from sklearn.ensemble import BaggingClassifier, RandomForestClassifier, GradientBoostingClassifier
    from sklearn.model_selection import GridSearchCV
    from sklearn.metrics import accuracy_score, f1_score

    # 6. (15 points) Evaluate the four tree and ensemble classifiers you used above on
    # the MNIST dataset from Project 2 (do not compute F1 scores on MNIST, just classification accuracy).
    from sklearn.datasets import fetch_openml
    # Load data from https://www.openml.org/d/554
    X, y = fetch_openml('mnist_784',
                        version=1,
                        return_X_y=True,
                        as_frame=False)
    y = [int(y) for y in y]

    X_valid, X_train, X_test = X[:10000], X[10000:60000], X[60000:]
    y_valid, y_train, y_test = y[:10000], y[10000:60000], y[60000:]

    search_model = GridSearchCV(
        DecisionTreeClassifier(),
        {
            "criterion": ["gini", "entropy", "log_loss"],
            "splitter": ["best", "random"],
            "max_depth": [5, 20, None],
            "max_features": ["sqrt", "log2", None],
        },
    )

    print(f"DecisionTreeClassifier mnist")

    search_model.fit(X_valid, y_valid)

    print(f"parameters: {search_model.best_params_}")

    model = DecisionTreeClassifier(**search_model.best_params_)
    model.fit([*X_valid, *X_train], [*y_valid, *y_train])

    predictions = model.predict(X_test)

    accuracy = accuracy_score(y_test, predictions)

    print(f"accuracy: {accuracy}")

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

    print(f"BaggingClassifier mnist")

    search_model.fit(X_valid, y_valid)

    print(f"parameters: {search_model.best_params_}")

    model = BaggingClassifier(**search_model.best_params_, )
    model.fit([*X_valid, *X_train], [*y_valid, *y_train])

    predictions = model.predict(X_test)

    accuracy = accuracy_score(y_test, predictions)

    print(f"accuracy: {accuracy}")

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

    print(f"RandomForestClassifier mnist")

    search_model.fit(X_valid, y_valid)

    print(f"parameters: {search_model.best_params_}")

    model = RandomForestClassifier(**search_model.best_params_)
    model.fit([*X_valid, *X_train], [*y_valid, *y_train])

    predictions = model.predict(X_test)

    accuracy = accuracy_score(y_test, predictions)

    print(f"accuracy: {accuracy}")

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

    print(f"GradientBoostingClassifier mnist")

    search_model.fit(X_valid, y_valid)

    print(f"parameters: {search_model.best_params_}")

    model = GradientBoostingClassifier(**search_model.best_params_)
    model.fit([*X_valid, *X_train], [*y_valid, *y_train])

    predictions = model.predict(X_test)

    accuracy = accuracy_score(y_test, predictions)

    print(f"accuracy: {accuracy}")