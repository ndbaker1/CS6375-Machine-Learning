if __name__ == "__main__":
    from sklearn.tree import DecisionTreeClassifier
    from sklearn.ensemble import BaggingClassifier, RandomForestClassifier, GradientBoostingClassifier
    from sklearn.model_selection import GridSearchCV
    from sklearn.metrics import accuracy_score

    from common import *

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
        DECISION_TREE_CLASSIFIER_PARAMS,
    )

    search_model.fit(X_valid, y_valid)

    model = DecisionTreeClassifier(**search_model.best_params_)
    model.fit([*X_valid, *X_train], [*y_valid, *y_train])

    predictions = model.predict(X_test)

    accuracy = accuracy_score(y_test, predictions)

    print_table(
        "DecisionTreeClassifier",
        "mnist",
        search_model.best_params_,
        accuracy,
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

    search_model.fit(X_valid, y_valid)

    model = BaggingClassifier(**search_model.best_params_, )
    model.fit([*X_valid, *X_train], [*y_valid, *y_train])

    predictions = model.predict(X_test)

    accuracy = accuracy_score(y_test, predictions)

    print_table(
        "BaggingClassifier",
        "mnist",
        search_model.best_params_,
        accuracy,
    )

    # 3. (15 points) Repeat the experiment described above using:
    # sklearn.ensemble.RandomForestClassifier.
    search_model = GridSearchCV(
        RandomForestClassifier(),
        RANDOM_FOREST_CLASSIFIER_PARAMS,
    )

    search_model.fit(X_valid, y_valid)

    model = RandomForestClassifier(**search_model.best_params_)
    model.fit([*X_valid, *X_train], [*y_valid, *y_train])

    predictions = model.predict(X_test)

    accuracy = accuracy_score(y_test, predictions)

    print_table(
        "RandomForestClassifier",
        "mnist",
        search_model.best_params_,
        accuracy,
    )

    # 4. (15 points) Repeat the experiment described above using:
    # sklearn.ensemble.GradientBoostingClassifier.
    search_model = GridSearchCV(
        GradientBoostingClassifier(),
        GRADIENT_BOOSTING_CLASSIFIER_PARAMS,
    )

    search_model.fit(X_valid, y_valid)

    model = GradientBoostingClassifier(**search_model.best_params_)
    model.fit([*X_valid, *X_train], [*y_valid, *y_train])

    predictions = model.predict(X_test)

    accuracy = accuracy_score(y_test, predictions)

    print_table(
        "GradientBoostingClassifier",
        "mnist",
        search_model.best_params_,
        accuracy,
    )