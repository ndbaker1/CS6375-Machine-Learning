if __name__ == "__main__":

    print('fetching dataset...')

    from sklearn.datasets import fetch_openml
    # Load data from https://www.openml.org/d/554
    X, y = fetch_openml('mnist_784',
                        version=1,
                        return_X_y=True,
                        as_frame=False)
    X = X / 255.0
    y = [int(y) for y in y]
    # rescale the data, use the traditional train/test split
    # (60K: Train) and (10K: Test)
    X_train, X_test = X[:60000], X[60000:]
    y_train, y_test = y[:60000], y[60000:]

    print('finished loading dataset.')

    from sklearn.svm import SVC
    from sklearn.neural_network import MLPClassifier
    from sklearn.neighbors import KNeighborsClassifier

    from sklearn.metrics import mean_absolute_error

    for kernel in ['linear', 'poly', 'rbf', 'sigmoid']:
        for C in [1.0, 0.1, 0.01]:

            print(f'running SVM classifier with [kernel={kernel}, C={C}]...')

            classifier = SVC(
                kernel=kernel,
                C=C,
            ).fit(X_train, y_train)

            print(
                f'error: {mean_absolute_error(y_test, classifier.predict(X_test))}'
            )

    for activation in ['identity', 'logistic', 'tanh', 'relu']:
        for solver in ['lbfgs', 'sgd', 'adam']:
            for learning_rate in ['constant', 'invscaling', 'adaptive']:
                for alpha in [0.0001, 0.001, 0.01]:
                    print(
                        f'running MLP classifier with [activation={activation}, solver={solver}, learning_rate={learning_rate}, alpha={alpha}]...'
                    )

                    classifier = MLPClassifier(
                        activation=activation,
                        solver=solver,
                        learning_rate=learning_rate,
                        alpha=alpha,
                    ).fit(X_train, y_train)

                    print(
                        f'error: {mean_absolute_error(y_test, classifier.predict(X_test))}'
                    )

    for weights in ['uniform', 'distance']:
        for algorithm in ['ball_tree', 'kd_tree', 'brute']:
            for n_neighbors in [2, 5, 10]:
                for p in [1, 1.5, 2]:

                    print(
                        f'running KNN classifier with [weights={weights}, algorithm={algorithm}, n_neighbors={n_neighbors}, p={p}]...'
                    )

                    classifier = KNeighborsClassifier(
                        weights=weights,
                        algorithm=algorithm,
                        n_neighbors=n_neighbors,
                        p=p,
                        n_jobs=-1,
                    ).fit(X_train, y_train)
                    print(
                        f'error: {mean_absolute_error(y_test, classifier.predict(X_test))}'
                    )
