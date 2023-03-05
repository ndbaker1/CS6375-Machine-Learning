# • Use the SVM classifier in scikit learn and try different kernels and values of penalty parameter. Important: Depending on your computer hardware, you may have to carefully select
# the parameters (see the documentation on scikit learn for details) in order to speed up the
# computation. Report the error rate for at least 10 parameter settings that you tried. Make
# sure to precisely describe the parameters used so that your results are reproducible.

# • Use the MLPClassifier in scikit learn and try different architectures, gradient descent schemes,
# etc. Depending on your computer hardware, you may have to carefully select the parameters
# of MLPClassifier in order to speed up the computation. Report the error rate for at least 10
# parameters that you tried. Make sure to precisely describe the parameters used so that your
# results are reproducible.

# • Use the k Nearest Neighbors classifier called KNeighborsClassifier in scikit learn and try different parameters (see the documentation for details). Again depending on your computer
# hardware, you may have to carefully select the parameters in order to speed up the computation. Report the error rate for at least 10 parameters that you tried. Make sure to precisely
# describe the parameters used so that your results are reproducible.

# • What is the best error rate you were able to reach for each of the three classifiers? Note that
# many parameters do not affect the error rate and we will deduct points if you try them. It
# is your duty to read the documentation and then employ your machine learning knowledge
# to determine whether a particular parameter will affect the error rate. Finally, don’t change
# just one parameter 10 times; we want to see diversity.

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
                for alpha in [0.00001, 0.0001, 0.001, 0.01]:
                    for early_stopping in [False, True]:

                        print(
                            f'running MLP classifier with [activation={activation}, solver={solver}, learning_rate={learning_rate}, alpha={alpha}, early_stopping={early_stopping}]...'
                        )

                        classifier = MLPClassifier(
                            activation=activation,
                            solver=solver,
                            learning_rate=learning_rate,
                            alpha=alpha,
                            early_stopping=early_stopping,
                        ).fit(X_train, y_train)

                        print(
                            f'error: {mean_absolute_error(y_test, classifier.predict(X_test))}'
                        )

    for weights in ['uniform', 'distance']:
        for algorithm in ['auto', 'ball_tree', 'kd_tree', 'brute']:
            for n_neighbors in [2, 5, 10, 50]:
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
