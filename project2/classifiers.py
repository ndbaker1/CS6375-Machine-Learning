from sklearn.datasets import fetch_openml
# Load data from https://www.openml.org/d/554
X, y = fetch_openml('mnist_784', version=1, return_X_y=True)
X = X / 255.0
# rescale the data, use the traditional train/test split
# (60K: Train) and (10K: Test)
X_train, X_test = X[:60000], X[60000:]
y_train, y_test = y[:60000], y[60000:]

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
    pass
