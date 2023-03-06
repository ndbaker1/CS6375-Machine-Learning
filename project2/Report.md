---
title: "Machine Learning Project 2 Report"
author: "Nicholas Baker - ndb180002"
geometry: margin=1in
---

# Collaborative Filtering

This collaborative filtering implementation improves runtime by only computing
correlation scores between the `5` nearest neighbors based on `cosine distance`
between binary vectors in a sparse `mxn` matrix between `MovieID` and `CustomerID`.

We do this by training a `NearestNeighbor` model on the dataset; ignoring the 
assigned `rating` of each interaction.

| MAE | RMSE |
|---|---|
| 0.7618265408304616 | 0.9660156340232393 |

# scikit-learn Classifiers

## SVM Classifier

| Parameters | MAE |
|---|---|
| kernel=linear, C=1.0 | 0.2153 |
| kernel=linear, C=0.1 | 0.1916 |
| kernel=linear, C=0.01 | 0.2068 |
| kernel=poly, C=1.0 | 0.0917 |
| kernel=poly, C=0.1 | 0.1695 |
| kernel=poly, C=0.01 | 0.3974 |
| kernel=rbf, C=1.0 | 0.0822 |
| kernel=rbf, C=0.1 | 0.1535 |
| kernel=rbf, C=0.01 | 0.2831 |
| kernel=sigmoid, C=1.0 | 0.8063 |
| kernel=sigmoid, C=0.1 | 0.5732 |
| kernel=sigmoid, C=0.01 | 0.4413 |

## MLP Classifier

| Parameters | MAE |
|---|---|
| activation=identity, solver=lbfgs, learning_rate=constant, alpha=0.0001 | 0.2712 |
| activation=identity, solver=lbfgs, learning_rate=constant, alpha=0.001 | 0.2711 |
| activation=identity, solver=lbfgs, learning_rate=constant, alpha=0.01 | 0.2694 |
| activation=identity, solver=lbfgs, learning_rate=invscaling, alpha=0.0001 | 0.2682 |
| activation=identity, solver=lbfgs, learning_rate=invscaling, alpha=0.001 | 0.2708 |
| activation=identity, solver=lbfgs, learning_rate=invscaling, alpha=0.01 | 0.267 |
| activation=identity, solver=lbfgs, learning_rate=adaptive, alpha=0.0001 | 0.2699 |
| activation=identity, solver=lbfgs, learning_rate=adaptive, alpha=0.001 | 0.2665 |
| activation=identity, solver=lbfgs, learning_rate=adaptive, alpha=0.01 | 0.2714 |
| activation=identity, solver=sgd, learning_rate=constant, alpha=0.0001 | 0.2756 |
| activation=identity, solver=sgd, learning_rate=constant, alpha=0.001 | 0.2744 |
| activation=identity, solver=sgd, learning_rate=constant, alpha=0.01 | 0.2722 |
| activation=identity, solver=sgd, learning_rate=invscaling, alpha=0.0001 | 0.5686 |
| activation=identity, solver=sgd, learning_rate=invscaling, alpha=0.001 | 0.5612 |
| activation=identity, solver=sgd, learning_rate=invscaling, alpha=0.01 | 0.5538 |
| activation=identity, solver=sgd, learning_rate=adaptive, alpha=0.0001 | 0.2763 |
| activation=identity, solver=sgd, learning_rate=adaptive, alpha=0.001 | 0.272 |
| activation=identity, solver=sgd, learning_rate=adaptive, alpha=0.01 | 0.276 |
| activation=identity, solver=adam, learning_rate=constant, alpha=0.0001 | 0.2682 |
| activation=identity, solver=adam, learning_rate=constant, alpha=0.001 | 0.2637 |
| activation=identity, solver=adam, learning_rate=constant, alpha=0.01 | 0.2672 |
| activation=identity, solver=adam, learning_rate=invscaling, alpha=0.0001 | 0.2701 |
| activation=identity, solver=adam, learning_rate=invscaling, alpha=0.001 | 0.2788 |
| activation=identity, solver=adam, learning_rate=invscaling, alpha=0.01 | 0.2715 |
| activation=identity, solver=adam, learning_rate=adaptive, alpha=0.0001 | 0.2794 |
| activation=identity, solver=adam, learning_rate=adaptive, alpha=0.001 | 0.2711 |
| activation=identity, solver=adam, learning_rate=adaptive, alpha=0.01 | 0.2873 |
| activation=logistic, solver=lbfgs, learning_rate=constant, alpha=0.0001 | 0.1 |
| activation=logistic, solver=lbfgs, learning_rate=constant, alpha=0.001 | 0.1011 |
| activation=logistic, solver=lbfgs, learning_rate=constant, alpha=0.01 | 0.0972 |
| activation=logistic, solver=lbfgs, learning_rate=invscaling, alpha=0.0001 | 0.096 |
| activation=logistic, solver=lbfgs, learning_rate=invscaling, alpha=0.001 | 0.1062 |
| activation=logistic, solver=lbfgs, learning_rate=invscaling, alpha=0.01 | 0.1004 |
| activation=logistic, solver=lbfgs, learning_rate=adaptive, alpha=0.0001 | 0.1015 |
| activation=logistic, solver=lbfgs, learning_rate=adaptive, alpha=0.001 | 0.1153 |
| activation=logistic, solver=lbfgs, learning_rate=adaptive, alpha=0.01 | 0.1076 |
| activation=logistic, solver=sgd, learning_rate=constant, alpha=0.0001 | 0.2429 |
| activation=logistic, solver=sgd, learning_rate=constant, alpha=0.001 | 0.2424 |
| activation=logistic, solver=sgd, learning_rate=constant, alpha=0.01 | 0.2448 |
| activation=logistic, solver=sgd, learning_rate=invscaling, alpha=0.0001 | 1.7575 |
| activation=logistic, solver=sgd, learning_rate=invscaling, alpha=0.001 | 1.6581 |
| activation=logistic, solver=sgd, learning_rate=invscaling, alpha=0.01 | 1.7623 |
| activation=logistic, solver=sgd, learning_rate=adaptive, alpha=0.0001 | 0.2399 |
| activation=logistic, solver=sgd, learning_rate=adaptive, alpha=0.001 | 0.241 |
| activation=logistic, solver=sgd, learning_rate=adaptive, alpha=0.01 | 0.2433 |
| activation=logistic, solver=adam, learning_rate=constant, alpha=0.0001 | 0.0821 |
| activation=logistic, solver=adam, learning_rate=constant, alpha=0.001 | 0.0713 |
| activation=logistic, solver=adam, learning_rate=constant, alpha=0.01 | 0.0737 |
| activation=logistic, solver=adam, learning_rate=invscaling, alpha=0.0001 | 0.0797 |
| activation=logistic, solver=adam, learning_rate=invscaling, alpha=0.001 | 0.0792 |
| activation=logistic, solver=adam, learning_rate=invscaling, alpha=0.01 | 0.0713 |
| activation=logistic, solver=adam, learning_rate=adaptive, alpha=0.0001 | 0.0872 |
| activation=logistic, solver=adam, learning_rate=adaptive, alpha=0.001 | 0.079 |
| activation=logistic, solver=adam, learning_rate=adaptive, alpha=0.01 | 0.0766 |
| activation=tanh, solver=lbfgs, learning_rate=constant, alpha=0.0001 | 0.1033 |
| activation=tanh, solver=lbfgs, learning_rate=constant, alpha=0.001 | 0.1027 |
| activation=tanh, solver=lbfgs, learning_rate=constant, alpha=0.01 | 0.0983 |
| activation=tanh, solver=lbfgs, learning_rate=invscaling, alpha=0.0001 | 0.0973 |
| activation=tanh, solver=lbfgs, learning_rate=invscaling, alpha=0.001 | 0.1025 |
| activation=tanh, solver=lbfgs, learning_rate=invscaling, alpha=0.01 | 0.1027 |
| activation=tanh, solver=lbfgs, learning_rate=adaptive, alpha=0.0001 | 0.1064 |
| activation=tanh, solver=lbfgs, learning_rate=adaptive, alpha=0.001 | 0.1023 |
| activation=tanh, solver=lbfgs, learning_rate=adaptive, alpha=0.01 | 0.1041 |
| activation=tanh, solver=sgd, learning_rate=constant, alpha=0.0001 | 0.1235 |
| activation=tanh, solver=sgd, learning_rate=constant, alpha=0.001 | 0.1281 |
| activation=tanh, solver=sgd, learning_rate=constant, alpha=0.01 | 0.133 |
| activation=tanh, solver=sgd, learning_rate=invscaling, alpha=0.0001 | 0.5837 |
| activation=tanh, solver=sgd, learning_rate=invscaling, alpha=0.001 | 0.5927 |
| activation=tanh, solver=sgd, learning_rate=invscaling, alpha=0.01 | 0.6417 |
| activation=tanh, solver=sgd, learning_rate=adaptive, alpha=0.0001 | 0.1297 |
| activation=tanh, solver=sgd, learning_rate=adaptive, alpha=0.001 | 0.1304 |
| activation=tanh, solver=sgd, learning_rate=adaptive, alpha=0.01 | 0.1246 |
| activation=tanh, solver=adam, learning_rate=constant, alpha=0.0001 | 0.0914 |
| activation=tanh, solver=adam, learning_rate=constant, alpha=0.001 | 0.0878 |
| activation=tanh, solver=adam, learning_rate=constant, alpha=0.01 | 0.0867 |
| activation=tanh, solver=adam, learning_rate=invscaling, alpha=0.0001 | 0.0866 |
| activation=tanh, solver=adam, learning_rate=invscaling, alpha=0.001 | 0.0885 |
| activation=tanh, solver=adam, learning_rate=invscaling, alpha=0.01 | 0.0854 |
| activation=tanh, solver=adam, learning_rate=adaptive, alpha=0.0001 | 0.0855 |
| activation=tanh, solver=adam, learning_rate=adaptive, alpha=0.001 | 0.0824 |
| activation=tanh, solver=adam, learning_rate=adaptive, alpha=0.01 | 0.0809 |
| activation=relu, solver=lbfgs, learning_rate=constant, alpha=0.0001 | 0.0907 |
| activation=relu, solver=lbfgs, learning_rate=constant, alpha=0.001 | 0.0942 |
| activation=relu, solver=lbfgs, learning_rate=constant, alpha=0.01 | 0.0851 |
| activation=relu, solver=lbfgs, learning_rate=invscaling, alpha=0.0001 | 0.0898 |
| activation=relu, solver=lbfgs, learning_rate=invscaling, alpha=0.001 | 0.0873 |
| activation=relu, solver=lbfgs, learning_rate=invscaling, alpha=0.01 | 0.0925 |
| activation=relu, solver=lbfgs, learning_rate=adaptive, alpha=0.0001 | 0.0877 |
| activation=relu, solver=lbfgs, learning_rate=adaptive, alpha=0.001 | 0.0879 |
| activation=relu, solver=lbfgs, learning_rate=adaptive, alpha=0.01 | 0.0921 |
| activation=relu, solver=sgd, learning_rate=constant, alpha=0.0001 | 0.1119 |
| activation=relu, solver=sgd, learning_rate=constant, alpha=0.001 | 0.1117 |
| activation=relu, solver=sgd, learning_rate=constant, alpha=0.01 | 0.1126 |
| activation=relu, solver=sgd, learning_rate=invscaling, alpha=0.0001 | 0.6479 |
| activation=relu, solver=sgd, learning_rate=invscaling, alpha=0.001 | 0.6473 |
| activation=relu, solver=sgd, learning_rate=invscaling, alpha=0.01 | 0.6874 |
| activation=relu, solver=sgd, learning_rate=adaptive, alpha=0.0001 | 0.1196 |
| activation=relu, solver=sgd, learning_rate=adaptive, alpha=0.001 | 0.1113 |
| activation=relu, solver=sgd, learning_rate=adaptive, alpha=0.01 | 0.1148 |
| activation=relu, solver=adam, learning_rate=constant, alpha=0.0001 | 0.0792 |
| activation=relu, solver=adam, learning_rate=constant, alpha=0.001 | 0.0795 |
| activation=relu, solver=adam, learning_rate=constant, alpha=0.01 | 0.0732 |
| activation=relu, solver=adam, learning_rate=invscaling, alpha=0.0001 | 0.0802 |
| activation=relu, solver=adam, learning_rate=invscaling, alpha=0.001 | 0.0765 |
| activation=relu, solver=adam, learning_rate=invscaling, alpha=0.01 | 0.0728 |
| activation=relu, solver=adam, learning_rate=adaptive, alpha=0.0001 | 0.0786 |
| activation=relu, solver=adam, learning_rate=adaptive, alpha=0.001 | 0.0813 |
| activation=relu, solver=adam, learning_rate=adaptive, alpha=0.01 | 0.0757 |

## kNN Classifier

| Parameters | MAE |
|---|---|
| weights=uniform, algorithm=ball_tree, n_neighbors=2, p=1 | 0.1771 |
| weights=uniform, algorithm=ball_tree, n_neighbors=2, p=1.5 | 0.156 |
| weights=uniform, algorithm=ball_tree, n_neighbors=2, p=2 | 0.1457 |
| weights=uniform, algorithm=ball_tree, n_neighbors=5, p=1 | 0.1502 |
| weights=uniform, algorithm=ball_tree, n_neighbors=5, p=1.5 | 0.1343 |
| weights=uniform, algorithm=ball_tree, n_neighbors=5, p=2 | 0.124 |
| weights=uniform, algorithm=ball_tree, n_neighbors=10, p=1 | 0.1611 |
| weights=uniform, algorithm=ball_tree, n_neighbors=10, p=1.5 | 0.1453 |
| weights=uniform, algorithm=ball_tree, n_neighbors=10, p=2 | 0.1325 |
| weights=uniform, algorithm=kd_tree, n_neighbors=2, p=1 | 0.1771 |
| weights=uniform, algorithm=kd_tree, n_neighbors=2, p=1.5 | 0.156 |
| weights=uniform, algorithm=kd_tree, n_neighbors=2, p=2 | 0.1457 |
| weights=uniform, algorithm=kd_tree, n_neighbors=5, p=1 | 0.1502 |
| weights=uniform, algorithm=kd_tree, n_neighbors=5, p=1.5 | 0.1343 |
| weights=uniform, algorithm=kd_tree, n_neighbors=5, p=2 | 0.124 |
| weights=uniform, algorithm=kd_tree, n_neighbors=10, p=1 | 0.1611 |
| weights=uniform, algorithm=kd_tree, n_neighbors=10, p=1.5 | 0.1453 |
| weights=uniform, algorithm=kd_tree, n_neighbors=10, p=2 | 0.1325 |
| weights=uniform, algorithm=brute, n_neighbors=2, p=1 | 0.1771 |
| weights=uniform, algorithm=brute, n_neighbors=2, p=1.5 | 0.156 |
| weights=uniform, algorithm=brute, n_neighbors=2, p=2 | 0.1457 |
| weights=uniform, algorithm=brute, n_neighbors=5, p=1 | 0.1502 |
| weights=uniform, algorithm=brute, n_neighbors=5, p=1.5 | 0.1343 |
| weights=uniform, algorithm=brute, n_neighbors=5, p=2 | 0.124 |
| weights=uniform, algorithm=brute, n_neighbors=10, p=1 | 0.1611 |
| weights=uniform, algorithm=brute, n_neighbors=10, p=1.5 | 0.1453 |
| weights=uniform, algorithm=brute, n_neighbors=10, p=2 | 0.1325 |
| weights=distance, algorithm=ball_tree, n_neighbors=2, p=1 | 0.1379 |
| weights=distance, algorithm=ball_tree, n_neighbors=2, p=1.5 | 0.1275 |
| weights=distance, algorithm=ball_tree, n_neighbors=2, p=2 | 0.115 |
| weights=distance, algorithm=ball_tree, n_neighbors=5, p=1 | 0.1437 |
| weights=distance, algorithm=ball_tree, n_neighbors=5, p=1.5 | 0.1293 |
| weights=distance, algorithm=ball_tree, n_neighbors=5, p=2 | 0.1207 |
| weights=distance, algorithm=ball_tree, n_neighbors=10, p=1 | 0.1503 |
| weights=distance, algorithm=ball_tree, n_neighbors=10, p=1.5 | 0.1396 |
| weights=distance, algorithm=ball_tree, n_neighbors=10, p=2 | 0.1256 |
| weights=distance, algorithm=kd_tree, n_neighbors=2, p=1 | 0.1379 |
| weights=distance, algorithm=kd_tree, n_neighbors=2, p=1.5 | 0.1275 |
| weights=distance, algorithm=kd_tree, n_neighbors=2, p=2 | 0.115 |
| weights=distance, algorithm=kd_tree, n_neighbors=5, p=1 | 0.1437 |
| weights=distance, algorithm=kd_tree, n_neighbors=5, p=1.5 | 0.1293 |
| weights=distance, algorithm=kd_tree, n_neighbors=5, p=2 | 0.1207 |
| weights=distance, algorithm=kd_tree, n_neighbors=10, p=1 | 0.1503 |
| weights=distance, algorithm=kd_tree, n_neighbors=10, p=1.5 | 0.1396 |
| weights=distance, algorithm=kd_tree, n_neighbors=10, p=2 | 0.1256 |
| weights=distance, algorithm=brute, n_neighbors=2, p=1 | 0.1379 |
| weights=distance, algorithm=brute, n_neighbors=2, p=1.5 | 0.1275 |
| weights=distance, algorithm=brute, n_neighbors=2, p=2 | 0.115 |
| weights=distance, algorithm=brute, n_neighbors=5, p=1 | 0.1437 |
| weights=distance, algorithm=brute, n_neighbors=5, p=1.5 | 0.1293 |
| weights=distance, algorithm=brute, n_neighbors=5, p=2 | 0.1207 |
| weights=distance, algorithm=brute, n_neighbors=10, p=1 | 0.1503 |
| weights=distance, algorithm=brute, n_neighbors=10, p=1.5 | 0.1396 |
| weights=distance, algorithm=brute, n_neighbors=10, p=2 | 0.1256 |

# Minimum Errors

## SVM Classifier

| Parameters | MAE |
|---|---|
| kernel=rbf, C=1.0 | 0.0822 |

The best performing SVM Classifier in this testing set used the `radial basis function`
kernel in combination with a high penalty term, causing the margin for this SVM to be 
very tight and quite possibly overfitted to the training set.

## MLP Classifier

| Parameters | MAE |
|---|---|
| activation=logistic, solver=adam, learning_rate=constant, alpha=0.001 | 0.0713 |
| activation=logistic, solver=adam, learning_rate=invscaling, alpha=0.01 | 0.0713 |

There were two intances of MLP Classifiers reaching the minimal mean absolute error over
the test set, and those models only differed between their `learning_rate` strategy and `alpha` parameter.

the discrepancy between the two models gives me an intuitition that a constant (seemingly low)
learning rate does not need a high `alpha`, which act as the L2 regularization term. Conversely,
the same score was achieved when `alpha` was slightly higher, but `learning_rate` was set to `invscaling` 
which gradually decreases the learning rate. This may be because large jumps along the gradient curve
require more regularization in order for performance to be stable.

Both models utilized the `adam` solver which extends classical stochastic gradint descent,
and favored the logiststic (or sigmoid) activation function. 


## KNN Classifier

| Parameters | MAE |
|---|---|
| weights=distance, algorithm=ball_tree, n_neighbors=2, p=2 | 0.115 |
| weights=distance, algorithm=brute, n_neighbors=2, p=2 | 0.115 |

Two instances of kNN Classifiers reached the minimal mean absolute error over the 
test set, any the only different in `algorithm`, which determines how neighbors on 
compute over the training set.

It is interesting that euclidean distance (`p=2`) and `2` neighbors performed most optimally
in both cases, possibly meaning there are small neighborhoods of similar users. 


