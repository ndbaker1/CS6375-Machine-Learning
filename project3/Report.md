---
title: "Machine Learning Project 3 Report"
author: "Nicholas Baker - ndb180002"
geometry: margin=1in
---

# Results

| Classifier | Dataset | Accuracy | F1 Score | Hyperparameters |
|---|---|---|---|---|
| DecisionTree | c1000_d100 | 0.69 | 0.7019 | criterion=gini, max_depth=None, max_features=None, splitter=best |
| DecisionTree | c1000_d1000 | 0.808 | 0.8102 | criterion=entropy, max_depth=5, max_features=None, splitter=random |
| DecisionTree | c1000_d5000 | 0.841 | 0.8446 | criterion=log_loss, max_depth=20, max_features=None, splitter=best |
| DecisionTree | c1500_d100 | 0.815 | 0.8102 | criterion=gini, max_depth=None, max_features=sqrt, splitter=random |
| DecisionTree | c1500_d1000 | 0.9175 | 0.9196 | criterion=log_loss, max_depth=None, max_features=None, splitter=random |
| DecisionTree | c1500_d5000 | 0.9533 | 0.9538 | criterion=log_loss, max_depth=None, max_features=None, splitter=random |
| DecisionTree | c1800_d100 | 0.935 | 0.9377 | criterion=entropy, max_depth=None, max_features=None, splitter=random |
| DecisionTree | c1800_d1000 | 0.975 | 0.9752 | criterion=log_loss, max_depth=None, max_features=None, splitter=random |
| DecisionTree | c1800_d5000 | 0.9836 | 0.9837 | criterion=log_loss, max_depth=20, max_features=None, splitter=random |
| DecisionTree | c300_d100 | 0.5 | 0.5145 | criterion=gini, max_depth=20, max_features=log2, splitter=random |
| DecisionTree | c300_d1000 | 0.6725 | 0.7095 | criterion=entropy, max_depth=5, max_features=None, splitter=best |
| DecisionTree | c300_d5000 | 0.7613 | 0.7588 | criterion=entropy, max_depth=20, max_features=None, splitter=random |
| DecisionTree | c500_d100 | 0.68 | 0.6923 | criterion=gini, max_depth=20, max_features=None, splitter=random |
| DecisionTree | c500_d1000 | 0.682 | 0.6870 | criterion=gini, max_depth=5, max_features=None, splitter=best |
| DecisionTree | c500_d5000 | 0.7745 | 0.7766 | criterion=log_loss, max_depth=None, max_features=None, splitter=best |
| Bagging | c1000_d100 | 0.885 | 0.8795 | max_features=1.0, max_samples=1.0, n_estimators=20 |
| Bagging | c1000_d1000 | 0.952 | 0.9518 | max_features=0.5, max_samples=1.0, n_estimators=20 |
| Bagging | c1000_d5000 | 0.9775 | 0.9774 | max_features=0.5, max_samples=1.0, n_estimators=20 |
| Bagging | c1500_d100 | 0.97 | 0.9702 | max_features=0.5, max_samples=1.0, n_estimators=20 |
| Bagging | c1500_d1000 | 0.9875 | 0.9874 | max_features=0.5, max_samples=0.5, n_estimators=20 |
| Bagging | c1500_d5000 | 0.9968 | 0.9967 | max_features=0.5, max_samples=0.5, n_estimators=20 |
| Bagging | c1800_d100 | 0.99 | 0.99 | max_features=0.5, max_samples=0.5, n_estimators=20 |
| Bagging | c1800_d1000 | 0.999 | 0.9989 | max_features=0.5, max_samples=0.5, n_estimators=20 |
| Bagging | c1800_d5000 | 0.9994 | 0.9993 | max_features=10, max_samples=1.0, n_estimators=20 |
| Bagging | c300_d100 | 0.71 | 0.6947 | max_features=1.0, max_samples=1.0, n_estimators=20 |
| Bagging | c300_d1000 | 0.8235 | 0.8196 | max_features=1.0, max_samples=1.0, n_estimators=20 |
| Bagging | c300_d5000 | 0.8927 | 0.8956 | max_features=1.0, max_samples=1.0, n_estimators=20 |
| Bagging | c500_d100 | 0.77 | 0.7578 | max_features=1.0, max_samples=1.0, n_estimators=20 |
| Bagging | c500_d1000 | 0.864 | 0.8595 | max_features=0.5, max_samples=1.0, n_estimators=20 |
| Bagging | c500_d5000 | 0.9108 | 0.9098 | max_features=1.0, max_samples=1.0, n_estimators=20 |
| RandomForest | c1000_d100 | 0.985 | 0.9850 | criterion=gini, max_depth=20, max_features=log2, min_samples_split=0.1, n_estimators=100 |
| RandomForest | c1000_d1000 | 0.9895 | 0.9894 | criterion=gini, max_depth=5, max_features=log2, min_samples_split=2, n_estimators=100 |
| RandomForest | c1000_d5000 | 0.9962 | 0.9962 | criterion=gini, max_depth=20, max_features=log2, min_samples_split=2, n_estimators=100 |
| RandomForest | c1500_d100 | 0.995 | 0.9950 | criterion=gini, max_depth=5, max_features=sqrt, min_samples_split=2, n_estimators=20 |
| RandomForest | c1500_d1000 | 1.0 | 1.0 | criterion=gini, max_depth=5, max_features=log2, min_samples_split=2, n_estimators=100 |
| RandomForest | c1500_d5000 | 1.0 | 1.0 | criterion=entropy, max_depth=20, max_features=log2, min_samples_split=2, n_estimators=100 |
| RandomForest | c1800_d100 | 1.0 | 1.0 | criterion=gini, max_depth=5, max_features=sqrt, min_samples_split=2, n_estimators=100 |
| RandomForest | c1800_d1000 | 1.0 | 1.0 | criterion=gini, max_depth=5, max_features=sqrt, min_samples_split=2, n_estimators=100 |
| RandomForest | c1800_d5000 | 0.9999 | 0.9998 | criterion=gini, max_depth=5, max_features=log2, min_samples_split=2, n_estimators=20 |
| RandomForest | c300_d100 | 0.75 | 0.7311 | criterion=log_loss, max_depth=None, max_features=sqrt, min_samples_split=2, n_estimators=100 |
| RandomForest | c300_d1000 | 0.8765 | 0.8774 | criterion=gini, max_depth=5, max_features=log2, min_samples_split=2, n_estimators=100 |
| RandomForest | c300_d5000 | 0.9168 | 0.9212 | criterion=gini, max_depth=20, max_features=None, min_samples_split=2, n_estimators=100 |
| RandomForest | c500_d100 | 0.845 | 0.8442 | criterion=entropy, max_depth=None, max_features=sqrt, min_samples_split=2, n_estimators=100 |
| RandomForest | c500_d1000 | 0.933 | 0.9339 | criterion=log_loss, max_depth=5, max_features=log2, min_samples_split=2, n_estimators=100 |
| RandomForest | c500_d5000 | 0.9342 | 0.9350 | criterion=gini, max_depth=5, max_features=log2, min_samples_split=2, n_estimators=100 |
| GradientBoosting | c1000_d100 | 0.97 | 0.9696 | criterion=friedman_mse, learning_rate=0.1, loss=exponential, max_features=log2, n_estimators=100, subsample=1.0 |
| GradientBoosting | c1000_d1000 | 0.998 | 0.9980 | criterion=friedman_mse, learning_rate=0.5, loss=log_loss, max_features=1.0, n_estimators=100, subsample=1.0 |
| GradientBoosting | c1000_d5000 | 0.9997 | 0.9997 | criterion=friedman_mse, learning_rate=0.5, loss=log_loss, max_features=1.0, n_estimators=100, subsample=1.0 |
| GradientBoosting | c1500_d100 | 0.99 | 0.9900 | criterion=friedman_mse, learning_rate=0.1, loss=log_loss, max_features=1.0, n_estimators=100, subsample=0.2 |
| GradientBoosting | c1500_d1000 | 1.0 | 1.0 | criterion=friedman_mse, learning_rate=0.1, loss=log_loss, max_features=1.0, n_estimators=100, subsample=0.2 |
| GradientBoosting | c1500_d5000 | 0.9997 | 0.9996 | criterion=friedman_mse, learning_rate=0.1, loss=log_loss, max_features=sqrt, n_estimators=100, subsample=0.05 |
| GradientBoosting | c1800_d100 | 0.995 | 0.9949 | criterion=friedman_mse, learning_rate=0.1, loss=log_loss, max_features=1.0, n_estimators=100, subsample=0.2 |
| GradientBoosting | c1800_d1000 | 1.0 | 1.0 | criterion=friedman_mse, learning_rate=0.1, loss=log_loss, max_features=1.0, n_estimators=100, subsample=0.05 |
| GradientBoosting | c1800_d5000 | 0.9999 | 0.9999 | criterion=friedman_mse, learning_rate=0.1, loss=log_loss, max_features=1.0, n_estimators=100, subsample=0.2 |
| GradientBoosting | c300_d100 | 0.715 | 0.7246 | criterion=squared_error, learning_rate=2, loss=exponential, max_features=1.0, n_estimators=20, subsample=1.0 |
| GradientBoosting | c300_d1000 | 0.986 | 0.9861 | criterion=friedman_mse, learning_rate=0.5, loss=exponential, max_features=1.0, n_estimators=100, subsample=1.0 |
| GradientBoosting | c300_d5000 | 0.9977 | 0.9977 | criterion=friedman_mse, learning_rate=0.5, loss=log_loss, max_features=1.0, n_estimators=100, subsample=1.0 |
| GradientBoosting | c500_d100 | 0.9 | 0.9019 | criterion=squared_error, learning_rate=0.5, loss=exponential, max_features=sqrt, n_estimators=100, subsample=1.0 |
| GradientBoosting | c500_d1000 | 0.9945 | 0.9945 | criterion=friedman_mse, learning_rate=0.5, loss=log_loss, max_features=1.0, n_estimators=100, subsample=1.0 |
| GradientBoosting | c500_d5000 | 0.9983 | 0.9983 | criterion=friedman_mse, learning_rate=0.5, loss=exponential, max_features=1.0, n_estimators=100, subsample=1.0 |

# Questions

* (5.a) Which classifier (among the four) yields the best overall generalization accuracy/F1 score?
Based on your ML knowledge, why do you think the "classifier" achieved the highest overall accuracy/F1 score.

    The RandomForestClassifier has the best general scores for accuracy/F1 score.

    This could be due to the advantage that the RandomForest strategy has over data with high dimensionality/feature count, since in RandomForest a subset of feature are selected. 

* (5.b) What is the impact of increasing the amount of training data on the
accuracy/F1 scores of each of the four classifiers.

    Increasing the number of data points in training results in an increase to the accuracy & F1 score.

    This is likely due to the upper bound on leaf nodes growing with the size of the training data,
    which increases tree depth until sample entropy collapses to 0.

* (5.c) What is the impact of increasing the number of features on the
accuracy/F1 scores of each of the four classifiers.

    Increasing the number of features results in an increase to the accuracy & F1 score.

    This is likely due to the upper bound on leaf nodes growing with the number of features,
    which also increases tree depth until sample entropy collapses to 0.

* (6) Which classifier among the four yields the best
classification accuracy on the MNIST dataset and why?

    | Classifier | Accuracy | Hyperparameters | 
    |---|---|---|
    | DecisionTree | 0.8861 | criterion=log_loss, max_depth=20, max_features=None, splitter=best |
    | Bagging | 0.9579 | max_features=0.5, max_samples=1.0, n_estimators=20 |
    | RandomForest | 0.9681 | criterion=log_loss, max_depth=20, max_features=sqrt, min_samples_split=2, n_estimators=100 |
    | GradientBoosting | 0.9405 | criterion=friedman_mse, learning_rate=0.1, loss=log_loss, max_features=sqrt, n_estimators=100, subsample=1.0 |

    RandomForestClassifier yields the best classification accuracy,
    which could be due to the advantage that the Random Forest strategy has over data with high dimensions.

    Decision trees normally exhibit difficulty locating the best features to split on in order to converge
    subtrees more quickly. Random Foresting mitigates part of these issues by using several trees with attention to different features to come to a final decision. This helps alleviate overfitting and gives more variety on higher dimensional data.

* (7) Compare the classification accuracy of tree and ensemble based classifiers with the (best)
accuracy you obtained using the MLPClassifier, SVMs and nearest-neighbors in Project 2 (best as in after tuning the hyperparameters).
Which classifier (or classifiers) among the seven has (have) the highest accuracy on the test set and why?

    $$Project2$$

    | Classifier | Accuracy | 
    |---|---|
    | SVC | 0.9178 | 
    | MLP | 0.9287 |
    | KNN | 0.885 |

    The RandomForestClassifier once again performs the best among the classifiers that we have for
    this comparison. 

    Because RandomForest is an improvement over Bagging where subsets of feature are selected, it is better able to handle high dimensional data and is less prone to overfitting. For Boosting there is a similar rationale; in which the weight of all features is difficult to determine with such a high number of options.

    Linear Classifiers may not to achieve below an arbitrary error rate because the mnist dataset is not linearly separable. Support Vectors  and MLP classifiers from project are linear classifiers, so it makes sense that their accuracy may not be able to outperform RandomForest. Finally, k-nearest-neighbors has difficulty with a high number of dimensions/features, which makes the mnist dataset very hard to work with since features of an image are the pixel matrix.

