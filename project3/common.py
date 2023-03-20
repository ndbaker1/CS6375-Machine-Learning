DECISION_TREE_CLASSIFIER_PARAMS = {
    "criterion": ["gini", "entropy", "log_loss"],
    "splitter": ["best", "random"],
    "max_depth": [5, 20, None],
    "max_features": ["sqrt", "log2", None],
}

BAGGING_CLASSIFIER_PARAMS = {
    "n_estimators": [5, 10, 20],
    "max_samples": [10, 0.5, 1.0],
    "max_features": [10, 0.5, 1.0],
}

RANDOM_FOREST_CLASSIFIER_PARAMS = {
    "n_estimators": [5, 20, 100],
    "criterion": ["gini", "entropy", "log_loss"],
    "max_depth": [5, 20, None],
    "max_features": ["sqrt", "log2", None],
    "min_samples_split": [2, 0.1, 0.5],
}

GRADIENT_BOOSTING_CLASSIFIER_PARAMS = {
    "n_estimators": [5, 20, 100],
    "loss": ['log_loss', 'exponential'],
    "criterion": ['friedman_mse', 'squared_error'],
    "max_features": [1.0, 'sqrt', 'log2'],
    "learning_rate": [0.1, 0.5, 2],
    "subsample": [0.05, 0.2, 1.0],
}


def print_table(classifier, dataset, params, accuracy, f1=None):
    param_expansion = ', '.join([f"{k}={v}" for k, v in params.items()])
    print(f"{classifier} | {dataset} | {param_expansion} | {accuracy}" +
          ("" if f1 is None else f" | {f1}"))
