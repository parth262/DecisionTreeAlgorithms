import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator


class Node:
    def __init__(self, val, nodes={}):
        self.val = val
        self.nodes = nodes

    def add_node(self, value, node):
        self.nodes[value] = node


def get_entropy(S):
    _, counts = np.unique(S, return_counts=True)

    def p_f(count):
        prob = float(count) / S.size
        return prob * np.log2(1/prob)

    return sum(map(p_f, counts))


def get_information_gain(dataset, column, target_column):
    column_values = dataset[:, column]
    target_values = dataset[:, target_column]
    decision_entropy = get_entropy(target_values)
    total_feature_entropy = 0

    for v in np.unique(column_values):
        target_subset = target_values[column_values == v]
        prob = float(target_subset.size) / target_values.size
        target_subset_entropy = get_entropy(target_subset)
        total_feature_entropy += prob * target_subset_entropy

    return decision_entropy - total_feature_entropy


def build_tree(dataset, features, tree=None):
    target_index = -1
    # target_values = dataset[:, target_index]
    # unique_target_values, counts = np.unique(target_values,
    #                                          return_counts=True)

    # if len(unique_target_values) == 1 or dataset.shape[1] == 1:
    #     return Node(unique_target_values[np.argmax(counts)])
    # else:
    information_gains = [get_information_gain(dataset, i, target_index)
                         for i in range(dataset.shape[1]-1)]

    best_feature_index = np.argmax(information_gains)
    best_feature = features[best_feature_index]

    if tree is None:
        tree = {}
        tree[best_feature] = {}

    best_feature_values = dataset[:, best_feature_index]
    unique_feature_values = np.unique(best_feature_values)

    for v in unique_feature_values:
        new_dataset = dataset[best_feature_values == v]

        target_values = new_dataset[:, target_index]
        unique_target_values, counts = np.unique(target_values,
                                                 return_counts=True)

        if len(counts) == 1:
            tree[best_feature][v] = unique_target_values[0]
        else:
            tree[best_feature][v] = build_tree(new_dataset,
                                               features)

    return tree


def fit(X, y):
    if isinstance(X, pd.DataFrame):
        X_values = X.values
        y_values = y.values
        features = X.columns
    else:
        X_values = X
        y_values = y
        features = ["C_" + i for i in range(X.shape[1])]

    dataset = np.hstack((X_values, y_values))

    tree = build_tree(dataset, features)
    return tree


def predict_for_instance(x, tree):
    if len(tree.nodes) > 0:
        feature_index = np.argwhere(features == tree.val).flatten()[0]
        sub_tree = tree.nodes[x[feature_index]]
        return predict_for_instance(x, sub_tree)
    else:
        return tree.val


def predict(X):
    if isinstance(X, pd.DataFrame):
        X_values = X.values
    else:
        X_values = X

    predictions = [predict_for_instance(
        x, tree) for x in X_values]

    return predictions
