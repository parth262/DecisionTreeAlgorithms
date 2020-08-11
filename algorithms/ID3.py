import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator


class Node:
    def __init__(self, val, nodes={}):
        self.val = val
        self.nodes = nodes

    def add_node(self, value, node):
        self.nodes[value] = node


class ID3(BaseEstimator):
    def __init__(self):
        self.tree = None
        self.target = "Decision"
        pass

    @staticmethod
    def get_entropy(S):
        _, counts = np.unique(S, return_counts=True)

        def p_f(count):
            prob = float(count) / S.size
            return prob * np.log2(1/prob)

        return sum(map(p_f, counts))

    @staticmethod
    def get_information_gain(dataset, column, target_column):
        column_values = dataset[:, column]
        target_values = dataset[:, target_column]
        decision_entropy = ID3.get_entropy(target_values)
        total_feature_entropy = 0

        for v in np.unique(column_values):
            target_subset = target_values[column_values == v]
            prob = float(target_subset.size) / target_values.size
            target_subset_entropy = ID3.get_entropy(target_subset)
            total_feature_entropy += prob * target_subset_entropy

        return decision_entropy - total_feature_entropy

    @staticmethod
    def train(dataset, features):
        target_index = -1
        target_values = dataset[:, target_index]
        unique_target_values, counts = np.unique(target_values,
                                                 return_counts=True)

        if len(unique_target_values) == 1 or dataset.shape[1] == 1:
            return Node(unique_target_values[np.argmax(counts)])

        information_gains = [ID3.get_information_gain(dataset, i, target_index)
                             for i in range(dataset.shape[1]-1)]

        best_feature_index = np.argmax(information_gains)
        best_feature = features[best_feature_index]
        root_node = Node(best_feature)

        best_feature_values = dataset[:, best_feature_index]
        unique_feature_values = np.unique(best_feature_values)

        for v in unique_feature_values:
            new_dataset = dataset[best_feature_values == v]
            new_dataset = np.delete(new_dataset, best_feature_index, 1)
            new_features = np.delete(features, best_feature_index)
            if len(new_dataset) > 0:
                pass
                # child_node = Node(v) #ID3.train(new_dataset, new_features)
            else:
                child_node = Node(unique_target_values[np.argmax(counts)])
                root_node.add_node(v, child_node)

        return root_node

    def fit(self, X, y):
        if isinstance(X, pd.DataFrame):
            X_values = X.values
            y_values = y.values
            self.features = X.columns
        else:
            X_values = X
            y_values = y
            self.features = ["C_" + i for i in range(X.shape[1])]

        dataset = np.hstack((X_values, y_values))

        self.tree = ID3.train(dataset, self.features)
        return self.tree

    def predict_for_instance(self, x, tree):
        if len(tree.nodes) > 0:
            feature_index = np.argwhere(self.features == tree.val).flatten()[0]
            sub_tree = tree.nodes[x[feature_index]]
            return self.predict_for_instance(x, sub_tree)
        else:
            return tree.val

    def predict(self, X):
        if isinstance(X, pd.DataFrame):
            X_values = X.values
        else:
            X_values = X

        predictions = [self.predict_for_instance(
            x, self.tree) for x in X_values]

        return predictions
