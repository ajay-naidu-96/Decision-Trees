import math
import numpy as np
import itertools
import graphviz
import pandas as pd
import random

class Node:
    def __init__(self, feature=None, value=None, left=None, right=None):
        self.feature = feature
        self.left = left
        self.right = right
        self.value = value

class ID3DecisionTree:
    def __init__(self, depth, cost_func, enable_categorical_splits):
        self.dtree = None
        self.max_depth = depth
        self.func = cost_func
        self.root = None
        self.enable_categorical_options = enable_categorical_splits
        self.dominant_class = None
        self.dominant_prob = None

    def fit(self, data, target_factor):

        self.dominant_class = data[target_factor].value_counts().nlargest(2).index[0]
        self.non_dominant_class = data[target_factor].value_counts().nlargest(2).index[-1]
        self.dominant_prob = data[target_factor].value_counts().max() / data.shape[0]
    
        self.root = self.train_tree(data, target_factor)

    def gini_impurity(self, y):
        
        if isinstance(y, pd.Series):
            p = y.value_counts()/y.shape[0]
            gini = 1-np.sum(p**2)
            return(gini)
        
        else:
            raise('Object must be a Pandas Series.')

    def entropy(self, y):

        if isinstance(y, pd.Series):
            a = y.value_counts()/y.shape[0]
            entropy = np.sum(-a*np.log2(a+1e-9))
            return(entropy)

        else:
            raise('Object must be a Pandas Series.')

    def information_gain(self, y, mask):
        
        a = sum(mask)
        b = mask.shape[0] - a
        
        if(a == 0 or b ==0): 
            ig = 0
        
        else:
        
            if self.func == 1:
                ig = self.entropy(y)-a/(a+b)*self.entropy(y[mask])-b/(a+b)*self.entropy(y[-mask])
            else:
                ig = self.gini_impurity(y)-a/(a+b)*self.gini_impurity(y[mask])-b/(a+b)*self.gini_impurity(y[-mask])

        return ig

    def categorical_options(self, a):
 
        a = a.unique()

        sample_options = []

        if not self.enable_categorical_options:
            for unique_val in a:
                sample_options.append([unique_val])

        else:
            for L in range(0, len(a)+1):
                for subset in itertools.combinations(a, L):
                    subset = list(subset)
                    sample_options.append(subset)

        return sample_options[1:-1]

    def max_information_gain_split(self, x, y):

        split_value = []
        ig = [] 
        
        options = self.categorical_options(x)

        # Calculate ig for all values
        for val in options:
            mask =  x.isin(val)
            val_ig = self.information_gain(y, mask)
            ig.append(val_ig)
            split_value.append(val)

        if len(ig) == 0:
            return(None,None,None, False)

        else:
            best_ig = max(ig)
            best_ig_index = ig.index(best_ig)
            best_split = split_value[best_ig_index]
            return(best_ig,best_split, False, True)

    def get_best_split(self, y, data):

        masks = data.drop(y, axis= 1).apply(self.max_information_gain_split, y = data[y])

        if sum(masks.loc[3,:]) == 0:
            return(None, None, None, None)

        else:
            masks = masks.loc[:,masks.loc[3,:]]

            split_variable = masks.iloc[0].astype(np.float32).idxmax()
            split_value = masks[split_variable][1] 
            split_ig = masks[split_variable][0]
            split_numeric = masks[split_variable][2]

            return(split_variable, split_value, split_ig, split_numeric)

    def make_split(self, variable, value, data, is_numeric):

        data_1 = data[data[variable].isin(value)]
        data_2 = data[(data[variable].isin(value)) == False]

        return(data_1,data_2)

    def train_tree(self, data, target_factor, depth=0):

        if (depth >= self.max_depth or len(data[target_factor].value_counts()) == 1):
            return Node(value=data[target_factor].value_counts().idxmax())

        var,val,ig,var_type = self.get_best_split(target_factor, data)

        left, right = self.make_split(var, val, data, var_type)

        left = self.train_tree(left, target_factor, depth + 1)
        right = self.train_tree(right, target_factor, depth + 1)

        return Node(var, val, left, right)

    def visualize_tree(self):
        dot = graphviz.Digraph()
        dot.attr(rankdir='TB')

        def add_node(node, parent_id=None, edge_label=None):
            if node is None:
                return

            node_id = str(id(node))
            
            if node.left is None or node.right is None:
                label = f"Class: {node.value}"
                dot.node(node_id, label, shape='box')
            else:
                label = f"{node.feature} = {"-".join(node.value)}"
                dot.node(node_id, label, shape='oval')
            
            if parent_id:
                dot.edge(parent_id, node_id, edge_label)

            add_node(node.left, node_id, "Yes")
            add_node(node.right, node_id, "No")

        add_node(self.root)
        
        dot.render("decision_tree", format="png", cleanup=True)
    
    def calculate_baseline_accuracy(self, y_true):
        y_pred = [random.choices([self.dominant_class, self.non_dominant_class], weights=[self.dominant_prob, 1-self.dominant_prob])[0] for _ in range(y_true.shape[0])]

        accuracy, error = self.calculate_accuracy(y_true, y_pred)
        print("Accuracy: ", accuracy)
        print("Error: ", error)
        print("Precision: ", self.calculate_precision(y_true, y_pred))
        print("Recall: ", self.calculate_recall(y_true, y_pred))

    def predict_sample(self, row, node):

        if node.left is None or node.right is None:
            return node.value

        if row[node.feature] in node.value:
            return self.predict_sample(row, node.left)
        
        return self.predict_sample(row, node.right)

    def predict(self, test, target):

        y_true = test[target].tolist()
        y_pred = []
        for idx, row in test.drop(target, axis=1).iterrows():
            y_pred.append(self.predict_sample(row, self.root))
        
        accuracy, error = self.calculate_accuracy(y_true, y_pred)
        
        print("Accuracy: ", accuracy)
        print("Error: ", error)
        print("Precision: ", self.calculate_precision(y_true, y_pred))
        print("Recall: ", self.calculate_recall(y_true, y_pred))

    def calculate_precision(self, y_true, y_pred):
    
        true_positives = sum(1 for true, pred in zip(y_true, y_pred) if true == "M" and pred == "M")
        false_positives = sum(1 for true, pred in zip(y_true, y_pred) if true == "B" and pred == "M")
        
        if (true_positives + false_positives) == 0:
            return 0
        
        precision = true_positives / (true_positives + false_positives)
        return precision

    def calculate_recall(self, y_true, y_pred):
    
        true_positives = sum(1 for true, pred in zip(y_true, y_pred) if true == "M" and pred == "M")
        false_negatives = sum(1 for true, pred in zip(y_true, y_pred) if true == "M" and pred == "B")
        
        if (true_positives + false_negatives) == 0:
            return 0
        
        recall = true_positives / (true_positives + false_negatives)

        return recall

    def calculate_accuracy(self, y_true, y_pred):

        correct = sum(1 for true, pred in zip(y_true, y_pred) if true == pred)
        incorrect = sum(1 for true, pred in zip(y_true, y_pred) if true != pred)

        accuracy = correct / len(y_true)
        error = incorrect/ len(y_true)

        return (accuracy, error)
