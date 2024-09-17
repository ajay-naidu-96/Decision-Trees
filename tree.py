import math
import numpy as np
import itertools
from collections import Counter
import graphviz
import pandas as pd

class Node:
    def __init__(self, feature=None, value=None, left=None, right=None):
        self.feature = feature
        self.left = left
        self.right = right
        self.value = value

class ID3DecisionTree:
    def __init__(self, depth, sample_split, info_gain, cost):
        self.dtree = None
        self.max_depth = depth
        self.min_information_gain = info_gain
        self.max_categories = 20
        self.counter = 0
        self.min_samples_split = sample_split
        self.func = cost
        self.root = None

    def fit(self, data, target_factor):
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

    def variance(self, y):
        '''
        Function to help calculate the variance avoiding nan.
        y: variable to calculate variance to. It should be a Pandas Series.
        '''
        if(len(y) == 1):
            return 0
        else:
            return y.var()

    def information_gain(self, y, mask):
        '''
        It returns the Information Gain of a variable given a loss function.
        y: target variable.
        mask: split choice.
        func: function to be used to calculate Information Gain in case os classification.
        '''
        
        a = sum(mask)
        b = mask.shape[0] - a
        
        if(a == 0 or b ==0): 
            ig = 0
        
        else:
            if y.dtypes != 'O':
                ig = variance(y) - (a/(a+b)* variance(y[mask])) - (b/(a+b)*variance(y[-mask]))
            else:
                if self.func == 1:
                    ig = self.entropy(y)-a/(a+b)*self.entropy(y[mask])-b/(a+b)*self.entropy(y[-mask])
                else:
                    ig = self.gini_impurity(y)-a/(a+b)*self.gini_impurity(y[mask])-b/(a+b)*self.gini_impurity(y[-mask])

        return ig

    def categorical_options(self, a):
        '''
        Creates all possible combinations from a Pandas Series.
        a: Pandas Series from where to get all possible combinations. 
        '''
        a = a.unique()

        opciones = []
        for L in range(0, len(a)+1):
            for subset in itertools.combinations(a, L):
                subset = list(subset)
                opciones.append(subset)

        return opciones[1:-1]

    def max_information_gain_split(self, x, y):
        '''
        Given a predictor & target variable, returns the best split, the error and the type of variable based on a selected cost function.
        x: predictor variable as Pandas Series.
        y: target variable as Pandas Series.
        func: function to be used to calculate the best split.
        '''

        split_value = []
        ig = [] 

        numeric_variable = True if x.dtypes != 'O' else False

        # Create options according to variable type
        if numeric_variable:
            options = x.sort_values().unique()[1:]
        else: 
            options = self.categorical_options(x)

        # Calculate ig for all values
        for val in options:
            mask =   x < val if numeric_variable else x.isin(val)
            val_ig = self.information_gain(y, mask)
            # Append results
            ig.append(val_ig)
            split_value.append(val)

        # Check if there are more than 1 results if not, return False
        if len(ig) == 0:
            return(None,None,None, False)

        else:
        # Get results with highest IG
            best_ig = max(ig)
            best_ig_index = ig.index(best_ig)
            best_split = split_value[best_ig_index]
            return(best_ig,best_split,numeric_variable, True)

    def get_best_split(self, y, data):
        '''
        Given a data, select the best split and return the variable, the value, the variable type and the information gain.
        y: name of the target variable
        data: dataframe where to find the best split.
        '''
        masks = data.drop(y, axis= 1).apply(self.max_information_gain_split, y = data[y])

        if sum(masks.loc[3,:]) == 0:
            return(None, None, None, None)

        else:
            # Get only masks that can be splitted
            masks = masks.loc[:,masks.loc[3,:]]

            # Get the results for split with highest IG
            split_variable = masks.iloc[0].astype(np.float32).idxmax()
            #split_valid = masks[split_variable][]
            split_value = masks[split_variable][1] 
            split_ig = masks[split_variable][0]
            split_numeric = masks[split_variable][2]

            return(split_variable, split_value, split_ig, split_numeric)

    def make_split(self, variable, value, data, is_numeric):
        '''
        Given a data and a split conditions, do the split.
        variable: variable with which make the split.
        value: value of the variable to make the split.
        data: data to be splitted.
        is_numeric: boolean considering if the variable to be splitted is numeric or not.
        '''
        if is_numeric:
            data_1 = data[data[variable] < value]
            data_2 = data[(data[variable] < value) == False]

        else:
            data_1 = data[data[variable].isin(value)]
            data_2 = data[(data[variable].isin(value)) == False]

        return(data_1,data_2)

    def make_prediction(self, data, target_factor):
        '''
        Given the target variable, make a prediction.
        data: pandas series for target variable
        target_factor: boolean considering if the variable is a factor or not
        '''

        # Make predictions
        if target_factor:
            pred = data.value_counts().idxmax()
        else:
            pred = data.mean()

        return pred

    def train_tree(self, data, target_factor, depth=0):

        if (depth >= self.max_depth or len(data[target_factor].value_counts()) == 1):
            return Node(value=data[target_factor].value_counts().idxmax())

        var,val,ig,var_type = self.get_best_split(target_factor, data)

        print("Var: {0}, Val: {1}, Ig: {2}".format(var, val, ig))

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

            # if node.value is not None:
            #     label = f"Class: {node.value}"
            #     dot.node(node_id, label, shape='box')
            # else:
            #     feature_name = node.feature
            #     # if tree.feature_types[node.feature] == 'categorical':

            #     print(feature_name)
            #     print(node.value)
            #     label = f"{feature_name} = {"-".join(node.value)}"
            #     # else:
            #     #     label = f"{feature_name} < {node.threshold:.2f}"
            label = f"{node.feature} = {"-".join(node.value)}"
            dot.node(node_id, label, shape='oval')
            
            if parent_id:
                dot.edge(parent_id, node_id, edge_label)

            add_node(node.left, node_id, "Yes")
            add_node(node.right, node_id, "No")

        add_node(self.root)
        
        dot.render("decision_tree", format="png", cleanup=True)


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
        accuracy = correct / len(y_true)

        return accuracy
