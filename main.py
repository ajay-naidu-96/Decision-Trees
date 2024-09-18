from tree import ID3DecisionTree
import pandas as pd
import numpy as np

if __name__ == "__main__":

    df_train = pd.read_csv("Dataset/train.csv")
    df_val = pd.read_csv("Dataset/val.csv")

    max_depth = 3
    cost = 2 # entropy = 1 gini impurity = 2

    id3 = ID3DecisionTree(max_depth, cost)

    id3.fit(df_train, 'Diagnosis')

    id3.visualize_tree()

    id3.predict(df_val, 'Diagnosis')


