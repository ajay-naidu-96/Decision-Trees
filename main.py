from tree import ID3DecisionTree
import pandas as pd
import numpy as np
import argparse

if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument('--data_root', type=str, required=True, default="./Dataset")
    parser.add_argument('--target_column_name', type=str, required=True, default="Diagnosis")
    parser.add_argument('--cost', type=int, default=2)
    parser.add_argument('--max_depth', type=int, default=3)
    parser.add_argument('--prune', type=bool, default=False)
    parser.add_argument('--enable_categorical_splits', type=bool, default=True)

    args = parser.parse_args()

    train_path = args.data_root + "/train.csv"
    val_path = args.data_root + "/val.csv"
    test_path = args.data_root + "/test.csv"

    df_train = pd.read_csv(train_path)
    df_val = pd.read_csv(val_path)
    df_test = pd.read_csv(test_path)

    # max_depth = 3
    # cost = 2 # entropy = 1 gini impurity = 2

    id3 = ID3DecisionTree(args.max_depth, args.cost)

    id3.fit(df_train, args.target_column_name)

    id3.visualize_tree()

    print("Val Accuracy Metrics: ")
    id3.predict(df_val, args.target_column_name)
    print("-"*50)
    print("Test Accuracy Metrics: ")
    id3.predict(df_test, args.target_column_name)
    print("-"*50)



