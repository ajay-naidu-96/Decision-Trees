from tree import ID3DecisionTree
import pandas as pd
import numpy as np

if __name__ == "__main__":

    # a = ["A SerieA CF Left yes True",
    #  "B LaLiga LW Right no True",
    #  "C LaLiga CF Right yes True",
    #  "D PremierLeague CF Left yes True",
    #  "E PremierLeague LW Left yes False",
    #  "F SerieA RW Left yes False",
    #  "G SerieA CF Right no True",
    #  "H PremierLeague LW Left no False",
    #  "I SerieA RW Right no True",
    #  "J LaLiga RW Left yes False",
    #  "K LaLiga RW Right no True",
    #  "L PremierLeague CF Right no False",
    #  "M LaLiga CF Right yes True",
    #  "N SerieA LW Left no False", 
    # ]

    # df = pd.DataFrame(columns=['Player', 'League', 'Position', 'PreferredFoot', 'Capped', 'Shortlisted'])

    # for item in a:
    #   df.loc[len(df)]= item.split(" ")


    df_train = pd.read_csv("Dataset/train.csv")
    df_val = pd.read_csv("Dataset/val.csv")

    max_depth = 3
    min_samples_split = 15
    min_information_gain  = 1e-5
    cost = 2 #entropy = 1 gini impurity = 2

    id3 = ID3DecisionTree(max_depth, min_samples_split, min_information_gain, cost)

    # masks = df_train.drop(['Diagnosis'], axis= 1).apply(id3.max_information_gain_split, y = df_train.Diagnosis)

    # # Get the results for split with highest IG
    # split_variable = masks.iloc[0].astype(np.float32).idxmax()
    # #split_valid = masks[split_variable][]
    # split_value = masks[split_variable][1] 
    # split_ig = masks[split_variable][0]
    # split_numeric = masks[split_variable][2]      
    
    # print(split_variable)    
    # print(split_ig)                                         

    # decisions = id3.train_tree(df.drop('Player', axis=1),'Shortlisted',True)

    id3.fit(df_train, 'Diagnosis')

    id3.visualize_tree()

    # predictions = id3.predict(df_val.drop('Diagnosis', axis=1))

    # test_val_y = df_val.Diagnosis.tolist()

    # print(id3.calculate_accuracy(test_val_y, predictions))
    # print(id3.calculate_precision(test_val_y, predictions))
    # print(id3.calculate_recall(test_val_y, predictions))


    # 
    # print(df_val.Diagnosis[:10])


