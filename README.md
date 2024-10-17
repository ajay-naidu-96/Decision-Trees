# Running Python Code 


## 1. Unzip the file
The folder consists of following structure
```
├── ps1-gopi
    ├── Assignment 1.ipynb 
    ├── Assignment Report 1.pdf
    ├── DecisionTree
    │   ├── Dataset
    │   │   ├── test.csv
    │   │   ├── train.csv
    │   │   └── val.csv
    │   ├── decision_tree.png
    │   ├── main.py
    │   └── tree.py
    └── ps1
        ├── Discretization_prepartion.ipynb
        ├── Final_data
        │   ├── wdbc_dev.csv
        │   ├── wdbc_dev_normalized.csv
        │   ├── wdbc_dev_raw.csv
        │   ├── wdbc_test.csv
        │   ├── wdbc_test_normalized.csv
        │   ├── wdbc_test_raw.csv
        │   ├── wdbc_train.csv
        │   ├── wdbc_train_normalized.csv
        │   └── wdbc_train_raw.csv
        ├── IntroToWDBC_DecisionTree.ipynb
        ├── README-610.md
        ├── requirements.txt
        └── wdbc.data

    
```    
Unzip the zip file
```
unzip ps1-gopi.zip
```

## 2. Install all the dependencies
Ensure you are within `ps1-gopi` folder.
```
pip install -r requirements.txt
```

This may take a minute or two.

## 3. Run the Tree

### 3.1 Change Directory to Decision Tree
```
cd DecisionTree
```

### 3.2 Run the Decision Tree Generator
```
python3 main.py --data_root ./Dataset/ --target_column_name Diagnosis --cost 2 --enable_categorical_splits False
```

List of Available Params
```
--data_root #root dir for train test and val set
--cost # 1 for Ig and 2 for Gini
--enable_categorical_split # for categorical feature subsetting to include multiple features value
--max_depth # for controlling depth of the tree
--target_column_name # ideally the decision column i.e y_true
--enable_prune # to enable or disable pruning

```

## 4. Question References

Most of the written summary are present in the report. The problem based ones are in 3 places, 
1. code for decision tree in the DecisionTree folder
2. Implementation of Q2 in Assignment 1.ipynb
3. Comparision of trees binned vs non binned is in the same original IntroToWDBC_DecisionTree notebook 







