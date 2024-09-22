import pandas as pd
from sklearn import tree
import numpy as np
from sklearn.utils import shuffle
import matplotlib.pyplot as plt
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer




csv_file = "exp2_D1_NEW_search__all.csv"
df = pd.read_csv(csv_file)

print(df.columns)

if 0: #assign labels
    dt_label = []
    cnt = 0
    for i in range(len(df)):
        if df["accuracy"][i] > 96 and df["human effort"][i] < 40:
            dt_label.append(1)
            cnt+=1
        else:
            dt_label.append(0)


    print(cnt)
    df["dt_label_96_40"] = dt_label
    df.to_csv(csv_file)




if 1:
    X = df[["threshold", "split_1", "split_2", "model"]]
    Y = df["dt_label_99_60"]

    preprocessor = ColumnTransformer(transformers=[('cat', OneHotEncoder(), ['model'])], 
                                     remainder='passthrough')  # Keep other columns (numerical) unchanged
    X, Y = shuffle(X, Y, random_state=5) #(720, 4) (720,)
    X = preprocessor.fit_transform(X)
    # Y = np.array(Y)
    clf = tree.DecisionTreeClassifier(max_depth=4)
    clf.fit(X, Y)
    plt.figure(figsize=(12,8))  # Set the figure size
    tree.plot_tree(clf, feature_names=["DT", "LR", "RF", "SVM", "ALPHA", "BETA", "SPLIT"], 
                   class_names=["No", "Yes"], filled=False,
                   label="all", proportion=True)
    plt.savefig("decision_tree_visualization_99_60_depth4.pdf")
    plt.show()




if 0:
    X = df[["threshold", "split_1", "split_2", "classifier"]]
    Y = df["dt_label_99_60"]

    X, Y = shuffle(X, Y, random_state=5) #(720, 4) (720,)

    clf = tree.DecisionTreeClassifier(max_depth=5)
    clf.fit(X, Y)
    plt.figure(figsize=(12,8))  # Set the figure size
    tree.plot_tree(clf, feature_names=["ALPHA", "BETA", "S2", "MODEL"], class_names=["No", "Yes"],
                   filled=False, label="all", proportion=True)
    # plt.title("")
    plt.show()
