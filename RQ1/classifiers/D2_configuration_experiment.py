from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.cluster import KMeans
from sklearn import tree
import pandas as pd
import numpy as np
from tqdm import tqdm
from sklearn.utils import shuffle
import pickle

from classification_all import create_x_y_v2, fit_and_acc


config_4 = ["Hist_int", "CS", "KL", "CPL"]

config_31 = ["CS", "KL", "CPL"]
config_32 = ["Hist_int", "KL", "CPL"]
config_33 = ["Hist_int", "CS", "CPL"]
config_34 = ["Hist_int", "CS", "KL"]

config_21 = ["Hist_int", "CS"]
config_22 = ["Hist_int", "KL"]
config_23 = ["Hist_int","CPL"]
config_24 = ["CS", "KL"]
config_25 = ["CS", "CPL"]
config_26 = ["KL", "CPL"]

config_11 = ["Hist_int"]
config_12 = ["CS"]
config_13 = ["KL"]
config_14 = ["CPL"]

configs = [config_11, config_12, config_13, config_14, config_21, config_22, config_23, config_24, config_25, config_26, config_31, config_32, config_33, config_34, config_4]
model_names = ["logistic-regression", "SVM", "random-forest", "decision-tree"] 
models = [LogisticRegression(), make_pipeline(StandardScaler(), SVC(gamma='auto')), RandomForestClassifier(n_estimators=200, random_state=0), tree.DecisionTreeClassifier()]#, KMeans(n_clusters=2, n_init=5)]

result = []
for j in range(len(configs)):
    csv_file = "C:\\Users\\ASUS\\Desktop\\research\\mitacs project\\paper experiments\\cifar dataset\\test_dataset.csv"
    X, Y = create_x_y_v2(csv_file, configs[j])
    n = int(0.75*X.shape[0])
    x_train = X[:n]
    x_test  = X[n:]
    y_train = Y[:n]
    y_test  = Y[n:]
    for i in range(len(models)):
        _, acc = fit_and_acc(models[i], model_names[i], x_train, y_train, x_test, y_test)
        result.append([model_names[i], acc, configs[j]])
        print("\n")


import csv
with open('exp1_D2test_configuration.csv', 'w') as f:
    write = csv.writer(f)
    write.writerow(["model", "accuracy", "config"])
    write.writerows(result)

