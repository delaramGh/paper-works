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

from classification_module import create_x_y_v2, fit_and_acc


config_4 = ["PSNR", "CS", "SSIM", "CPL"]

config_31 = ["CS", "SSIM", "CPL"]
config_32 = ["PSNR", "SSIM", "CPL"]
config_33 = ["PSNR", "CS", "CPL"]
config_34 = ["PSNR", "CS", "SSIM"]

config_21 = ["PSNR", "CS"]
config_22 = ["PSNR", "SSIM"]
config_23 = ["PSNR","CPL"]
config_24 = ["CS", "SSIM"]
config_25 = ["CS", "CPL"]
config_26 = ["SSIM", "CPL"]

config_11 = ["PSNR"]
config_12 = ["CS"]
config_13 = ["SSIM"]
config_14 = ["CPL"]


# configs = [config_11, config_12, config_13, config_14, config_21, config_22, config_23, config_24, config_25, config_26, config_31, config_32, config_33, config_34, config_4]
# model_names = ["logistic-regression", "SVM", "random-forest", "decision-tree"] 
# models = [LogisticRegression(), make_pipeline(StandardScaler(), SVC(gamma='auto')), RandomForestClassifier(n_estimators=200, random_state=0), tree.DecisionTreeClassifier()]#, KMeans(n_clusters=2, n_init=5)]


configs = [["PSNR", "CS", "CPL"]]
model_names = ["SVM"]
models = [make_pipeline(StandardScaler(), SVC(gamma='auto'))]



result = []
for j in range(len(configs)):
    csv_file = "C:\\Users\\ASUS\\Desktop\\research\\mitacs project\\paper experiments\\smartInside dataset\\test_dataset.csv"
    X, Y = create_x_y_v2(csv_file, configs[j])

    effort = 0.75

    n = int(effort*X.shape[0])
    x_train = X[:n]
    x_test  = X[n:]
    y_train = Y[:n]
    y_test  = Y[n:]
    for i in range(len(models)):
        print("\n")
        _, acc, precision, recall = fit_and_acc(models[i], model_names[i], x_train, y_train, x_test, y_test)
        result.append([model_names[i], acc, precision, recall, configs[j]])
        print("input parameters: ", configs[j])
        print("human effort: ", effort)
        print("\n")


# import csv
# with open('exp1_D1_configuration_0_25.csv', 'w') as f:
#     write = csv.writer(f)
#     write.writerow(["model", "accuracy", "precision", "recall", "config"])
#     write.writerows(result)

