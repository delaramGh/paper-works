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



def create_x_y_v2(csv_name, metric_list):
    X = []
    Y = []
    df = pd.read_csv(csv_name)
    for i in range(df.shape[0]):
        x = []
        for metric in metric_list:
            x.append(df[metric][i])
        Y.append(df["label"][i]) 
        X.append(x)
    
    X = np.array(X)
    Y = np.array(Y)
    X, Y = shuffle(X, Y, random_state=5)#5
    return X, Y


######################################################################################
def fit_and_acc(model, model_name, x_train, y_train, x_test, y_test):
    model.fit(x_train, y_train)

    pred = model.predict(x_test)
    correct = np.sum(pred == y_test)
    acc = 1-(y_test.shape[0]-correct)/y_test.shape[0]
    tp = np.sum(pred & y_test)

    if np.sum(pred) == 0:
        precision = -1
    else:
        precision = tp/np.sum(pred)

    if np.sum(y_test) == 0:
        recall = -1
    else:
        recall = tp/np.sum(y_test)
        
    if acc < 0.5:
        acc = 1 - acc
    print(model_name, "Accuracy is: ", str(acc)[:7])
    print("Precision is: ", str(precision)[:7])
    print("Recall is: ", str(recall)[:7])
    return model, acc, precision, recall

######################################################################################

if __name__ == "__main__":
    csv_file = "C:\\Users\\ASUS\\Desktop\\research\\mitacs project\\paper experiments\\smartInside dataset\\configuration_dataset.csv"
    X, Y = create_x_y_v2(csv_file, ["PSNR", "CS", "SSIM"])

    n = 800 #all: 1298
    x_train = X[:n]
    x_test  = X[n:]
    y_train = Y[:n]
    y_test  = Y[n:]
    print("Train data shape: ", x_train.shape, ", ", y_train.shape)
    print("Test data shape: ", x_test.shape, ", ", y_test.shape)

    clf = LogisticRegression()
    clf = fit_and_acc(clf, "logistic-regression", x_train, y_train, x_test, y_test)

    clf = make_pipeline(StandardScaler(), SVC(gamma='auto'))
    clf = fit_and_acc(clf, "SVM", x_train, y_train, x_test, y_test)

    clf = RandomForestClassifier(n_estimators=200, random_state=0)
    clf = fit_and_acc(clf, "random-forest", x_train, y_train, x_test, y_test)

    clf = tree.DecisionTreeClassifier()
    clf = fit_and_acc(clf, "decision-tree", x_train, y_train, x_test, y_test)

    # clf = KMeans(n_clusters=2, n_init=)
    # clf = fit_and_acc(clf, "2-means", x_train, y_train, x_test, y_test)


    if 0: #to save a model
        with open('models\\random_forest.pkl','wb') as f:
            pickle.dump(clf, f)

