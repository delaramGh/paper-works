# step 1: extract features from the data (preprocessing) 
# step 2: split the dataset into train (10%) and test (90%)
# step 3: train the SVM model on the training data
# step 4: caculate models confidence on the test data and decide which ones can be labeled
# step 5: choose 2% samples for human annotation
# step 6: re-train the model with the new annotated data
# step 7: if any sample is unlabeled, go to step 4


import pandas as pd
import numpy as np
from tqdm import tqdm
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.cluster import KMeans
from sklearn import tree


class Active_Learning():
    def __init__(self, csv_file, model, inputs, threshold, s1, s2, print_):
        self.csv_file = csv_file
        self.inputs = inputs
        self. model = model
        self.threshold = threshold
        self.split1 = s1
        self.split2 = s2
        self.print = print_



    def model_train(self, x_train, y_train):
        if self.model == "SVM":
            clf = make_pipeline(StandardScaler(), SVC(gamma='auto', probability=True))
        if self.model == "Logistic Regression":
            clf = LogisticRegression()
        if self.model == "Random Forest":
            clf = RandomForestClassifier(n_estimators=200, random_state=0)
        if self.model == "Decision Tree":
            clf = tree.DecisionTreeClassifier()
        if self.model == "Kmeans":
            clf = KMeans(n_clusters=2)
        clf.fit(x_train, y_train)
        return clf


    def create_dataset(self):
        df = pd.read_csv(self.csv_file)
        size = df.shape[0]

        x_train = []
        for i in range(int(size*self.split1)):
            x = []
            for metric in self.inputs:
                x.append(df[metric][i])
            x_train.append(x)
        x_train = np.array(x_train)

        y_train = np.array(df["label"][:int(self.split1*size)])
        if self.print:
            print(y_train.shape[0], " new data are labeled manually!")

        label = np.concatenate((y_train , -1 * np.ones(size-len(y_train))))
        df["machine_labels"] = label
        df.to_csv(self.csv_file)

        return size, (x_train, y_train)
        

    def predict(self, model):
        #this function uses the trained model to label the data that we are confident about.
        #it writes the labels in the csv file.
        df = pd.read_csv(self.csv_file)

        cnt = 0
        labels = np.copy(df["machine_labels"])
        for i in range(len(labels)):
            if labels[i] == -1:
                x = []
                for metric in self.inputs:
                    x.append(df[metric][i])
                x = np.array([x])
                pred = model.predict_proba(x)
                if pred[0, 1] > self.threshold:
                    labels[i] = 1
                    cnt += 1
                elif pred[0, 0] > self.threshold:
                    labels[i] = 0
                    cnt += 1
        df["machine_labels"] = labels
        df.to_csv(self.csv_file)
        if self.print:
            print(cnt, " new data are labeled automatically! :)")
        return cnt
                    

    def new_training_data(self):
        df = pd.read_csv(self.csv_file)
        labels = np.copy(df["machine_labels"])

        x_train = []
        y_train = []
        cnt = 0
        for i in range(len(labels)):
            if labels[i] == -1:
                cnt += 1
                x_train.append(np.array([df[metric][i] for metric in self.inputs]))
                y_train.append(int(df["label"][i]))
                labels[i] = df["label"][i]
                if cnt >= int(len(labels)*self.split2):
                    break
        x_train = np.array(x_train)
        y_train = np.array(y_train)
        df["machine_labels"] = labels
        df.to_csv(self.csv_file)
        if self.print:
            print(y_train.shape[0], " new data are labeled manually!")
        return x_train, y_train
    

    def report(self, manual):
        ## final eval
        df = pd.read_csv(self.csv_file)
        df.drop(df.columns[df.columns.str.contains('unnamed', case=False)], axis=1, inplace=True)
        df.to_csv(self.csv_file)
        true_label = df["label"]
        active_learning = df["machine_labels"]
        ok = np.sum(true_label==active_learning)
        all = len(true_label)

        human_effort = (100*manual/all)
        acc = 100*ok/all
        tp = 0
        for i in range(len(active_learning)):
            if active_learning[i] == 1 and true_label[i] == 1:
                tp += 1
        precision = 100*tp / np.sum(active_learning)
        recall = 100*tp / np.sum(true_label)

        print("***    REPORT    ***")
        # print("+ number of automatically labeled data: ", all - manual, " out of ", all)
        print("+ human effort is:    ", str(human_effort)[:6])
        print("+ final accuracy is:  ", str(acc)[:6])
        print("+ final precision is: ", str(precision)[:6])
        print("+ final recall is:    ", str(recall)[:6])
        # print("+ number of missed data: ", all-ok, "(", str(100*(all-ok)/all)[:6], "%)")
        print("\n")
        
        return human_effort, acc, precision, recall


###############################################################
def active_labeling(csv_name, model_, inputs, threshold, split1=0.2, split2=0.05, print_=False):
    module = Active_Learning(csv_name, model_, inputs, threshold, split1, split2, print_)

    print("***  PARAMETERS  ***\nmodel: ", model_, ", inputs: ", inputs, ", Th: ", threshold, ", split_1: ", split1, ", split_2: ", split2)
    number_of_samples, (x_train, y_train) = module.create_dataset()
    labeled_samples = len(y_train)
    if print_:
        print("shape: ", x_train.shape, y_train.shape)
    model = module.model_train(x_train, y_train)
    labeled_samples += module.predict(model)
    if print_:
        print("\n")
    for i in range(500):
        if labeled_samples == number_of_samples:
            human_effort, acc, precision, recall = module.report(y_train.shape[0])
            param = [model_, threshold, split1, split2]
            return human_effort, acc, precision, recall, param
        
        if print_:
            print("* MAIN LOOP - ", i, "th iteration - ", str(100*labeled_samples/number_of_samples)[:2], "% progress")
        x, y = module.new_training_data()
        labeled_samples += len(y)
        x_train = np.concatenate((x_train, x))
        y_train = np.concatenate((y_train, y))
        model = module.model_train(x_train, y_train)
        labeled_samples += module.predict(model)
        if print_:
            print("\n")

            
    
###############################################################
if __name__ == "__main__":
    models = ["Logistic Regression"]
    active_labeling("..\\cifar dataset\\test_dataset.csv", models[0], ["PSNR", "CPL"], 0.9, split1=0.2, split2=0.05, print_=True)


