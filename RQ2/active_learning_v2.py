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
from sklearn.utils import shuffle
from svm_classifier_function import preprocessing

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.cluster import KMeans
from sklearn import tree


def model_train(x_train, y_train, model):
    if model == "SVM":
        clf = make_pipeline(StandardScaler(), SVC(gamma='auto', probability=True))
    if model == "Logistic Regression":
        clf = LogisticRegression()
    if model == "Random Forest":
        clf = RandomForestClassifier(n_estimators=200, random_state=0)
    if model == "Decision Tree":
        clf = tree.DecisionTreeClassifier()
    if model == "Kmeans":
        clf = KMeans(n_clusters=2)
    clf.fit(x_train, y_train)
    return clf


def create_dataset(df_name, split, print_):
    df = pd.read_csv(df_name)
    size = df.shape[0]
    #preprocessing
    if 0:
        psnr = []
        ssim = []
        cpl = []
        cs = []
        gen_path = "C:\\Users\\ASUS\\Desktop\\research\\mitacs project\\paper experiments\\taoyu dataset\\taoyu_gen_dataset\\"
        org_path = "C:\\Users\\ASUS\\Desktop\\research\\mitacs project\\paper experiments\\taoyu dataset\\taoyu_org_dataset\\"
        for i in tqdm(range(size)):
            psnr_, ssim_, cpl_, cs_ = preprocessing(org_path+df["original"][i], gen_path+df["gen"][i])
            psnr.append(psnr_)
            ssim.append(ssim_)
            cpl.append(cpl_)
            cs.append(cs_)
        df["PSNR"] = psnr 
        df["SSIM"] = ssim 
        df["CPL"] = cpl 
        df["CS"] = cs 
        df.to_csv(df_name)

    x_train = np.array([np.array([df["PSNR"][i], df["SSIM"][i], df["CPL"][i], df["CS"][i]]) for i in range(int(size*split))])
    y_train = np.array(df["label"][:int(split*size)])
    if print_:
        print(y_train.shape[0], " new data are labeled manually!")

    label = np.concatenate((y_train , -1 * np.ones(size-len(y_train))))
    df["machine_labels"] = label
    df.to_csv(df_name)

    return size, (x_train, y_train)
    

def predict(model, df_name, threshold, print_):
    #this function uses the trained model to label the data that we are confident about.
    #it writes the labels in the csv file.
    df = pd.read_csv(df_name)

    cnt = 0
    labels = np.copy(df["machine_labels"])
    for i in range(len(labels)):
        if labels[i] == -1:
            x = np.array([[df["PSNR"][i], df["SSIM"][i], df["CPL"][i], df["CS"][i]]])
            pred = model.predict_proba(x)
            if pred[0, 1] > threshold:
                labels[i] = 1
                cnt += 1
            elif pred[0, 0] > threshold:
                labels[i] = 0
                cnt += 1
    df["machine_labels"] = labels
    df.to_csv(df_name)
    if print_:
        print(cnt, " new data are labeled automatically! :)")
    return cnt
                

def new_training_data(df_name, split, print_):
    df = pd.read_csv(df_name)
    labels = np.copy(df["machine_labels"])

    x_train = []
    y_train = []
    cnt = 0
    for i in range(len(labels)):
        if labels[i] == -1:
            cnt += 1
            x_train.append(np.array([df["PSNR"][i], df["SSIM"][i], df["CPL"][i], df["CS"][i]]))
            y_train.append(int(df["label"][i]))
            labels[i] = df["label"][i]
            if cnt >= int(len(labels)*split):
                break
    x_train = np.array(x_train)
    y_train = np.array(y_train)
    df["machine_labels"] = labels
    df.to_csv(df_name)
    if print_:
        print(y_train.shape[0], " new data are labeled manually!")
    return x_train, y_train


def __new_model_eval__(model, df_name):
    df = pd.read_csv(df_name)
    size = df.shape[0]
    x_test = np.array([np.array([df["PSNR"][i], df["SSIM"][i], df["CPL"][i], df["CS"][i]]) for i in range(size)]) 
    y_test = np.array(df["label"])
    ans = model.predict(x_test)
    print("accuracy of this model is: ", 100*np.sum(ans == y_test)/size, "\n\n")


def report(df_name, manual):
    ## final eval
    df = pd.read_csv(df_name)
    df.drop(df.columns[df.columns.str.contains('unnamed', case=False)], axis=1, inplace=True)
    df.to_csv(df_name)
    true_label = df["label"]
    active_learning = df["machine_labels"]
    ok = np.sum(true_label==active_learning)
    all = len(true_label)

    human_effort = (100*manual/all)
    acc = 100*ok/all

    print("***    REPORT    ***")
    print("+ number of automatically labeled data: ", all - manual, " out of ", all)
    print("+ human effort is: ", str(human_effort)[:4])
    print("+ final accuracy is: ", str(acc)[:4])
    print("+ number of missed data: ", all-ok, "(", str(100*(all-ok)/all)[:4], "%)")
    print("\n")
    
    return human_effort, acc


###############################################################
def active_labeling(csv_name, model_, threshold, split1=0.2, split2=0.05, print_=False):
    print("***  PARAMETERS  ***\nmodel: ", model_, ", Th: ", threshold, ", split_1: ", split1, ", split_2: ", split2)
    number_of_samples, (x_train, y_train) = create_dataset(csv_name, split1, print_)
    labeled_samples = len(y_train)
    if print_:
        print("shape: ", x_train.shape, y_train.shape)
    model = model_train(x_train, y_train, model_)
    labeled_samples += predict(model, csv_name, threshold, print_)
    if print_:
        print("\n")
    for i in range(500):
        if labeled_samples == number_of_samples:
            human_effort, acc = report(csv_name, y_train.shape[0])
            param = [model_, threshold, split1, split2]
            return human_effort, acc, param
        
        if print_:
            print("* MAIN LOOP - ", i, "th iteration - ", str(100*labeled_samples/number_of_samples)[:2], "% progress")
        x, y = new_training_data(csv_name, split2, print_)
        labeled_samples += len(y)
        x_train = np.concatenate((x_train, x))
        y_train = np.concatenate((y_train, y))
        model = model_train(x_train, y_train, model_)
        labeled_samples += predict(model, csv_name, threshold, print_)
        if print_:
            print("\n")

            
    
###############################################################
if __name__ == "__main__":
    models = ["Logistic Regression"]
    active_labeling("..\\cifar dataset\\test_dataset.csv", models[0], 0.95, split1=0.1, split2=0.01, print_=True)


