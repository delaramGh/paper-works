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


def model_train(x_train, y_train):
    clf = make_pipeline(StandardScaler(), SVC(gamma='auto', probability=True))
    clf.fit(x_train, y_train)
    return clf


def create_dataset(df_name, split=0.1):
    df = pd.read_csv(df_name)
    size = df.shape[0]
    #preprocessing
    if 0:
        psnr = []
        ssim = []
        cpl = []
        cs = []
        for i in tqdm(range(size)):
            psnr_, ssim_, cpl_, cs_ = preprocessing(df["original"][i], df["gen"][i])
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
    print(y_train.shape[0], " new data are labeled manually!")

    label = np.concatenate((y_train , -1 * np.ones(size-len(y_train))))
    df["machine_labels"] = label
    df.to_csv(df_name)

    return size, (x_train, y_train)
    

def predict(model, df_name, threshold=0.95):
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
    print(cnt, " new data are labeled automatically! :)")
    return cnt
                

def new_train_data(df_name, split=0.05):
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
    print("***    REPORT    ***")
    print("+ number of automatically labeled data: ", len(true_label) - manual, " out of ", len(true_label))
    print("+ human effort is: ", str(100*manual/len(true_label))[:4])
    print("+ final accuracy is: ", str(100*ok/len(true_label))[:4])
    print("+ number of missed data: ", len(true_label)-ok)
    print("\n")


###############################################################
def main():
    csv_name = "test_dataset.csv"
    threshold = 0.95

    number_of_samples, (x_train, y_train) = create_dataset(csv_name, split=0.2)
    labeled_samples = len(y_train)
    print("shape: ", x_train.shape, y_train.shape)
    model = model_train(x_train, y_train)
    # __new_model_eval__(model, csv_name)
    labeled_samples += predict(model, csv_name, threshold)
    print("\n")


    for i in range(10):
        print("* MAIN LOOP - ", i, "th iteration - ", str(100*labeled_samples/number_of_samples)[:2], "% progress")
        x, y = new_train_data(csv_name, split=0.05)
        labeled_samples += len(y)
        x_train = np.concatenate((x_train, x))
        y_train = np.concatenate((y_train, y))
        model = model_train(x_train, y_train)
        labeled_samples += predict(model, csv_name, threshold)
        print("\n")

        if labeled_samples == number_of_samples:
            report(csv_name, y_train.shape[0])
            return 0
    

###############################################################
main()

