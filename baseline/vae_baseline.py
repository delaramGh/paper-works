from sklearn import svm
import numpy as np
from sewar.full_ref import vifp
import cv2
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm
from sklearn.utils import shuffle


def shuffle_dataset(dataset):
    df = pd.read_csv(dataset)
    df = shuffle(df)
    df.to_csv(dataset, index=False)


# csv_file = "C:\\Users\\ASUS\\Desktop\\research\\mitacs project\\paper experiments\\cifar dataset\\concept1_cifar_all.csv"
csv_file = "C:\\Users\\ASUS\\Desktop\\research\\mitacs project\\paper experiments\\cifar dataset\\concept1_cifar_all.csv"




if 0:  #plot
    df = pd.read_csv(csv_file)
    human_labels = df["label"]
    human_labels = pd.Series(human_labels)
    ok_imgs = df["VAE"][human_labels == 1]
    lost_imgs = df["VAE"][human_labels == 0]

    plt.plot(ok_imgs, "g*", lost_imgs, "r.")
    plt.title("smartInside test-dataset VAE Compared to Human Label")
    plt.xlabel("Image Index")
    plt.ylabel("VAE")
    plt.legend(["Ok", "Lost"])
    plt.show()


if 0: #ُSVM Threshold
    human_effort_list = []
    acc_list = []
    precision_list = []
    recall_list = []
    for effort in [0.25, 0.5, 0.75]:
        for _ in tqdm(range(20)):
            shuffle_dataset(csv_file)
            df = pd.read_csv(csv_file) 

            X = np.expand_dims(np.array(df["VAE"]), axis=1)
            y = np.array(df["label"])
            
            # clf = svm.SVC(kernel="linear", class_weight='balanced')
            clf = svm.LinearSVC(class_weight='balanced', C=1000, loss='hinge')
            X_train = X[:int(effort*len(y))]
            Y_train = y[:int(effort*len(y))]
            clf.fit(X_train, Y_train)

            pred = clf.predict(X)
            acc = 100*np.sum(pred==y)/len(y)
            tp = np.sum(pred & y)

            if np.sum(pred) != 0:
                precision = 100*tp/np.sum(pred)
            else:
                precision = "NAN"
            recall =  100*tp/np.sum(y)

            human_effort_list.append(effort)
            acc_list.append(acc)
            precision_list.append(precision)
            recall_list.append(recall)
            print('-'*50)
            print('َAccuracy:', acc)
            print('Precision:', precision)
            print('Recall:', recall)

    df = pd.DataFrame({"accuracy":acc_list, "precision":precision_list, 
                        "recall":recall_list, "human effort":human_effort_list})

    df.to_csv(f"D2_concept2_baseline_VAE.csv")


def find_best_th(th_list, x, y):
    accs = []
    for th in th_list:
        pred = np.array(x > th) * 1.0
        accs.append(100*np.sum(pred==y)/len(y))
    accs = np.array(accs)
    return th_list[accs.argmax()]


if 1: #dasti Threshold
    human_effort_list = []
    acc_list = []
    precision_list = []
    recall_list = []
    threshold_list = []
    for effort in [0.25, 0.5, 0.75]:
        for _ in tqdm(range(20)):
            shuffle_dataset(csv_file)
            df = pd.read_csv(csv_file) 

            n = int(effort*len(df["VAE"]))
            x_train = np.array(df["VAE"])[:n]
            y_train = np.array(df["label"])[:n]
            x_test = np.array(df["VAE"])[n:]
            y_test = np.array(df["label"])[n:]

            th_list = np.linspace(x_train.min(), x_train.max(), num=100)
            th = find_best_th(th_list, x_train, y_train)

            pred = np.array(x_test > th) 
            acc = 100*np.sum(pred==y_test)/len(y_test)

            # tp = 0
            # for i in range(len(pred)):
            #     if pred[i] == 1 and y_test[i] == 1:
            #         tp += 1
            tp = np.sum(pred & y_test)

            if np.sum(pred) != 0:
                precision = 100*tp/np.sum(pred)
            else:
                precision = "NAN"
            recall =  100*tp/np.sum(y_test)

            human_effort_list.append(effort)
            acc_list.append(acc)
            precision_list.append(precision)
            recall_list.append(recall)
            threshold_list.append(th)

    df = pd.DataFrame({"accuracy":acc_list, "precision":precision_list, 
                        "recall":recall_list, "human effort":human_effort_list,
                        "threshold": threshold_list})

    df.to_csv(f"D2_concept1_baseline_VAE.csv")
        