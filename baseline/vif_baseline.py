from sewar.full_ref import vifp
import cv2
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm



csv_file = "C:\\Users\\ASUS\\Desktop\\research\\mitacs project\\paper experiments\\smartInside dataset\\configuration_dataset.csv"
df = pd.read_csv(csv_file) 


if 1: #VIF calculation
    vif = []
    for i in tqdm(range(df.shape[0])):
        vif.append(vifp(cv2.imread(df["original"][i]), cv2.imread(df["gen"][i])))
    df["VIF"] = vif
    df.to_csv(csv_file)


if 0: #plot
    human_labels = df["label"]
    human_labels = pd.Series(human_labels)
    ok_imgs = df["VIF"][human_labels == 1]
    lost_imgs = df["VIF"][human_labels == 0]

    plt.plot(ok_imgs, "g*", lost_imgs, "r*")
    plt.title("cifar test-dataset VIF Compared to Human Label")
    plt.xlabel("Image Index")
    plt.ylabel("PSNR")
    plt.legend(["Ok", "Lost"])
    plt.show()


if 0: #treshold
    from sklearn import svm
    import numpy as np
    X = np.expand_dims(np.array(df["VIF"]), axis=1)
    y = np.array(df["label"])
    clf = svm.SVC(kernel="linear")
    clf.fit(X, y)

    pred = clf.predict(X)
    print("Accuracy: ", 100*np.sum(pred==y)/len(y))
