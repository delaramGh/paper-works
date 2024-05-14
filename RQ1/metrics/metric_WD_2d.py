import cv2
import pandas as pd
from tqdm import tqdm
import numpy as np
from scipy.stats import wasserstein_distance


df = pd.read_csv("ref_csv_.csv")
dataset_length = df.shape[0]


# image to signature for color image
def img_to_sig(img):
    sig = np.empty((img.size, 4), dtype=np.float32)
    idx = 0
    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            for k in range(img.shape[2]):
                sig[idx] = np.array([img[i,j,k], i, j,k])
                idx += 1
    return sig

wd = []
for i in tqdm(range(dataset_length)):
    img_org = cv2.imread(df["original"][i])# cv2.IMREAD_GRAYSCALE)
    #**************CHANGE**************
    img_gen = cv2.imread(df["moderate_fog"][i])# cv2.IMREAD_GRAYSCALE)
    sig1 = img_to_sig(img_gen)
    sig2 = img_to_sig(img_org)
    print(sig1.shape)
    distance, _, _ = cv2.EMD(sig1, sig2, cv2.DIST_L1,lowerBound=0)
    wd.append(distance)



#back-up
df.to_csv("ref_csv_BACKUP.csv")
#**************CHANGE**************
df["MF_WD"] = wd
print(df["MF_WD"])
df.to_csv("ref_csv_.csv")


