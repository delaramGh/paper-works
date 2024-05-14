import cv2
import pandas as pd
from tqdm import tqdm
import numpy as np
from scipy.stats import wasserstein_distance


df = pd.read_csv("ref_csv_.csv")
dataset_length = df.shape[0]

wd = []
for i in tqdm(range(dataset_length)):
    img_org = cv2.imread(df["original"][i], cv2.IMREAD_GRAYSCALE).flatten()
    #**************CHANGE**************
    img_gen = cv2.imread(df["moderate_fog"][i], cv2.IMREAD_GRAYSCALE).flatten()
    wd.append(wasserstein_distance(img_org, img_gen))

#back-up
df.to_csv("ref_csv_BACKUP.csv")
#**************CHANGE**************
df["MF_WD"] = wd
print(df["MF_WD"])
df.to_csv("ref_csv_.csv")


