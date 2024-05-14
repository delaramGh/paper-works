import cv2
import pandas as pd
from tqdm import tqdm
import numpy as np

def my_mse(img1, img2):
    h, w, c = img1.shape
    err = np.sum(cv2.subtract(img1, img2)**2)
    return err/float(w*h*c)


df = pd.read_csv("ref_csv_.csv")
dataset_length = df.shape[0]

ssim = []
for i in tqdm(range(dataset_length)):
    img_org = cv2.imread(df["original"][i]) #, cv2.IMREAD_GRAYSCALE)
    #**************CHANGE**************
    img_gen = cv2.imread(df["moderate_fog"][i]) #, cv2.IMREAD_GRAYSCALE))
    ssim.append(my_mse(img_org, img_gen))



#back-up
df.to_csv("ref_csv_BACKUP.csv")
#**************CHANGE**************
df["MF_MSE"] = ssim
print(df["MF_MSE"])
df.to_csv("ref_csv_.csv")


