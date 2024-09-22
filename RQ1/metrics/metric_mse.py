import cv2
import pandas as pd
from tqdm import tqdm
import numpy as np


def my_mse(img1, img2):
    h, w, c = img1.shape
    err = np.sum(cv2.subtract(img1, img2)**2)
    return err/float(w*h*c)

csv_file = "C:\\Users\\ASUS\Desktop\\research\\mitacs project\\paper experiments\\smartInside dataset\\test_dataset.csv"


df = pd.read_csv(csv_file)
dataset_length = df.shape[0]

ssim = []
for i in tqdm(range(dataset_length)):
    img_org = cv2.imread(df["original"][i]) #, cv2.IMREAD_GRAYSCALE)
    img_gen = cv2.imread(df["gen"][i]) #, cv2.IMREAD_GRAYSCALE))
    ssim.append(my_mse(img_org, img_gen))


df["MSE"] = ssim
print(df["MSE"])
df.to_csv(csv_file)


