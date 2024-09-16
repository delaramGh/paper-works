import cv2
import pandas as pd
from tqdm import tqdm
import numpy as np
from scipy.stats import wasserstein_distance

csv_file = "C:\\Users\\ASUS\Desktop\\research\\mitacs project\\paper experiments\\smartInside dataset\\test_dataset.csv"

df = pd.read_csv(csv_file)
dataset_length = df.shape[0]

wd = []
for i in tqdm(range(dataset_length)):
    img_org = cv2.imread(df["original"][i], cv2.IMREAD_GRAYSCALE).flatten()
    img_gen = cv2.imread(df["gen"][i], cv2.IMREAD_GRAYSCALE).flatten()
    wd.append(wasserstein_distance(img_org, img_gen))


df["WD"] = wd
print(df["WD"])
df.to_csv(csv_file)


