import cv2
import pandas as pd
from tqdm import tqdm

df = pd.read_csv("ref_csv_snow.csv")
dataset_length = df.shape[0]

psnr = []
for i in tqdm(range(dataset_length)):
    img_org = cv2.imread(df["original"][i])
    #**************CHANGE**************
    img_gen = cv2.imread(df["high_snow"][i])
    psnr.append(cv2.PSNR(img_org, img_gen))

#**************CHANGE**************
df["HS_PSNR"] = psnr
print(df["HS_PSNR"])
df.to_csv("ref_csv_snow.csv")





