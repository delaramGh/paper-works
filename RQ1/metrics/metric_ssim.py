from skimage.metrics import structural_similarity 
from skimage import data, img_as_float
import cv2
import pandas as pd
from tqdm import tqdm


df = pd.read_csv("ref_csv_snow.csv")
dataset_length = df.shape[0]

ssim = []
for i in tqdm(range(dataset_length)):
    img_org = img_as_float(cv2.imread(df["original"][i], cv2.IMREAD_GRAYSCALE))
    #**************CHANGE**************
    img_gen = img_as_float(cv2.imread(df["moderate_snow"][i], cv2.IMREAD_GRAYSCALE))
    ssim.append(structural_similarity (img_org, img_gen, data_range=img_gen.max()-img_gen.min()))

#**************CHANGE**************
df["MS_SSIM"] = ssim
print(df["MS_SSIM"])
df.to_csv("ref_csv_snow.csv")


