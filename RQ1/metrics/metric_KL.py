import matplotlib.pyplot as plt
import numpy as np
from scipy.special import rel_entr
from tqdm import tqdm
import cv2
import pandas as pd


def rgb2gray(rgb):
    return np.dot(rgb[...,:3], [0.299, 0.587, 0.144])


csv_file = "C:\\Users\\ASUS\Desktop\\research\\mitacs project\\paper experiments\\smartInside dataset\\test_dataset.csv"



df = pd.read_csv(csv_file)
dataset_length = df.shape[0]

kl = []
for i in tqdm(range(dataset_length)):
    img_org = rgb2gray(plt.imread(df["original"][i]))
    out_org = plt.hist(x=img_org.ravel(), bins=256, range=[0, 256])
    dist_org = out_org[0]/(img_org.shape[0]*img_org.shape[1])

    img_gen = rgb2gray(plt.imread(df["gen"][i]))
    out_gen = plt.hist(x=img_gen.ravel(), bins=256, range=[0, 256])
    dist_gen = out_gen[0]/(img_gen.shape[0]*img_gen.shape[1])
    kl.append(sum(rel_entr(dist_org, dist_gen)))



df["KL"] = kl
# print(df["KL"])
df.to_csv(csv_file)
    