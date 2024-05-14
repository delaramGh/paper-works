import cv2
import pandas as pd
from tqdm import tqdm
import matplotlib.pyplot as plt


df = pd.read_csv("ref_csv_.csv")
dataset_length = df.shape[0]


hist_org = []
hist_gen = []
for i in tqdm(range(dataset_length)):
    img_org = cv2.imread(df["original"][i], cv2.IMREAD_GRAYSCALE)
    hist = cv2.calcHist([img_org], [0], None, [256], [0, 256])
    hist_org.append(hist)
    #**************CHANGE**************
    img_gen = cv2.imread(df["moderate_fog"][i], cv2.IMREAD_GRAYSCALE)
    hist = cv2.calcHist([img_gen], [0], None, [256], [0, 256])
    hist_gen.append(hist)

correlation = []
Chi_Squared = []
intersection = []
overlap = [] 
for i in tqdm(range(dataset_length)):
    correlation.append(cv2.compareHist(hist_gen[i], hist_org[i], cv2.HISTCMP_CORREL))
    # Chi_Squared.append(cv2.compareHist(hist_gen[i], hist_org[i], cv2.HISTCMP_CHISQR))
    intersection.append(cv2.compareHist(hist_gen[i], hist_org[i], cv2.HISTCMP_INTERSECT))
    # overlap.append(cv2.compareHist(hist_gen[i], hist_org[i], cv2.HISTCMP_BHATTACHARYYA))

df["MF_HistCmp_correlation"] = correlation
# df["MF_HistCmp_chi_squared"] = Chi_Squared
df["MF_HistCmp_intersection"] = intersection
# df["MF_HistCmp_overlap"] = overlap

df.to_csv("ref_csv_histcmp.csv")


