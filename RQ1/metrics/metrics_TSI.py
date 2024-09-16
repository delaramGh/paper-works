import cv2
import pandas as pd
from tqdm import tqdm
import numpy as np
from skimage import io, color, img_as_ubyte
from skimage.feature import graycomatrix, graycoprops

def compute_glcm_features(image, distances, angles):
    # Compute GLCM
    gray_image = img_as_ubyte(color.rgb2gray(image))
    glcm = graycomatrix(gray_image, distances=distances, angles=angles, levels=256, symmetric=True, normed=True)
    
    # Compute GLCM properties
    contrast = graycoprops(glcm, prop='contrast')
    dissimilarity = graycoprops(glcm, prop='dissimilarity')
    homogeneity = graycoprops(glcm, prop='homogeneity')
    energy = graycoprops(glcm, prop='energy')
    correlation = graycoprops(glcm, prop='correlation')
    # metrics = (contrast, dissimilarity, homogeneity, energy, correlation)
    metrics = (dissimilarity,)
    return metrics

##  Data...
csv_file = "C:\\Users\\ASUS\Desktop\\research\\mitacs project\\paper experiments\\smartInside dataset\\test_dataset.csv"

df = pd.read_csv(csv_file)
img_shape = (512, 320)  #  (224, 224)
dataset_length = len(df)

img_org, img_gen = [], []
similarity = []

for i in tqdm(range(dataset_length)):
    img_org_ = cv2.resize(cv2.imread(df["original"][i]), img_shape)
    img_org.append(img_org_)
    img_gen_ = cv2.resize(cv2.imread(df["gen"][i]), img_shape)
    img_gen.append(img_gen_)

    f1 = compute_glcm_features(img_org_, distances=[1], angles=[0])
    f2 = compute_glcm_features(img_gen_, distances=[1], angles=[0])
    similarity.append(np.linalg.norm(np.array(f1) - np.array(f2)))
    
img_org = np.array(img_org)
img_gen = np.array(img_gen)
print(img_org.shape, img_gen.shape)


df["TSI"] = similarity
print(df["TSI"])
df.to_csv(csv_file, index=False)


