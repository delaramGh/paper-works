import cv2
import pandas as pd
from tqdm import tqdm
import numpy as np
import tensorflow.keras
from tensorflow.keras.applications.vgg16 import VGG16
from tensorflow.keras.applications.vgg16 import preprocess_input
from sklearn.metrics.pairwise import cosine_similarity

##  Data...
csv_file = "C:\\Users\\ASUS\\Desktop\\research\\mitacs project\\paper experiments\\cifar dataset\\cifar_dataset_modified.csv"
df = pd.read_csv(csv_file)
img_shape = (32, 32) #(512, 320)  #  (224, 224)
dataset_length = len(df)

img_org, img_gen = [], []

for i in tqdm(range(dataset_length)):
    img_org_ = cv2.resize(cv2.imread(df["original"][i]), img_shape).astype(float)
    img_org.append(img_org_)
    img_gen_ = cv2.resize(cv2.imread(df["gen"][i]), img_shape).astype(float)
    img_gen.append(img_gen_)
    
img_org = np.array(img_org)
img_gen = np.array(img_gen)
print(img_org.shape, img_gen.shape)

##  Model...
model = VGG16(weights='imagenet', include_top=False, input_shape=(*img_shape[::-1], 3))

X_org = preprocess_input(img_org)
X_gen = preprocess_input(img_gen)
X_org = model.predict(X_org)
X_gen = model.predict(X_gen)

X_org = X_org.reshape(len(X_org), -1)
X_gen = X_gen.reshape(len(X_gen), -1)

cs = cosine_similarity(X_org, X_gen)
cs = [cs[i, i] for i in range(len(cs))]


df["CS"] = cs
print(df["CS"])
df.to_csv(csv_file)


