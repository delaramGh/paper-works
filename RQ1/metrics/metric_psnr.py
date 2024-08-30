import cv2
import pandas as pd
from tqdm import tqdm


csv_file = "C:\\Users\\ASUS\\Desktop\\research\\mitacs project\\paper experiments\\cifar dataset\\cifar_dataset_modified.csv"
df = pd.read_csv(csv_file)
dataset_length = df.shape[0]

psnr = []
for i in tqdm(range(dataset_length)):
    img_org = cv2.imread(df["original"][i])
    #**************CHANGE**************
    img_gen = cv2.imread(df["gen"][i])
    psnr.append(cv2.PSNR(img_org, img_gen))

#**************CHANGE**************
df["PSNR"] = psnr
print(df["PSNR"])
df.to_csv(csv_file)





