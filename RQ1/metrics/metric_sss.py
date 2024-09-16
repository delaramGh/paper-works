from torchvision.io.image import read_image
from torchvision.io.image import ImageReadMode
from torchvision.models.segmentation import fcn_resnet50, FCN_ResNet50_Weights
from torchvision.transforms.functional import to_pil_image
import pandas as pd
from tqdm import tqdm
import numpy as np


# Step 1: Initialize model with the best available weights
weights = FCN_ResNet50_Weights.DEFAULT
model = fcn_resnet50(weights=weights)
model.eval()

# # Step 2: Initialize the inference transforms
preprocess = weights.transforms()

csv_file = "C:\\Users\\ASUS\Desktop\\research\\mitacs project\\paper experiments\\smartInside dataset\\test_dataset.csv"


df = pd.read_csv(csv_file)
dataset_length = df.shape[0]

sss_1 = []
sss_2 = []
for i in tqdm(range(dataset_length)):
    img_org = read_image(df["original"][i]) 
    img_gen = read_image(df["gen"][i])

    # # Step 3: Apply inference preprocessing transforms
    batch_org = preprocess(img_org).unsqueeze(0)
    batch_gen = preprocess(img_gen).unsqueeze(0)

    # # Step 4: Use the model and visualize the prediction
    prediction_org = model(batch_org)["out"]#[0,0]
    normalized_masks_org = prediction_org.softmax(dim=1)[0, 0]

    prediction_gen = model(batch_gen)["out"]#[0,0]
    normalized_masks_gen = prediction_gen.softmax(dim=1)[0, 0]
    
    gen = normalized_masks_gen.detach().numpy() 
    org = normalized_masks_org.detach().numpy() 
    loss = np.square(np.subtract(org, gen)).mean() 
    sss_1.append(loss)
    # print(loss)
    # to_pil_image(normalized_masks).show()


df["SSS1"] = sss_1
print(df["SSS1"])
df.to_csv(csv_file)
