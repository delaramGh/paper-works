from PIL import Image
import os
import random
import pandas as pd
from tqdm import tqdm



org_path = "C:\\Users\\ASUS\\Desktop\\research\\mitacs project\\paper experiments\\cifar dataset\\cifar_org_dataset\\"
gen_path = "C:\\Users\\ASUS\\Desktop\\research\\mitacs project\\paper experiments\\cifar dataset\\cifar_gen_dataset\\"

df = pd.read_csv("cifar_dataset_modified.csv")
df = df.drop(df.columns[0], axis=1)
df2 = df[df["label"] == 0].reset_index()



for i in tqdm(range(len(df2))):    
    angle = random.randint(45, 135)
    with Image.open(os.path.join(org_path, df2["original_name"][i])) as img:
        rotated_img = img.rotate(angle)
        name_org = org_path + "_rotated" + df2["original_name"][i] 
        rotated_img.save(name_org)

    with Image.open(os.path.join(org_path, df2["transformed_name"][i])) as img:
        rotated_img = img.rotate(angle)
        name_gen = gen_path + "_rotated" + df2["transformed_name"][i] 
        rotated_img.save(name_gen)


    new_row = {'original': name_org,
               'gen'     : name_gen,
               'transformation' : df2['transformation'][i],
               'org label' : df2['org label'][i],
               'trns label' : df2['trns label'][i],
               'label' : 0,
               'original_name' : "_rotated" + df2["original_name"][i] ,
               'transformed_name' : "_rotated" + df2["transformed_name"][i] }
    
    df.loc[len(df)] = new_row


# df_shuffled = df.sample(frac=1).reset_index(drop=True)
# print(df_shuffled)
# df_shuffled.to_csv("cifar_dataset_modified.csv")
