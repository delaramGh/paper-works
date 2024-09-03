from PIL import Image
import os
import random
import pandas as pd
from tqdm import tqdm



org_path = "C:\\Users\\ASUS\\Desktop\\research\\mitacs project\\paper experiments\\cifar dataset\\cifar_org_dataset\\"
gen_path = "C:\\Users\\ASUS\\Desktop\\research\\mitacs project\\paper experiments\\cifar dataset\\cifar_gen_dataset\\"

df = pd.read_csv("cifar_dataset_modified.csv")
# df = df.drop(df.columns[0], axis=1)
# df2 = df[df["label"] == 0].reset_index()



# for i in tqdm(range(len(df2))):    
#     angle1 = random.randint(5, 20)
#     angle2 = random.randint(5, 20)

#     with Image.open(os.path.join(org_path, df2["original_name"][i])) as img:
#         rotated_img1 = img.rotate(angle1)
#         name_org1 = org_path + "_rotated1_" + df2["original_name"][i] 
#         rotated_img1.save(name_org1)

#         rotated_img2 = img.rotate(angle2)
#         name_org2 = org_path + "_rotated2_" + df2["original_name"][i] 
#         rotated_img2.save(name_org2)

#     with Image.open(os.path.join(gen_path, df2["transformed_name"][i])) as img:
#         rotated_img1 = img.rotate(angle1)
#         name_gen1 = gen_path + "_rotated1_" + df2["transformed_name"][i] 
#         rotated_img1.save(name_gen1)

#         rotated_img2 = img.rotate(angle2)
#         name_gen2 = gen_path + "_rotated2_" + df2["transformed_name"][i] 
#         rotated_img2.save(name_gen2)


#     new_row1 = {'original': name_org1,
#                'gen'     : name_gen1,
#                'transformation' : df2['transformation'][i],
#                'org label' : df2['org label'][i],
#                'trns label' : df2['trns label'][i],
#                'label' : 0,
#                'original_name' : "_rotated1_" + df2["original_name"][i] ,
#                'transformed_name' : "_rotated1_" + df2["transformed_name"][i] }
    
#     new_row2 = {'original': name_org2,
#                'gen'     : name_gen2,
#                'transformation' : df2['transformation'][i],
#                'org label' : df2['org label'][i],
#                'trns label' : df2['trns label'][i],
#                'label' : 0,
#                'original_name' : "_rotated2_" + df2["original_name"][i] ,
#                'transformed_name' : "_rotated2_" + df2["transformed_name"][i] }
    
#     df.loc[len(df)] = new_row1
#     df.loc[len(df)] = new_row2



df_shuffled = df.sample(frac=1).reset_index(drop=True)
print(df_shuffled)
df_shuffled.to_csv("cifar_dataset_modified.csv")
