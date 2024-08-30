import pandas as pd
import urllib.request 
from tqdm import tqdm


# df = pd.read_csv("cifar dataset\\original_pair_with_Transformed_cifar.csv")


# original = []
# transformed = []
# for i in tqdm(range(len(df))):
#     org_url = df["original_image_link"][i]
#     gen_url = df["transformed_image_link"][i]

#     indx1 = gen_url.find("TRANSFORMED")
#     index2 = gen_url[indx1+12:].find('/')
#     n1 = gen_url[indx1+index2+13:]
#     gen_name = "cifar dataset\\cifar_gen_dataset\\" + n1

#     indx1 = org_url.find("ORIGINAL")
#     index2 = org_url[indx1+9:].find('/')
#     n2 = org_url[indx1+index2+10:]
#     org_name = "cifar dataset\\cifar_org_dataset\\" + n2

#     if n1 != n2:
#         print(n1, "\n", n2)
#         raise Exception(f"{i}th row name does not match! \n")
#     # (_, info_org) = urllib.request.urlretrieve(org_url, org_name)
#     # (_, info_gen) = urllib.request.urlretrieve(gen_url, gen_name)
        
#     original.append(n2)
#     transformed.append(n1)


# df["original_name"] = original
# df["transformed_name"] = transformed
# df.to_csv("cifar_dataset_modified.csv")


################################################################################
################################################################################
# import shutil 

# df = pd.read_csv("correlation_dataset.csv")

# for i in range(len(df)):
#     index = df["gen"][i].find("cifar_gen_dataset")
#     name = df["gen"][i][index+18:]
#     # print(name)
#     shutil.copy(df["gen"][i], f"cifar_correlation_dataset//{name}") 


################################################################################
################################################################################
csv_file = "cifar_dataset_modified.csv"
df = pd.read_csv(csv_file)

org_names = []
gen_names = []
gen_path = "C:\\Users\\ASUS\\Desktop\\research\\mitacs project\\paper experiments\\cifar dataset\\cifar_gen_dataset\\"
org_path = "C:\\Users\\ASUS\\Desktop\\research\\mitacs project\\paper experiments\\cifar dataset\\cifar_org_dataset\\"

for i in range(df.shape[0]):
    i1 = df["original"][i].find("ORIGINAL")
    i2 = df["original"][i][i1+9:].find("/")
    org_name = df["original"][i][i1+9+i2+1:]

    i1 = df["gen"][i].find("TRANSFORMED")
    i2 = df["gen"][i][i1+12:].find("/")
    gen_name = df["gen"][i][i1+12+i2+1:]

    org_names.append(org_path + org_name)
    gen_names.append(gen_path + gen_name)
    

print(len(org_names))
print(len(gen_names))
df["original"] = org_names
df["gen"] = gen_names

df.to_csv(csv_file)



