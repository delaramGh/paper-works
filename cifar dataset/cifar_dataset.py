import pandas as pd
import urllib.request 
from tqdm import tqdm


df = pd.read_csv("cifar dataset\\original_pair_with_Transformed_cifar.csv")


original = []
transformed = []
for i in tqdm(range(len(df))):
    org_url = df["original_image_link"][i]
    gen_url = df["transformed_image_link"][i]

    indx1 = gen_url.find("TRANSFORMED")
    index2 = gen_url[indx1+12:].find('/')
    n1 = gen_url[indx1+index2+13:]
    gen_name = "cifar dataset\\cifar_gen_dataset\\" + n1

    indx1 = org_url.find("ORIGINAL")
    index2 = org_url[indx1+9:].find('/')
    n2 = org_url[indx1+index2+10:]
    org_name = "cifar dataset\\cifar_org_dataset\\" + n2

    if n1 != n2:
        print(n1, "\n", n2)
        raise Exception(f"{i}th row name does not match! \n")
    # (_, info_org) = urllib.request.urlretrieve(org_url, org_name)
    # (_, info_gen) = urllib.request.urlretrieve(gen_url, gen_name)
        
    original.append(n2)
    transformed.append(n1)



df["original_name"] = original
df["transformed_name"] = transformed
df.to_csv("cifar_dataset_modified.csv")




