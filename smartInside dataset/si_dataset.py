import pandas as pd
import shutil 

df = pd.read_csv("test_dataset.csv")

for i in range(len(df)):
    index = df["gen"][i].find("images")
    name = df["gen"][i][index+7:] 
    name = name[:len(name)-4] + "___" + str(i) + ".jpg"
    shutil.copy(df["gen"][i], f"smartInside_test_dataset//{name}") 

