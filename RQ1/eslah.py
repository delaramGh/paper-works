import pandas as pd


df = pd.read_csv("results\\smartInside dataset\\D1_test_vae1D_result.csv")

print(df.columns)

names = []
for i in range(df.shape[0]):
    index = df['name'][i].find('___')
    name = df['name'][i][:index]+'.jpg'
    names.append(name)

df['good name'] = names
df.to_csv("results\\smartInside dataset\\D1_test_vae1D_result.csv")
    