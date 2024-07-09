import pandas as pd
import numpy as np

csv_file = "C:\\Users\\ASUS\Desktop\\research\\mitacs project\\paper experiments\\cifar dataset\\correlation_dataset.csv"

df = pd.read_csv(csv_file)

psnr = df["PSNR"]
psnr = psnr/max(psnr)

cs = df["CS"]
cs = cs/max(cs)

cpl = df["CPL"]
cpl = cpl/max(cpl)

ssim = df["SSIM"]
ssim = ssim/max(ssim)

mse = df["MSE"]
mse = mse/max(mse)

wd = df["WD"]
wd = wd/max(wd)

kl = np.copy(df["KL"])
inf_ = kl[18]
for i in range(len(kl)):
    if kl[i] == inf_:
        kl[i] = 100
kl = kl/max(kl)

sss = df["SSS1"]
sss = sss/max(sss)

tsi = df["TSI"]
tsi = tsi/max(tsi)

vae = df["VAE"]
vae = vae/max(vae)

hist_cor = df["Hist_cor"]
hist_cor = hist_cor/max(hist_cor)

hist_int = df["Hist_int"]
hist_int = hist_int/max(hist_int)

vif = df["VIF"]
vif = vif/max(vif)

labels = df["label"]
labels = [int(i) for i in labels]

metrics = [psnr, cpl, cs, ssim, mse, wd, kl, sss, tsi, hist_cor, hist_int, vae, vif]
# metrics = [psnr, cpl, cs, ssim, vif]

matrix = np.zeros((len(metrics)))
for i in range(len(metrics)):
        matrix[i] = np.corrcoef(metrics[i], labels)[1, 0]
df2 = pd.DataFrame(matrix)
print(df2)
df2.to_csv("exp1_D2_corrolation_label.csv")


matrix3 = np.corrcoef(metrics)

df3 = pd.DataFrame(matrix3)
df3.to_csv("exp1_D2_corrolation_metrics.csv")

