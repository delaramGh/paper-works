import pandas as pd
import numpy as np
from scipy.stats import pearsonr

dataset = 'smartInside'  ##  cifar or smartInside

csv_file = f"C:\\Users\\ASUS\Desktop\\research\\mitacs project\\paper experiments\\{dataset} dataset\\correlation_dataset.csv"

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

hist_cor = df["Hist_correlation"]
hist_cor = hist_cor/max(hist_cor)

hist_int = df["Hist_intersection"]
hist_int = hist_int/max(hist_int)

vif = df["VIF"]
vif = vif/max(vif)

# vae2d = df["VAE2D"]
# vae2d = vae2d/max(vae2d)

labels = df["label"]
labels = [int(i) for i in labels]


metrics = [psnr, cpl, cs, ssim, mse, wd, kl, sss, tsi, vae, hist_cor, hist_int, vif]
metric_labels = ['PSNR', 'CS', 'CPL', 'SSIM', 'MSE', 'WD', 'KL', 'SSS1', 'TSI', 'VAE', 'Hist_correlation', 'Hist_intersection', 'VIF']
# metrics = [psnr, cpl, cs, ssim, vif]

matrix = np.zeros((len(metrics)))
res_dict = {'Correlation': [], 'p_val': []}


for i in range(len(metrics)):
        matrix[i] = np.corrcoef(metrics[i], labels)[1, 0]
        corr = pearsonr(metrics[i], labels)
        res_dict["Correlation"].append(corr.statistic)
        res_dict['p_val'].append(corr.pvalue)

df_res = pd.DataFrame(res_dict, index=metric_labels)
df_res.to_csv(f'corr_{dataset}_pvals.csv')
df2 = pd.DataFrame(matrix)
print(df2)
df2.to_csv(f"exp1_{dataset}_corrolation_label.csv")


##  Inner correlation...
corr_matrix = np.empty((len(metrics), len(metrics)))
pval_matrix = np.empty((len(metrics), len(metrics)))

for i in range(len(metrics)):
        for j in range(len(metrics)):
                corr = pearsonr(metrics[i], metrics[j])
                corr_matrix[i, j] = corr.statistic
                pval_matrix[i, j] = corr.pvalue

df_corr = pd.DataFrame(corr_matrix, columns=metric_labels, index=metric_labels)
df_pval = pd.DataFrame(pval_matrix, columns=metric_labels, index=metric_labels)
df_corr.to_csv(f'Inner_corr_{dataset}.csv')
df_pval.to_csv(f'Inner_pval_{dataset}.csv')

