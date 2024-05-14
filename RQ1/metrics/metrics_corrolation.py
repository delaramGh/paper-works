import pandas as pd
import numpy as np


df = pd.read_csv("C:\\Users\\ASUS\\Desktop\\research\\mitacs project\\Delaram_work\\ref_csv_fog___.csv")

mf_psnr = df["MF_PSNR"][df["Final"]!="Disagree"]
mf_psnr = mf_psnr/max(mf_psnr)

mf_cs = df["MF_CS"][df["Final"]!="Disagree"]
mf_cs = mf_cs/max(mf_cs)

mf_cpl = df["MF_CPL"][df["Final"]!="Disagree"]
mf_cpl = mf_cpl/max(mf_cpl)

mf_ssim = df["MF_SSIM"][df["Final"]!="Disagree"]
mf_ssim = mf_ssim/max(mf_ssim)

mf_mse = df["MF_MSE"][df["Final"]!="Disagree"]
mf_mse = mf_mse/max(mf_mse)

mf_wd = df["MF_WD"][df["Final"]!="Disagree"]
mf_wd = mf_wd/max(mf_wd)

mf_kl = np.copy(df["MF_KL"][df["Final"]!="Disagree"])
inf_ = mf_kl[6]
for i in range(len(mf_kl)):
    if mf_kl[i] == inf_:
        mf_kl[i] = 100
mf_kl = mf_kl/max(mf_kl)

mf_sss = df["MF_SSS1"][df["Final"]!="Disagree"]
mf_sss = mf_sss/max(mf_sss)

mf_tsi = df["MF_TSI"][df["Final"]!="Disagree"]
mf_tsi = mf_tsi/max(mf_tsi)

mf_vae = df["MF_VAE"][df["Final"]!="Disagree"]
mf_vae = mf_vae/max(mf_vae)

mf_hist_cor = df["MF_HistCmp_correlation"][df["Final"]!="Disagree"]
mf_hist_cor = mf_hist_cor/max(mf_hist_cor)

mf_hist_int = df["MF_HistCmp_intersection"][df["Final"]!="Disagree"]
mf_hist_int = mf_hist_int/max(mf_hist_int)

labels = df["Final"][df["Final"]!="Disagree"]
labels = [int(i) for i in labels]

metrics = [mf_psnr, mf_cpl, mf_cs, mf_ssim, mf_mse, mf_wd, mf_kl, mf_sss, mf_tsi, mf_hist_cor, mf_hist_int, mf_vae]
print(len(metrics))
matrix = np.zeros((len(metrics)))
for i in range(len(metrics)):
        matrix[i] = np.corrcoef(metrics[i], labels)[1, 0]

print(matrix)
df2 = pd.DataFrame(matrix)
df2.to_csv("corrolation_label.csv")

