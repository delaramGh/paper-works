import matplotlib.pyplot as plt
import pandas as pd


path = "C:\\Users\\ASUS\\Desktop\\research\\mitacs project\\paper experiments\\RQ2\\results\\cifar_dataset\\"
df_svm = pd.read_csv(f"{path}exp2_D2_NEW_search__SVM.csv")
df_rf = pd.read_csv(f"{path}exp2_D2_NEW_search__Random Forest.csv")
df_lr = pd.read_csv(f"{path}exp2_D2_NEW_search__Logistic Regression.csv")
df_dt = pd.read_csv(f"{path}exp2_D2_NEW_search__Decision Tree.csv")

# plt.figure(figsize=(10, 6))
# plt.scatter(svm['human effort'], svm['accuracy'], color='blue', alpha=0.7, label='SVM')
# plt.scatter(rf['human effort'], rf['accuracy'], color='red', alpha=0.7, label='Random Forest')
# plt.scatter(lr['human effort'], lr['accuracy'], color='Green', alpha=0.7, label='Logistic Regression')
# plt.scatter(dt['human effort'], dt['accuracy'], color='Orange', alpha=0.7, label='Decision Tree')
# plt.title('Accuracy vs Human Effort for Different Classifiers on cifar Dataset')
# plt.xlabel('Human Effort')
# plt.ylabel('Accuracy (%)')
# plt.grid(True)
# plt.legend()
# plt.show()



import seaborn as sns


sns.set_theme(style="whitegrid", font_scale=1)
plt.figure(figsize=(10, 5))
# ax = sns.boxplot(x="dataset", y="error", data=df, palette='tab10')
df_all = pd.concat([df_svm, df_rf, df_lr, df_dt])
ax = sns.scatterplot(data=df_all, y="accuracy", x="human effort", hue="model")

##  Display mean values on the plot...
# for i, ds_name in enumerate(ds_names):
#     plt.text(i, means[ds_name], f'{means[ds_name]:.3f}', horizontalalignment='center', size='medium', color='black', weight='semibold')
ax.set_xlabel('Human Effort (%)', fontweight='bold')
ax.set_ylabel('Accuracy (%)', fontweight='bold')
plt.legend(title='Classifiers')
# plt.xticks(ticks=range(len(ds_names)), labels=[r'$\mathbf{D}_{\mathbf{real\_test}}$', r'$\mathbf{D}_{\mathbf{sim}}$', r'SAEVAE', r'cycleG', r'styleT'])
ax.set_xticklabels(ax.get_xticklabels(), weight='bold')
ax.set_yticklabels(ax.get_yticklabels(), weight='bold')
plt.savefig('D2_grid_search.pdf')
# plt.show()
# plt.close()

