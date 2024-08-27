import pandas as pd
import matplotlib.pyplot as plt 


def func(df, p1_name, p1_value, p2_name, p2_value):
    df = df.loc[df[p1_name]==p1_value]
    df = df.loc[df[p2_name]==p2_value]
    return df


path = "C:\\Users\\ASUS\\Desktop\\research\\mitacs project\\paper experiments\\RQ2\\results\\smartInside_dataset\\"

svm = pd.read_csv(f"{path}exp2_d1_search__SVM.csv")
svm = func(svm, "split_1", 0.2, "split_2", 0.05)

rf = pd.read_csv(f"{path}exp2_d1_search__Random Forest.csv")
rf = func(rf, "split_1", 0.2, "split_2", 0.05)

lr = pd.read_csv(f"{path}exp2_d1_search__Logistic Regression.csv")
lr = func(lr, "split_1", 0.2, "split_2", 0.05)

dt = pd.read_csv(f"{path}exp2_d1_search__Decision Tree.csv")
dt = func(dt, "split_1", 0.2, "split_2", 0.05)


plt.scatter(lr["human effort"], lr["accuracy"], color='Green', alpha=0.7, label='Logistic Regression')
plt.scatter(svm['human effort'], svm['accuracy'], color='blue', alpha=0.7, label='SVM')
plt.scatter(rf['human effort'], rf['accuracy'], color='red', alpha=0.7, label='Random Forest')
# plt.scatter(dt['human effort'], dt['accuracy'], color='Orange', alpha=0.7, label='Decision Tree')

# thresholds = ['th=0.8', 'th=0.85', 'th=0.9', 'th=0.95', 'th=0.99']
# split_1 = [0.1, 0.15, 0.2, 0.25, 0.3, 0.4]

# # Plotting
# models = [lr, svm, rf, dt]
# colors = ['Green', 'Blue', 'Red', 'Orange']
# labels = ['Logistic Regression', 'SVM', 'Random Forest', 'Decision Tree']

# for model, color, label in zip(models, colors, labels):
#     plt.scatter(model['human effort'], model['accuracy'], color=color, alpha=0.7, label=label)
#     for i, (x, y) in enumerate(zip(model['human effort'], model['accuracy'])):
#         if i < len(split_1):
#             plt.text(x, y, split_1[i], color='black', ha='center', va='bottom')


plt.ylabel("Accuracy %")
plt.xlabel("Human Effort %")
plt.legend()
plt.title("Humman Effort vs. Accuracy w.r.t. alpha for SmartInside Dataset\nbeta=.05, initial=.2")
plt.grid(True)
plt.show()


