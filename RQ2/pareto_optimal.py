import pandas as pd
import matplotlib.pyplot as plt
import numpy as np


def pareto_optimal(df):
    pareto = []
    for i in range(df.shape[0]):
        pareto_acc = df["accuracy"][i]
        pareto_he = df["human effort"][i]
        cnt = 0
        for j in range(df.shape[0]):
            if (pareto_acc <= df["accuracy"][j] and pareto_he > df["human effort"][j]) or (pareto_acc < df["accuracy"][j] and pareto_he >= df["human effort"][j]):
                break
            else:
                cnt += 1
        if cnt == df.shape[0]:
            pareto.append([pareto_acc, pareto_he])
    
    pareto = np.array(pareto)
    print(pareto.shape)
    return pareto
#####################################################################

path = "C:\\Users\\ASUS\\Desktop\\research\\mitacs project\\paper experiments\\RQ2\\results\\cifar_dataset\\"
svm = pd.read_csv(f"{path}exp2_d2_search__SVM.csv")
rf = pd.read_csv(f"{path}exp2_d2_search__Random Forest.csv")
lr = pd.read_csv(f"{path}exp2_d2_search__Logistic Regression.csv")
dt = pd.read_csv(f"{path}exp2_d2_search__Decision Tree.csv")

svm = pareto_optimal(svm)
rf = pareto_optimal(rf)
lr = pareto_optimal(lr)
dt = pareto_optimal(dt)


plt.figure(figsize=(10, 6))
plt.scatter(svm[:, 1], svm[:, 0], color='blue', alpha=0.7, label='SVM')
plt.scatter(rf[:, 1], rf[:, 0], color='red', alpha=0.7, label='Random Forest')
plt.scatter(lr[:, 1], lr[:, 0], color='Green', alpha=0.7, label='Logistic Regression')
plt.scatter(dt[:, 1], dt[:, 0], color='Orange', alpha=0.7, label='Decision Tree')
plt.title('Pareto Optimal of Accuracy vs Human Effort for Different Classifiers on cifar Dataset')
plt.xlabel('Human Effort')
plt.ylabel('Accuracy (%)')
plt.grid(True)
plt.legend()
plt.show()
            
        