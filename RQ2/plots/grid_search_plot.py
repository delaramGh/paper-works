import matplotlib.pyplot as plt
import pandas as pd


path = "C:\\Users\\ASUS\\Desktop\\research\\mitacs project\\paper experiments\\RQ2\\results\\smartInside_dataset\\"
svm = pd.read_csv(f"{path}exp2_d1_search__SVM.csv")
rf = pd.read_csv(f"{path}exp2_d1_search__Random Forest.csv")
lr = pd.read_csv(f"{path}exp2_d1_search__Logistic Regression.csv")
dt = pd.read_csv(f"{path}exp2_d1_search__Decision Tree.csv")

plt.figure(figsize=(10, 6))
plt.scatter(svm['human effort'], svm['accuracy'], color='blue', alpha=0.7, label='SVM')
plt.scatter(rf['human effort'], rf['accuracy'], color='red', alpha=0.7, label='Random Forest')
plt.scatter(lr['human effort'], lr['accuracy'], color='Green', alpha=0.7, label='Logistic Regression')
plt.scatter(dt['human effort'], dt['accuracy'], color='Orange', alpha=0.7, label='Decision Tree')
plt.title('Accuracy vs Human Effort for Different Classifiers on SmartInside Dataset')
plt.xlabel('Human Effort')
plt.ylabel('Accuracy (%)')
plt.grid(True)
plt.legend()
plt.show()

