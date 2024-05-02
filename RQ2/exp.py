from active_learning_v2 import active_labeling
import pandas as pd


models = ["SVM", "Logistic Regression", "Random Forest", "Decision Tree"] #"Kmeans"
thresholds = [0.8, 0.85, 0.9, 0.95, 0.99]


he = []
accs = []
params = []
for model in models:
    for th in thresholds:
        human_effort, acc, param = active_labeling("test_dataset.csv", model, th)
        he.append(human_effort)
        accs.append(acc)
        params.append(param)

df = pd.DataFrame({"parameters":params, "accuracy":accs, "human effort":he})
df.to_csv("raw_results.csv")



