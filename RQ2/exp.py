from active_learning_v2 import active_labeling
import pandas as pd


models = ["Decision Tree", "SVM"]#, "Logistic Regression", "Random Forest"] #"Kmeans"
thresholds = [0.9] #[0.8, 0.85, 0.9, 0.95, 0.99]
split1 = [0.1, 0.15, 0.2, 0.25, 0.3, 0.4]



he = []
accs = []
params = []
for model in models:
    for th in thresholds:
        for s in split1:
            human_effort, acc, param = active_labeling("test_dataset.csv", model, th, split1=s, print_=False)
            he.append(human_effort)
            accs.append(acc)
            params.append(param)


model_name = [p[0] for p in params]
threshold = [p[1] for p in params]
split_1 = [p[2] for p in params]
split_2 = [p[3] for p in params]

df = pd.DataFrame({"model":model_name, "threshold":threshold, 
                   "split_1":split_1, "split_2":split_2, 
                   "accuracy":accs, "human effort":he})
df.to_csv("results\\exp_s1_changing.csv")



