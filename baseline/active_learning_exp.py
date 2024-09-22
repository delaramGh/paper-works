from active_learning_module import active_labeling
import pandas as pd
from tqdm import tqdm
from sklearn.utils import shuffle



def shuffle_dataset(dataset):
    df = pd.read_csv(dataset)
    df = shuffle(df)
    df.to_csv(dataset, index=False)



# dataset = "..\\smartInside dataset\\test_dataset - Copy.csv"
dataset = "..\\cifar dataset\\concept1_cifar_all.csv"
input = ["VIF", "MSE", "PSNR", "Hist_cor", "Hist_int", "TSI", "SSIM", "CS", "CPL", "WD", "KL", "SSS1", "VAE"]
model = "Random Forest"

l = []
# threshold = 0.75
# split1 = 0.1
# split2 = 0.01
# effort = 0.25
# l.append([threshold, split1, split2])

threshold = 0.8
split1 = 0.3
split2 = 0.03
effort = 0.50
l.append([threshold, split1, split2])

threshold = 0.9
split1 = 0.4
split2 = 0.1
effort = 0.75
l.append([threshold, split1, split2])


model_list = []
threshold_list = []
s1_list = []
s2_list = []
acc_list = []
precision_list = []
recall_list = []
he_list = []



human_effort_list = []
acc_list = []
precision_list = []
recall_list = []
model_name_list = []
th_list = []
s1_list = []
s2_list = []

for item in l:
    th = item[0]
    s1 = item[1]
    s2 = item[2]
    for _ in range(20):
        shuffle_dataset(dataset)

        human_effort, acc, precision, recall, param = active_labeling(dataset, model, input, th, s1, s2, print_=False)

        human_effort_list.append(human_effort)
        acc_list.append(acc)
        precision_list.append(precision)
        recall_list.append(recall)
        model_name_list.append(param[0])
        th_list.append(param[1])
        s1_list.append(param[2])
        s2_list.append(param[3])


df = pd.DataFrame({"model":model_name_list, "threshold":th_list, 
                    "split_1":s1_list, "split_2":s2_list, 
                    "accuracy":acc_list, "precision":precision_list, 
                    "recall":recall_list, "human effort":human_effort_list})

df.to_csv(f"D2_NEW_baseline_exp2_5_75.csv")



