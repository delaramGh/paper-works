from active_learning_v3 import active_labeling
import pandas as pd
from tqdm import tqdm
import numpy as np



# def replace_inf_in_csv(input_file, output_file, replace_value=10):
#     # Read the CSV file into a DataFrame
#     df = pd.read_csv(input_file)

#     # Replace infinite values with the specified value
#     df.replace([np.inf, -np.inf], replace_value, inplace=True)

#     # Write the updated DataFrame to a new CSV file
#     df.to_csv(output_file, index=False)
#     print(f"Infinite values replaced and saved to {output_file}")

# replace_inf_in_csv("..\\smartInside dataset\\all_dataset.csv", "..\\smartInside dataset\\all_good.csv")


metrics = ["VIF", "MSE", "PSNR", "Hist_cor", "Hist_int", "TSI", "SSIM", "CS", "CPL", "WD", "KL", "SSS1"]
inputs = [metrics, metrics, metrics, metrics] #cifar

# inputs = [["PSNR", "CS", "CPL", "SSIM"], 
#           ["PSNR", "CS", "CPL"], 
#           ["PSNR", "CS", "CPL", "SSIM"],
#           ["PSNR", "CS", "CPL", "SSIM"]] #smartInside

models = ["Decision Tree", "SVM", "Logistic Regression", "Random Forest"]

thresholds = [0.8, 0.85, 0.9, 0.95, 0.99]
split1 = [0.1, 0.15, 0.2, 0.25, 0.3, 0.4]
split2 = [0.01, 0.03, 0.05, 0.08, 0.1, 0.15]


dataset = "..\\smartInside dataset\\all_good.csv"
# dataset = "..\\cifar dataset\\correlation_dataset_old.csv"
# dataset = "..\\smartInside dataset\\test_dataset.csv"

for model_number in range(len(models)):
    human_effort_list = []
    acc_list = []
    precision_list = []
    recall_list = []
    model_name_list = []
    th_list = []
    s1_list = []
    s2_list = []

    for th in tqdm(thresholds):
        for s2 in tqdm(split2):
            for s1 in split1:
                human_effort, acc, precision, recall, param = active_labeling(dataset, models[model_number], inputs[model_number], th, s1, s2, print_=False)
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
    df.to_csv(f"results\\smartInside_dataset\\exp2_D1_NEW_search__{models[model_number]}.csv")



