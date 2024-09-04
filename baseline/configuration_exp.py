from sklearn.ensemble import RandomForestClassifier
from sklearn.utils import shuffle
from tqdm import tqdm 
from classification_module import create_x_y_v2, fit_and_acc

from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC



config = ["VIF", "SSIM", "CS"]
model_name = "SVM"
# model = RandomForestClassifier(n_estimators=200, random_state=0)
model = make_pipeline(StandardScaler(), SVC(gamma='auto', probability=True))
efforts = [0.25, 0.5, 0.75]

csv_file = "C:\\Users\\ASUS\\Desktop\\research\\mitacs project\\paper experiments\\cifar dataset\\concept2_cifar_all.csv"


result = []
X, Y = create_x_y_v2(csv_file, config)
for effort in efforts:
    for _ in tqdm(range(20)):
        X, Y = shuffle(X, Y)
        n = int(effort*X.shape[0])
        x_train = X[:n]
        x_test  = X[n:]
        y_train = Y[:n]
        y_test  = Y[n:]

        _, acc, precision, recall = fit_and_acc(model, model_name, x_train, y_train, x_test, y_test)
        result.append([model_name, config, effort, acc, precision, recall])



import csv
with open('D2_concept2_baseline_exp1.csv', 'w') as f:
    write = csv.writer(f)
    write.writerow(["model", "config", "effort", "accuracy", "precision", "recall"])
    write.writerows(result)

