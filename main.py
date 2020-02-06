import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn import svm
from sklearn.metrics import accuracy_score, confusion_matrix,classification_report
import pickle

Data_path = "/home/himanshu/Downloads/Airplane_Severity/"


data = pd.read_csv(Data_path + "train.csv")

le = LabelEncoder()

data["Severity"] = le.fit_transform(data["Severity"])

target = data["Severity"]
in_data = data.drop("Severity", axis=1)
in_data = in_data.drop("Accident_ID", axis=1)

X_train, X_test, y_train, y_test = train_test_split(in_data, target, test_size= 0.2)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

clf = svm.SVC(kernel="rbf")

clf.fit(X_train_scaled, y_train)

Y_pred = clf.predict(X_test_scaled)

print(accuracy_score(y_test, Y_pred))
print(confusion_matrix(y_test, Y_pred))
print(classification_report(y_test, Y_pred))
print(clf.score(X_test_scaled, y_test))

