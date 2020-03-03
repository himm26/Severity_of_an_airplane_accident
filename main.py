import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split, RandomizedSearchCV
# from sklearn import svm
# from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, confusion_matrix,classification_report
import pickle

Data_path = "/home/himanshu/Downloads/Airplane_Severity/"


data = pd.read_csv(Data_path + "train.csv")
testData = pd.read_csv(Data_path + "test.csv")

# print(data.shape)
# print(data.columns)


le = LabelEncoder()

data["Severity"] = le.fit_transform(data["Severity"])

target = data["Severity"]
in_data = data.drop("Severity", axis=1)
in_data = in_data.drop("Accident_ID", axis=1)

test_data = testData.drop("Accident_ID", axis=1)

X_train, X_test, y_train, y_test = train_test_split(in_data, target, test_size= 0.1)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

test_data_scaled = scaler.transform(test_data)

clf = DecisionTreeClassifier(max_features='auto', max_depth=180, criterion='entropy')

clf.fit(X_train_scaled, y_train)

Y_pred = clf.predict(X_test_scaled)
out_pred = clf.predict(test_data_scaled)
Y_pred_label = list(le.inverse_transform(out_pred))

output = list(testData["Accident_ID"])
out_data = pd.DataFrame(list(zip(output, Y_pred_label)), columns=['Accident_ID', 'Severity'])
out_data.to_csv(Data_path + "predictions.csv", index=None, header=True)

with open('model.pkl', 'wb') as model_file:
    pickle.dump(clf, model_file)

print(accuracy_score(y_test, Y_pred))
print(confusion_matrix(y_test, Y_pred))
print(classification_report(y_test, Y_pred))
print(clf.score(X_test_scaled, y_test))

