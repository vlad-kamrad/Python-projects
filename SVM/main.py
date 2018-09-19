from pandas import read_csv
from sklearn.svm import SVC
import numpy as np
location = "E:\Project\PythonPr\SVM\svm-data.csv"
headers = ["Target", "Prop1", "Prop2"]

print("K P A C N B O")

df = read_csv(location, names = headers)
y = df["Target"]
X = df[["Prop1", "Prop2"]]

clf = SVC(kernel = "linear", C = 100000, random_state = 241)
clf.fit(X, y)

arr = np.array(clf.support_)
arr = arr + 1
print  arr