from pandas import read_csv
from sklearn.linear_model import Perceptron
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler

LocationTrain = "E:\Project\PythonPr\Percrptron\perceptron-train.csv"
LocationTest  = "E:\Project\PythonPr\Percrptron\perceptron-test.csv"
headers = ["Target", "Prop1", "Prop2"]

Train = read_csv(LocationTrain, names = headers)
Test  = read_csv(LocationTest, names = headers)
scaler = StandardScaler()
prcTron  = Perceptron(random_state = 241, max_iter = 5)
prcTron2 = Perceptron(random_state = 241, max_iter = 5)
print("K P A C N B O")

yTrain = Train["Target"]
xTrain = Train[["Prop1", "Prop2"]]
yTest = Test["Target"]
xTest = Test[["Prop1", "Prop2"]]
prcTron.fit(xTrain, yTrain)
accuracy = accuracy_score(yTest, prcTron.predict(xTest))

xScalTrain = scaler.fit_transform(xTrain)
xScalTest = scaler.transform(xTest)

prcTron2.fit(xScalTrain, yTrain)
accuracy2 = accuracy_score(yTest, prcTron2.predict(xScalTest))
print accuracy
print accuracy2
print accuracy2 - accuracy