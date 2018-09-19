from pandas import DataFrame, read_csv
from sklearn.tree import DecisionTreeClassifier		#Dlya klasifikatsii
from sklearn.tree import DecisionTreeRegressor		#Dlya regresii

import matplotlib.pyplot as plt
import pandas as pd
import matplotlib as mpl
import numpy as np

print("K P A C N B O")

Location = r'E:\Project\PythonPr\Titanic2\titanic.csv'	#Location to file
df = pd.read_csv(Location)	#Open file

#notnull() - function from pandas, wh search NaN
datataX = df[ df['Age'].notnull() ][["Pclass", "Fare", "Age", "Sex"]]	#Viborka
datataX = datataX.replace(to_replace = ['male', 'female'], value = [1, 0])		#Zamena znachenii
datataY = df[ df['Age'].notnull() ][['Survived']]

X = np.array(datataX)
Y = np.array(datataY)

clf = DecisionTreeClassifier(random_state = 241)
clf.fit(X, Y)		#poshla jara

#Vajnost priznakov
importances = clf.feature_importances_
print importances