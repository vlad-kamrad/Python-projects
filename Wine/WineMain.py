from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import scale

import pandas as pd
import matplotlib as mpl
import numpy as np

Location = r'E:\Project\PythonPr\Wine\wine.data'	#Location to file
headers = ["Class", "Alcohol", "MalicAcid", "Ash", "Alcalinity", "Magnesium", "Phenols", "Flavanoids", "Nonflavanoid", "Proanthocyanins", "Color", "Hue", "OD280", "Proline"]
df = pd.read_csv(Location,  names = headers)
WineClass = df["Class"]#Opredelenie classa vina i ego priznakov
WinePr = df[["Alcohol", "MalicAcid", "Ash", "Alcalinity", "Magnesium", "Phenols", "Flavanoids", "Nonflavanoid", "Proanthocyanins", "Color", "Hue", "OD280", "Proline"]]

X = WinePr
y = WineClass
scaledDate = scale(X)	#Mashtabiruemost

def keywithmaxval(d): 
    v = list(d.values())
    k = list(d.keys())
    return k[v.index(max(v))]

kf = KFold(n_splits = 5, shuffle = True, random_state = 42)
#results = cross_val_score(knn, X, y, cv = kf)
#print results.mean()
dictonary = dict()
dictonary2 = dict()
i = 1
while i <= 50:
	knn = KNeighborsClassifier(n_neighbors = i)
	results = cross_val_score(knn, X, y, cv = kf)
	results2 = cross_val_score(knn, scaledDate, y ,cv = kf)
	dictonary[i] = results.mean()
	dictonary2[i] = results2.mean()
	i = i + 1 

print("{0} {1}".format(keywithmaxval(dictonary), max(dictonary.values())))
print("{0} {1}".format(keywithmaxval(dictonary2), max(dictonary2.values())))
print "All"

#print df
