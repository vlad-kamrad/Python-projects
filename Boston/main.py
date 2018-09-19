from sklearn.neighbors import KNeighborsRegressor
from sklearn.preprocessing import scale
from sklearn.model_selection import KFold
from sklearn.datasets import load_boston
from sklearn.model_selection import cross_val_score

import numpy as np

print "K P A C N B O"

def keywithmaxval(d): 
    v = list(d.values())
    k = list(d.keys())
    return k[v.index(max(v))]

kf = KFold(n_splits = 5, shuffle = True, random_state = 42)

df = load_boston()

X = df.data
y = df.target
X = scale(X)

dictonary = dict()

for i in np.linspace(1.0, 10.0, 200):
	#print i
	knn = KNeighborsRegressor(metric = 'minkowski', p = i, n_neighbors = 5, weights = 'distance')
	results = cross_val_score(knn, X, y, cv = kf, scoring = 'neg_mean_squared_error') 	#scoring='neg_mean_squared_error' / mean_squared_error
	dictonary[i] = results.mean()
	#	print("{0}  {1}".format(i, results.mean()))

print keywithmaxval(dictonary)
print "All"