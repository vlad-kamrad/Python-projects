from sklearn import datasets
from sklearn.svm import SVC
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import GridSearchCV
#from sklearn.grid_search import GridSearchCV
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
import numpy as np
cv = KFold(n_splits = 5, shuffle = True, random_state = 241)
newsgroups = datasets.fetch_20newsgroups(subset = 'all', categories = ['alt.atheism', 'sci.space'])

y = newsgroups.target
X = newsgroups.data
vec = TfidfVectorizer()
X = vec.fit_transform(X)

print("K P A C N B O")
grid = {"C" : np.power(10.0, np.arange(-5, 6))}
clf = SVC(kernel = "linear",  random_state = 241)
gs = GridSearchCV(clf, grid, scoring = "accuracy", cv = cv)

#gs.fit(X, y)
#C_best = gs.best_params_.get('C')
#print C_best 
# lutshiy C = 1.0

clf = SVC(kernel = "linear",  C = 1.0, random_state = 241)
clf.fit(X, y)

row = clf.coef_.getrow(0).toarray()[0].ravel()
top_ten_indicies = np.argsort(abs(row))[-10:]
top_ten_values = row[top_ten_indicies]

otvets = []
for a in top_ten_indicies:
	#print(feature_mapping[a])
	otvets.append(feature_mapping[a])
	
otvets.sort()
print otvets