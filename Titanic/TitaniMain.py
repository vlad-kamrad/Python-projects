from pandas import DataFrame, read_csv
from collections import Counter		#Posmotrel na saite

import matplotlib.pyplot as plt
import pandas as pd
import matplotlib

print("K P A C N B O")

Location = r'E:\Project\PythonPr\Titanic\titanic.csv'	#Location to file
df = pd.read_csv(Location)	#Open file

print "(1)"
countMale = len(df[df['Sex'] == 'male'].index)		#kolvo mujikov na korable
countFemale = len(df[df['Sex'] == 'female'].index)	#kolvo bab na korable
print("{0} {1}".format(countMale, countFemale))

print "(2)"
survived = len(df[df["Survived"] == 1].index)		#kolvo vijivshih
allCount = len(df.index)							#kolvo all humans
print survived
print allCount
dolyaSurvived = (100. * survived)/allCount
print dolyaSurvived

print "(3)"
oneClass = len(df[df["Pclass"] == 1].index)			#kolvo humans one class
dolyaOnesClass = (100. * oneClass)/allCount
print dolyaOnesClass

print "(4)"
medianAge = df["Age"].median()
meanAge = df["Age"].mean()
print("{0} {1}".format(meanAge, medianAge))

print "(5)"
#obshaya corr
#print df.corr()
corr = df['SibSp'].corr(df['Parch'])
print corr

print "(6)"
famaleName = df[df['Sex'] == 'female']	#array bab
arr = famaleName['Name'].str.split(r"Mrs.*\(|Miss.\s*").str[1]
arr1 = arr.str.split(" |\)").str[0]
c = Counter(arr1)
print c.most_common(1)[0][0]	#Potomuchto tak rabotaet, i voobshe pofig





