# -*- coding: utf-8 -*-
"""
Created on Sat Nov 23 13:31:46 2019

@author: lanziyun
"""
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
import seaborn as sns

breast = pd.read_csv("C:/Users/lanziyun/Desktop/data.csv")

Class = {'M': 0, 'B':1}
breast['Diagnosis'] = breast['Diagnosis'].map(Class)
print(breast)

sns.pairplot(breast, hue="Diagnosis", size=2)

X = breast[['3','4','5','6','7','8','9','10','11','12','13','14','15','16','17','18','19','20','21','22','23','24','25','26','27','28','29','30','31','32']]
y = breast[['Diagnosis']]

X_train, X_test, y_train, y_test = train_test_split(X, y, random_state = 5, test_size = 0.2)

clf = LogisticRegression()
clf2 = KNeighborsClassifier()
clf3 = LinearDiscriminantAnalysis()

clf.fit(X_train, y_train)
clf2.fit(X_train, y_train)
clf3.fit(X_train, y_train)

predictclass = clf.predict(X_test)
predictclass2 = clf2.predict(X_test)
predictclass3 = clf3.predict(X_test)
 
print("Logistic acc is: %.4f" % accuracy_score(y_test, predictclass))
print(confusion_matrix(y_test, predictclass))
print(classification_report(y_test, predictclass))
print(clf)

print("KNN acc is: %.4f" % accuracy_score(y_test, predictclass2))
print(confusion_matrix(y_test, predictclass2))
print(classification_report(y_test, predictclass2))
print(clf2)
print("DA acc is: %.4f" % accuracy_score(y_test, predictclass3))
print(confusion_matrix(y_test, predictclass3))
print(classification_report(y_test, predictclass3))
print(clf3)


