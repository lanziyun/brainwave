# -*- coding: utf-8 -*-
"""
Created on Thu Nov 21 20:57:14 2019

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
import warnings
warnings.filterwarnings("ignore")
iris = pd.read_csv("C:/Users/lanziyun/Desktop/機器學習/iris.csv")

Class = {'Iris-setosa': 0, 'Iris-versicolor':1, 'Iris-virginica':2}
iris['class'] = iris['class'].map(Class)
print(iris)

sns.pairplot(iris, hue="class", size=2)

X = iris[['sepal length','sepal width','petal length','petal width']]
y = iris[['class']]

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