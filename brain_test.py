# -*- coding: utf-8 -*-
"""
Created on Wed Jan 15 20:57:35 2020

@author: lanziyun
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.svm import SVC
from sklearn import model_selection
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
#from sklearn.metrics import confusion_matrix
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
import warnings
warnings.filterwarnings("ignore")

brainwave = pd.read_csv("0517-sis1.csv", delimiter=',', header=None, skiprows=1, names=['user_id','time','song','Delta','Theta','Low_Alpha','High_Alpha','Low_Beta',
                            'High_Beta','Low_Gamma','Mid_Gamma','Attenttion','Meditation','flag'])

array = brainwave.values
print(array)
attention = array[:,11]
meditation = array[:,12]

att = [attention[i:i+25] for i in range(0,len(attention),25)]
att_avg = []
for i in range (len(att)):
    att_sum = 0
    for j in range(len(att[0])):
        att_sum+=att[i][j]
    att_avg.append(att_sum/25)
#print(att_avg)

med = [meditation[i:i+25] for i in range(0,len(meditation),25)]
med_avg = []
for i in range (len(med)):
    med_sum = 0
    for j in range(len(med[0])):
        med_sum+=med[i][j]
    med_avg.append(med_sum/25)
#print(med_avg)

att3 = []
for i in range (len(att)):
    attt = att[i]
    att2 = [attt[j:j+5] for j in range(0,len(attt),5)]
    att3.append(att2)

att_uptimes = []
att_downtimes = []
for i in range(len(att3)):
    att_slope = 0
    att_uptime = 0
    att_downtime = 0
    for j in range(len(att3[0])):
        att_slope = (att3[i][0][4] - att3[i][0][0])/5-1
        att_slope2 = (att3[i][1][4] - att3[i][1][0])/5-1
        if(att_slope > 0.8):
            att_uptime+=1
        if(att_slope2 > 0.8):
            att_uptime+=1
        if(att_slope < -0.8):
            att_downtime+=1
        if(att_slope2 < -0.8):
            att_downtime+=1
    att_uptimes.append(att_uptime)
    att_downtimes.append(att_downtime)
#print(att_uptimes)
#print(att_downtimes)

med3 = []
for i in range (len(med)):
    medd = med[i]
    med2 = [medd[j:j+5] for j in range(0,len(medd),5)]
    med3.append(med2)

med_uptimes = []
med_downtimes = []
for i in range(len(med3)):
    med_slope = 0
    med_uptime = 0
    med_downtime = 0
    for j in range(len(med3[0])):
        med_slope = (med3[i][0][4] - med3[i][0][0])/5-1
        med_slope2 = (med3[i][1][4] - med3[i][1][0])/5-1
        if(med_slope > 0.8):
            med_uptime+=1
        if(med_slope2 > 0.8):
            med_uptime+=1
        if(med_slope < -0.8):
            med_downtime+=1
        if(med_slope2 < -0.8):
            med_downtime+=1
    med_uptimes.append(med_uptime)
    med_downtimes.append(med_downtime)
#print(med_uptimes)
#print(med_downtimes)

pram = np.array([att_avg,med_avg,att_uptimes,att_downtimes,med_uptimes,med_downtimes])
arr0 = []
for i in range(len(pram[0])):
    for j in range(len(pram)):    
        arr0.append(pram[j][i])
arr1 = [arr0[j:j+6] for j in range(0,len(arr0),6)]    
print('特徵值= ', arr1)
y_value = []
for i in range(len(arr1)):
    if(att_avg[i] >= 50 and med_avg[i] >= 50 and att_uptimes[i] > 1 and med_uptimes[i] > 1):
        y_value.append(1)
    elif((att_avg[i] < 50 and med_avg[i] >= 50) or (att_uptimes[i] < 1 and med_uptimes[i] > 1)):
        y_value.append(2)
    elif(att_avg[i] >= 50 and med_avg[i] < 50 or (att_uptimes[i] > 1 and med_uptimes[i] < 1)):
        y_value.append(4)
    elif(att_avg[i] < 50 or med_avg[i] < 50 or (att_uptimes[i] < 1 and med_uptimes[i] < 1)):
        y_value.append(3)
print('預測值= ', y_value)

x = arr1
y = y_value
validation_size = 0.4
seed = 7
X_train, X_validation, Y_train, Y_validation = model_selection.train_test_split(x, y, random_state = seed, test_size = validation_size)

# Models
models = []
models.append(('SVM', SVC()))
models.append(('DecisionTree', DecisionTreeClassifier()))
models.append(('RandomForest', RandomForestClassifier(n_estimators=100, random_state=0)))
models.append(('KNN', KNeighborsClassifier(n_neighbors=3)))
models.append(('Naive Bayes', GaussianNB()))
models.append(('LogisticRegression', LogisticRegression()))
models.append(('LinearDiscriminantAnalysis', LinearDiscriminantAnalysis()))

scoring = 'accuracy'

# evaluate each model in turn
results = []
names = []
for name, model in models:
	kfold = model_selection.KFold(n_splits=6, random_state=None)
	cv_results = model_selection.cross_val_score(model, X_train, Y_train, cv=kfold, scoring=scoring)
	results.append(cv_results)
	names.append(name)
for i in range (len(results)):
    print("%s: %.4f (%.4f)" %(names[i], results[i].mean(), cv_results.std()))


'''
print('----------------------')
clf = SVC()
clf.fit(X_train,Y_train)
predictclass1 = clf.predict(X_validation)
print("SVM: %.4f" % accuracy_score(Y_validation, predictclass1))
print(Y_validation,predictclass1)
#print(confusion_matrix(Y_validation, predictclass))
print(classification_report(Y_validation, predictclass1))


print('----------------------')
clf1 = DecisionTreeClassifier()
clf1.fit(X_train,Y_train)
predictclass2 = clf1.predict(X_validation)
print("決策樹: %.4f" % accuracy_score(Y_validation, predictclass2))
print(Y_validation, predictclass2)
#print(classification_report(Y_validation, predictclass2))

print('----------------------')
clf2 = RandomForestClassifier(n_estimators=100, random_state=0)
clf2.fit(X_train,Y_train)
predictclass3 = clf2.predict(X_validation)
print("隨機森林: %.4f" % accuracy_score(Y_validation, predictclass3))
print(Y_validation, predictclass3)
#print(classification_report(Y_validation, predictclass3))

print('----------------------')
clf3 = LogisticRegression()
clf3.fit(X_train,Y_train)
predictclass4 = clf3.predict(X_validation)
print("羅吉斯回歸: %.4f" % accuracy_score(Y_validation, predictclass4))
print(Y_validation, predictclass4)
#print(classification_report(Y_validation, predictclass4))

print('----------------------')
clf4 = KNeighborsClassifier()
clf4.fit(X_train,Y_train)
predictclass5 = clf4.predict(X_validation)
print("KNN: %.4f" % accuracy_score(Y_validation, predictclass5))
print(Y_validation, predictclass5)
#print(classification_report(Y_validation, predictclass5))

print('----------------------')
clf5 = LinearDiscriminantAnalysis()
clf5.fit(X_train,Y_train)
predictclass6 = clf5.predict(X_validation)
print("LDA區別分析: %.4f" % accuracy_score(Y_validation, predictclass6))
print(Y_validation, predictclass6)
#print(classification_report(Y_validation, predictclass6))
'''
'''
## 參數權重 
weight = pd.read_excel('X_test.xls',index_col=0)
weight_name = weight[['A_avg','M_avg','A_ups','A_downs','M_ups','M_downs']]

brain_features = [x for i,x in enumerate(weight.columns)]

def plot_feature_importances(model):
    plt.figure(figsize=(10,6))
    n_features = 6
    plt.barh(range(n_features), model.feature_importances_, align='center')
    plt.yticks(np.arange(n_features), brain_features)
    plt.xlabel("Feature importance")
    plt.ylabel("Feature")
    plt.ylim(-1, n_features)

plot_feature_importances(clf1)
plt.title("DecisionTree")
plt.savefig('feature_importance')
plt.show()

plot_feature_importances(clf2)
plt.title("RandomForest")
plt.savefig('feature_importance')
plt.show()

## 線性圖
plt.plot(Y_validation, predictclass2, 'g', label='optimal value')
plt.xlabel("type")
plt.ylabel("predit")
plt.title("Decision Tree")
plt.legend()
plt.show()

c = np.array(y_value) # 使用原本 arr_y值
d = np.array(clf.predict(x)) # 使用predict值


## 離散圖
plt.scatter(c[c==1],c[c==1],s=30,c='red',marker='o',alpha=0.5,label='1')
plt.scatter(c[c==2],c[c==2],s=30,c='green',marker='o',alpha=0.5,label='2')
plt.scatter(c[c==3],c[c==3],s=30,c='blue',marker='x',alpha=0.5,label='3')
plt.scatter(c[c==4],c[c==4],s=30,c='yellow',marker='o',alpha=0.5,label='4')

plt.title('basic scatter plot ')
plt.xlabel('type')
plt.ylabel('predictclass')
plt.legend(loc='upper left')
plt.show()

## 直方圖
print('\n')
print('------ 使用原本 arr_y值 -----')
print('arr_y', c)
plt.hist(c, rwidth=0.8, range=(1,4))
plt.xlabel('type')
plt.ylabel('count')
plt.title("arr_y histogram") 
plt.show()

print('------ 使用predict值 -----')
print('clf.predict(x)', d)
plt.hist(d, rwidth=0.8, range=(1,4))
plt.xlabel('type')
plt.ylabel('count')
plt.title("predict histogram") 
plt.show()
'''