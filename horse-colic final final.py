# -*- coding: utf-8 -*-
"""
Created on Wed Jun 27 01:46:51 2018

@author: Sidharth
"""
#
import numpy as np
from pandas import read_csv
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
filename='horse-colic.data.csv'
filename2='horse-colic.test.csv'
names=['Surgery','Age','Hosp_no','RT','Pulse','RR','TE','PP','MM','CRT','Pain','Peristalsis','AD','NT','NR','NRPH','RE','abdm','PV','TP','AA','ATP','Outcome','SL','TP1','TP2','Tp3','cP_dat']
dataframe=read_csv(filename,names=names)
dataframe=dataframe.fillna(dataframe.mean())
dataframe1=read_csv(filename2,names=names)
dataframe1=dataframe1.fillna(dataframe.mean())
print(dataframe.mean())
array=dataframe.values
array1=dataframe1.values
np.isnan(array)
shapes=array.shape
print(shapes)
X=array[:,[0,1,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,23,24,25,26]]
Y=array[:,22]
Y=Y.astype('int')
X1=array1[:,[0,1,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,23,24,25,26]]
Y1=array1[:,22]
Y1=Y1.astype('int')
model=DecisionTreeClassifier()
model.fit(X,Y)
result=model.score(X1,Y1)
print(round(result*100),3)
predicted1=model.predict(X1)
matrix=confusion_matrix(Y1,predicted1)
print(matrix)
plt.xlabel('x axis')
plt.ylabel('y axis')
plt.scatter(Y1,predicted1)
print(model.feature_importances_)
model.predict_proba(X1)

#random forest classifier
import numpy as np
from pandas import read_csv
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
filename='horse-colic.data.csv'
filename2='horse-colic.test.csv'
names=['Surgery','Age','Hosp_no','RT','Pulse','RR','TE','PP','MM','CRT','Pain','Peristalsis','AD','NT','NR','NRPH','RE','abdm','PV','TP','AA','ATP','Outcome','SL','TP1','TP2','Tp3','cP_dat']
dataframe=read_csv(filename,names=names)
dataframe=dataframe.fillna(dataframe.mean())
dataframe1=read_csv(filename2,names=names)
dataframe1=dataframe1.fillna(dataframe.mean())
array=dataframe.values
array1=dataframe1.values
np.isnan(array)
shapes=array.shape
X=array[:,[0,1,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,23,24,25,26]]
Y=array[:,22]
Y=Y.astype('int')
X1=array1[:,[0,1,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,23,24,25,26]]
Y1=array1[:,22]
Y1=Y1.astype('int')
model=RandomForestClassifier()
model.fit(X,Y)
predicted=model.predict(X1)
matrix=confusion_matrix(Y1,predicted)
print(matrix)
result=model.score(X1,Y1)
print(round(result*100,3))
plt.xlabel('x axis')
plt.ylabel('y axis')
plt.show
plt.scatter(Y1,predicted)
print(model.feature_importances_)
model.predict_proba(X1)