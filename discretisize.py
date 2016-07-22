# -*- coding: utf-8 -*-
"""
Created on Fri Jan 15 09:15:27 2016

@author: huanghai
"""
import numpy as np
from sklearn import preprocessing
import string
import gc
#输入 ndarray
def disc(X):
    input1 = open("features_type.csv","rb")
    lines = input1.readlines()
    type_list=[]
    for line in lines[1:]:
        type_list.append(line.split(",")[1].strip("\n").strip("\""))
    gc.collect()
    for i in range(X.shape[1]):
        if type_list[i]=='category':
            le = preprocessing.LabelEncoder()
            le.fit(X[:,i])
            X[:,i]=le.transform(X[:,i])
    dim=[]
    for i in range(X.shape[1]):
        dim.append(X[:,i].max())
    dis=[]
    for i in range(X.shape[0]):
        temp=[]
        for j in range(X.shape[1]):
            if type_list[j]=='category':
                list1=np.zeros(dim[j]+1)
                list1.astype(np.byte)
                list1[int(X[i][j])]=1
                temp.extend(list1)
            else:
                temp.append(X[i][j])
        dis.append(temp)
    return np.array(dis)
"""
gc.collect()
 #readdata
input1 = open("train_x.csv","rb")
input2 = open("train_y.csv","rb")
lines = input1.readlines()
data=[]
for line in lines[1:]:
    word = line.split(",")
    temp=[]
    for i in range(1,len(word)):
        temp.append(string.atof(word[i].strip("\"")))
    data.append(temp)
input1.close()
target=[]
lines = input2.readlines()
for line in lines[1:]:
    word = line.split(",")
    target.append([string.atoi(word[1].strip("\n"))])
input2.close()
dis=disc(np.array(data))
"""

