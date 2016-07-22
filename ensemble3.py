# -*- coding: utf-8 -*-
"""
Created on Thu Feb 18 09:58:34 2016

@author: think
"""

from sklearn import preprocessing
import string
file1=open("temp2.csv","rb")
file2=open("NN_result.csv","rb")
output=open("result.csv","wb")
lines1=file1.readlines()
lines2=file2.readlines()
output.write(lines1[0])

uid=[]
score1=[]
score2=[]
for i in range(1,len(lines1)):
    line1=lines1[i]
    line2=lines2[i]
    seg1=line1.strip("\n").split(",")
    seg2=line2.strip("\n").split(",")
    uid.append(seg1[0])
    score1.append(string.atof(seg1[1]))
    score2.append(string.atof(seg2[1]))
c1=preprocessing.MinMaxScaler(feature_range=(0,1))
score1=c1.fit_transform(score1)
c2=preprocessing.MinMaxScaler(feature_range=(0,1))
score2=c2.fit_transform(score2)

score=[]
for i in range(len(score1)):
    score.append(score1[i]*0.9+score2[i]*0.1) 
c3=preprocessing.MinMaxScaler(feature_range=(0,1))
score=c2.fit_transform(score)   
for i in range(len(score)):
    output.write(uid[i]+","+str(score[i])+"\n\r")
file1.close()
file2.close()
output.close()
