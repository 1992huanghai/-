# -*- coding: utf-8 -*-
"""
Created on Tue Jan 12 20:23:06 2016

@author: huanghai
"""
import numpy as np
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import f_classif
import discretisize
from sklearn.preprocessing import Imputer
if __name__ == '__main__':
    output = open("gbdt_result.csv","wb")
    
    #readdata
    print "start readdata"
    data=np.loadtxt("train_x.csv",dtype=np.str,delimiter=',')
    data2=np.loadtxt("test_x.csv",dtype=np.str,delimiter=',')
    data=np.concatenate((data,data2[1:]))
    target = np.loadtxt("train_y.csv",dtype=np.str,delimiter=',')
    target = target[1:,1:].astype(np.int)
    data=data[1:,1:]
    for i in range(len(data)):
        for j in range(len(data[0])):
            data[i][j]=data[i][j].strip("\"")
    data=data.astype(np.float)
    data=discretisize.disc(data)
    imputer = Imputer(missing_values=-1,strategy="mean",axis=0)
    data=imputer.fit_transform(data)
    print "finish readdata"
    
    #feature selection
    skb=SelectKBest(f_classif,k=600)
    skb.fit(data[0:10000],target[0:10000])
    data=skb.transform(data)
    """
    weight=[]
    for i in range(len(target)):
        if target[i][0]==0:
            weight.append(1)
        else:
            weight.append(1)
    """   
    #train
    data =np.array(data)
    target=np.array(target)
    print "start"
    clf = GradientBoostingClassifier(loss="exponential",n_estimators=260,max_depth=5,random_state=2)
    clf.fit(data[0:15000],target[0:15000])
    print "finish"
    
    #test
    data_test=data[15000:]
    uid = data2[1:,0] 
    y=clf.predict_proba(data_test)
    
    output.write("\"uid\""+","+"\"score\""+'\n\r')
    for i in range(len(y)):
        output.write(str(uid[i])+","+str(y[i][1])+"\n\r")
    output.close()
