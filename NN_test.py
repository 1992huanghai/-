# -*- coding: utf-8 -*-
"""
Created on Sat Jan 16 21:34:44 2016

@author: huanghai
"""
import sys
sys.path.append("E:\\program\\xgboost-0.47\\wrapper")
from kaggler.preprocessing.data import Normalizer
from kaggler.preprocessing.data import OneHotEncoder
from kaggler.online_model import NN
import numpy as np
from scipy.sparse import coo_matrix,hstack
from sklearn.metrics import roc_curve
from sklearn.metrics import roc_auc_score
import xgboost as xgb
from sklearn.externals import joblib
if __name__ == '__main__':
    output = open("NN_result.csv","wb")
    
    #readdata
    print "start readdata"
    data=np.loadtxt("train_x.csv",dtype=np.str,delimiter=',')
    data2=np.loadtxt("test_x.csv",dtype=np.str,delimiter=',')
    data=np.concatenate((data,data2[1:]))
    target = np.loadtxt("train_y.csv",dtype=np.str,delimiter
    =',')
    target = target[1:,1:].astype(np.int)
    data=data[1:,1:]
    for i in range(len(data)):
        for j in range(len(data[0])):
            data[i][j]=data[i][j].strip("\"")
    data=data.astype(np.float)
    print "finish readdata"
    
    #create target
    print "start create"
    xgboost2=xgb.Booster(model_file="xgboost_best2.model")
    datatrain=xgb.DMatrix(data[0:15000])
    target_test=xgboost2.predict(datatrain)
    print "finish create"
    #preprocessing
    print "start preprocessing"
    
    input1 = open("features_type.csv","rb")
    lines = input1.readlines()
    type_list=[]
    for line in lines[1:]:
        type_list.append(line.split(",")[1].strip("\n").strip("\""))
    numerical_data=[]
    numerical_data=np.reshape(numerical_data,(20000,0))
    category_data=[]
    category_data=np.reshape(category_data,(20000,0))    
    for i in range(len(type_list)):
        if type_list[i]=='numeric':
            numerical_data=np.column_stack((numerical_data,data[:,i]))
        else:
            category_data=np.column_stack((category_data,data[:,i]))
    normalizer=Normalizer()
    numerical_data=normalizer.fit_transform(numerical_data)
    ohe=OneHotEncoder(min_obs=2)
    category_data=ohe.fit_transform(category_data)
    data=hstack([numerical_data,category_data])
    data=coo_matrix(data).tocsr()
    data_train=data[0:15000,:]
    data_test=data[15000:,:]
    print "finish preprocessing"
    
    #train
    print "start train"
    clf=NN(n=1e5,epoch=20,h=1000,a=0.01,l2=1e-6)
    clf.fit(data_train,target_test)
    print "finish train"
    
    #test
    print "start test"
    uid = data2[1:,0]   
    ypred = clf.predict(data_test) 
    print "finish test"
    output.write("\"uid\""+","+"\"score\""+'\n\r')
    for i in range(len(ypred)):
        output.write(str(uid[i])+","+str(ypred[i])+"\n\r")
    output.close()
    print "finish test"
    
    