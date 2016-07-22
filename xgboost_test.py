# -*- coding: utf-8 -*-
"""
Created on Sat Jan 16 21:34:44 2016

@author: huanghai
"""
import sys
sys.path.append("E:\\program\\xgboost-0.47\\wrapper")
import xgboost as xgb
import numpy as np

if __name__ == '__main__':
    output = open("xgboost_result.csv","wb")
    
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
    #data=discretisize.disc(data)
    #imputer = Imputer(missing_values=-1,strategy="most_frequent",axis=0)
    #data=imputer.fit_transform(data)
    print "finish readdata"
    
    #train
    print "start train"
    param={
        'booster':'gbtree',
        'objective': 'binary:logistic',
        'early_stopping_rounds':100,
        'scale_pos_weight': 1400.0/13458.0,
        'eval_metric': 'auc',
        'gamma':0.1,
        'max_depth':8,
        'lambda':550,
        'subsample':0.7,
        'colsample_bytree':0.4,
        'min_child_weight':3,
        'eta': 0.02,
        'seed':'random_seed',
        'nthread':4
    }
    dtrain=xgb.DMatrix(np.concatenate((data[0:15000,:],data[0:15000,:])),label=np.concatenate((target[0:15000],target[0:15000])))
    dval=xgb.DMatrix(data[14000:15000,:],label=target[14000:15000])
    evallist  = [(dval,'eval'), (dtrain,'train')]
    num_round =10000
    bst = xgb.train(param, dtrain, num_round, evallist )
    
    bst.save_model("xgboost_best2.model")
    print "finish train"    
    
    #test
    print "start test"
    data_test=data[15000:]
    uid = data2[1:,0]   
    #data_test = skb.transform(data_test)
    dtest = xgb.DMatrix(data_test)
    ypred = bst.predict(dtest,ntree_limit=bst.best_iteration) 
    print "finish test"
    output.write("\"uid\""+","+"\"score\""+'\n\r')
    for i in range(len(ypred)):
        output.write(str(uid[i])+","+str(ypred[i])+"\n\r")
    output.close()
    
