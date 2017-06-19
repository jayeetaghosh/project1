# Auto insurance model - Jayeeta Ghosh
import pandas as pd
import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
import operator
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn import datasets, linear_model
from sklearn.model_selection import train_test_split
from sklearn import cross_validation
from sklearn.metrics import confusion_matrix
from numpy import genfromtxt, savetxt
import logloss


from sklearn import cross_validation, ensemble
from Utils import prepare_data
from time import time

import sys


logf = open('log.txt','w')  # File where you need to keep the logs

class Unbuffered:
   def __init__(self, stream):
       self.stream = stream

   def write(self, data):
       self.stream.write(data)
       self.stream.flush()
       logf.write(data)    # Write the data of stdout here to a text file as well

sys.stdout = Unbuffered(sys.stdout)


if __name__ == '__main__':

    # all_data,con,cat_int, cat_txt, extra,prev,conf,conf_f,encoders = prepare_data()
    all_data,con,cat_int, cat_txt, extra,prev,conf,conf_f = prepare_data()


    # create training and testing vars
    #https://medium.com/towards-data-science/train-test-split-and-cross-validation-in-python-80b61beca4b6
    y=all_data['A']
    X_train, X_test, y_train, y_test = train_test_split(all_data, y, test_size=0.2)
    print(X_train.shape, y_train.shape)
    print(X_test.shape, y_test.shape)


    all_data['is_train'] =  np.random.uniform(0, 1, len(all_data)) <= 0.8

    data_offered = all_data[all_data.RecordType == 0]
    final_purchase = all_data[all_data.RecordType == 1]

    # features only, target is assigned in "for" loop
    model_on = data_offered[data_offered['is_train']==True]
    ext_test_on = data_offered[data_offered['is_train']==False]

    X = model_on[con+cat_int+cat_txt+conf+extra+prev]
    X_ext = ext_test_on[con+cat_int+cat_txt+conf+extra+prev]
    
    #y = data['A_f'] # 3 class
    #y = data['B_f'] # 2 class
    #y = data['C_f'] # 4 class
    #y = data['D_f'] # 3 class
    #y = data['E_f'] # 2 class
    #y = data['F_f'] # 4 class
    #y = data['G_f'] # 4 class
    

    print ("Final data set going into model building ")
    print(X.describe())
    print(X.shape)
    print(model_on.isnull().sum().to_string())

    n_estim = 50; n_fld = 10; m_feature=6; leafsz=20
    print ("RandomForestClassifier Parameters")
    print ("---------------------------------")
    print ("No of estimators = ", n_estim)
    print ("No of features = ", m_feature)
    print ("No of Min Samples Leaf = ", leafsz)
    print ("CrossValidation -> No of folds = ", n_fld)
    conf = ['A']

    for col in conf: #['A','B','C','D','E','F','G']:
        y = model_on[col+'_f']
        y = np.array(y, dtype=pd.Series)
        # print(np.isnan(y).sum())
        print(X.shape)
        y_ext = ext_test_on[col+'_f']
        print ("\r\n****************************")
        print ("Building model for option ",col)
        
        # clfr = RandomForestClassifier(n_estimators = n_estim, max_features=m_feature,min_samples_leaf = leafsz, max_depth = None,min_samples_split = 2,n_jobs=-1)
        # train_fit = clfr.fit(X,y)
        # trainpreds = train_fit.predict(X)
        # ct = pd.crosstab(y, trainpreds, rownames=['actual'], colnames=['preds'])
        # print("Confusion Matrix for Training set")
        # print(ct)
        # print("Model Done")

        ##### Cross validation is not working
        from sklearn.model_selection import KFold
        kf = KFold(n_splits=n_fld)
        # kf.get_n_splits(X)
        results_pred = sp.zeros(len(y))
        testpreds = sp.zeros(len(y))
        for train_index, test_index in kf.split(X, y):
            print("Train:", train_index, "Test:", test_index)
            X_train, X_test = X.iloc[train_index], X.iloc[test_index]
            y_train, y_test = y[train_index], y[test_index]
            clfr = RandomForestClassifier(n_estimators=n_estim, max_features=m_feature, min_samples_leaf=leafsz,
                                          max_depth=None, min_samples_split=2, n_jobs=-1)
            print(X_train.isnull().sum().to_string())
            # print(np.isnan(y_train).sum())
            print(y[24381])
            print("Now y_train")

            print(y_train)

            y_train = pd.Series(y_train)
            print(y_train.value_counts())
            train_fit = clfr.fit(X_train, y_train)
            results_pred[test_index] = train_fit.predict_proba(X_test)
            testpreds[test_index] = train_fit.predict(X_test)
            print('lets see')

        # # cv = cross_validation.KFold(len(X), n_folds = n_fld, indices = False)
        # # cv = cross_validation.KFold(len(X), n_folds = n_fld)
        # from sklearn.model_selection import KFold
        # cv = KFold(n_splits=n_fld)
        # results_pred = sp.zeros(len(y))
        # testpreds = sp.zeros(len(y))
        # for k, (traincv,testcv) in enumerate(cv.split(X,y)):
        #     # for ifold, (traincv, testcv) in enumerate(cv):
        #
        #     print("TRAIN:", traincv, "TEST:", testcv)
        #     X_train, X_test = X[traincv], X[testcv]
        #     y_train, y_test = y[traincv], y[testcv]
        #     train_fit = clfr.fit(X_train, y_train)
        #     # train_fit = clfr.fit(X[traincv],y[traincv])
        #     results_pred[testcv] = train_fit.predict_proba(X[testcv])
        #     testpreds[testcv] = train_fit.predict(X[testcv])

        #best array contains information about dominant features
        # #ToDo: Make sure to process this to see the effect of important features whether that makes sense or not
        # best=clfr.feature_importances_
        # print("feature_importances:",best)
        #
        ct = pd.crosstab(y, testpreds, rownames=['actual'], colnames=['preds'])
        print ("Confusion Matrix for Training set")
        print(ct)

        # Now calculate % accuracy
        conf_matrix=ct.get_values()
        sum_diag = np.diag(conf_matrix).sum()
        total_sum = conf_matrix.sum()
        acc = 100*sum_diag/total_sum
        print ("Training set accuracy : ",acc,"%")

        # Now apply to external test set
        ext_testpred = train_fit.predict(X_ext)
        ct = pd.crosstab(y_ext, ext_testpred, rownames=['actual'], colnames=['preds'])
        print ("Confusion Matrix for External Test set")
        print(ct)

        # Now calculate % accuracy
        conf_matrix=ct.get_values()
        sum_diag = np.diag(conf_matrix).sum()
        total_sum = conf_matrix.sum()
        acc = 100*sum_diag/total_sum
        print ("External Test accuracy : ",acc,"%")



    

    # Following code calculates sens, spec etc for two class model 
    #accuracy = 100*(conf_matrix[0,0]+conf_matrix[1,1])/(conf_matrix[0,0]+conf_matrix[1,0]+conf_matrix[0,1]+conf_matrix[1,1])
    #sensitivity = 100*(conf_matrix[1,1])/(conf_matrix[1,1]+conf_matrix[0,1])
    #specificity = 100*(conf_matrix[0,0])/(conf_matrix[0,0]+conf_matrix[1,0])
    
    #print accuracy,"---",sensitivity,"---",specificity
    ## The feature importances (the higher, the more important the feature).
    #best=clfr.feature_importances_
    

    #yarr = np.asarray(y)
    #round_pred = np.around(results_pred,decimals=0).astype(int)
    #conf_mat = confusion_matrix(yarr, round_pred)

    ## | True Neg   | False Pos
    ##-------------------------
    ## | False Neg  | True Pos

    #nTrueNeg = conf_mat[0,0]; nFalsePos = conf_mat[0,1]
    #nFalseNeg = conf_mat[1,0]; nTruePos = conf_mat[1,1]

    #print ("From Confusion Matrix")
    #print ("True Pos   |   True Neg   |   False Pos   |   False Neg")
    #print (nTruePos, '   |   ', nTrueNeg, '   |   ', nFalsePos, '   |   ', nFalseNeg)

    ### Sensitivity, Specificity can be calculated only for 2 class model
    ### TODO: calculate Accuracy/Concordance from sum of diagonal elements / number of obs
    #sens = float (nTruePos) / float (nTruePos + nFalseNeg)
    #spec = float (nTrueNeg) / (float (nTrueNeg + nFalsePos))
    #conc = float (nTruePos + nTrueNeg) / float (nTruePos + nTrueNeg + nFalsePos + nFalseNeg)

    #print ("Sensitivity:   ", sens)
    #print ("Specificity:   ", spec)
    #print ("Concordance:   ", conc)

    ##print out the mean of the cross-validated results
    #print "Results: " + str( np.array(results).mean() )

print("Done models A-G")
logf.close()

    