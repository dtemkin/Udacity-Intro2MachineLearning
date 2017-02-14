#!/usr/bin/python

""" 
    This is the code to accompany the Lesson 2 (SVM) mini-project.

    Use a SVM to identify emails from the Enron corpus by their authors:    
    Sara has label 0
    Chris has label 1
"""
    
import sys
from time import time
sys.path.append("../tools/")
from email_preprocess import preprocess
from sklearn.metrics import accuracy_score
from sklearn.svm import SVC

### features_train and features_test are the features for the training
### and testing datasets, respectively
### labels_train and labels_test are the corresponding item labels
features_train, features_test, labels_train, labels_test = preprocess()




#########################################################
### your code goes here ###
#features_train = features_train[:len(features_train)/100]
#labels_train = labels_train[:len(labels_train)/100]

cls = SVC(C=10000., kernel="rbf")
t0 = time()
cls.fit(features_train, labels_train)
print("training time with SVM's linear kernel", time() - t0)
t1 = time()
pred = cls.predict(features_test)
print("prediction time with SVM's linear kernel", time() - t1)
acc = accuracy_score(labels_test, pred)

## For number of emails labelled Chris
print(len(filter(lambda x: x==1, pred)))


#########################################################


