# -*- coding: utf-8 -*-
"""
Created on Tue May 21 15:56:46 2019

@author: Ishan Yash
"""


import numpy as np
import sklearn
from sklearn.svm import SVC




#print("pyts: {0}".format(pyts.__version__))

PATH = "G:/Coding/ML/UCRArchive_2018/" # Change this value if necessary
dataset = "FacesUCR" 

file_train = PATH + str(dataset) + "/" + str(dataset) + "_TRAIN.tsv"
file_test = PATH + str(dataset) + "/" + str(dataset) + "_TEST.tsv"

train = np.genfromtxt(fname=file_train, delimiter="\t", skip_header=0)
test = np.genfromtxt(fname=file_test, delimiter="\t", skip_header=0)

X_train, y_train = train[:, 1:], train[:, 0]
X_test, y_test = test[:, 1:], test[:, 0]

clf = SVC(gamma='auto')
clf.fit(X_train, y_train)

#print(clf.predict(X_test))

print(1- clf.score(X_test, y_test))



    



