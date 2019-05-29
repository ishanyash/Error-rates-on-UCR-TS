# -*- coding: utf-8 -*-
"""
Created on Fri May 24 17:44:56 2019

@author: Ishan Yash
"""

import numpy as np
from pyts.transformation import BOSS
from pyts.transformation import WEASEL
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression


import pyts
print("pyts: {0}".format(pyts.__version__))

PATH = "G:/Coding/ML/UCRArchive_2018/"
clf = LogisticRegression(penalty='l2', C=1, fit_intercept=True,
                         solver='liblinear', multi_class='ovr')
dataset_adiac = "Car"
file_train_adiac = PATH + str(dataset_adiac) + "/" + str(dataset_adiac) + "_TRAIN.tsv"
file_test_adiac = PATH + str(dataset_adiac) + "/" + str(dataset_adiac) + "_TEST.tsv"

train_adiac = np.genfromtxt(fname=file_train_adiac, delimiter="\t", skip_header=0)
test_adiac = np.genfromtxt(fname=file_test_adiac, delimiter="\t", skip_header=0)

X_train_adiac, y_train_adiac = train_adiac[:, 1:], train_adiac[:, 0]
X_test_adiac, y_test_adiac = test_adiac[:, 1:], test_adiac[:, 0]

weasel_adiac = WEASEL(word_size=5, window_sizes=np.arange(6, X_train_adiac.shape[1]))

pipeline_adiac = Pipeline([("weasel", weasel_adiac), ("clf", clf)])

accuracy_adiac = pipeline_adiac.fit(
    X_train_adiac, y_train_adiac).score(X_test_adiac, y_test_adiac)

print("Dataset: {}".format(dataset_adiac))
print("Accuracy on the testing set: {0:.3f}".format(accuracy_adiac))