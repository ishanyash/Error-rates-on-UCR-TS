# -*- coding: utf-8 -*-
"""
Created on Tue May 21 15:56:46 2019

@author: Ishan Yash
"""


import numpy as np
import pyts
from pyts.classification import KNeighborsClassifier
import xlwt
from xlwt import Workbook 
from pyts.transformation import BOSS
from pyts.classification import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.ensemble import VotingClassifier
from sklearn.preprocessing import FunctionTransformer




#print("pyts: {0}".format(pyts.__version__))

PATH = "G:/Coding/ML/UCRArchive_2018/" # Change this value if necessary
dataset = ["CBF"] #, "ECG200", "GunPoint", "MiddlePhalanxTW", "Plane", "SyntheticControl"]
nor = 1
warping_window_list = [0.05] #, 0., 0., 0.03, 0.05, 0.06]


error_ed_list = []
error_dtw_list = []
error_dtw_w_list = []
error_boss_list = []
#default_rate_list = []

file_train = PATH + str(dataset) + "/" + str(dataset) + "_TRAIN.tsv"
file_test = PATH + str(dataset) + "/" + str(dataset) + "_TEST.tsv"

for i, (dataset, warping_window) in enumerate(zip(dataset, warping_window_list)):
    print("Dataset: {}".format(dataset))
    
    file_train = PATH + str(dataset) + "/" + str(dataset) + "_TRAIN.tsv"
    file_test = PATH + str(dataset) + "/" + str(dataset) + "_TEST.tsv"
    
    train = np.genfromtxt(fname=file_train, delimiter="\t", skip_header=0)
    test = np.genfromtxt(fname=file_test, delimiter="\t", skip_header=0)

    X_train, y_train = train[:, 1:], train[:, 0]
    X_test, y_test = test[:, 1:], test[:, 0]

    clf_ed = KNeighborsClassifier(metric='euclidean')
    clf_dtw = KNeighborsClassifier(metric='dtw')
    clf_dtw_w = KNeighborsClassifier(metric='dtw_sakoechiba',
                                     metric_params={'window_size': warping_window})

    # Euclidean Distance
    error_ed = 1 - clf_ed.fit(X_train, y_train).score(X_test, y_test)
    print('Accuracy ED: ', 1 - error_ed )
    print("Error rate with Euclidean Distance: {0:.4f}".format(error_ed))
    error_ed_list.append(error_ed)
    
    # Dynamic Time Warping
    error_dtw = 1 - clf_dtw.fit(X_train, y_train).score(X_test, y_test)
    print('Accuracy DTW: ', 1 - error_dtw )
    print("Error rate with Dynamic Time Warping: {0:.4f}".format(error_dtw))
    error_dtw_list.append(error_dtw)
    
    # Dynamic Time Warping with a learned warping window
    error_dtw_w = 1- clf_dtw_w.fit(X_train, y_train).score(X_test, y_test)
    print('Accuracy DTW_W: ', 1 - error_dtw_w )
    print("Error rate with Dynamic Time Warping with a learned warping "
          "window: {0:.4f}".format(error_dtw_w))
    error_dtw_w_list.append(error_dtw_w)
    
    print()

#BOSS

transformer = FunctionTransformer(func=lambda x: x.toarray(),
                                  validate=False, check_inverse=False)
knn = KNeighborsClassifier(n_neighbors=1, metric='boss')


'''*******************************************************************  SAX'''
#train = read_ucr_data(r'G:\Coding\ML\UCRArchive_2018\ECG200\ECG200_TRAIN.tsv')
#test = read_ucr_data(r'G:\Coding\ML\UCRArchive_2018\ECG200\ECG200_TEST.tsv')

train = read_ucr_data(file_train)
test = read_ucr_data(file_test)

win = 30
paa = 6
alp = 6
na_strategy = "exact"
zthresh = 0.01

def test_accuracy(dd_train, dd_test, sax_win, sax_paa, sax_alp, sax_strategy, z_threshold):
    
    train_bags = {}
    for key, arr in dd_train.items():
        train_bags[key] = manyseries_to_wordbag(dd[key], sax_win, sax_paa,
                                                sax_alp, sax_strategy, z_threshold)
    
    tfidf_vectors = bags_to_tfidf(train_bags)

    correct = 0
    count = 0

    for cls in [*dd_test.copy()]:
        for s in dd_test[cls]:
            sim = cosine_similarity(tfidf_vectors, 
                                    series_to_wordbag(s, sax_win, sax_paa,
                                                      sax_alp, sax_strategy, z_threshold))
            res = class_for_bag(sim)
            if res == cls:
                correct = correct + 1
            count = count + 1
    
    return correct / count

accuracy = test_accuracy(train, test, win, paa, alp, na_strategy, zthresh)
errorr = 1 - accuracy



print('Accuracy SAX:',accuracy)
print('Error Rate SAX: ', errorr)



er_final = [error_ed,error_dtw,error_dtw_w,errorr]
acc_final = [1-error_ed,1-error_dtw,1-error_dtw_w,1-errorr]

print('Minimum Error rate: ', min(er_final))
print('Maximum Accuracy: ', max(acc_final))

wb = Workbook()
sheet1=wb.add_sheet('sheet 1')
sheet1.write(0,1,'ED error rate')
sheet1.write(0,2,'DTW error rate')
sheet1.write(0,3,'DTW_W error rate')
sheet1.write(0,4,'SAX error rate')
sheet1.write(0,5,'Minimum Error rate')
sheet1.write(0,6,'Maximum Accuracy')
sheet1.write(0,0, 'Dataset Name')
sheet1.write(int(nor),0, str(dataset))
sheet1.write(int(nor),1, float(error_ed))
sheet1.write(int(nor),2, float(error_dtw))
sheet1.write(int(nor),3, float(error_dtw_w))
sheet1.write(int(nor),4, float(errorr))
sheet1.write(int(nor),5, float(min(er_final)))
sheet1.write(int(nor),6, float(max(acc_final)))

file_ER_ALL = 'G:/Coding/ML' +"/" + str(dataset) + ".csv"

wb.save(file_ER_ALL) 








