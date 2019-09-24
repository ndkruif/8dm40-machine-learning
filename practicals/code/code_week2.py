# -*- coding: utf-8 -*-
"""
Created on Sat Sep 21 15:00:03 2019

@author: s146958
"""

def perf_measures(prediction, y_test):
    TN = 0
    TP = 0
    FN = 0
    FP =0
    for i in range(len(prediction)):
        if prediction[i] == 0 and y_test[i] == 0:
            TN = TN+1
        if prediction[i] == 1 and y_test[i] == 1:
            TP = TP+1
        if prediction[i] == 0 and y_test[i] == 1:
            FN = FN+1
        if prediction[i] == 1 and y_test[i] == 0:
            FP = FP+1
    return (TN, TP, FN, FP)