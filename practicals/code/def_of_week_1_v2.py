# -*- coding: utf-8 -*-
"""
Created on Wed Sep 11 14:29:48 2019

@author: s166895
"""
import numpy as np
from sklearn.datasets import load_diabetes, load_breast_cancer
import operator
import matplotlib.pyplot as plt

"""
Linear regression"""

 
def lsq(X, y):
    """
    Least squares linear regression
    :param X: Input data matrix
    :param y: Target vector
    :return: Estimated coefficient vector for the linear regression
    """

    # add column of ones for the intercept
    ones = np.ones((len(X), 1))
    X = np.concatenate((ones, X), axis=1)

    # calculate the coefficients
    beta = np.dot(np.linalg.inv(np.dot(X.T, X)), np.dot(X.T, y))

    return beta


 
""" 
Knn classifier """

def distance(X_train, X_test):  #X_train is the complete training set, and X_test complete all features for one individual
    """
    This function calculates the distance between points in
    the form of Euclidian distance"""
    return np.sqrt(np.sum(np.power(X_train-X_test, 2),1))   # calculates the distance between two points

def get_neighbours_index1(X_train, X_test_individual, k):
    """
    This function calculates all the distances of an individual
    point of X_test with the X_train data. Then sort the
    distances and return a list with the index value of the
    nearest neighbours relative to the individual X_test point""" 
    neighbors_index = []                # list for the neighbors
    #for i in range(X_train.shape[0]):
        #dist = distance(X_train, X_test_individual)  #calculates the distance  between the x_test point and all the x_train points
    dist = np.sqrt(np.sum(np.power(X_train-X_test_individual, 2),1))
    dist = np.absolute(dist)                        #makes sure it is not negative
    dist_patient = np.argsort(dist)     #train patienten van dichtbij naar ver
    neighbors_index = np.argsort(dist_patient)[:k]        #list only the index of the nearest neighbours and not their distance
        
        
    return neighbors_index 


def predictkNNLabels(closest_neighbors, y_train):
    """This function predicts the label of a individual point
    in X_test based on the labels of the nearest neighbour(s).
    And sums up the total of appearences of the labels and 
    returns the label that occurs the most """

    label_train = y_train[closest_neighbors]
    if np.mean(label_train)<0.5:
        label_p = 0
    else:
        label_p=1
    return label_p
    
    

def kNN_test(X_train, X_test, Y_train, Y_test, k):
    """ This function will give a list of predicted labels
    of the x_test data with the use of the aforementioned
    definitions"""
    predicted_labels = np.zeros(Y_test.shape[0])
    for point in range(0, X_test.shape[0]):
        closest_neighbours = get_neighbours_index1(X_train, X_test[point,:], k)  #you get a list with the index of k nearest neighbours
        y_predicted = predictkNNLabels(closest_neighbours, Y_train)       #you get the predicted label that has counted the most by the neighbours
        predicted_labels[point] = y_predicted                           #makes a new list of predicted labels which corresponds with the x_test
    return predicted_labels


def error_squared(true_labels, predicted_labels):
    """ This function calculates the error between the 
    predicted labels and the true labels """
    predictionmeasure = predicted_labels ==true_labels.T #whuch predictions were true or false
    right = sum(predictionmeasure)  #nr of correct predictions
    error1 = len(true_labels)-right #nr of false predictions
    error2 = error1/len(true_labels)
    return error2