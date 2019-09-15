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

def distance(X_train, X_test):
    """
    This function calculates the distance between points in
    the form of Euclidian distance"""
    return np.sqrt(np.sum(np.power(X_train-X_test, 2)))    #calculates the distance between two points

def get_neighbours_index(X_train, X_test_individual, k):
    """
    This function calculates all the distances of an individual
    point of X_test with the X_train data. Then sort the
    distances and return a list with the index value of the
    nearest neighbours relative to the individual X_test point""" 
    distances = []                # list for the distances
    neighbors_index = []                # list for the neighbors
    for i in range(0, X_train.shape[0]):
        dist = distance(X_train[i], X_test_individual)  #calculates the distance  between the x_test point and all the x_train points
        dist = np.absolute(dist)                        #makes sure it is not negative
        distances.append((i, dist))
    distances.sort(key=operator.itemgetter(1))          #sorted the list of distances, wherein the distance and INDEX is listed
    for x in range(k):
        neighbors_index.append(distances[x][0])         #list only the index of the nearest neighbours and not their distance

    return neighbors_index


def predictkNNLabels(closest_neighbors, y_train):
    """This function predicts the label of a individual point
    in X_test based on the labels of the nearest neighbour(s).
    And sums up the total of appearences of the labels and 
    returns the label that occurs the most """
    labelPrediction = {}
    for i in range(len(closest_neighbors)):
        if y_train[closest_neighbors[i]][0] in labelPrediction:
            labelPrediction[y_train[closest_neighbors[i]][0]] += 1
        else:
            labelPrediction[y_train[closest_neighbors[i]][0]] = 1        
    sortedLabelPrediction = sorted(labelPrediction.items(), key=operator.itemgetter(1), reverse=True)
    return sortedLabelPrediction[0][0]        # gives the most in common label

def predictkNNLabelsReg(closest_neighbors, y_train): 
    """predictKNNLabelsReg is a function that calculates the predicted label
    with the regression method. It uses the closest_neighbors, which consists of all the
    indices of the nearest neighbors"""
    total = 0;
    for i in range(len(closest_neighbors)):
        total = total + y_train[closest_neighbors[i]][0];
    LabelPrediction = total/len(closest_neighbors)
    return LabelPrediction

def kNN_test(X_train, X_test, Y_train, Y_test, k):
    """ This function will give a list of predicted labels
    of the x_test data with the use of the aforementioned
    definitions"""
    predicted_labels = []
    for point in range(0, X_test.shape[0]):
        closest_neighbours = get_neighbours_index(X_train, X_test[point], k)  #you get a list with the index of k nearest neighbours
        predictedLabels = predictkNNLabels(closest_neighbours, Y_train)       #you get the predicted label that has counted the most by the neighbours
        predicted_labels.append(np.array(predictedLabels))                             #makes a new list of predicted labels which corresponds with the x_test
    predicted_labels = np.array(predicted_labels)
    return predicted_labels

def kNN_test_reg(X_train_d, X_test_d, y_train_d, y_test_d, k):
    "This function gives a list of predicted labels of the x_test data with the kNN-regression method"
    reg_labels = []
    for point in range(0, X_test_d.shape[0]):
        closest_neighbours = get_neighbours_index(X_train_d, X_test_d[point], k)
        print(closest_neighbours)
        predictedLabel = predictkNNLabelsReg(closest_neighbours, y_train_d)
        reg_labels.append(np.array(predictedLabel))
    return reg_labels


def error_squared(true_labels, predicted_labels):
    """ This function calculates the error between the 
    predicted labels and the true labels """
    error = (1/len(predicted_labels))*np.sum(np.square(np.subtract(true_labels[:,0],predicted_labels)))
    return error