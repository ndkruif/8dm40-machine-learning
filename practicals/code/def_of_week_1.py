# -*- coding: utf-8 -*-
"""
Created on Wed Sep 11 14:29:48 2019

@author: s166895
"""
import numpy as np
from sklearn.datasets import load_diabetes, load_breast_cancer
import operator
from scipy.special import expit
 


def distance(X_train, X_test):
    return np.sqrt(np.sum(np.power(X_train-X_test, 2)))    #calculates the distance between two points

def get_neighbours_index(X_train, X_test_individual, k):
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
    labelPrediction = {}
    for i in range(len(closest_neighbors)):
        if y_train[closest_neighbors[i]][0] in labelPrediction:
            labelPrediction[y_train[closest_neighbors[i]][0]] += 1
        else:
            labelPrediction[y_train[closest_neighbors[i]][0]] = 1        
    sortedLabelPrediction = sorted(labelPrediction.items(), key=operator.itemgetter(1), reverse=True)
    return sortedLabelPrediction[0][0]        # gives the most in common label


def kNN_test(X_train, X_test, Y_train, Y_test, k):
    predicted_labels = []
    for point in range(0, X_test.shape[0]):
        closest_neighbours = get_neighbours_index(X_train, X_test[point], k)  #you get a list with the index of k nearest neighbours
        predictedLabels = predictkNNLabels(closest_neighbours, Y_train)       #you get the predicted label that has counted the most by the neighbours
        predicted_labels.append(np.array(predictedLabels))                             #makes a new list of predicted labels which corresponds with the x_test
    predicted_labels = np.array(predicted_labels)
    return predicted_labels

def error_squared(true_labels, predicted_labels):
    error = (1/len(predicted_labels))*np.sum(np.square(np.subtract(true_labels[:,0],predicted_labels)))
    return error