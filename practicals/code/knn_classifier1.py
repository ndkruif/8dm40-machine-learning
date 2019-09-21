<<<<<<< HEAD
# -*- coding: utf-8 -*-
"""
Created on Wed Sep  4 18:36:00 2019

@author: s166895
"""

import numpy as np
from sklearn.datasets import load_diabetes, load_breast_cancer
import operator
from scipy.special import expit

diabetes = load_diabetes()
breast_cancer = load_breast_cancer()


X_train = breast_cancer.data[:350, np.newaxis, 3]
y_train = breast_cancer.target[:350, np.newaxis]
X_test = breast_cancer.data[350:, np.newaxis, 3]
y_test = breast_cancer.target[350:, np.newaxis]

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

predicted_labels = kNN_test(X_train, X_test, y_train, y_test, 5)
print (predicted_labels)


def error_squared(true_labels, predicted_labels):
    error = (1/len(predicted_labels))*np.sum(np.square(np.subtract(true_labels[:,0],predicted_labels)))
    return error

knn_error = error_squared(y_test,predicted_labels) 
print (knn_error)



=======
# -*- coding: utf-8 -*-
"""
Created on Wed Sep  4 18:36:00 2019

@author: s166895
"""

import numpy as np
from sklearn.datasets import load_diabetes, load_breast_cancer
import operator
import matplotlib.pyplot as plt

diabetes = load_diabetes()
breast_cancer = load_breast_cancer()

x = breast_cancer.data[:]
# normalize the data
x = (x-np.mean(x))/np.std(x)   
    
X_train = x[:350]
y_train = breast_cancer.target[:350, np.newaxis]
X_test = x[350:]
y_test = breast_cancer.target[350:, np.newaxis]

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


def error_squared(true_labels, predicted_labels):
    """ This function calculates the error between the 
    predicted labels and the true labels """
    error = (1/len(predicted_labels))*np.sum(np.square(np.subtract(true_labels[:,0],predicted_labels)))
    return error

dict_of_errors ={}
# the value of k needs always to be an odd number
for k in range(1,len(X_train)+1,2):        
    predicted_labels = kNN_test(X_train, X_test, y_train, y_test, k)
    #calculates the error of every k
    knn_error = error_squared(y_test,predicted_labels)   
    if k in dict_of_errors:
        print (k)
    # creates a dictionary with the k as key and the error as value
    else:
        dict_of_errors[k]=knn_error
    
# It plots all the errors (y-as) against the k value's (x-as)
plt.plot(list(dict_of_errors.keys()), list(dict_of_errors.values()))
plt.xlabel('value of k')
plt.ylabel('error')
plt.title('predicting which k is optimal for the lowest error')
plt.show()

# It will print the value of K with the lowest error
whichK = sorted(dict_of_errors.items(), key=operator.itemgetter(1))
bestKvalue = whichK[0][0]
print (bestKvalue)
>>>>>>> 257bdf16c0296fd6ce5c8b7277adf11ccb0a930f
