# -*- coding: utf-8 -*-
"""
Created on Sat Feb  1 01:29:09 2020

@author: jonik
"""
from sklearn import preprocessing
import csv
import numpy as np


def classes_to_int(classes):
    """
    Maps different class names to integers
    ::param classes:: list of classes
    ::output y:: classes mapped to integers
    """
    le = preprocessing.LabelEncoder()
    le.fit(classes)
    y = le.transform(classes)
    return y
    

def str_column_to_float(dataset, column):
    for row in dataset:
        row[column]=float(row[column].strip())
        

def load_csv(path, nroAttributes):
    """
    loads csv files from UCI Machine learning repository
    (https://archive.ics.uci.edu/ml/index.php) to numpy matrices
    ::param path:: path of csv file
    ::param nroAttributes:: number of attributes in dataset
    ::output attributes:: attributes in numpy array
    ::output classes:: classes in numpy array
    """
    attList = []
    classList = []
   
    with open(path, 'r') as csvfile:
        reader = csv.reader(csvfile, delimiter=',', quotechar='|')
        for row in reader:
            if len(row)==0:
                break
            attList.append(row[2:nroAttributes+1])
            classList.append(row[1])
    
    #convert attributes to floats
    for i in range(len(attList[0])):
        str_column_to_float(attList, i)
    
    # convert list to two numpy arrays, attributes and classes
    attributes = np.asarray(attList)
    classes = np.asarray(classList)
            
    return attributes, classes
