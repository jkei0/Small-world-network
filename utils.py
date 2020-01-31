# -*- coding: utf-8 -*-
"""
Created on Sat Feb  1 01:29:09 2020

@author: jonik
"""
from sklearn import preprocessing
import csv
import numpy as np


def classes_to_int(classes):
    le = preprocessing.LabelEncoder()
    le.fit(classes)
    y = le.transform(classes)
    return y
    

def str_column_to_float(dataset, column):
    for row in dataset:
        row[column]=float(row[column].strip())
        

def load_csv(path, nroAttributes):
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
