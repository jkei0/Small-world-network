# -*- coding: utf-8 -*-
"""
Created on Thu Jan 23 19:23:27 2020

@author: jonik
"""

import csv
import numpy as np
from keras.models import Sequential
from keras.layers import Dense, Dropout, LSTM
from keras.optimizers import Adam, SGD
from keras.utils import to_categorical
from keras import regularizers
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from blocksparse.matmul import BlocksparseMatMul
import tensorflow as tf

NUMBER_OF_ATTRIBUTES = 32
NUMBER_OF_INSTANCES = 150
PATH = 'brestcancer/wdbc.data'



def plot_training_history(history):
    # summarize history for accuracy
    plt.plot(history.history['accuracy'])
    plt.title('model accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['train'], loc='upper left')
    plt.show()
    # summarize history for loss
    plt.plot(history.history['loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train'], loc='upper left')
    plt.show()

def classes_to_int(classes):
    le = preprocessing.LabelEncoder()
    le.fit(classes)
    y = le.transform(classes)
    return y
    

def str_column_to_float(dataset, column):
    for row in dataset:
        row[column]=float(row[column].strip())

def load_csv():
    attList = []
    classList = []
   
    with open(PATH, 'r') as csvfile:
        reader = csv.reader(csvfile, delimiter=',', quotechar='|')
        for row in reader:
            if len(row)==0:
                break
            attList.append(row[2:NUMBER_OF_ATTRIBUTES+1])
            classList.append(row[1])
    
    #convert attributes to floats
    for i in range(len(attList[0])):
        str_column_to_float(attList, i)
    
    # convert list to two numpy arrays, attributes and classes
    attributes = np.asarray(attList)
    classes = np.asarray(classList)
            
    return attributes, classes

def get_model_dense():
    model = Sequential()
    model.add(Dense(output_dim=16, activation='relu', input_dim=30, init='uniform'))   
    model.add(Dropout(0.1))
    model.add(Dense(16, activation='relu',init='uniform')) 
    model.add(Dropout(0.1))
    model.add(Dense(1, activation='sigmoid',init='uniform'))
    
    model.compile(optimizer='adam', 
                  loss='binary_crossentropy', metrics=['accuracy'])
    return model

def test_model(model):
    
    #train model
    history = model.fit(X_train, y_train, nb_epoch=100, verbose=1, 
                        batch_size=100)
    
    #test
    results = model.evaluate(X_test, y_test)
    
    print("Final test set loss {}".format(results[0]))
    print("Final test set accuracy {}".format(results[1]))
    
    plot_training_history(history)

if __name__ == "__main__":
    
    #load dataset
    x,y = load_csv()
    
    #convert labels to integers
    y = classes_to_int(y)
    #y = to_categorical(y, num_classes=2)
    
    #split to trainign and testing data
    X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.2,
                                                        random_state=42)
    #get neural network
    
    
    
