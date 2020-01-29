# -*- coding: utf-8 -*-
"""
Created on Thu Jan 23 19:23:27 2020

@author: jonik
"""

import csv
import numpy as np
import keras.backend as K
from keras.models import Sequential, Model
from keras.layers import add, Dense, Dropout, LSTM, Input, Lambda, concatenate
from keras.optimizers import Adam, SGD
from keras.utils import to_categorical
from keras import regularizers
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
#from blocksparse.matmul import BlocksparseMatMul
import tensorflow as tf
from keras.utils.vis_utils import plot_model
import keras.initializers


NUMBER_OF_ATTRIBUTES = 32
NUMBER_OF_INSTANCES = 150
PATH = 'brestcancer/wdbc.data'



def plot_training_history(history):
    # summarize history for accuracy
    plt.plot(history.history['acc'])
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


def model_skip_connection():
    
    # input tensor
    inputs = Input(shape=(30,))
    
    #layers
    output_1 = Dense(16, activation='relu', init='uniform')(inputs)
    output_2 = Dense(16, activation='relu', init='uniform')(output_1)
    z = add([output_1, output_2])
    predictions = Dense(1, activation='sigmoid', init='uniform')(z)

    
    #create model
    model = Model(inputs=inputs, outputs=predictions)
    model.compile(optimizer='adam', 
                  loss='binary_crossentropy', metrics=['accuracy'])
    
    return model
    

def skip_connection_layer():
    inp = Input(shape=(30,), name='i1')
    inp2 = Lambda(lambda x: x[:,1:2], name='i2')(inp)   # get the second neuron
    
    h1_out = Dense(1, activation='relu', name='h1', init='uniform')(inp2)  # only connected to the second neuron
    h2_out = Dense(1, activation='relu', name='h2', init='uniform')(inp)  # connected to both neurons
    h_out = concatenate([h1_out, h2_out])
    
    d1 = Dense(1, activation='relu', init='uniform')(h_out)
    d2 = Dense(1, activation='relu', init='uniform')(inp2)
    d_out = concatenate([d1, d2])
    
    out = Dense(1, activation='sigmoid', init='uniform')(d_out)
    
    
    
    model = Model(inp, out)
    
    model.compile(optimizer='adam', 
                  loss='binary_crossentropy', metrics=['accuracy'])
    return model
    

def test_model(model):
 
    #train model
    history = model.fit(X_train, y_train, epochs=200, verbose=0, 
                        batch_size=100)
    
    #test
    results = model.evaluate(X_test, y_test)
    
    print("Final test set loss {}".format(results[0]))
    print("Final test set accuracy {}".format(results[1]))
    
    plot_training_history(history)
    return history



def small_model():
    #input tensor
    inp = Input(shape=(30,))
    
    #first layer
    l1 = Dense(16, activation='relu')(inp)
    d1 = Dropout(0.2)(l1)
    
    #second layer
    mat = np.eye(16)
    l2 = CustomConnected(16, connections=mat, activation='relu')(d1)

    d2 = Dropout(0.2)(l2)
    
    #output node
    mat = np.zeros((16,1))
    out = CustomConnected(1, activation='sigmoid', connections=mat, name="output")(d2)
    
    model = Model(inp, out)
    
    model.compile(optimizer='adam', 
                  loss='binary_crossentropy', metrics=['accuracy'])
    return model


class CustomConnected(Dense):
    
    def __init__(self, units, connections, **kwargs):
        
        # connection matrix
        self.connections = connections
        
        super(CustomConnected, self).__init__(units,**kwargs)
                
    def call(self, inputs):
        output = K.dot(inputs, self.kernel * self.connections)
        if self.use_bias:
            output = K.bias_add(output, self.bias)
        if self.activation is not None:
            output = self.activation(output)
        return output
    
    def build(self, input_shape):
        super(CustomConnected, self).build(input_shape)
        weights = self.get_weights()
        weights[0] = self.get_weights()[0] * self.connections
        self.set_weights(weights)

    

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
    model = small_model()
    
    s = model.get_weights()
    
    model.summary()
    
    history = test_model(model)
    
    weights = model.get_weights()
    
    aps = w[2] - s[2]
    
    

   
    
    
