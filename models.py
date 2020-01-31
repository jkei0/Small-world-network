# -*- coding: utf-8 -*-
"""
Created on Sat Feb  1 01:24:56 2020

@author: jonik
"""

from keras.models import Sequential, Model
from keras.layers import Dense, Dropout, Lambda, Input, concatenate
import numpy as np
import sparseconnection as sp


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
    output_1 = Dense(4, activation='relu')(inputs)
    output_2 = Dense(3, activation='relu')(output_1)
    z = concatenate([output_1, output_2])
    predictions = Dense(1, activation='sigmoid')(z)

    
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
    

def small_model():
    #input tensor
    inp = Input(shape=(30,))
    
    #first layer
    l1 = Dense(16, activation='relu')(inp)
    d1 = Dropout(0.2)(l1)
    
    #second layer
    mat = np.eye(16)
    l2 = sp.CustomConnected(16, connections=mat, activation='relu')(d1)

    d2 = Dropout(0.2)(l2)
    
    #output node
    mat = np.zeros((16,1))
    out = sp.CustomConnected(1, activation='sigmoid', connections=mat, name="output")(d2)
    
    model = Model(inp, out)
    
    model.compile(optimizer='adam', 
                  loss='binary_crossentropy', metrics=['accuracy'])
    return model

def sparseSkipModel():
    
    inp = Input(shape=(30,))
    
    # First layer
    l1 = Dense(4, activation='relu')(inp)
    
    #second layer
    mat = np.zeros((4,4))
    mat[0,0] = 1
    mat[3,0] = 1
    mat[0,1] = 1
    mat[2,1] = 1
    mat[1,2] = 1
    mat[2,2] = 1
    mat[1,3] = 1
    mat[3,3] = 1
    l2 = sp.CustomConnected(4, activation='relu', connections=mat)(l1)
    
    #third layer
    z = concatenate([l2, l1])
    mat2 = np.zeros((4*2,4))
    mat2[0,0:] = 1
    mat2[4,0] = 1
    mat2[7,3] = 1

    l3 = sp.CustomConnected(4, activation='relu', connections=mat2)(z)
    
    #output layer
    mat3 = np.zeros((4*3, 1))
    mat3[0:4] = 1
    mat3[6] = 1
    z = concatenate([l3, z])
    out = sp.CustomConnected(1, activation='sigmoid', connections=mat3)(z)
    
    model = Model(inp, out)
    
    model.compile(optimizer='adam', 
                  loss='binary_crossentropy', metrics=['accuracy'])
    return model
