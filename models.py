# -*- coding: utf-8 -*-
"""
Created on Sat Feb  1 01:24:56 2020

@author: jonik

Different keras models used for testing
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


def model_orig():
    
    inp = Input(shape=(30,))
    
    #first layer
    l1 = Dense(16, activation='relu')(inp)
    
    #second layer
    mat2 = np.ones((16,16))
    l2 = sp.CustomConnected(16, connections=mat2, activation='relu')(l1)
    z = concatenate([l1, l2])
    
    #third layer
    mat3 = np.zeros((16*2, 16))
    mat3[16:,:] = 1
    l3 = sp.CustomConnected(16,connections=mat3, activation='relu')(z)
    z = concatenate([l1, l2, l3])
    
    #forth layer
    mat4 = np.zeros((16*3, 16))
    mat4[16*2:,:] = 1
    l4 = sp.CustomConnected(16,connections=mat4, activation='relu')(z)
    z = concatenate([l1, l2, l3, l4])
    
    #fifth layer
    mat5 = np.zeros((16*4, 16))
    mat5[16*3:,:] = 1
    l5 = sp.CustomConnected(16,connections=mat5, activation='relu')(z)
    z = concatenate([l1, l2, l3, l4, l5])
    
    #sixth layer
    mat6 = np.zeros((16*5, 16))
    mat6[16*4:,:] = 1
    l6 = sp.CustomConnected(16,connections=mat6, activation='relu')(z)
    z = concatenate([z, l6])
    
    #seventh layer
    mat7 = np.zeros((16*6, 16))
    mat7[16*5:,:] = 1
    l7 = sp.CustomConnected(16,connections=mat7, activation='relu')(z)
    z = concatenate([z, l7])
    
    #output layer
    mat8 = np.zeros((16*7, 1))
    mat8[16*6:] = 1
    out = sp.CustomConnected(1, connections=mat8, activation='sigmoid')(z)
    
    model = Model(inp, out)
    model.compile(optimizer='adam', 
                  loss='binary_crossentropy', metrics=['accuracy'])


    layers = [mat8, mat7, mat6, mat5, mat4, mat3, mat2]
    
    return model, layers

def model_rewired(layers):
    inp = Input(shape=(30,))
    
    #first layer
    l1 = Dense(16, activation='relu')(inp)
    
    #second
    l2 = sp.CustomConnected(16, connections=layers[6], activation='relu')(l1)
    z = concatenate([l1, l2])
    
    #third
    l3 = sp.CustomConnected(16, connections=layers[5], activation='relu')(z)
    z = concatenate([z,l3])
    
    #forth 
    l4 = sp.CustomConnected(16,connections=layers[4], activation='relu')(z)
    z = concatenate([z, l4])
    
    #fifth
    l5 = sp.CustomConnected(16, connections=layers[3], activation='relu')(z)
    z = concatenate([z, l5])
    
    #sixth
    l6 = sp.CustomConnected(16, connections=layers[2], activation='relu')(z)
    z = concatenate([z, l6])
    
    #seventh 
    l7 = sp.CustomConnected(16, connections=layers[1], activation='relu')(z)
    z = concatenate([z, l7])    
    
    #output
    out = sp.CustomConnected(1, connections=layers[0], activation='sigmoid')(z)
    
    model = Model(inp, out)
    model.compile(optimizer='adam', 
                  loss='binary_crossentropy', metrics=['accuracy'])
    
    return model
    

def model_orig_compare():
    model = Sequential()
    model.add(Dense(output_dim=16, activation='relu', input_dim=30))   
    model.add(Dense(16, activation='relu'))
    model.add(Dense(16, activation='relu')) 
    model.add(Dense(1, activation='sigmoid'))
    
    model.compile(optimizer='adam', 
                  loss='binary_crossentropy', metrics=['accuracy'])
    return model

    
    
    
    
    

