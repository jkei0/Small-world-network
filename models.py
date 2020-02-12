# -*- coding: utf-8 -*-
"""
Created on Sat Feb  1 01:24:56 2020

@author: jonik

Different keras models used for testing
"""

from keras.models import Sequential, Model
from keras.layers import Dense, Dropout, Input, concatenate
from keras import regularizers
import numpy as np
import sparseconnection as sp


def model_weight_reg():
    model = Sequential()
    model.add(Dense(561, activation='relu'))
    
    model.add(Dense(64, activation='relu', kernel_regularizer=regularizers.l2(0.01), 
                    activity_regularizer=regularizers.l1(0.01)))
    
    model.add(Dense(64, activation='relu', kernel_regularizer=regularizers.l2(0.01), 
                    activity_regularizer=regularizers.l1(0.01)))

    model.add(Dense(64, activation='relu', kernel_regularizer=regularizers.l2(0.01), 
                    activity_regularizer=regularizers.l1(0.01)))
    
    model.add(Dense(64, activation='relu', kernel_regularizer=regularizers.l2(0.01), 
                    activity_regularizer=regularizers.l1(0.01)))
    
    model.add(Dense(64, activation='relu', kernel_regularizer=regularizers.l2(0.01), 
                    activity_regularizer=regularizers.l1(0.01)))
    
    model.add(Dense(64, activation='relu', kernel_regularizer=regularizers.l2(0.01), 
                    activity_regularizer=regularizers.l1(0.01)))
    
    model.add(Dense(64, activation='relu', kernel_regularizer=regularizers.l2(0.01), 
                    activity_regularizer=regularizers.l1(0.01)))
    
    model.add(Dense(11, activation='softmax'))
    
    model.compile(optimizer='adam', 
            loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    
    return model

        
    

def model_dropout():
    model = Sequential()
    model.add(Dense(561, input_shape=(48,), activation='relu'))
    model.add(Dropout(0.2))
    
    model.add(Dense(16, activation='relu'))
    model.add(Dropout(0.2))
    
    model.add(Dense(16, activation='relu'))
    model.add(Dropout(0.2))
    
    model.add(Dense(16, activation='relu'))
    model.add(Dropout(0.2))

    
    model.add(Dense(11, activation='softmax'))
    
    model.compile(optimizer='adam', 
            loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    
    return model
    

def model_orig():
    
    
    inp = Input(shape=(48,))
    
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
    
    #eight layer
    mat8 = np.zeros((16*7, 16))
    mat8[16*6:,:] = 1
    l8 = sp.CustomConnected(16,connections=mat8, activation='relu')(z)
    z = concatenate([z, l8])
    
    #nineth layer
    mat9 = np.zeros((16*8, 16))
    mat9[16*7:,:] = 1
    l9 = sp.CustomConnected(16,connections=mat9, activation='relu')(z)
    z = concatenate([z, l9])
    
    #output layer
    mat10 = np.zeros((16*9, 11))
    mat10[16*8:, :] = 1
    out = sp.CustomConnected(11, connections=mat10, activation='softmax')(z)
    
    model = Model(inp, out)
    model.compile(optimizer='adam', 
                  loss='sparse_categorical_crossentropy', metrics=['accuracy'])


    layers = [mat10, mat9, mat8, mat7, mat6, mat5, mat4, mat3, mat2]
    
    return model, layers

def model_rewired(layers):
    inp = Input(shape=(48,))
    
    #first layer
    l1 = Dense(16, activation='relu')(inp)
    
    #second
    l2 = sp.CustomConnected(16, connections=layers[8], activation='relu')(l1)
    z = concatenate([l1, l2])
    
    #third
    l3 = sp.CustomConnected(16, connections=layers[7], activation='relu')(z)
    z = concatenate([z,l3])
    
    #forth 
    l4 = sp.CustomConnected(16,connections=layers[6], activation='relu')(z)
    z = concatenate([z, l4])
    
    #fifth
    l5 = sp.CustomConnected(16, connections=layers[5], activation='relu')(z)
    z = concatenate([z, l5])
    
    #sixth
    l6 = sp.CustomConnected(16, connections=layers[4], activation='relu')(z)
    z = concatenate([z, l6])
    
    #seventh 
    l7 = sp.CustomConnected(16, connections=layers[3], activation='relu')(z)
    z = concatenate([z, l7])

    #eighth 
    l8 = sp.CustomConnected(16, connections=layers[2], activation='relu')(z)
    z = concatenate([z, l8])
    
    #ninth 
    l9 = sp.CustomConnected(16, connections=layers[1], activation='relu')(z)
    z = concatenate([z, l9]) 
    
    #output
    out = sp.CustomConnected(11, connections=layers[0], activation='softmax')(z)
    
    model = Model(inp, out)
    model.compile(optimizer='adam', 
                  loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    
    return model


    
    
    
    
    

