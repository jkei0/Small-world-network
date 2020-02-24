# -*- coding: utf-8 -*-
"""
Created on Sat Feb  1 01:24:56 2020

@author: jonik

Different keras models used for testing
"""

from keras.models import Sequential, Model
from keras.layers import Dense, Dropout, Input, concatenate
from keras import regularizers
from keras.optimizers import Adam
import numpy as np
import sparseconnection as sp

INPUT = 3072
OUTPUT = 10

def model_dense():
    model = Sequential()
    model.add(Dense(32, activation='relu'))
    model.add(Dense(32, activation='relu'))
    model.add(Dense(32, activation='relu'))
    model.add(Dense(32, activation='relu'))
    model.add(Dense(32, activation='relu'))
    model.add(Dense(32, activation='relu'))


    
    model.add(Dense(OUTPUT, activation='softmax'))
    
    model.compile(optimizer='adam', 
            loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    
    return model



def model_weight_reg():
    l1 = 0.000001
    l2 = 0.00001
    
    model = Sequential()
    model.add(Dense(32, activation='relu', kernel_regularizer=regularizers.l2(l2), 
                    activity_regularizer=regularizers.l1(l1)))
    
    model.add(Dense(32, activation='relu', kernel_regularizer=regularizers.l2(l2), 
                    activity_regularizer=regularizers.l1(l1)))
    
    model.add(Dense(32, activation='relu', kernel_regularizer=regularizers.l2(l2), 
                    activity_regularizer=regularizers.l1(l1)))

    model.add(Dense(32, activation='relu', kernel_regularizer=regularizers.l2(l2), 
                    activity_regularizer=regularizers.l1(l1)))
    
    model.add(Dense(32, activation='relu', kernel_regularizer=regularizers.l2(l2), 
                    activity_regularizer=regularizers.l1(l1)))
    
    model.add(Dense(32, activation='relu', kernel_regularizer=regularizers.l2(l2), 
                    activity_regularizer=regularizers.l1(l1)))
 
    
    model.add(Dense(OUTPUT, activation='softmax'))
    
    model.compile(optimizer='adam', 
            loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    
    return model

        
    

def model_dropout():
    dropout = 0.3
    
    model = Sequential()
    
    model.add(Dense(32, activation='relu'))
    model.add(Dropout(dropout))
    model.add(Dense(32, activation='relu'))
    model.add(Dropout(dropout))
    model.add(Dense(32, activation='relu'))
    model.add(Dropout(dropout))
    model.add(Dense(32, activation='relu'))
    model.add(Dropout(dropout))
    model.add(Dense(32, activation='relu'))
    model.add(Dropout(dropout))
    model.add(Dense(32, activation='relu'))


    
    model.add(Dense(OUTPUT, activation='softmax'))
    
    model.compile(optimizer='adam', 
            loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    
    return model
    

def model_orig():
    
    
    inp = Input(shape=(INPUT,))
    
    #first layer
    l1 = Dense(32, activation='relu')(inp)
    
    #second layer
    mat2 = np.ones((32,32))
    l2 = sp.CustomConnected(32, connections=mat2, activation='relu')(l1)
    z = concatenate([l1, l2])
    
    #third layer
    mat3 = np.zeros((32*2, 32))
    mat3[32:,:] = 1
    l3 = sp.CustomConnected(32,connections=mat3, activation='relu')(z)
    z = concatenate([l1, l2, l3])
    
    #forth layer
    mat4 = np.zeros((32*3, 32))
    mat4[32*2:,:] = 1
    l4 = sp.CustomConnected(32,connections=mat4, activation='relu')(z)
    z = concatenate([l1, l2, l3, l4])
    
    #fifth layer
    mat5 = np.zeros((32*4, 32))
    mat5[32*3:,:] = 1
    l5 = sp.CustomConnected(32,connections=mat5, activation='relu')(z)
    z = concatenate([l1, l2, l3, l4, l5])
    
    #sixth layer
    mat6 = np.zeros((32*5, 32))
    mat6[32*4:,:] = 1
    l6 = sp.CustomConnected(32,connections=mat6, activation='relu')(z)
    z = concatenate([z, l6])
    
    
    #output layer
    mat7 = np.zeros((32*6, OUTPUT))
    mat7[32*5:, :] = 1
    out = sp.CustomConnected(OUTPUT, connections=mat7, activation='softmax')(z)
    
    model = Model(inp, out)
    model.compile(optimizer='adam', 
                  loss='sparse_categorical_crossentropy', metrics=['accuracy'])


    layers = [mat7, mat6, mat5, mat4, mat3, mat2]
    
    return model, layers

def model_rewired(layers):
    inp = Input(shape=(INPUT,))
    
    
    #first layer
    l1 = Dense(32, activation='relu')(inp)
    
    #second
    l2 = sp.CustomConnected(32, connections=layers[5], activation='relu')(l1)
    z = concatenate([l1, l2])
    
    #third
    l3 = sp.CustomConnected(32, connections=layers[4], activation='relu')(z)
    z = concatenate([z,l3])
    
    #forth 
    l4 = sp.CustomConnected(32,connections=layers[3], activation='relu')(z)
    z = concatenate([z, l4])
    
    #fifth
    l5 = sp.CustomConnected(32, connections=layers[2], activation='relu')(z)
    z = concatenate([z, l5])
    
    #sixth
    l6 = sp.CustomConnected(32, connections=layers[1], activation='relu')(z)
    z = concatenate([z, l6])
    
    
    #output
    out = sp.CustomConnected(OUTPUT, connections=layers[0], activation='softmax')(z)
    
    model = Model(inp, out)
    model.compile(optimizer='adam', 
                  loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    
    return model

def model_rewired_weight_reg(layers):

    la1 = 0.000001
    la2 = 0.00001    

    inp = Input(shape=(INPUT,))
    
    
    #first layer
    l1 = Dense(32, activation='relu', kernel_regularizer=regularizers.l2(la2), 
                    activity_regularizer=regularizers.l1(la1))(inp)
    
    #second
    l2 = sp.CustomConnected(32, connections=layers[5], activation='relu', kernel_regularizer=regularizers.l2(la2), 
                    activity_regularizer=regularizers.l1(la1))(l1)
    z = concatenate([l1, l2])
    
    #third
    l3 = sp.CustomConnected(32, connections=layers[4], activation='relu', kernel_regularizer=regularizers.l2(la2), 
                    activity_regularizer=regularizers.l1(la1))(z)
    z = concatenate([z,l3])
    
    #forth 
    l4 = sp.CustomConnected(32,connections=layers[3], activation='relu', kernel_regularizer=regularizers.l2(la2), 
                    activity_regularizer=regularizers.l1(la1))(z)
    z = concatenate([z, l4])
    
    #fifth
    l5 = sp.CustomConnected(32, connections=layers[2], activation='relu', kernel_regularizer=regularizers.l2(la2), 
                    activity_regularizer=regularizers.l1(la1))(z)
    z = concatenate([z, l5])
    
    #sixth
    l6 = sp.CustomConnected(32, connections=layers[1], activation='relu', kernel_regularizer=regularizers.l2(la2), 
                    activity_regularizer=regularizers.l1(la1))(z)
    z = concatenate([z, l6])
    
    
    #output
    out = sp.CustomConnected(OUTPUT, connections=layers[0], activation='softmax')(z)
    
    model = Model(inp, out)
    model.compile(optimizer='adam', 
                  loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    
    return model


    
    
    
    
    

