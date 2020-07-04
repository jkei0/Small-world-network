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
from keras.constraints import maxnorm

INPUT = 48
OUTPUT = 11


optimizer='adam'

def model_dense():
    model = Sequential()
    model.add(Dense(128, input_shape = (INPUT,), activation='relu'))
    model.add(Dense(128, activation='relu'))
    model.add(Dense(128, activation='relu'))
    model.add(Dense(128, activation='relu'))
    model.add(Dense(128, activation='relu'))
    model.add(Dense(128, activation='relu'))
    
    model.add(Dense(OUTPUT, activation='softmax'))
    
    model.compile(optimizer=optimizer, 
            loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    
    return model



def model_weight_reg():
    l1 = 1e-07
    l2 = 1e-06
    
    model = Sequential()
    model.add(Dense(128, input_shape = (INPUT,), activation='relu', kernel_regularizer=regularizers.l2(l2), 
                    activity_regularizer=regularizers.l1(l1)))
    
    model.add(Dense(128, activation='relu', kernel_regularizer=regularizers.l2(l2), 
                    activity_regularizer=regularizers.l1(l1)))
    
    model.add(Dense(128, activation='relu', kernel_regularizer=regularizers.l2(l2), 
                    activity_regularizer=regularizers.l1(l1)))

    model.add(Dense(128, activation='relu', kernel_regularizer=regularizers.l2(l2), 
                    activity_regularizer=regularizers.l1(l1)))
    
    model.add(Dense(128, activation='relu', kernel_regularizer=regularizers.l2(l2), 
                    activity_regularizer=regularizers.l1(l1)))
    
    model.add(Dense(128, activation='relu', kernel_regularizer=regularizers.l2(l2), 
                    activity_regularizer=regularizers.l1(l1)))
 
    
    model.add(Dense(OUTPUT, activation='softmax'))
    opt = Adam(lr=0.01)
    
    model.compile(optimizer=opt, 
            loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    
    return model

        
    

def model_dropout():
    dropout = 0.1
    
    model = Sequential()
    model.add(Dropout(0.3, input_shape=(INPUT,)))
    model.add(Dense(128, input_shape=(INPUT,), activation='relu'))
    model.add(Dropout(dropout))
    model.add(Dense(128, activation='relu', kernel_constraint=maxnorm(4)))
    model.add(Dropout(dropout))
    model.add(Dense(128, activation='relu', kernel_constraint=maxnorm(4)))
    model.add(Dropout(dropout))
    model.add(Dense(128, activation='relu', kernel_constraint=maxnorm(4)))
    model.add(Dropout(dropout))
    model.add(Dense(128, activation='relu', kernel_constraint=maxnorm(4)))
    model.add(Dropout(dropout))
    model.add(Dense(128, activation='relu', kernel_constraint=maxnorm(4)))
    model.add(Dropout(dropout))
    model.add(Dense(OUTPUT, activation='softmax'))
    
    opt = Adam(lr=0.001)
    
    model.compile(optimizer=opt, 
            loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    
    return model
    

def model_orig():
    
    NUM_NODES = 128
    inp = Input(shape=(INPUT,))
    
    #first layer
    l1 = Dense(NUM_NODES, activation='relu')(inp)
    
    #second layer
    mat2 = np.ones((NUM_NODES,NUM_NODES))
    l2 = sp.CustomConnected(NUM_NODES, connections=mat2, activation='relu')(l1)
    z = concatenate([l1, l2])
    
    #third layer
    mat3 = np.zeros((NUM_NODES*2, NUM_NODES))
    mat3[NUM_NODES:,:] = 1
    l3 = sp.CustomConnected(NUM_NODES,connections=mat3, activation='relu')(z)
    z = concatenate([l1, l2, l3])
    
    #forth layer
    mat4 = np.zeros((NUM_NODES*3, NUM_NODES))
    mat4[NUM_NODES*2:,:] = 1
    l4 = sp.CustomConnected(NUM_NODES,connections=mat4, activation='relu')(z)
    z = concatenate([l1, l2, l3, l4])
    
    #fifth layer
    mat5 = np.zeros((NUM_NODES*4, NUM_NODES))
    mat5[NUM_NODES*3:,:] = 1
    l5 = sp.CustomConnected(NUM_NODES,connections=mat5, activation='relu')(z)
    z = concatenate([l1, l2, l3, l4, l5])
    
    #sixth layer
    mat6 = np.zeros((NUM_NODES*5, NUM_NODES))
    mat6[NUM_NODES*4:,:] = 1
    l6 = sp.CustomConnected(NUM_NODES,connections=mat6, activation='relu')(z)
    z = concatenate([z, l6])
    
    
    #output layer
    mat7 = np.zeros((NUM_NODES*6, OUTPUT))
    mat7[NUM_NODES*5:, :] = 1
    out = sp.CustomConnected(OUTPUT, connections=mat7, activation='softmax')(z)
    
    model = Model(inp, out)
    model.compile(optimizer=optimizer, 
                  loss='sparse_categorical_crossentropy', metrics=['accuracy'])


    layers = [mat7, mat6, mat5, mat4, mat3, mat2]
    
    return model, layers
#
#def model_rewired(layers):
#    inp = Input(shape=(INPUT,))
#    
#    
#    #first layer
#    l1 = Dense(32, activation='relu')(inp)
#    
#    #second
#    l2 = sp.CustomConnected(32, connections=layers[5], activation='relu')(l1)
#    z = concatenate([l1, l2])
#    
#    #third
#    l3 = sp.CustomConnected(32, connections=layers[4], activation='relu')(z)
#    z = concatenate([z,l3])
#    
#    #forth 
#    l4 = sp.CustomConnected(32,connections=layers[3], activation='relu')(z)
#    z = concatenate([z, l4])
#    
#    #fifth
#    l5 = sp.CustomConnected(32, connections=layers[2], activation='relu')(z)
#    z = concatenate([z, l5])
#    
#    #sixth
#    l6 = sp.CustomConnected(32, connections=layers[1], activation='relu')(z)
#    z = concatenate([z, l6])
#    
#    
#    #output
#    out = sp.CustomConnected(OUTPUT, connections=layers[0], activation='softmax')(z)
#    
#    model = Model(inp, out)
#    model.compile(optimizer=optimizer, 
#                  loss='sparse_categorical_crossentropy', metrics=['accuracy'])
#    
#    return model

def model_rewired_weight_reg(layers):

    NUM_NODES = 128
    la1 = 1e-07
    la2 = 1e-06   

    inp = Input(shape=(INPUT,))
    
    
    #first layer
    l1 = Dense(NUM_NODES, activation='relu', kernel_regularizer=regularizers.l2(la2), 
                    activity_regularizer=regularizers.l1(la1))(inp)
    
    #second
    l2 = sp.CustomConnected(NUM_NODES, connections=layers[5], activation='relu', kernel_regularizer=regularizers.l2(la2), 
                    activity_regularizer=regularizers.l1(la1))(l1)
    z = concatenate([l1, l2])
    
    #third
    l3 = sp.CustomConnected(NUM_NODES, connections=layers[4], activation='relu', kernel_regularizer=regularizers.l2(la2), 
                    activity_regularizer=regularizers.l1(la1))(z)
    z = concatenate([z,l3])
    
    #forth 
    l4 = sp.CustomConnected(NUM_NODES,connections=layers[3], activation='relu', kernel_regularizer=regularizers.l2(la2), 
                    activity_regularizer=regularizers.l1(la1))(z)
    z = concatenate([z, l4])
    
    #fifth
    l5 = sp.CustomConnected(NUM_NODES, connections=layers[2], activation='relu', kernel_regularizer=regularizers.l2(la2), 
                    activity_regularizer=regularizers.l1(la1))(z)
    z = concatenate([z, l5])
    
    #sixth
    l6 = sp.CustomConnected(NUM_NODES, connections=layers[1], activation='relu', kernel_regularizer=regularizers.l2(la2), 
                    activity_regularizer=regularizers.l1(la1))(z)
    z = concatenate([z, l6])
    
    
    #output
    out = sp.CustomConnected(OUTPUT, connections=layers[0], activation='softmax')(z)
    
    model = Model(inp, out)
    opt = Adam(lr=0.01)
    model.compile(optimizer=opt, 
                  loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    
    return model


def model_orig_8neurons():
    
    
    inp = Input(shape=(INPUT,))
    
    #first layer
    l1 = Dense(8, activation='relu')(inp)
    
    #second layer
    mat2 = np.ones((8,8))
    l2 = sp.CustomConnected(8, connections=mat2, activation='relu')(l1)
    z = concatenate([l1, l2])
    
    #third layer
    mat3 = np.zeros((8*2, 8))
    mat3[8:,:] = 1
    l3 = sp.CustomConnected(8,connections=mat3, activation='relu')(z)
    z = concatenate([l1, l2, l3])
    
    #forth layer
    mat4 = np.zeros((8*3, 8))
    mat4[8*2:,:] = 1
    l4 = sp.CustomConnected(8,connections=mat4, activation='relu')(z)
    z = concatenate([l1, l2, l3, l4])
    
    #fifth layer
    mat5 = np.zeros((8*4, 8))
    mat5[8*3:,:] = 1
    l5 = sp.CustomConnected(8,connections=mat5, activation='relu')(z)
    z = concatenate([l1, l2, l3, l4, l5])
    
    #sixth layer
    mat6 = np.zeros((8*5, 8))
    mat6[8*4:,:] = 1
    l6 = sp.CustomConnected(8,connections=mat6, activation='relu')(z)
    z = concatenate([z, l6])
    
    
    #output layer
    mat7 = np.zeros((8*6, OUTPUT))
    mat7[8*5:, :] = 1
    out = sp.CustomConnected(OUTPUT, connections=mat7, activation='softmax')(z)
    
    model = Model(inp, out)
    model.compile(optimizer=optimizer, 
                  loss='sparse_categorical_crossentropy', metrics=['accuracy'])


    layers = [mat7, mat6, mat5, mat4, mat3, mat2]
    
    return model, layers

def model_rewired(layers):
    
    NUM_NODES = 128
    
    inp = Input(shape=(INPUT,))
    
    
    #first layer
    l1 = Dense(NUM_NODES, activation='relu')(inp)
    
    #second
    l2 = sp.CustomConnected(NUM_NODES, connections=layers[5], activation='relu')(l1)
    z = concatenate([l1, l2])
    
    #third
    l3 = sp.CustomConnected(NUM_NODES, connections=layers[4], activation='relu')(z)
    z = concatenate([z,l3])
    
    #forth 
    l4 = sp.CustomConnected(NUM_NODES,connections=layers[3], activation='relu')(z)
    z = concatenate([z, l4])
    
    #fifth
    l5 = sp.CustomConnected(NUM_NODES, connections=layers[2], activation='relu')(z)
    z = concatenate([z, l5])
    
    #sixth
    l6 = sp.CustomConnected(NUM_NODES, connections=layers[1], activation='relu')(z)
    z = concatenate([z, l6])
    
    
    #output
    out = sp.CustomConnected(OUTPUT, connections=layers[0], activation='softmax')(z)
    
    model = Model(inp, out)
    model.compile(optimizer=optimizer, 
                  loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    
    return model


def model_rewired_dropout(layers):
    
    dropout = 0.1
    NUM_NODES = 128
 
    #input layer
    inp = Input(shape=(INPUT,))
    dinp = Dropout(0.3)(inp)
    
    #first layer
    l1 = Dense(NUM_NODES, activation='relu', kernel_constraint=maxnorm(4))(dinp)
    d1 = Dropout(dropout)(l1)
    
    #second layer
    l2 = sp.CustomConnected(NUM_NODES, connections=layers[5], activation='relu', kernel_constraint=maxnorm(4))(d1)
    z = concatenate([l1,l2])
    d2 = Dropout(dropout)(z)
    
    #third layer
    l3 = sp.CustomConnected(NUM_NODES, connections=layers[4], activation='relu', kernel_constraint=maxnorm(4))(d2)
    z = concatenate([z, l3])
    d3 = Dropout(dropout)(z)
    
    #fourth layer
    l4 = sp.CustomConnected(NUM_NODES, connections=layers[3], activation='relu', kernel_constraint=maxnorm(4))(d3)
    z = concatenate([z, l4])
    d4 = Dropout(dropout)(z)
    
    #fifth layer
    l5 = sp.CustomConnected(NUM_NODES, connections=layers[2], activation='relu', kernel_constraint=maxnorm(4))(d4)
    z = concatenate([z, l5])
    d5 = Dropout(dropout)(z)
    
    #sixth layer
    l6 = sp.CustomConnected(NUM_NODES, connections=layers[1], activation='relu', kernel_constraint=maxnorm(4))(d5)
    z = concatenate([z, l6])
    d6 = Dropout(dropout)(z)
    
    #output layer
    out = sp.CustomConnected(OUTPUT, connections=layers[0], activation='softmax')(d6)
    
    model = Model(inp,out)
    
    opt = Adam(lr=0.001)
    
    model.compile(optimizer=opt, 
            loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    
    return model
    
    
    
    

