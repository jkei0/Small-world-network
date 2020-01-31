# -*- coding: utf-8 -*-
"""
Created on Thu Jan 23 19:23:27 2020

@author: jonik
"""

import numpy as np
from keras.models import Sequential, Model
from keras.layers import add, Dense, Dropout, LSTM, Input, Lambda, concatenate
from keras.optimizers import Adam, SGD
from keras.utils import to_categorical
from keras import regularizers
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from keras.utils.vis_utils import plot_model
import keras.initializers
import networkx as nx
import pydot

import models
import utils


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

    
def ann_to_graph(layers):
    
    #Generate adjacency matrix
    mat = np.zeros((layers[0].shape[0]+1,layers[0].shape[0]+1))
    i = -1
    for layer in layers:
        if layer.ndim == 1:
            layer = layer.reshape((layer.shape[0], 1))
        
        if layer.shape[1] == 1:
            mat[0:layer.shape[0],i:] = layer
            mat[i, 0:layer.shape[0]] = np.transpose(layer)
            
        else:
            mat[0:layer.shape[0], -layer.shape[1]+i+1:i+1] = layer
            mat[i-layer.shape[1]+1:i+1, 0:layer.shape[0]] = np.transpose(layer)
            
        i = i - layer.shape[1]
        
    # turn into networkx graph
    graph = nx.from_numpy_matrix(mat)
    
    #vizualise NN
    vis = nx.nx_pydot.to_pydot(graph)
    vis.write_png('example2_graph.png')
    
    return graph
    
        
if __name__ == "__main__":
    
    #load dataset
    x,y = utils.load_csv(PATH, NUMBER_OF_ATTRIBUTES)
    
    #convert labels to integers
    y = utils.classes_to_int(y)
    #y = to_categorical(y, num_classes=2)
    
    #split to trainign and testing data
    X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.2,
                                                        random_state=42)
    #get neural network
    model = models.sparseSkipModel()
    
    s = model.get_weights()
    
    model.summary()
    
    history = test_model(model)
    
    weights = model.get_weights()
    
   
    
    
