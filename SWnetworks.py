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
import random

import models
import utils
from networkx.exception import NetworkXNoPath
from itertools import permutations

NUMBER_OF_ATTRIBUTES = 49
NUMBER_OF_INSTANCES = 58509

PATH = 'DriveDiagnosis/Sensorless_drive_diagnosis.txt'



def plot_training_history(history):
    """
    Function plots training loss and accuracy for keras model
    ::param history:: keras history object 

    """
    
    # summarize history for accuracy
    plt.plot(history.history['acc'])
    plt.title('Model accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['train'], loc='upper left')
    plt.show()
    # summarize history for loss
    plt.plot(history.history['loss'])
    plt.title('Model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train'], loc='upper left')
    plt.show()
    

def test_model(model, X_train, y_train, X_test, y_test):
    """
    Trains keras model
    ::param model:: keras model
    ::output history:: keras history object
    """
 
    #train model
    history = model.fit(X_train, y_train, epochs=500, verbose=1, 
                        batch_size=124, validation_split=0.3)

    #test
    results = model.evaluate(X_test, y_test)
    
    print("Final test set loss {}".format(results[0]))
    print("Final test set accuracy {}".format(results[1]))
    
    plot_training_history(history)
    return history

    
def ann_to_graph(layers):
    """
    Generates graph from ANN connection matrices
    ::param layers:: list of numpy matrices that maps connections between ANN neurons
    ::output graph:: networkx graph object
    ::output mat:: graph adjacency matrix
    """
       
    #Generate adjacency matrix
    mat = np.zeros((layers[0].shape[0]+11,layers[0].shape[0]+11))
    i = -1
    for layer in layers:
        if layer.ndim == 1:
            layer = layer.reshape((layer.shape[0], 1))
        
        if layer.shape[1] == 1:
            mat[0:layer.shape[0],i:] = layer
            mat[i, 0:layer.shape[0]] = np.transpose(layer)
            
        elif i == -1:
            mat[0:layer.shape[0], -layer.shape[1]:] = layer
            mat[-layer.shape[1]:, 0:layer.shape[0]] = np.transpose(layer)
            
        else:
            mat[0:layer.shape[0], -layer.shape[1]+i+1:i+1] = layer
            mat[i-layer.shape[1]+1:i+1, 0:layer.shape[0]] = np.transpose(layer)
            
        i = i - layer.shape[1]
        
    #mat[0:16, 16:32] = 1
        
    # turn into networkx graph
    graph = nx.from_numpy_matrix(mat)
    
    #vizualise NN
    vis = nx.nx_pydot.to_pydot(graph)
    vis.write_png('example2_graph.png')
    
    return graph, mat


def graph_to_ann(mat, layers):
    """
    Generates ANN connection matrices from graph adjacency matrix
    ::param mat:: adjacency matrix
    ::param layers:: list of correct size numpy matrices
    ::output layers:: list of numpy matrices that maps connections between ANN neurons 
    """

    i = -1
    for j in range(len(layers)):
        
        if layers[j].shape[1] == 1:
            layers[j] = mat[0:layers[j].shape[0], i:]
            
        elif i == -1:
            layers[j] = mat[0:layers[j].shape[0], -layers[j].shape[1]:]
        
        else:
            layers[j] = mat[0:layers[j].shape[0], -layers[j].shape[1]+i+1:i+1]
        
        i = i - layers[j].shape[1]
        
    return layers


def measure_small_worldness(mat):
    """
    Measures small-wordliness of graph
    ::param mat:: graph adjacency matrix
    ::output Dglobal:: global efficiensy of graph
    ::output Dlocal:: local efficiensy of graph
    """

    graph = nx.from_numpy_matrix(mat)
    """
    
    random_clustering, random_path, lattice_cluster, lattice_path = get_random_graph_coeffs(mat)
    
    clustering_coeff = nx.algorithms.cluster.average_clustering(graph)
    shortest_path = nx.average_shortest_path_length(graph)
    small_world_coeff = (random_path/shortest_path) - (clustering_coeff/lattice_cluster)
    """
    Dlocal = utils.local_efficiency(graph)
    Dglobal = utils.global_efficiency(graph)
    
    return Dglobal, Dlocal
    

        

def find_smallnetwork(mat, layers):
    """
    Function rewires given graph with multiple probabilites p, and stores 
    Dglobal and Dlocal values, tries to find values p which gives best 
    representation of small-worldiness of graph
    ::param mat:: graph adjacency matrix
    ::param layers:: numpy matrices that maps connections between neurons in ANN
    ::output Dglobals:: list of Dglobal values
    ::output Dlocal:: list of Dlocal values
    ::output ps:: list of probabilities
    """

    Dglobals = []
    Dlocals = []
    ps = []
    p = 0.0
    while p<=0.76:
        mat2 = np.array(mat)
        mat1 = utils.rewire_to_smallworld(mat2,layers,p)
        graph = nx.from_numpy_matrix(mat1, create_using=nx.MultiDiGraph())
        #graph = nx.from_numpy_matrix(mat1)
        utils.remove_wrong_edges(graph)
        global_eff = utils.global_efficiency(graph)
        local_eff = utils.local_efficiency(graph)
        Dglobals.append(global_eff)
        Dlocals.append(local_eff)
        ps.append(p)
        p = p+0.02
            
    return Dglobals, Dlocals, ps
            
        
if __name__ == "__main__":
    
    #load dataset
    x,y =utils.load_csv(PATH, NUMBER_OF_ATTRIBUTES)
    
    #convert labels to integers
    y = utils.classes_to_int(y)
    #y = to_categorical(y, num_classes=2)
    
    #split to trainign and testing data
    X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.70)
    
    #get neural network
    model, layers = models.model_orig()
#    model = models.model_dropout()

#    test_model(model, X_train, y_train, X_test, y_test)

    
    
    
#    graph, mat = ann_to_graph(layers)
    
    
     #rewire connections
#    Dglobals, Dlocals, p = find_smallnetwork(mat, layers)
#    
#    for i in range(len(Dglobals)):
#        Dglobals[i] = 1/Dglobals[i]
#        Dlocals[i] = 1/Dlocals[i]
#               
#    plt.scatter(p, Dglobals)
#    plt.show()
#    plt.scatter(p, Dlocals)
#    plt.show()
#    
#    rewired_mat = utils.rewire_to_smallworld(mat, layers, 0.3)
#    new_layers = graph_to_ann(rewired_mat, layers)
#    new_model = models.model_rewired(new_layers)
#    test_model(new_model, X_train, y_train, X_test, y_test)

    
    
    
    