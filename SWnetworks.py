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

NUMBER_OF_ATTRIBUTES = 32
NUMBER_OF_INSTANCES = 150
PATH = 'brestcancer/wdbc.data'



def plot_training_history(history):
    """
    Function plots training loss and accuracy for keras model
    ::param history:: keras history object 

    """
    
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
    """
    Trains keras model
    ::param model:: keras model
    ::output history:: keras history object
    """
 
    #train model
    history = model.fit(X_train, y_train, epochs=200, verbose=1, 
                        batch_size=100)
    
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
        
        else:
            layers[j] = mat[0:layers[j].shape[0], -layers[j].shape[1]+i+1:i+1]
        
        i = i - layers[j].shape[1]
        
    return layers
            
        

def decision(propability):
    return random.random() < propability


def rewire_to_smallworld(adjmat, layers, p):
    """
    Function that rewires connections in graph according to Wattz-Strogatz model,
    creates small-world networks
    ::param adjmat:: graph adjacency matrix
    ::param layers:: numpy matrices that maps connections between neurons in ANN
    ::param p:: probability of rewiring
    ::output mat:: new adjacency matrix
    """
    
    mat = np.array(adjmat)
    rewired = [] #list of already rewired connections
    
    for row in range(mat.shape[0]):
        column = row+1
        while column < mat.shape[0]-2:
            
            if mat[row, column] == 1 and decision(p) and (row,column) not in rewired:
                while True:
                    new_col = random.randint(0, mat.shape[1]-1)
                    new_ind = (row, new_col)
                    if (check_if_neighbour(row, new_col, layers)) and mat[new_ind]==0:
                        rewired.append(new_ind)
                        mat[row,column] = 0
                        mat[column,row] = 0
                        
                        mat[row,new_col] = 1
                        mat[new_col,row] = 1
                        break
            column = column+1
    return mat
                        
            
def check_if_neighbour(neuron1, neuron2, layers):
    """
    Checks if two neurons are neighbours in ANN
    ::param neuron1:: coordinates of neuron 1 in adjacency matrix
    ::param neuron2:: coordinates of neuron 2 in adjacency matrix
    ::param layers:: numpy matrices that maps connections between neurons in ANN
    ::output:: True if neurons are neighbours otherwise false
    """
    for layer in reversed(layers):
        size = layer.shape[0]
        if ((neuron1-size > 0) and (neuron2-size < 0)) or ((neuron1-size < 0) and (neuron2-size > 0)):
            return True
    return False


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
    Dlocal = local_efficiency(graph)
    Dglobal = global_efficiency(graph)
    
    return Dglobal, Dlocal
    
    
def get_random_graph_coeffs(mat):
    
    graph = nx.from_numpy_matrix(mat)

    rand_graph = nx.algorithms.smallworld.random_reference(graph)
    lattice_graph = nx.algorithms.smallworld.lattice_reference(graph)
        
    avg_path_length = nx.average_shortest_path_length(rand_graph)
    cluster_coeff = nx.algorithms.cluster.average_clustering(rand_graph)
        
    avg_path_length_l = nx.average_shortest_path_length(lattice_graph)
    cluster_coeff_l = nx.algorithms.cluster.average_clustering(lattice_graph)
    
    return cluster_coeff, avg_path_length, cluster_coeff_l, avg_path_length_l
        

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
    while p<=1.0:
        mat2 = np.array(mat)
        mat1 = rewire_to_smallworld(mat2,layers,p)
        graph = nx.from_numpy_matrix(mat1, create_using=nx.MultiDiGraph())
        #graph = nx.from_numpy_matrix(mat1)
        remove_wrong_edges(graph)
        global_eff = global_efficiency(graph)
        local_eff = local_efficiency(graph)
        Dglobals.append(global_eff)
        Dlocals.append(local_eff)
        ps.append(p)
        p = p+0.05
            
    return Dglobals, Dlocals, ps
        

def remove_wrong_edges(graph):
    """
    Removes wrong edges from directed graph
    ::param graph:: networkx graph object
    """
    remove_edges = []
    for it in graph.edges():
        if it[0] >= it[1]:
            remove_edges.append(it)
    
    graph.remove_edges_from(remove_edges)
            


def efficiency(G,u,v):
    try:
        eff = 1 / nx.shortest_path_length(G, u, v)
    
    except NetworkXNoPath:
        eff = 0
    return eff  
    

def global_efficiency(G):
    """
    Calculates global efficiency (Dglobal) from graph
    """
    n = len(G)
    denom = n * (n - 1)
    if denom != 0:
        
        g_eff = sum(efficiency(G, u, v) for u, v in permutations(G, 2)) / denom
    else:
        g_eff = 0

    return g_eff

def local_efficiency(G):
    """
    Calculates local efficiensy (Dlocal) from graph
    """

    return sum(global_efficiency(nx.ego_graph(G, v)) for v in G) / len(G)
        
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
    model, layers = models.model_orig()
    graph, mat = ann_to_graph(layers)
    
    #random_clustering, random_path,  cluster_coeff_l, avg_path_length_l = get_random_graph_coeffs(mat)
    
    # rewire connections
    Dglobals, Dlocals, p = find_smallnetwork(mat, layers)
    
    for i in range(len(Dglobals)):
        Dglobals[i] = 1/Dglobals[i]
        Dlocals[i] = 1/Dlocals[i]
        
        
    plt.scatter(p, Dglobals)
    plt.scatter(p, Dlocals)

    
    #layers = graph_to_ann(mat1, layers)
    
    #get model
    #model1 = models.model_rewired(layers)
    #test_model(model1)
    
    
    