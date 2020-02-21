# -*- coding: utf-8 -*-
"""
Created on Sat Feb  1 01:29:09 2020

@author: jonik
"""
from sklearn import preprocessing
import csv
import numpy as np
from networkx.exception import NetworkXNoPath
from itertools import permutations
import networkx as nx
import random


def classes_to_int(classes):
    """
    Maps different class names to integers
    ::param classes:: list of classes
    ::output y:: classes mapped to integers
    """
    le = preprocessing.LabelEncoder()
    le.fit(classes)
    y = le.transform(classes)
    return y
    

def str_column_to_float(dataset, column):
    for row in dataset:
        row[column]=float(row[column].strip())
                
        
def load_csv(path, nroAttributes):
    """
    loads csv files from UCI Machine learning repository
    (https://archive.ics.uci.edu/ml/index.php) to numpy matrices
    ::param path:: path of csv file
    ::param nroAttributes:: number of attributes in dataset
    ::output attributes:: attributes in numpy array
    ::output classes:: classes in numpy array
    """
    attList = []
    classList = []
   
    with open(path, 'r') as csvfile:
        reader = csv.reader(csvfile, delimiter=' ', quotechar='|')
        next(reader)
        for row in reader:
            if len(row)==0:
                break
            
#            result = [x for x in row if x != ''] 
            attList.append(row[1:nroAttributes-1])
            classList.append(row[-1])

    #convert attributes to floats
    for i in range(len(attList[0])):
        str_column_to_float(attList, i)

    
    # convert list to two numpy arrays, attributes and classes
    attributes = np.asarray(attList)
    #classes = np.asarray(classList)
    #classes = np.array(classes)
            
    return attributes, classList

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

def decision(propability):
    return random.random() < propability


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
    
    
def get_random_graph_coeffs(mat):
    
    graph = nx.from_numpy_matrix(mat)

    rand_graph = nx.algorithms.smallworld.random_reference(graph)
    lattice_graph = nx.algorithms.smallworld.lattice_reference(graph)
        
    avg_path_length = nx.average_shortest_path_length(rand_graph)
    cluster_coeff = nx.algorithms.cluster.average_clustering(rand_graph)
        
    avg_path_length_l = nx.average_shortest_path_length(lattice_graph)
    cluster_coeff_l = nx.algorithms.cluster.average_clustering(lattice_graph)
    
    return cluster_coeff, avg_path_length, cluster_coeff_l, avg_path_length_l


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
        while column < mat.shape[0]-layers[0].shape[1]:
            
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
