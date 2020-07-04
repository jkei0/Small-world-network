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


def normalize(x):
    """
        argument
            - x: input image data in numpy array [32, 32, 3]
        return
            - normalized x 
    """
    min_val = np.min(x)
    max_val = np.max(x)
    x = (x-min_val) / (max_val-min_val)
    return x


def unpickle(file):
    import pickle
    with open(file, 'rb') as fo:
        dict = pickle.load(fo, encoding='bytes')
    return dict

def get_cifar_data(path1, path2, path3, path4, path5, path6):
    dic1 = unpickle(path1)
    X_train = dic1[b'data']
    y_train = dic1[b'labels']
    
    dic2 = unpickle(path2)
    X_train = np.append(X_train, dic2[b'data'], axis=0)
    y_train = np.append(y_train, dic2[b'labels'], axis=0)
    
    dic3 = unpickle(path3)
    X_train = np.append(X_train, dic3[b'data'], axis=0)
    y_train = np.append(y_train, dic3[b'labels'], axis=0)
    
    dic4 = unpickle(path4)
    X_train = np.append(X_train, dic4[b'data'], axis=0)
    y_train = np.append(y_train, dic4[b'labels'], axis=0)
    
    dic5 = unpickle(path5)
    X_train = np.append(X_train, dic5[b'data'], axis=0)
    y_train = np.append(y_train, dic5[b'labels'], axis=0)
    
    dic6 = unpickle(path6)
    X_test = dic6[b'data']
    y_test = dic6[b'labels']
    
    return X_train, y_train, X_test, y_test


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
        
        if(row[column] == ""):
            row[column] = "0"
        
        row[column] = row[column].replace(",", ".")
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
            
            try: 
               # result = [x for x in row if x != ''] 
                attList.append(row[0:nroAttributes-1])
                classList.append(row[-1])
            except IndexError:
                print(row)
                        
    #classList = classList[0:len(classList)-3]

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
        while column < mat.shape[0]:
            
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
    #vis = nx.nx_pydot.to_pydot(graph)
    #vis.write_png('example2_graph.png')
    
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
    Dlocal = local_efficiency(graph)
    Dglobal = global_efficiency(graph)
    
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
    while p<=1:
        mat2 = np.array(mat)
        mat1 = rewire_to_smallworld(mat2,layers,p)
        graph = nx.from_numpy_matrix(mat1, create_using=nx.MultiDiGraph())
        remove_wrong_edges(graph)
        global_eff = global_efficiency(graph)
        local_eff = local_efficiency(graph)
        Dglobals.append(global_eff)
        Dlocals.append(local_eff)
        ps.append(p)
        p = p+0.02
            
    return Dglobals, Dlocals, ps

