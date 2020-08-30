# -*- coding: utf-8 -*-
"""
Created on Sat Mar 28 15:50:01 2020

@author: jonik
"""


import csv
from sklearn import preprocessing
import numpy as np
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Dropout, Input, concatenate
from keras import regularizers
from keras.optimizers import Adam
from keras.constraints import maxnorm
import tensorflow.keras.backend as K
import random
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import networkx as nx


PATHDRIVE = 'avila/avila/avila-tr.txt'
PATH = 'dataframe/data11.pkl'
PATH_ACC = 'dataframe/data_acc11.pkl'
NUMBER_OF_ATTRIBUTES = 10
NUMBER_OF_INSTANCES = 58509
EPOCHS = 400
INPUT = 4
OUTPUT = 4
optimizer='adam'
NUM_NODES = 128
NUM_MODELS = 4
TEST_SIZE = 0.8
SWPARA = 0.7


class CustomConnected(Dense):
    
    def __init__(self, units, connections, **kwargs):
        
        # connection matrix
        self.connections = connections
        
        super(CustomConnected, self).__init__(units,**kwargs)
                
    def call(self, inputs):
        output = K.dot(inputs, self.kernel * self.connections)
        if self.use_bias:
            output = K.bias_add(output, self.bias)
        if self.activation is not None:
            output = self.activation(output)
        return output
    
    def build(self, input_shape):
        super(CustomConnected, self).build(input_shape)
        weights = self.get_weights()
        weights[0] = self.get_weights()[0] * self.connections
        self.set_weights(weights)


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
        reader = csv.reader(csvfile, delimiter=',', quotechar='|')
        next(reader)
        next(reader)

        for row in reader:
            if len(row)==0:
                break
            
            attList.append(row[0:nroAttributes])
            classList.append(row[-1])

    #convert attributes to floats
    for i in range(len(attList[0])):
        str_column_to_float(attList, i)

    
    # convert list to two numpy arrays, attributes and classes
    attributes = np.asarray(attList)
    #classes = np.asarray(classList)
    #classes = np.array(classes)
            
    return attributes, classList

def normalize(x):
    """
        argument
            - x: input image data in numpy array [32, 32, 3]
        return
            - normalized x 
    """
    x = x / np.max(np.abs(x))
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

def model_orig():
    
    
    inp = Input(shape=(INPUT,))
    
    #first layer
    l1 = Dense(NUM_NODES, activation='relu')(inp)
    
    #second layer
    mat2 = np.ones((NUM_NODES,NUM_NODES))
    l2 = CustomConnected(NUM_NODES, connections=mat2, activation='relu')(l1)
    z = concatenate([l1, l2])
    
    #third layer
    mat3 = np.zeros((NUM_NODES*2, NUM_NODES))
    mat3[NUM_NODES:,:] = 1
    l3 = CustomConnected(NUM_NODES,connections=mat3, activation='relu')(z)
    z = concatenate([l1, l2, l3])
    
    #forth layer
    mat4 = np.zeros((NUM_NODES*3, NUM_NODES))
    mat4[NUM_NODES*2:,:] = 1
    l4 = CustomConnected(NUM_NODES,connections=mat4, activation='relu')(z)
    z = concatenate([l1, l2, l3, l4])
    
    #fifth layer
    mat5 = np.zeros((NUM_NODES*4, NUM_NODES))
    mat5[NUM_NODES*3:,:] = 1
    l5 = CustomConnected(NUM_NODES,connections=mat5, activation='relu')(z)
    z = concatenate([l1, l2, l3, l4, l5])
    
    #sixth layer
    mat6 = np.zeros((NUM_NODES*5, NUM_NODES))
    mat6[NUM_NODES*4:,:] = 1
    l6 = CustomConnected(NUM_NODES,connections=mat6, activation='relu')(z)
    z = concatenate([z, l6])
    
    
    #output layer
    mat7 = np.zeros((NUM_NODES*6, OUTPUT))
    mat7[NUM_NODES*5:, :] = 1
    out = CustomConnected(OUTPUT, connections=mat7, activation='softmax')(z)
    
    model = Model(inp, out)
    model.compile(optimizer=optimizer, 
                  loss='sparse_categorical_crossentropy', metrics=['accuracy'])


    layers = [mat7, mat6, mat5, mat4, mat3, mat2]
    
    return model, layers


def model_dropout():
    dropout = 0.1
    
    model = Sequential()
    model.add(Dropout(0.1, input_shape=(INPUT,)))
    model.add(Dense(NUM_NODES, input_shape=(INPUT,), activation='relu'))
    model.add(Dropout(dropout))
    model.add(Dense(NUM_NODES, activation='relu', kernel_constraint=maxnorm(4)))
    model.add(Dropout(dropout))
    model.add(Dense(NUM_NODES, activation='relu', kernel_constraint=maxnorm(4)))
    model.add(Dropout(dropout))
    model.add(Dense(NUM_NODES, activation='relu', kernel_constraint=maxnorm(4)))
    model.add(Dropout(dropout))
    model.add(Dense(NUM_NODES, activation='relu', kernel_constraint=maxnorm(4)))
    model.add(Dropout(dropout))
    model.add(Dense(NUM_NODES, activation='relu', kernel_constraint=maxnorm(4)))
    model.add(Dense(OUTPUT, activation='softmax'))

    opt = Adam(lr=0.001)

    
    model.compile(optimizer=opt, 
            loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    
    return model

def model_dropout_0():
    dropout = 0.0
    
    model = Sequential()
    model.add(Dropout(0.0, input_shape=(INPUT,)))
    model.add(Dense(NUM_NODES, input_shape=(INPUT,), activation='relu'))
    model.add(Dropout(dropout))
    model.add(Dense(NUM_NODES, activation='relu'))
    model.add(Dropout(dropout))
    model.add(Dense(NUM_NODES, activation='relu'))
    model.add(Dropout(dropout))
    model.add(Dense(NUM_NODES, activation='relu'))
    model.add(Dropout(dropout))
    model.add(Dense(NUM_NODES, activation='relu'))
    model.add(Dropout(dropout))
    model.add(Dense(NUM_NODES, activation='relu'))
    model.add(Dense(OUTPUT, activation='softmax'))

    
    model.compile(optimizer='adam', 
            loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    
    return model
    

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
        if(row[column] == "?"):
            row[column] = "0"
        row[column] = row[column].replace(",", ".")
        row[column]=float(row[column].strip())
                

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
                    if (check_if_neighbour(row, new_col, layers)) and mat[new_ind]==0 and new_ind not in rewired:
                        rewired.append(new_ind)
                        rewired.append((row,column))
                        mat[row,column] = 0
                        mat[column,row] = 0
                        
                        mat[row,new_col] = 1
                        mat[new_col,row] = 1
                        break
            column = column+1
    return mat

def rewire_num_connections(adjmat, layers, num_connections):
    
    mat = np.array(adjmat)
    rewired = []
    
    rew_con = 0
    
    while rew_con < num_connections:
        
        row = random.randint(0, mat.shape[1]-1)
        col = random.randint(0, mat.shape[1]-1)
        
        new_col = random.randint(0, mat.shape[1]-1)
        
        if mat[row,col] == 1 and (row,col) not in rewired and (row,new_col) not in rewired:
            if check_if_neighbour(row, new_col, layers) and mat[row, new_col] == 0:
                
                rewired.append((row,col))
                rewired.append((row, new_col))
                rewired.append((col, row))
                rewired.append((new_col, row))
                
                mat[row,col] = 0
                mat[col, row] = 0
                
                mat[row, new_col] = 1
                mat[new_col, row] = 1
                
                rew_con = rew_con + 1
                
    return mat


    
def ann_to_graph(layers):
    """
    Generates graph from ANN connection matrices
    ::param layers:: list of numpy matrices that maps connections between ANN neurons
    ::output graph:: networkx graph object
    ::output mat:: graph adjacency matrix
    """
       
    #Generate adjacency matrix
    mat = np.zeros((layers[0].shape[0]+OUTPUT,layers[0].shape[0]+OUTPUT))
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
        

    #vizualise NN
#    graph = nx.from_numpy_matrix(mat)
#    vis = nx.nx_pydot.to_pydot(graph)
#    vis.write_png('example2_graph.png')
#    
    return mat


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


def model_dense():
    model = Sequential()
    model.add(Dense(NUM_NODES, input_shape = (INPUT,), activation='relu'))
    model.add(Dense(NUM_NODES, activation='relu'))
    model.add(Dense(NUM_NODES, activation='relu'))
    model.add(Dense(NUM_NODES, activation='relu'))
    model.add(Dense(NUM_NODES, activation='relu'))
    model.add(Dense(NUM_NODES, activation='relu'))


    
    model.add(Dense(OUTPUT, activation='softmax'))
    
    model.compile(optimizer=optimizer, 
            loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    
    return model

def model_weight_reg():
    l1 = 1e-07
    l2 = 1e-06
    
    model = Sequential()
    model.add(Dense(NUM_NODES, input_shape = (INPUT,), activation='relu', kernel_regularizer=regularizers.l2(l2), 
                    activity_regularizer=regularizers.l1(l1)))
    
    model.add(Dense(NUM_NODES, activation='relu', kernel_regularizer=regularizers.l2(l2), 
                    activity_regularizer=regularizers.l1(l1)))
    
    model.add(Dense(NUM_NODES, activation='relu', kernel_regularizer=regularizers.l2(l2), 
                    activity_regularizer=regularizers.l1(l1)))

    model.add(Dense(NUM_NODES, activation='relu', kernel_regularizer=regularizers.l2(l2), 
                    activity_regularizer=regularizers.l1(l1)))
    
    model.add(Dense(NUM_NODES, activation='relu', kernel_regularizer=regularizers.l2(l2), 
                    activity_regularizer=regularizers.l1(l1)))
    
    model.add(Dense(NUM_NODES, activation='relu', kernel_regularizer=regularizers.l2(l2), 
                    activity_regularizer=regularizers.l1(l1)))
 
    
    model.add(Dense(OUTPUT, activation='softmax'))
    
    opt = Adam(lr=0.01)
    
    model.compile(optimizer=opt, 
            loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    
    return model

def model_rewired(layers):
    
    
    inp = Input(shape=(INPUT,))
    
    
    #first layer
    l1 = Dense(NUM_NODES, activation='relu')(inp)
    
    #second
    l2 = CustomConnected(NUM_NODES, connections=layers[5], activation='relu')(l1)
    z = concatenate([l1, l2])
    
    #third
    l3 = CustomConnected(NUM_NODES, connections=layers[4], activation='relu')(z)
    z = concatenate([z,l3])
    
    #forth 
    l4 = CustomConnected(NUM_NODES,connections=layers[3], activation='relu')(z)
    z = concatenate([z, l4])
    
    #fifth
    l5 = CustomConnected(NUM_NODES, connections=layers[2], activation='relu')(z)
    z = concatenate([z, l5])
    
    #sixth
    l6 = CustomConnected(NUM_NODES, connections=layers[1], activation='relu')(z)
    z = concatenate([z, l6])
    
    
    #output
    out = CustomConnected(OUTPUT, connections=layers[0], activation='softmax')(z)
    
    model = Model(inp, out)
    model.compile(optimizer=optimizer, 
                  loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    
    return model


def model_rewired_weight_reg(layers):

    la1 = 1e-05
    la2 = 1e-05

    inp = Input(shape=(INPUT,))
    
    
    #first layer
    l1 = Dense(NUM_NODES, activation='relu', kernel_regularizer=regularizers.l2(la2), 
                    activity_regularizer=regularizers.l1(la1))(inp)
    
    #second
    l2 = CustomConnected(NUM_NODES, connections=layers[5], activation='relu', kernel_regularizer=regularizers.l2(la2), 
                    activity_regularizer=regularizers.l1(la1))(l1)
    z = concatenate([l1, l2])
    
    #third
    l3 = CustomConnected(NUM_NODES, connections=layers[4], activation='relu', kernel_regularizer=regularizers.l2(la2), 
                    activity_regularizer=regularizers.l1(la1))(z)
    z = concatenate([z,l3])
    
    #forth 
    l4 = CustomConnected(NUM_NODES,connections=layers[3], activation='relu', kernel_regularizer=regularizers.l2(la2), 
                    activity_regularizer=regularizers.l1(la1))(z)
    z = concatenate([z, l4])
    
    #fifth
    l5 = CustomConnected(NUM_NODES, connections=layers[2], activation='relu', kernel_regularizer=regularizers.l2(la2), 
                    activity_regularizer=regularizers.l1(la1))(z)
    z = concatenate([z, l5])
    
    #sixth
    l6 = CustomConnected(NUM_NODES, connections=layers[1], activation='relu', kernel_regularizer=regularizers.l2(la2), 
                    activity_regularizer=regularizers.l1(la1))(z)
    z = concatenate([z, l6])
    
    
    #output
    out = CustomConnected(OUTPUT, connections=layers[0], activation='softmax')(z)
    
    model = Model(inp, out)
    opt = Adam(lr=0.01)
    model.compile(optimizer=opt, 
                  loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    
    return model

def model_rewired_dropout(layers):
    
    dropout = 0.1
 
    #input layer
    inp = Input(shape=(INPUT,))
    dinp = Dropout(0.1)(inp)
    
    #first layer
    l1 = Dense(NUM_NODES, activation='relu', kernel_constraint=maxnorm(4))(dinp)
    d1 = Dropout(dropout)(l1)
    
    #second layer
    l2 = CustomConnected(NUM_NODES, connections=layers[5], activation='relu', kernel_constraint=maxnorm(4))(d1)
    z = concatenate([l1,l2])
    d2 = Dropout(dropout)(z)
    
    #third layer
    l3 = CustomConnected(NUM_NODES, connections=layers[4], activation='relu', kernel_constraint=maxnorm(4))(d2)
    z = concatenate([z, l3])
    d3 = Dropout(dropout)(z)
    
    #fourth layer
    l4 = CustomConnected(NUM_NODES, connections=layers[3], activation='relu', kernel_constraint=maxnorm(4))(d3)
    z = concatenate([z, l4])
    d4 = Dropout(dropout)(z)
    
    #fifth layer
    l5 = CustomConnected(NUM_NODES, connections=layers[2], activation='relu', kernel_constraint=maxnorm(4))(d4)
    z = concatenate([z, l5])
    d5 = Dropout(dropout)(z)
    
    #sixth layer
    l6 = CustomConnected(NUM_NODES, connections=layers[1], activation='relu', kernel_constraint=maxnorm(4))(d5)
    z = concatenate([z, l6])
    d6 = Dropout(dropout)(z)
    
    #output layer
    out = CustomConnected(OUTPUT, connections=layers[0], activation='softmax')(d6)
    
    model = Model(inp,out)
    
    opt = Adam(lr=0.001)
    
    model.compile(optimizer=opt, 
            loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    
    return model
    

def test_model(model, X_train, y_train, X_test, y_test):
    """
    Trains keras model
    ::param model:: keras model
    ::output history:: keras history object
    """
    
    #train model
    history = model.fit(X_train, y_train, epochs=EPOCHS, verbose=1,
                        batch_size=128, validation_data=(X_test, y_test))
    
    return history

def model_orig1(obs_space, num_outputs):
        
    #input layer
    input_layer = Input(shape=(obs_space,), name="observation")
        
    #first layer
    mat1 = np.ones((obs_space, NUM_NODES))
    layer_1 = CustomConnected(NUM_NODES, connections=mat1, activation='tanh')(input_layer)
    z = concatenate([input_layer, layer_1])
      
    #second layer
    mat2 = np.zeros((obs_space+NUM_NODES, NUM_NODES))
    mat2[obs_space:,:] = 1
    layer_2 = CustomConnected(NUM_NODES, connections=mat2, activation='tanh')(z)
    z = concatenate([z, layer_2])
        
    #output layer
    mat3 = np.zeros((obs_space+NUM_NODES*2, num_outputs))
    mat3[obs_space+NUM_NODES:,:] = 1
    layer_out = CustomConnected(num_outputs, connections=mat3, activation=None)(z)
       
    model = Model(input_layer, layer_out)
    layers = [mat3, mat2, mat1]
      
    return model, layers 

if __name__ == "__main__":
    
    model, layers = model_orig1(4, 2)
    mat = ann_to_graph(layers)
    swmat = rewire_num_connections(mat, layers, 400)
    new_layers = graph_to_ann(swmat, layers)
    #model = model_rewired(new_layers)
    
    

#    x,y =load_csv(PATHDRIVE, NUMBER_OF_ATTRIBUTES)
#    #x = normalize(x)
#    
#    
#    #convert labels to integers
#    y = classes_to_int(y)
#
#    #split to trainign and testing data
#    X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=TEST_SIZE)
#    X_test = X_test[0:1000]
#    y_test = y_test[0:1000]
#    
#
#    model, layers = model_orig()
#    mat = ann_to_graph(layers)
#    
#        
#    index = []
#    data = []
#    modelss = []
#    accu = []
#    
#    # Dense
#    for i in range(NUM_MODELS):
#        
#        
#        
#        model = model_dense()
#        
#        history = test_model(model, X_train, y_train, X_test, y_test)
#        
#        index = index + list(range(0, EPOCHS))
#        data = data + history.history['val_acc']
#        modelss = modelss + ["Dense"] * EPOCHS
#        accu = accu + history.history['acc']
#
#
#    dataframe = pd.DataFrame((index,data), dtype=float).T
#    dataframe.columns=["epochs", "Test accuracy"]
#    dataframe['Model'] = modelss
#    
#    dataframe_acc = pd.DataFrame((index, accu)).T
#    dataframe_acc.columns=["epochs", "Train accuracy"]
#    dataframe_acc['Model'] = modelss
#    
#
#    modelss = []
#    index = []
#    data = []
#    accu = []
#    
#    #dropout
#    for i in range(NUM_MODELS):
#        
#               
#        model = model_dropout()
#        
#        history = test_model(model, X_train, y_train, X_test, y_test)
#        
#        index = index + list(range(0, EPOCHS))
#        data = data + history.history['val_acc']
#        modelss = modelss + ["Dropout"] * EPOCHS
#        accu = accu + history.history['acc']
#        
#
#    p = pd.DataFrame((index,data)).T
#    p.columns=["epochs", "Test accuracy"]
#    p['Model'] = modelss
#    
#    d = pd.DataFrame((index,accu)).T
#    d.columns=["epochs", "Train accuracy"]
#    d['Model'] = modelss
#    
#    dataframe_acc = dataframe_acc.append(d)
#    
#    dataframe = dataframe.append(p)
#    
#    
#    modelss = []
#    index = []
#    data = []
#    accu = []
#        
#
#    #dropout 0
#    for i in range(NUM_MODELS):
#        
#               
#        model = model_dropout_0()
#        
#        history = test_model(model, X_train, y_train, X_test, y_test)
#        
#        index = index + list(range(0, EPOCHS))
#        data = data + history.history['val_acc']
#        modelss = modelss + ["Dropout 0"] * EPOCHS
#        accu = accu + history.history['acc']
#        
#
#    p = pd.DataFrame((index,data)).T
#    p.columns=["epochs", "Test accuracy"]
#    p['Model'] = modelss
#    
#    d = pd.DataFrame((index,accu)).T
#    d.columns=["epochs", "Train accuracy"]
#    d['Model'] = modelss
#    
#    dataframe_acc = dataframe_acc.append(d)
#    
#    dataframe = dataframe.append(p)
#    
#        
#
#    plot_testacc= sns.lineplot(x="epochs", y="Test accuracy", hue='Model', data=dataframe)
#    plt.show()
#    plot_trainacc = sns.lineplot(x="epochs", y="Train accuracy", hue='Model', data=dataframe_acc)
#    plt.show()

