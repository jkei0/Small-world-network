# -*- coding: utf-8 -*-
"""
Created on Wed May  6 17:57:01 2020

@author: jonik
"""

from sklearn.model_selection import GridSearchCV
from keras.wrappers.scikit_learn import KerasClassifier
from keras.models import Sequential, Model
from keras.layers import Dense, Dropout, Input, concatenate
from keras.optimizers import Adam
import numpy as np
from sklearn import preprocessing
import csv
from keras import regularizers
from sklearn.model_selection import train_test_split
from keras.constraints import maxnorm
import random
import keras.backend as K

PATH = 'DriveDiagnosis/Sensorless_drive_diagnosis.txt'


NUMBER_OF_ATTRIBUTES = 49
INPUT = 48
OUTPUT = 11
NUM_NODES = 64
optimizer = 'adam'


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
        #next(reader)
        for row in reader:
            if len(row)==0:
                break
            
#            result = [x for x in row if x != ''] 
            attList.append(row[0:nroAttributes-1])
            classList.append(row[-1])

    #convert attributes to floats
    for i in range(len(attList[0])):
        str_column_to_float(attList, i)

    
    # convert list to two numpy arrays, attributes and classes
    attributes = np.asarray(attList)
    #classes = np.asarray(classList)
    #classes = np.array(classes)
            
    return attributes, classList

def create_model(layers, learning_rate=0.001, dropout=0.1, dropout_in=0.1):
    
 
    #input layer
    inp = Input(shape=(INPUT,))
    dinp = Dropout(dropout_in)(inp)
    
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
        

    #vizualise NN
    #vis = nx.nx_pydot.to_pydot(graph)
    #vis.write_png('example2_graph.png')
    
    return mat

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

    

if __name__ == '__main__':
    
    # get data
    x,y =load_csv(PATH, NUMBER_OF_ATTRIBUTES)
    
    #convert labels to integers
    y = classes_to_int(y)

    #split to trainign and testing data
    X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.97)
    X_test = X_test[0:1000]
    y_test = y_test[0:1000]    

    
    # get model in scikit-learn wrapper
    model1, layers = model_orig()
    mat = ann_to_graph(layers)
    m = rewire_to_smallworld(mat, layers, 0.8)
    new_layers = graph_to_ann(m, layers)
    
    model2 = create_model(new_layers)
    
    model = KerasClassifier(build_fn=model2,verbose=0, epochs=400, batch_size=128)
    
    dropout = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
    dropout_in = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]


    
    param_grid = dict(dropout=dropout, dropout_in=dropout_in)
    
    grid = GridSearchCV(estimator=model, param_grid=param_grid, n_jobs=1, cv=3)
    grid_result = grid.fit(X_train, y_train)
    
    means = grid_result.cv_results_['mean_test_score']
    params = grid_result.cv_results_['params']
    
    file = open('results.txt', 'w')
    for mean, param in zip(means, params):
        print("%f with: %r" % (mean, param))
        file.write("%f with: %r \n" % (mean, param))
        
    file.write("Best: %f using %s" % (grid_result.best_score_, grid_result.best_params_))
    file.close()
        

