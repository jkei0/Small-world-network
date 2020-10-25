# -*- coding: utf-8 -*-
"""
Created on Wed Oct 21 17:03:07 2020

@author: jonik
"""

import numpy as np
import random
import tensorflow as tf
import tensorflow.keras.backend as K
from tensorflow.keras.layers import Conv2D, Input, Add, Flatten, Dense
from tensorflow.keras.models import Model
from tensorflow.keras import datasets

tf.random.set_seed(1000)

class CustomConv(Conv2D):
    
    def __init__(self, filters, kernel_size, connections, **kwargs):
        
        # connection matrix
        self.connections = connections
        
        super(CustomConv, self).__init__(filters, kernel_size,**kwargs)     
    
    def call(self, inputs):
        output = self._convolution_op(inputs, self.kernel*self.connections)
        if self.use_bias:
            output = K.bias_add(output, self.bias)
        if self.activation is not None:
            output = self.activation(output)
        return output
            
    def build(self, input_shape):
        super(CustomConv, self).build(input_shape)
        weights = self.get_weights()
        weights[0] = self.get_weights()[0] * self.connections
    
        self.set_weights(weights)

        


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


def orig_conv_net(conv_layers):
    layers_fin = []
    laSum = conv_layers[0]
    j = 0
    for i in conv_layers:
        
        if(laSum == conv_layers[0]):
            mat = np.ones((laSum, i))
            layers_fin.append(mat)
        else: 
            mat = np.zeros((laSum, i))
            mat[laSum-conv_layers[j-1]:,:] = 1
            layers_fin.append(mat)
        
        laSum = laSum + i
        j = j+1
        
    layers_fin.reverse()
    return layers_fin


def ann_to_graph(layers, OUTPUT):
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


def model(layers):
    inp = Input(shape=(28,28,1))
    
    layer1 = Conv2D(filters=5, kernel_size=(3,3), name='conv1', activation='relu', padding='same', input_shape=(32,32,3))(inp)
    
    layer2 = CustomConv(filters=5, kernel_size=(3,3), connections=layers[0], name='conv2', activation='relu', padding='same')(layer1)
    
    layer3 = CustomConv(filters=6, kernel_size=(3,3),name='conv3', connections=layers[1], activation='relu', padding='same')(layer2)
    skip2 = CustomConv(filters=6, kernel_size=(1,1), connections=layers[2])(layer1)
    layer3 = Add()([layer3, skip2]) 
    
    layer4 = CustomConv(filters=6, kernel_size=(3,3),name='conv4', connections=layers[3], activation='relu', padding='same')(layer3)
    skip2 = CustomConv(filters=6, kernel_size=(1,1), connections=layers[4])(layer1) 
    skip3 = CustomConv(filters=6, kernel_size=(1,1), connections=layers[5])(layer2) 
    layer4 = Add()([layer4, skip2, skip3])
    
    layer5 = CustomConv(filters=5, kernel_size=(3,3),name='conv5', activation='relu', connections=layers[6], padding='same')(layer4)
    skip2 = CustomConv(filters=5, kernel_size=(1,1), connections=layers[7])(layer1)
    skip3 = CustomConv(filters=5, kernel_size=(1,1), connections=layers[8])(layer2)
    skip4 = CustomConv(filters=5, kernel_size=(1,1), connections=layers[9])(layer3)
    layer5 = Add()([layer5, skip2, skip3, skip4])
    
    flat = Flatten()(layer5)
    d1 = Dense(64, activation='relu')(flat)
    d2 = Dense(10)(d1)
       
    model = Model(inp,d2)
    
    return model

def skipConnection(inp, out):
    
    x = Conv2D(out, (3,3), padding='same', activation='relu')(inp)
    x = Conv2D(out, (3,3), padding='same', activation='relu')(x)
    
    skip = Conv2D(out, (1,1), padding='same')(inp)
    m = Add()([x, skip])
    return m

    

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


def graph_to_CNN(layers): # layers = [5, 5, 5, 5, 5]
    
    new_layers = []
    
    
    for i in range(len(layers)):
        if (i == 0):
            new_layers.append(layers[i])
            
        else:
           a = layers[i][layers[i-1].shape[0]:,:]
           new_layers.append(a)
           
           for j in range(i):
               if (j == 0): 
                   a = layers[i][0:layers[j].shape[0]]
                   new_layers.append(a)
               else:
                   a = layers[i][layers[j-1].shape[0]:layers[j].shape[0],:]
                   new_layers.append(a)
           
        
    return new_layers


layers = orig_conv_net([5,6,6,5])
mat = ann_to_graph(layers, 5)
swmat = rewire_to_smallworld(mat, layers, 0)
new_layers = graph_to_ann(swmat, layers)
new_layers.reverse()

CNN_layers = graph_to_CNN(layers)
model = model(CNN_layers)
weights = model.get_weights()

(train_images, train_labels), (test_images, test_labels) = datasets.mnist.load_data()

# Normalize pixel values to be between 0 and 1
train_images, test_images = train_images / 255.0, test_images / 255.0

model.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])

history = model.fit(train_images, train_labels, epochs=10, 
                    validation_data=(test_images, test_labels))

weights_aft = model.get_weights()





                