#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jul 12 23:55:21 2020

@author: joni
"""

# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

import tensorflow as tf
import ray
from ray import tune
from ray.rllib.agents.ppo import PPOTrainer
from ray.rllib.agents.dqn import DQNTrainer
from ray.rllib.models import ModelCatalog
from ray.rllib.models.tf.tf_modelv2 import TFModelV2
from tensorflow.keras.layers import Dense, Input, concatenate
from tensorflow.keras.models import Sequential, Model
from ray.rllib.models.tf.misc import normc_initializer
import tensorflow.keras.backend as K
import numpy as np
import random


NUM_NODES = 256


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

def model_orig(obs_space, num_outputs):

    #first layer
    mat1 = np.ones((obs_space.shape[0], NUM_NODES))
     
    #second layer
    mat2 = np.zeros((obs_space.shape[0]+NUM_NODES, NUM_NODES))
    mat2[obs_space.shape[0]:,:] = 1
    
    #third layer
    mat3 = np.zeros((obs_space.shape[0]+NUM_NODES*2, NUM_NODES))
    mat3[obs_space.shape[0]+NUM_NODES:,:] = 1
        
    #output layer
    mat4 = np.zeros((obs_space.shape[0]+NUM_NODES*3, num_outputs))
    mat4[obs_space.shape[0]+NUM_NODES*2:,:] = 1
        
    #model = Model(input_layer, layer_out)
    layers = [mat4, mat3, mat2, mat1]
        
    return layers
    
    
def model_orig_value(obs_space):
        
        
    #first layer
    mat1 = np.ones((obs_space.shape[0], NUM_NODES))

        
    #second layer
    mat2 = np.zeros((obs_space.shape[0]+NUM_NODES, NUM_NODES))
    mat2[obs_space.shape[0]:,:] = 1

    #third layer
    mat3 = np.zeros((obs_space.shape[0]+NUM_NODES*2, NUM_NODES))
    mat3[obs_space.shape[0]+NUM_NODES:,:] = 1
        
    #output layer
    mat4 = np.zeros((obs_space.shape[0]+NUM_NODES*3, 1))
    mat4[obs_space.shape[0]+NUM_NODES*2:,:] = 1
    #layer_out = CustomConnected(1, connections=mat3, activation=None)(z)
        
   # model = Model(input_layer, layer_out)
    layers = [mat4, mat3, mat2, mat1]
        
    return layers

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
    

class DenseModel(TFModelV2):
    def __init__(self,  obs_space, action_space, num_outputs, 
                 model_config, name):
        super(DenseModel, self).__init__(obs_space, action_space, 
             num_outputs, model_config, name)
        
        input_layer = Input(shape=obs_space.shape, name="observations")
        
        layer_1 = Dense(NUM_NODES, activation='tanh', 
                        kernel_initializer=normc_initializer(1.0))(input_layer)
        value_1 = Dense(NUM_NODES, activation='tanh',
                        kernel_initializer=normc_initializer(1.0))(input_layer)
        
        layer_2 = Dense(NUM_NODES, activation='tanh',
                        kernel_initializer=normc_initializer(1.0))(layer_1)
        value_2 = Dense(NUM_NODES, activation='tanh',
                        kernel_initializer=normc_initializer(1.0))(value_1)
        
        layer_3 = Dense(NUM_NODES, activation='tanh',
                        kernel_initializer=normc_initializer(1.0))(layer_2)
        value_3 = Dense(NUM_NODES, activation='tanh',
                        kernel_initializer=normc_initializer(1.0))(value_2)
        
        layer_out = Dense(NUM_NODES, activation=None,
                          kernel_initializer=normc_initializer(0.01))(layer_3)
        value_out = Dense(1, activation=None,
                          kernel_initializer=normc_initializer(0.01))(value_3)
        self.base_model = Model(input_layer, [layer_out, value_out])
        
        self.register_variables(self.base_model.variables)
            
    def forward(self, input_dict, state, seq_lens):
        model_out, self._value_out = self.base_model(input_dict["obs"])
        return model_out, state
        
    def value_function(self):
        return tf.reshape(self._value_out, [-1])
    
    def metrics(self):
        return {"foo": tf.constant(42.0)}


class SmallWorldModel(TFModelV2):
    
    def __init__(self, obs_space, action_space, num_outputs,
                 model_config, name):
        super(SmallWorldModel, self).__init__(obs_space, action_space,
             num_outputs, model_config, name)
        
        layers = model_orig(obs_space, num_outputs)
        value_layers = model_orig_value(obs_space)
        
        mat = ann_to_graph(layers, num_outputs)
        value_mat = ann_to_graph(value_layers, 1)
        
        swmat = rewire_to_smallworld(mat, layers, 0.0)
        swmat_value = rewire_to_smallworld(value_mat, value_layers, 0.6)
        
        layers = graph_to_ann(swmat, layers)
        value_layers = graph_to_ann(swmat_value, value_layers)
        
        #input layer
        self.inputs = Input(shape=obs_space.shape)
        
        #first layer
        layer_1 = CustomConnected(NUM_NODES, connections=layers[3], activation='tanh',
                                  kernel_initializer=normc_initializer(1.0))(self.inputs)
        value_1 = CustomConnected(NUM_NODES, connections=value_layers[3], activation='tanh',
                                  kernel_initializer=normc_initializer(1.0))(self.inputs)
        z_layer = concatenate([self.inputs, layer_1])
        z_value = concatenate([self.inputs, value_1])
        
        #second layer
        layer_2 = CustomConnected(NUM_NODES, connections=layers[2], activation='tanh',
                                  kernel_initializer=normc_initializer(1.0))(z_layer)
        value_2 = CustomConnected(NUM_NODES, connections=value_layers[2], activation='tanh',
                                  kernel_initializer=normc_initializer(1.0))(z_value)     
        z1_layer = concatenate([z_layer, layer_2])
        z1_value = concatenate([z_value, value_2])
        
        #third layer
        layer_3 = CustomConnected(NUM_NODES, connections=layers[1], activation='tanh',
                                  kernel_initializer=normc_initializer(1.0))(z1_layer)
        value_3 = CustomConnected(NUM_NODES, connections=value_layers[1], activation='tanh',
                                  kernel_initializer=normc_initializer(1.0))(z1_value)     
        z2_layer = concatenate([z1_layer, layer_3])
        z2_value = concatenate([z1_value, value_3])
        
        #output layer
        layer_out = CustomConnected(NUM_NODES, connections=layers[0], activation=None,
                                    kernel_initializer=normc_initializer(0.01))(z2_layer)
        value_out = CustomConnected(1, connections=value_layers[0], activation=None,
                                    kernel_initializer=normc_initializer(0.01))(z2_value)
        
        self.base_model = Model(self.inputs, [layer_out, value_out])
        
        self.register_variables(self.base_model.variables)
    
 
        
    def forward(self, input_dict, state, seq_lens):
        model_out, self._value_out = self.base_model(input_dict["obs"])
        return model_out, state
        
    
    def value_function(self):
        return tf.reshape(self._value_out, [-1])
    
    
    def metrics(self):
        return {"foo": tf.constant(42.0)}


if __name__== "__main__":

    #ray.init()
    
    ModelCatalog.register_custom_model(
            "keras_model", SmallWorldModel)
    
    tune.run(
            run_or_experiment=DQNTrainer,
            verbose=1,
            stop={"training_iteration": 25},
            name="fully connected Small-world",
            config={                    
                    "env": "CartPole-v0",
                    "framework": "tf",
                    "model": {
                            "custom_model" : "keras_model"}
                    })
        
    #ray.shutdown()
