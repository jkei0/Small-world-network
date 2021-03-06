"""
Created on Mon Jul 20 15:35:27 2020

@author: joni
"""

import tensorflow as tf
import ray
from ray import tune
from ray.rllib.agents.ppo import PPOTrainer
from ray.rllib.models import ModelCatalog
from ray.rllib.models.tf.tf_modelv2 import TFModelV2
from tensorflow.keras.layers import Dense, Input, concatenate
from tensorflow.keras.models import Sequential, Model
from ray.rllib.models.tf.misc import normc_initializer
import tensorflow.keras.backend as K
import numpy as np
import random


NUM_NODES = 128
SEED = 12345
NET_SEED = 22345
VALUE_LAYERS = []
LAYERS = []
NUM_OUTPUTS = 8
NUM_INPUTS = 24

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

def model_orig_incInputs():


    #first layer
    mat1 = np.ones((NUM_INPUTS, NUM_NODES))
    
    #second layer
    mat2 = np.zeros((NUM_NODES+NUM_INPUTS, NUM_NODES))
    mat2[NUM_INPUTS:, :] = 1

    #third layer
    mat3 = np.zeros((NUM_NODES*2+NUM_INPUTS, NUM_NODES))
    mat3[NUM_NODES+NUM_INPUTS:,:] = 1

    #forth layer
    mat4 = np.zeros((NUM_NODES*3+NUM_INPUTS, NUM_NODES))
    mat4[NUM_NODES*2+NUM_INPUTS:,:] = 1

    #fifth layer
    mat5 = np.zeros((NUM_NODES*4+NUM_INPUTS, NUM_NODES))
    mat5[NUM_NODES*3+NUM_INPUTS:,:] = 1

    #output layer
    mat6 = np.zeros((NUM_NODES*5+NUM_INPUTS, NUM_OUTPUTS))
    mat6[NUM_NODES*4+NUM_INPUTS:,:] = 1

    #model = Model(input_layer, layer_out)
    layers = [mat6, mat5, mat4, mat3, mat2, mat1]

    return layers


def model_orig_value_incInputs():


    #first layer
    mat1 = np.ones((NUM_INPUTS, NUM_NODES))
    
    #second layer
    mat2 = np.zeros((NUM_NODES+NUM_INPUTS, NUM_NODES))
    mat2[NUM_INPUTS:, :] = 1

    #third layer
    mat3 = np.zeros((NUM_NODES*2+NUM_INPUTS, NUM_NODES))
    mat3[NUM_NODES+NUM_INPUTS:,:] = 1

    #forth layer
    mat4 = np.zeros((NUM_NODES*3+NUM_INPUTS, NUM_NODES))
    mat4[NUM_NODES*2+NUM_INPUTS:,:] = 1

    #fifth layer
    mat5 = np.zeros((NUM_NODES*4+NUM_INPUTS, NUM_NODES))
    mat5[NUM_NODES*3+NUM_INPUTS:,:] = 1

    #output layer
    mat6 = np.zeros((NUM_NODES*5+NUM_INPUTS, 1))
    mat6[NUM_NODES*4+NUM_INPUTS:,:] = 1

    #model = Model(input_layer, layer_out)
    layers = [mat6, mat5, mat4, mat3, mat2, mat1]

    return layers


def model_orig():


    #second layer
    mat2 = np.ones((NUM_NODES, NUM_NODES))

    #third layer
    mat3 = np.zeros((NUM_NODES*2, NUM_NODES))
    mat3[NUM_NODES:,:] = 1

    #forth layer
    mat4 = np.zeros((NUM_NODES*3, NUM_NODES))
    mat4[NUM_NODES*2:,:] = 1

    #fifth layer
    mat5 = np.zeros((NUM_NODES*4, NUM_NODES))
    mat5[NUM_NODES*3:,:] = 1

    #output layer
    mat6 = np.zeros((NUM_NODES*5, 1))
    mat6[NUM_NODES*4:,:] = 1

    #model = Model(input_layer, layer_out)
    layers = [mat6, mat5, mat4, mat3, mat2]

    return layers


def model_orig_value():


    #second layer
    mat2 = np.ones((NUM_NODES, NUM_NODES))

    #third layer
    mat3 = np.zeros((NUM_NODES*2, NUM_NODES))
    mat3[NUM_NODES:,:] = 1

    #forth layer
    mat4 = np.zeros((NUM_NODES*3, NUM_NODES))
    mat4[NUM_NODES*2:,:] = 1

    #fifth layer
    mat5 = np.zeros((NUM_NODES*4, NUM_NODES))
    mat5[NUM_NODES*3:,:] = 1

    #output layer
    mat6 = np.zeros((NUM_NODES*5, 1))
    mat6[NUM_NODES*4:,:] = 1

    #model = Model(input_layer, layer_out)
    layers = [mat6, mat5, mat4, mat3, mat2]

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

        layer_4 = Dense(NUM_NODES, activation='tanh',
                        kernel_initializer=normc_initializer(1.0))(layer_3)
        value_4 = Dense(NUM_NODES, activation='tanh',
                        kernel_initializer=normc_initializer(1.0))(value_3)

        layer_5 = Dense(NUM_NODES, activation='tanh',
                        kernel_initializer=normc_initializer(1.0))(layer_4)
        value_5 = Dense(NUM_NODES, activation='tanh',
                        kernel_initializer=normc_initializer(1.0))(value_4)

        layer_out = Dense(num_outputs, activation=None,
                          kernel_initializer=normc_initializer(0.01))(layer_5)
        value_out = Dense(1, activation=None,
                          kernel_initializer=normc_initializer(0.01))(value_5)
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
               
        #input layer
        input_layer = Input(shape=obs_space.shape)

        #first layer
        layer_1 = CustomConnected(NUM_NODES, connections=LAYERS[5], activation='tanh',
                                  kernel_initializer=normc_initializer(1.0))(input_layer)
        value_1 = CustomConnected(NUM_NODES, connections=VALUE_LAYERS[5], activation='tanh',
                                  kernel_initializer=normc_initializer(1.0))(input_layer)
        z0_layer = concatenate([input_layer, layer_1])
        z0_value = concatenate([input_layer, value_1])

        #second layer
        layer_2 = CustomConnected(NUM_NODES, connections=LAYERS[4], activation='tanh',
                                  kernel_initializer=normc_initializer(1.0))(z0_layer)
        value_2 = CustomConnected(NUM_NODES, connections=VALUE_LAYERS[4], activation='tanh',
                                  kernel_initializer=normc_initializer(1.0))(z0_value)
        z1_layer = concatenate([z0_layer, layer_2])
        z1_value = concatenate([z0_value, value_2])

        #third layer
        layer_3 = CustomConnected(NUM_NODES, connections=LAYERS[3], activation='tanh',
                                  kernel_initializer=normc_initializer(1.0))(z1_layer)
        value_3 = CustomConnected(NUM_NODES, connections=VALUE_LAYERS[3], activation='tanh',            
                                  kernel_initializer=normc_initializer(1.0))(z1_value)
        z2_layer = concatenate([z1_layer, layer_3])
        z2_value = concatenate([z1_value, value_3])

        #forth layer
        layer_4 = CustomConnected(NUM_NODES, connections=LAYERS[2], activation='tanh',
                                  kernel_initializer=normc_initializer(1.0))(z2_layer)
        value_4 = CustomConnected(NUM_NODES, connections=VALUE_LAYERS[2], activation='tanh',
                                  kernel_initializer=normc_initializer(1.0))(z2_value)
        z3_layer = concatenate([z2_layer, layer_4])
        z3_value = concatenate([z2_value, value_4])

        #fifth layer
        layer_5 = CustomConnected(NUM_NODES, connections=LAYERS[1], activation='tanh',
                                  kernel_initializer=normc_initializer(1.0))(z3_layer)
        value_5 = CustomConnected(NUM_NODES, connections=VALUE_LAYERS[1], activation='tanh',
                                  kernel_initializer=normc_initializer(1.0))(z3_value)
        z4_layer = concatenate([z3_layer, layer_5])
        z4_value = concatenate([z3_value, value_5])

        #output layer
        layer_out = CustomConnected(num_outputs, connections=LAYERS[0], activation=None,
                                    kernel_initializer=normc_initializer(0.01))(z4_layer)
        value_out = CustomConnected(1, connections=VALUE_LAYERS[0], activation=None,
                                    kernel_initializer=normc_initializer(0.01))(z4_value)

        self.base_model = Model(input_layer, [layer_out, value_out])

        self.register_variables(self.base_model.variables)


    def forward(self, input_dict, state, seq_lens):
        model_out, self._value_out = self.base_model(input_dict["obs"])
        return model_out, state


    def value_function(self):
        return tf.reshape(self._value_out, [-1])


    def metrics(self):
        return {"foo": tf.constant(42.0)}


if __name__== "__main__":


    random.seed(a=NET_SEED)
    value_layers = model_orig_value_incInputs()

    value_mat = ann_to_graph(value_layers, 1)
    swmat_value = rewire_num_connections(value_mat, value_layers, 10)
    value_layers = graph_to_ann(swmat_value, value_layers)
    
    
    layers = model_orig_incInputs()
    mat = ann_to_graph(layers, NUM_OUTPUTS)
    swmat = rewire_num_connections(mat, layers, 0)
    layers = graph_to_ann(swmat, layers)

    def change_glob_value(value_layers):
        global VALUE_LAYERS
        VALUE_LAYERS = value_layers
        
    def change_layer_value(layers):
        global LAYERS
        LAYERS = layers

    change_glob_value(value_layers)
    change_layer_value(layers)
    
    ray.init()

    ModelCatalog.register_custom_model(
            "keras_model", SmallWorldModel)

    tune.run(
            run_or_experiment=PPOTrainer,
            verbose=1,
            stop={"training_iteration": 500},
            name="BipedalWalker smallworld both",
            config={
                    "env": "BipedalWalker-v3",
                    "seed" : SEED,
                    "framework": "tf",
                    "model": {
                            "custom_model" : "keras_model",
                            },
                    })


