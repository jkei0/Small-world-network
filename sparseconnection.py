# -*- coding: utf-8 -*-
"""
Created on Sat Feb  1 01:12:38 2020

@author: jonik
"""
import keras.backend as K
from keras.layers import Dense

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
        
