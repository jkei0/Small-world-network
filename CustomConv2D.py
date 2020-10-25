# -*- coding: utf-8 -*-
"""
Created on Sun Oct 25 15:53:56 2020

@author: jonik
"""

import tensorflow.keras.backend as K
from tensorflow.keras.layers import Conv2D

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
