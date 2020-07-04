
"""
Created on Sat Mar  7 20:43:34 2020

@author: jonik
"""

import numpy as np
from keras.models import Model
from keras.layers import Dense, Input, concatenate
import keras.backend as K
import random
import matplotlib.pyplot as plt


PATH1 = 'cifar-10-batches-py/data_batch_1'
PATH2 = 'cifar-10-batches-py/data_batch_2'
PATH3 = 'cifar-10-batches-py/data_batch_3'
PATH4 = 'cifar-10-batches-py/data_batch_4'
PATH5 = 'cifar-10-batches-py/data_batch_5'
PATH6 = 'cifar-10-batches-py/test_batch'
INPUT = 3072
OUTPUT = 10
EPOCHS = 350
NUM_NODES = 128


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
    model.compile(optimizer='adam', 
                  loss='sparse_categorical_crossentropy', metrics=['accuracy'])


    layers = [mat7, mat6, mat5, mat4, mat3, mat2]
    
    return model, layers


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
    model.compile(optimizer='adam', 
                  loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    
    return model


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
    
    return mat


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


def test_model(model, X_train, y_train, X_test, y_test):
    """
    Trains keras model
    ::param model:: keras model
    ::output history:: keras history object
    """
 
    #train model
    history = model.fit(X_train, y_train, epochs=EPOCHS, verbose=1, 
                        batch_size=124, validation_data=(X_test, y_test))
    
    #test
    results = model.evaluate(X_test, y_test)

    return history, results



if __name__ == "__main__":
    
    X_train, y_train, X_test, y_test = get_cifar_data(PATH1,PATH2,PATH3,PATH4,PATH5,PATH6)
    X_train = normalize(X_train)
    X_test = normalize(X_test)
    
    results = []
    
    #get neural network
    model, layers = model_orig()
    mat = ann_to_graph(layers)
    hist_00 = test_model(model, X_train, y_train, X_test, y_test)
    
    # ANN with rewiring p = 0.1
    rewired_mat = rewire_to_smallworld(mat, layers, 0.1)
    new_layers = graph_to_ann(rewired_mat, layers)
    model = model_rewired(new_layers)
    hist_01, result = test_model(model, X_train, y_train, X_test, y_test)
    results.append(result)
    
    # ANN with rewiring p = 0.2
    rewired_mat = rewire_to_smallworld(mat, layers, 0.2)
    new_layers = graph_to_ann(rewired_mat, layers)
    model = model_rewired(new_layers)
    hist_02, result = test_model(model, X_train, y_train, X_test, y_test)
    results.append(result)
    
    # ANN with rewiring p = 0.3
    rewired_mat = rewire_to_smallworld(mat, layers, 0.3)
    new_layers = graph_to_ann(rewired_mat, layers)
    model = model_rewired(new_layers)
    hist_03, result = test_model(model, X_train, y_train, X_test, y_test)
    results.append(result)
    
    # ANN with rewiring p = 0.4
    rewired_mat = rewire_to_smallworld(mat, layers, 0.4)
    new_layers = graph_to_ann(rewired_mat, layers)
    model = model_rewired(new_layers)
    hist_04, result = test_model(model, X_train, y_train, X_test, y_test)
    results.append(result)
    
    # ANN with rewiring p = 0.5
    rewired_mat = rewire_to_smallworld(mat, layers, 0.5)
    new_layers = graph_to_ann(rewired_mat, layers)
    model = model_rewired(new_layers)
    hist_05, result = test_model(model, X_train, y_train, X_test, y_test)
    results.append(result)
    
    # ANN with rewiring p = 0.6
    rewired_mat = rewire_to_smallworld(mat, layers, 0.6)
    new_layers = graph_to_ann(rewired_mat, layers)
    model = model_rewired(new_layers)
    hist_06, result = test_model(model, X_train, y_train, X_test, y_test)
    results.append(result)
    
    # ANN with rewiring p = 0.7
    rewired_mat = rewire_to_smallworld(mat, layers, 0.7)
    new_layers = graph_to_ann(rewired_mat, layers)
    model = model_rewired(new_layers)
    hist_07, result = test_model(model, X_train, y_train, X_test, y_test)
    results.append(result)
    
    # ANN with rewiring p = 0.8
    rewired_mat = rewire_to_smallworld(mat, layers, 0.8)
    new_layers = graph_to_ann(rewired_mat, layers)
    model = model_rewired(new_layers)
    hist_08, result = test_model(model, X_train, y_train, X_test, y_test)
    results.append(result)
    
    # ANN with rewiring p = 0.9
    rewired_mat = rewire_to_smallworld(mat, layers, 0.9)
    new_layers = graph_to_ann(rewired_mat, layers)
    model = model_rewired(new_layers)
    hist_09, result = test_model(model, X_train, y_train, X_test, y_test)
    results.append(result)
    
    # ANN with rewiring p = 1
    rewired_mat = rewire_to_smallworld(mat, layers, 1.0)
    new_layers = graph_to_ann(rewired_mat, layers)
    model = model_rewired(new_layers)
    hist_10, result = test_model(model, X_train, y_train, X_test, y_test)
    results.append(result)
    
    # summarize history for accuracy
    plt.plot(hist_00.history['acc'], color='lightblue', label='p = 0')
    plt.plot(hist_01.history['acc'], color='lightgreen', label='p = 0.1')
    plt.plot(hist_02.history['acc'], color='red', label='p = 0.2')
    plt.plot(hist_03.history['acc'], color='cyan', label='p = 0.3')
    plt.plot(hist_04.history['acc'], color='magenta', label='p = 0.4')
    plt.plot(hist_05.history['acc'], color='yellow', label='p = 0.5')
    plt.plot(hist_06.history['acc'], color='black', label='p = 0.6')
    plt.plot(hist_07.history['acc'], color='0.50', label='p = 0.7')
    plt.plot(hist_08.history['acc'], color='darkgreen', label='p = 0.8')
    plt.plot(hist_09.history['acc'], color='darkblue', label='p = 0.9')
    plt.plot(hist_10.history['acc'], color='#6C0F44', label='p = 1.0')
    plt.title('Training accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(loc='lower right')
    plt.savefig('train accuracy')
    plt.show()
    
    # summarize history for validation accuracy
    plt.plot(hist_00.history['val_acc'], color='lightblue', label='p = 0')
    plt.plot(hist_01.history['val_acc'], color='lightgreen', label='p = 0.1')
    plt.plot(hist_02.history['val_acc'], color='red', label='p = 0.2')
    plt.plot(hist_03.history['val_acc'], color='cyan', label='p = 0.3')
    plt.plot(hist_04.history['val_acc'], color='magenta', label='p = 0.4')
    plt.plot(hist_05.history['val_acc'], color='yellow', label='p = 0.5')
    plt.plot(hist_06.history['val_acc'], color='black', label='p = 0.6')
    plt.plot(hist_07.history['val_acc'], color='0.50', label='p = 0.7')
    plt.plot(hist_08.history['val_acc'], color='darkgreen', label='p = 0.8')
    plt.plot(hist_09.history['val_acc'], color='darkblue', label='p = 0.9')
    plt.plot(hist_10.history['val_acc'], color='#6C0F44', label='p = 1.0')
    plt.title('Test accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(loc='lower right')
    plt.savefig('test accuracy')
    plt.show()
    
    # summarize history for loss
    plt.plot(hist_00.history['loss'], color='lightblue', label='p = 0')
    plt.plot(hist_01.history['loss'], color='lightgreen', label='p = 0.1')
    plt.plot(hist_02.history['loss'], color='red', label='p = 0.2')
    plt.plot(hist_03.history['loss'], color='cyan', label='p = 0.3')
    plt.plot(hist_04.history['loss'], color='magenta', label='p = 0.4')
    plt.plot(hist_05.history['loss'], color='yellow', label='p = 0.5')
    plt.plot(hist_06.history['loss'], color='black', label='p = 0.6')
    plt.plot(hist_07.history['loss'], color='0.50', label='p = 0.7')
    plt.plot(hist_08.history['loss'], color='darkgreen', label='p = 0.8')
    plt.plot(hist_09.history['loss'], color='darkblue', label='p = 0.9')
    plt.plot(hist_10.history['loss'], color='#6C0F44', label='p = 1.0')
    plt.title('Model loss')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(loc='lower right')
    plt.savefig('loss')
    plt.show()
    
    models = ["0", "0.1", "0.2","0.3","0.4","0.5","0.6","0.7","0.8","0.9","1.0"]
    
    file = open('small_world_comp_results.txt', 'w')
    for i in range(len(models)):
        file.write("Accuracy with %s = %s \n" % (models[i], results[i]))
    file.close()
        
    
