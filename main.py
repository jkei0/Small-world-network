# -*- coding: utf-8 -*-
"""
Created on Thu Jan 23 19:23:27 2020

@author: jonik
"""

import numpy as np
from keras.models import Sequential, Model
from keras.layers import add, Dense, Dropout, LSTM, Input, Lambda, concatenate
from keras.optimizers import Adam, SGD
from keras.utils import to_categorical
from keras import regularizers
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from keras.utils.vis_utils import plot_model
import keras.initializers
import pandas as pd
import seaborn as sns

import utils
from itertools import permutations

NUMBER_OF_ATTRIBUTES = 49
NUMBER_OF_INSTANCES = 2123

EPOCHS = 400
NUM_MODELS = 8

PATHDRIVE = 'DriveDiagnosis/Sensorless_drive_diagnosis.txt'
PATH1 = 'cifar-10-batches-py/data_batch_1'
PATH2 = 'cifar-10-batches-py/data_batch_2'
PATH3 = 'cifar-10-batches-py/data_batch_3'
PATH4 = 'cifar-10-batches-py/data_batch_4'
PATH5 = 'cifar-10-batches-py/data_batch_5'
PATH6 = 'cifar-10-batches-py/test_batch'
PATHCARD = 'cardiodata/CTG.csv'
PATHMICE = 'micedata/Data_Cortex_Nuclear.csv'

NUM_NODES = 128

def model_orig(obs_space, num_outputs):

     
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
    mat6 = np.zeros((NUM_NODES*5, num_outputs))
    mat6[NUM_NODES*4:,:] = 1
        
    #model = Model(input_layer, layer_out)
    layers = [mat6, mat5, mat4, mat3, mat2]
        
    return layers


def print_training_phase(model_name, i):
    print("{} {}/{}".format(model_name,i,NUM_MODELS))


def plot_training_history(history):
    """
    Function plots training loss and accuracy for keras model
    ::param history:: keras history object 

    """
    
    # summarize history for accuracy
    plt.plot(history.history['acc'])
    plt.title('Model train accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['train'], loc='upper left')
    plt.show()
    # summarize history for loss
    plt.plot(history.history['loss'])
    plt.title('Model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train'], loc='upper left')
    plt.show()
    # validation accuracy
    plt.plot(history.history['val_acc'])
    plt.title('Model test accuracy')
    plt.ylabel('Validation accuracy')
    plt.xlabel('epoch')
    plt.legend(['train'], loc='upper left')
    plt.show()
    

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


if __name__ == "__main__":
    
        
    layers = model_orig(4, 2)
        
    _, mat = utils.ann_to_graph(layers)
    
    #rewire connections
    Dglobals, Dlocals, p = utils.find_smallnetwork(mat, layers)
    
    for i in range(len(Dglobals)):
        Dglobals[i] = 1/Dglobals[i]
        Dlocals[i] = 1/Dlocals[i]
               
    fig, ax1 = plt.subplots()
    ax1.set_xlabel('rewiring probability (p)')
    ax1.set_ylabel('Dglobal', color='b')
    ax1.scatter(p, Dglobals, color='b', marker='.')
    ax1.set_ylabel("Dglobal", color='b')
    
    ax2 = ax1.twinx()
    ax2.scatter(p, Dlocals, color='r', marker='.')
    ax2.set_ylabel("Dlocal", color='r')
    
    plt.show()
    
    #load dataset
#    x,y =utils.load_csv(PATHDRIVE, NUMBER_OF_ATTRIBUTES)
#    X_train, y_train, X_test, y_test = utils.get_cifar_data(PATH1,PATH2,PATH3,PATH4,PATH5,PATH6)
#    X_train = utils.normalize(X_train)
#    X_test = utils.normalize(X_test)
#    x,y = utils.load_csv(PATHDRIVE, NUMBER_OF_ATTRIBUTES)
    

    #convert labels to integers
#    y = utils.classes_to_int(y)
    #y = to_categorical(y, num_classes=2)
    
    #split to trainign and testing data
#    X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.97)
#    X_test = X_test[0:1000]
#    y_test = y_test[0:1000]
#    
#    
#    model = models.model_dense()
#    test_model(model, X_train, y_train, X_test, y_test)


    #get neural network
#    model, layers = models.model_orig()
#    _, mat = utils.ann_to_graph(layers)
#    
#    model = models.model_weight_reg()
#    hist_mod1 = test_model(model, X_train, y_train, X_test, y_test)
#
#    rewired_mat = utils.rewire_to_smallworld(mat, layers, 0.7)
#    new_layers = utils.graph_to_ann(rewired_mat, layers)    
#    model2 = models.model_rewired_weight_reg(new_layers)
#    hist_mod2 = test_model(model2, X_train, y_train, X_test, y_test)
    
    
#    m = utils.rewire_to_smallworld(mat, layers, 0.7)
#    new_layers = utils.graph_to_ann(m, layers)
#
#    modelW = models.model_weight_reg()  
#    histW = test_model(modelW, X_train, y_train, X_test, y_test)
#    
#    modelSW = models.model_rewired_weight_reg(new_layers)
#    histSW = test_model(modelSW, X_train, y_train, X_test, y_test)
    
#    # summarize history for accuracy
#    plt.plot(hist_mod1.history['acc'], color='yellow', label='weight regularization')
#    plt.plot(hist_mod2.history['acc'], color='magenta', label='small-world with weight regularization')
#    plt.title('Model accuracy')
#    plt.ylabel('accuracy')
#    plt.xlabel('epoch')
#    plt.legend(loc='lower right')
#    plt.show()
#    
#
#    plt.plot(hist_mod1.history['loss'], label='weight regularization')
#    plt.plot(hist_mod2.history['loss'], label='small-world with weight regularization')
#    plt.title('Model loss')
#    plt.ylabel('loss')
#    plt.xlabel('epoch')
#    plt.legend(loc='upper right')
#    plt.show()
#    
#    # summarize history for validation accuracy
#    plt.plot(hist_mod1.history['val_acc'], label='weight regularization')
#    plt.plot(hist_mod2.history['val_acc'], label='small-world with weight regularization')
#    plt.title('Validation accuracy')
#    plt.ylabel('accuracy')
#    plt.xlabel('epoch')
#    plt.legend(loc='lower right')
#    plt.show()
#    
    #model_dense = models.model_dense()
#    results = []
#    modelss = ['Dense', 'Small-world with weight reg', 'Small-world', 'Dropout', 'Weight reg']

#    history_dense, result = test_model(model, X_train, y_train, X_test, y_test)
#    results.append(result)
#        
#    
#    graph, mat = utils.ann_to_graph(layers)
#    
#    m = utils.rewire_to_smallworld(mat, layers, 0.7)
#    
    #vizualise NN
#    graph = nx.from_numpy_matrix(m)
#    vis = nx.nx_pydot.to_pydot(graph)
#   vis.write_png('example2_graph.png')
    
#    model = models.model_dropout()
#    test_model(model, X_train, y_train, X_test, y_test)
    

#    
#    index = []
#    data = []
#    modelss = []
#    accu = []
#    
#    for i in range(NUM_MODELS):
#        
#        rewired_mat = utils.rewire_to_smallworld(mat, layers, 0.7)
#        new_layers = graph_to_ann(rewired_mat, layers)
#        
#        model = models.model_rewired(new_layers)
#        
#        history = test_model(model, X_train, y_train, X_test, y_test)
#        
#        index = index + list(range(0, EPOCHS))
#        data = data + history.history['val_acc']
#        modelss = modelss + ["Small-world"] * EPOCHS
#        
#        accu = accu + history.history['acc']
#        print_training_phase("Small-world", i)
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
#    modelss = []
#    index = []
#    data = []
#    accu = []
#    
#    for i in range(NUM_MODELS):
#        
#        model = models.model_dense()
#        
#        history = test_model(model, X_train, y_train, X_test, y_test)
#        
#        index = index + list(range(0, EPOCHS))
#        data = data + history.history['val_acc']
#        modelss = modelss + ["Dense"] * EPOCHS
#        accu = accu + history.history['acc']
#        
#        print_training_phase("Dense", i)
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
#    modelss = []
#    index = []
#    data = []
#    accu = []
#    
#    for i in range(NUM_MODELS):
#               
#        model = models.model_dropout()
#        
#        history = test_model(model, X_train, y_train, X_test, y_test)
#        
#        index = index + list(range(0, EPOCHS))
#        data = data + history.history['val_acc']
#        modelss = modelss + ["Dropout"] * EPOCHS
#        accu = accu + history.history['acc']
#        
#        print_training_phase("Dropout", i)
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
#    modelss = []
#    index = []
#    data = []
#    accu = []
#    
#    for i in range(NUM_MODELS):
#               
#        
#        model = models.model_weight_reg()
#        
#        history = test_model(model, X_train, y_train, X_test, y_test)
#        
#        index = index + list(range(0, EPOCHS))
#        data = data + history.history['val_acc']
#        modelss = modelss + ["Weight reg"] * EPOCHS
#        accu = accu + history.history['acc']
#        
#        print_training_phase("Weight reg", i)
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
#    modelss = []
#    index = []
#    data = []
#    accu = []
#    
#    for i in range(NUM_MODELS):
#               
#        rewired_mat = utils.rewire_to_smallworld(mat, layers, 0.7)
#        new_layers = graph_to_ann(rewired_mat, layers)    
#    
#        model = models.model_rewired_weight_reg(new_layers)
#        
#        history = test_model(model, X_train, y_train, X_test, y_test)
#        
#        index = index + list(range(0, EPOCHS))
#        data = data + history.history['val_acc']
#        modelss = modelss + ["Small-world with weight reg"] * EPOCHS
#        accu = accu + history.history['acc']
#        
#        print_training_phase("Small-world with weight reg", i)
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
#    modelss = []
#    index = []
#    data = []
#    accu = []
#    
#    
#    
#    sns.lineplot(x="epochs", y="Test accuracy", hue='Model', data=dataframe)
#    plt.show()
#    sns.lineplot(x="epochs", y="Train accuracy", hue='Model', data=dataframe_acc)
#    
    
#    
#    model_small_weight = models.model_rewired_weight_reg(new_layers)
#    history_small_weight, result = test_model(model_small_weight, X_train, y_train, X_test, y_test)
#    results.append(result)
#    
#    new_model = models.model_rewired(new_layers)
#    history_small, result = test_model(new_model, X_train, y_train, X_test, y_test)
#    results.append(result)
#    
#    model_dropout = models.model_dropout()
#    history_dropout, result = test_model(model_dropout,X_train, y_train, X_test, y_test)
#    results.append(result)
#    
#    model_weight_reg = models.model_weight_reg()
#    history_weight, result = test_model(model_weight_reg, X_train, y_train, X_test, y_test)
#    results.append(result)
#    

#    # summarize history for accuracy
#    plt.plot(history_dense.history['acc'], color='green', label='Dense')
#    plt.plot(history_small.history['acc'], color='red', label='small-world')
#    plt.plot(history_dropout.history['acc'], color='blue', label='Dropout')
#    plt.plot(history_weight.history['acc'], color='yellow', label='weight regularization')
#    plt.plot(history_small_weight.history['acc'], color='magenta', label='small-world with weight regularization')
#    plt.title('Model accuracy')
#    plt.ylabel('accuracy')
#    plt.xlabel('epoch')
#    plt.legend(loc='lower right')
#    plt.show()
#    
#    plt.plot(history_dense.history['loss'], color='green', label='Dense')
#    plt.plot(history_small.history['loss'], color='red', label='small-world')
#    plt.plot(history_dropout.history['loss'], color='blue', label='Dropout')
#    plt.plot(history_weight.history['loss'], color='yellow', label='weight regularization')
#    plt.plot(history_small_weight.history['loss'], color='magenta', label='small-world with weight regularization')
#    plt.title('Model loss')
#    plt.ylabel('loss')
#    plt.xlabel('epoch')
#    plt.legend(loc='upper right')
#    plt.show()
#    
#    # summarize history for validation accuracy
#    plt.plot(history_dense.history['val_acc'], color='green', label='Dense')
#    plt.plot(history_small.history['val_acc'], color='red', label='small-world')
#    plt.plot(history_dropout.history['val_acc'], color='blue', label='Dropout')
#    plt.plot(history_weight.history['val_acc'], color='yellow', label='weight regularization')
#    plt.plot(history_small_weight.history['val_acc'], color='magenta', label='small-world with weight regularization')
#    plt.title('Validation accuracy')
#    plt.ylabel('accuracy')
#    plt.xlabel('epoch')
#    plt.legend(loc='lower right')
#    plt.show()
#    
#    for i in range(len(results)):
#        print('Test accuracy with {} network {}'.format(modelss[i], results[i]))
#    
#    