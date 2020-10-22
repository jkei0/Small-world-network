#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Aug  7 00:03:52 2020

@author: joni
"""

import json
import pandas as pd
import os
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

NUM_ITER = 500
# small world path
path = "../Downloads/PPO_prob_value"

#dense path
path2 = "../Downloads/PPODense"

directory = os.fsencode(path)
dir2 = os.fsencode(path2)

data = pd.DataFrame().T
dataDense = pd.DataFrame().T

model = ["Small-world"] * NUM_ITER * 8
modelD = ["Dense"] * NUM_ITER * 8

unit = ["a"]*NUM_ITER + ["b"]*NUM_ITER + ["c"]*NUM_ITER + ["d"]*NUM_ITER + ["e"]*NUM_ITER + ["f"] *NUM_ITER +\
        ["g"]*NUM_ITER + ["h"]*NUM_ITER + ["i"]*NUM_ITER + ["j"]*NUM_ITER + ["k"]*NUM_ITER + ["l"] *NUM_ITER +\
        ["m"]*NUM_ITER + ["n"]*NUM_ITER + ["o"]*NUM_ITER + ["p"]*NUM_ITER

for file in os.listdir(directory):
    filename = os.fsdecode(file)
    dat = pd.read_json(path + "/" + filename)
    data = pd.concat([data, dat])
    
for file in os.listdir(dir2):
    filename = os.fsdecode(file)
    dat = pd.read_json(path2 + "/" + filename)
    dataDense = pd.concat([dataDense, dat])

data.columns = ["len", "Timestep", "Reward"]
data["Model"] = model

dataDense.columns = ["len", "Timestep", "Reward"]
dataDense["Model"] = modelD

result = data["Reward"]
result2 = dataDense["Reward"]

data = pd.concat([dataDense, data])

# episodes standard length
plt.figure()
sns.lineplot(x="Timestep", y="Reward", hue="Model", data=data, units=unit, estimator=None)
plt.figure()
sns.lineplot(x="Timestep", y="Reward", hue="Model", data=data)

# episodes length vary
#plt.figure()
#sns.lineplot(x=data.index, y="Reward", hue="Model", data=data, units=unit, estimator=None)
#plt.figure()
#sns.lineplot(x=data.index, y="Reward", hue="Model", data=data)




