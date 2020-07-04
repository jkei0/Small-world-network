# -*- coding: utf-8 -*-
"""
Created on Mon Apr 27 18:00:34 2020

@author: jonik
"""

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


data1 = pd.read_pickle('dataframe/data1.pkl')
data2 = pd.read_pickle('dataframe/data2.pkl')
data3 = pd.read_pickle('dataframe/data3.pkl')
data4 = pd.read_pickle('dataframe/data4.pkl')
data5 = pd.read_pickle('dataframe/data5.pkl')
data6 = pd.read_pickle('dataframe/data6.pkl')
data7 = pd.read_pickle('dataframe/data7.pkl')
data8 = pd.read_pickle('dataframe/data8.pkl')
data9 = pd.read_pickle('dataframe/data9.pkl')
data10 = pd.read_pickle('dataframe/data10.pkl')

data1 = data1.append(data2)
data1 = data1.append(data3)
data1 = data1.append(data4)
data1 = data1.append(data5)
data1 = data1.append(data6)
data1 = data1.append(data7)
data1 = data1.append(data8)
data1 = data1.append(data9)
data1 = data1.append(data10)

sns.lineplot(x="epochs", y="Test accuracy", hue='Model', data=data1)
plt.show()


#data1 = pd.read_pickle('dataframe/data_acc1.pkl')
#data2 = pd.read_pickle('dataframe/data_acc2.pkl')
#data3 = pd.read_pickle('dataframe/data_acc3.pkl')
#data4 = pd.read_pickle('dataframe/data_acc4.pkl')
#data5 = pd.read_pickle('dataframe/data_acc5.pkl')
#data6 = pd.read_pickle('dataframe/data_acc6.pkl')
#data7 = pd.read_pickle('dataframe/data_acc7.pkl')
#data8 = pd.read_pickle('dataframe/data_acc8.pkl')
#data9 = pd.read_pickle('dataframe/data_acc9.pkl')
#data10 = pd.read_pickle('dataframe/data_acc10.pkl')
#
#data1 = data1.append(data2)
#data1 = data1.append(data3)
#data1 = data1.append(data4)
#data1 = data1.append(data5)
#data1 = data1.append(data6)
#data1 = data1.append(data7)
#data1 = data1.append(data8)
#data1 = data1.append(data9)
#data1 = data1.append(data10)
#
#sns.lineplot(x="epochs", y="Train accuracy", hue='Model', data=data1)
#plt.show()


