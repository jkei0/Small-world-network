# -*- coding: utf-8 -*-
"""
Created on Sun Apr  5 23:30:13 2020

@author: jonik
"""

import networkx as nx
import numpy as np

g = nx.from_numpy_matrix(a)
vis = nx.nx_pydot.to_pydot(g)
vis.write_png('test_graph.png')