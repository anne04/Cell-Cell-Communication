import os
import sys
import matplotlib
matplotlib.use('Agg')
#matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import pylab as pl
from mpl_toolkits.axes_grid1.inset_locator import inset_axes

from sklearn import metrics
from scipy import interp
from sklearn.metrics import roc_curve, auc, roc_auc_score

import numpy as np
from scipy import sparse
import pickle
import pandas as pd
import scanpy as sc
import anndata as ad

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, ChebConv, GATConv, DeepGraphInfomax, global_mean_pool, global_max_pool  # noqa
from torch_geometric.data import Data, DataLoader

from CCST import get_graph, train_DGI, train_DGI, PCA_process, Kmeans_cluster

rootPath = os.path.dirname(sys.path[0])
os.chdir(rootPath+'/CCST')

data_file = 'generated_data/V1_Breast_Cancer_Block_A_Section_1/'
with open(data_file + 'Adjacent', 'rb') as fp:
    adj_0 = pickle.load(fp)
X_data = np.load(data_file + 'features.npy')

lambda_I=0.3
num_points = X_data.shape[0]
adj_I = np.eye(num_points)
adj_I = sparse.csr_matrix(adj_I)
adj = (1-lambda_I)*adj_0 + lambda_I*adj_I

row_col = []
edge_weight = []
rows, cols = adj.nonzero() # index of non zero rows,cols
edge_nums = adj.getnnz()  # number of entries in adj having non zero weights

for i in range(edge_nums):
    row_col.append([rows[i], cols[i]])
    edge_weight.append(adj.data[i])
    
edge_index = torch.tensor(np.array(row_col), dtype=torch.long).T
edge_attr = torch.tensor(np.array(edge_weight), dtype=torch.float)

graph = Data(x=torch.tensor(X_data, dtype=torch.float), edge_index=edge_index, edge_attr=edge_attr)

graph_bags = []
graph_bags.append(graph)

