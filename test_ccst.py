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



class Encoder(nn.Module):
    def __init__(self, in_channels, hidden_channels):
        super(Encoder, self).__init__()
        self.conv = GCNConv(in_channels, hidden_channels)
        self.conv_2 = GCNConv(hidden_channels, hidden_channels)
        self.conv_3 = GCNConv(hidden_channels, hidden_channels)
        self.conv_4 = GCNConv(hidden_channels, hidden_channels)
        
        self.prelu = nn.PReLU(hidden_channels)
        
    def forward(self, data):
        x, edge_index, edge_weight = data.x, data.edge_index, data.edge_attr
        x = self.conv(x, edge_index, edge_weight=edge_weight)
        x = self.conv_2(x, edge_index, edge_weight=edge_weight)
        x = self.conv_3(x, edge_index, edge_weight=edge_weight)
        x = self.conv_4(x, edge_index, edge_weight=edge_weight)
        x = self.prelu(x)

        return x


#rootPath = os.path.dirname(sys.path[0])
#os.chdir(rootPath+'/CCST')

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

num_cell = X_data.shape[0]
num_feature = X_data.shape[1]
print('Adj:', adj.shape, 'Edges:', len(adj.data))
print('X:', X_data.shape)



graph = Data(x=torch.tensor(X_data, dtype=torch.float), edge_index=edge_index, edge_attr=edge_attr)

graph_bags = []
graph_bags.append(graph)

batch_size=1
data_list=graph_bags
data_loader = DataLoader(data_list, batch_size=batch_size)

in_channels=num_feature # length of node attribute vector
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

hidden_channels=num_feature
Encoder_DGI=Encoder(in_channels=in_channels, hidden_channels=hidden_channels)
corruption_DGI=corruption

DGI_model = DeepGraphInfomax(hidden_channels=args.hidden, encoder=Encoder_DGI,
        summary=lambda z, *args, **kwargs: torch.sigmoid(z.mean(dim=0)),
        corruption=corruption_DGI).to(device)


