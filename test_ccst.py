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

import argparse
parser = argparse.ArgumentParser()
# ================Specify data type firstly===============
parser.add_argument( '--data_type', default='nsc', help='"sc" or "nsc", \
    refers to single cell resolution datasets(e.g. MERFISH) and \
    non single cell resolution data(e.g. ST) respectively') 
# =========================== args ===============================
parser.add_argument( '--data_name', type=str, default='V1_Breast_Cancer_Block_A_Section_1', help="'MERFISH' or 'V1_Breast_Cancer_Block_A_Section_1") 
parser.add_argument( '--lambda_I', type=float, default=0.3) #0.8 on MERFISH, 0.3 on ST
parser.add_argument( '--data_path', type=str, default='generated_data/', help='data path')
parser.add_argument( '--model_path', type=str, default='model') 
parser.add_argument( '--embedding_data_path', type=str, default='Embedding_data') 
parser.add_argument( '--result_path', type=str, default='results') 
parser.add_argument( '--DGI', type=int, default=1, help='run Deep Graph Infomax(DGI) model, otherwise direct load embeddings')
parser.add_argument( '--load', type=int, default=0, help='Load pretrained DGI model')
parser.add_argument( '--num_epoch', type=int, default=5000, help='numebr of epoch in training DGI')
parser.add_argument( '--hidden', type=int, default=256, help='hidden channels in DGI') 
parser.add_argument( '--PCA', type=int, default=1, help='run PCA or not')   
parser.add_argument( '--cluster', type=int, default=1, help='run cluster or not')
parser.add_argument( '--n_clusters', type=int, default=5, help='number of clusters in Kmeans, when ground truth label is not avalible.') #5 on MERFISH, 20 on Breast
parser.add_argument( '--draw_map', type=int, default=1, help='run drawing map')
parser.add_argument( '--diff_gene', type=int, default=0, help='Run differential gene expression analysis')
args = parser.parse_args() 


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

def corruption(data):
    x = data.x[torch.randperm(data.x.size(0))]
    return my_data(x, data.edge_index, data.edge_attr)




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


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

in_channels=num_feature # length of node attribute vector
hidden_channels=256 # length of node embedding vector
Encoder_DGI=Encoder(in_channels=in_channels, hidden_channels=hidden_channels)

corruption_DGI=corruption

DGI_model = DeepGraphInfomax(hidden_channels=hidden_channels, encoder=Encoder_DGI,
        summary=lambda z, *args, **kwargs: torch.sigmoid(z.mean(dim=0)),
        corruption=corruption_DGI).to(device)


