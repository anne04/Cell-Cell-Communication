# adapted from https://github.com/dmlc/dgl/blob/master/examples/pytorch/vgae/model.py
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
#from dgl.nn.pytorch import GraphConv
from train import device


class VGAEModel(nn.Module):
    def __init__(self, in_dim, hidden1_dim, hidden2_dim):
        super(VGAEModel, self).__init__()
        self.in_dim = in_dim
        self.hidden1_dim = hidden1_dim
        self.hidden2_dim = hidden2_dim

        layers = [
            GCNConv(
                self.in_dim,
                self.hidden1_dim #,
                #activation=F.relu
            ),
            GCNConv(
                self.hidden1_dim,
                self.hidden2_dim #,
#                activation=lambda x: x,
            ),
            GCNConv(
                self.hidden1_dim,
                self.hidden2_dim #,
#                activation=lambda x: x,
            ),
        ]
        self.layers = nn.ModuleList(layers)

    def encoder(self, data): #g, features):
        x, edge_index, edge_weight = data.x, data.edge_index, data.edge_attr
        x = self.layers[0](x, edge_index, edge_weight=edge_weight)
        x = self.relu(x)
        #h = self.layers[0](g, features)
        self.mean = self.layers[1](x, edge_index, edge_weight=edge_weight)
        self.log_std = self.layers[2](x, edge_index, edge_weight=edge_weight)
        gaussian_noise = torch.randn(x.size(0), self.hidden2_dim).to(
            device
        )
        sampled_z = self.mean + gaussian_noise * torch.exp(self.log_std).to(
            device
        )
        '''
        gaussian_noise = torch.randn(features.size(0), self.hidden2_dim).to(
            device
        )        
        sampled_z = self.mean + gaussian_noise * torch.exp(self.log_std).to(
            device
        )'''
        return sampled_z

    def decoder(self, z):
        adj_rec = torch.sigmoid(torch.matmul(z, z.t()))
        return adj_rec

    def forward(self, data):
        z = self.encoder(data)
        adj_rec = self.decoder(z)
        return adj_rec
    '''
    def forward(self, g, features):
        z = self.encoder(g, features)
        adj_rec = self.decoder(z)
        return adj_rec
    '''
