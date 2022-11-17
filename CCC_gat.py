  GNU nano 2.3.1                                                                                                                                                                                              File: CCC_gat.py                                                                                                                                                                                                                                                                                                                                                                                                    

##exocrine GCNG with normalized graph matrix
import os
import sys
import matplotlib
matplotlib.use('Agg')
#matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import gzip

from sklearn import metrics
from scipy import sparse
import pickle
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import RGCNConv, TAGConv, GCNConv, Linear, DeepGraphInfomax, global_mean_pool, global_max_pool, SAGEConv, ChebConv, RGATConv, GATv2Conv, GraphConv
from torch_geometric.data import Data, DataLoader



def get_graph(adj, X, training_data_name):
    # create sparse matrix

    '''row_col = []
    edge_weight = []
    edge_type = []
    rows, cols = adj.nonzero()
    edge_nums = adj.getnnz()
    for i in range(edge_nums):
        row_col.append([rows[i], cols[i]])
        edge_weight.append(adj.data[i])
        edge_type.append(1)'''

    '''f = gzip.open("/cluster/projects/schwartzgroup/fatema/find_ccc/" + 'adjacency_records_2RGAT_synthetic_region1_STnCCC_70', 'rb') #normalized
    row_col, edge_weight, edge_rtype = pickle.load(f)

    edge_index = torch.tensor(np.array(row_col), dtype=torch.long).T
    edge_attr = torch.tensor(np.array(edge_weight), dtype=torch.float)
    edge_type = torch.tensor(np.array(edge_rtype), dtype=torch.int) #float)
    print('X shape ')
    print(X.shape)
    graph_bags = []
    graph = Data(x=torch.tensor(X, dtype=torch.float), edge_index=edge_index, edge_attr=edge_attr, edge_type=edge_type)'''


#    f = gzip.open("/cluster/projects/schwartzgroup/fatema/find_ccc/" + 'adjacency_records_GAT_onlyccc_97', 'rb') #normalized
#    f = gzip.open("/cluster/projects/schwartzgroup/fatema/find_ccc/" + 'adjacency_records_GAT_STnCCC_97', 'rb')
#    f = gzip.open("/cluster/projects/schwartzgroup/fatema/find_ccc/"+'adjacency_records_GAT_synthetic_region1_onlyccc_70', 'rb')
#    f = gzip.open("/cluster/projects/schwartzgroup/fatema/find_ccc/"+'adjacency_records_GAT_synthetic_region1_STnCCC_70', 'rb')
#    f = gzip.open("/cluster/projects/schwartzgroup/fatema/find_ccc/"+'adjacency_records_GAT_synthetic_region1_STnCCC_70', 'rb')
    f = gzip.open("/cluster/projects/schwartzgroup/fatema/find_ccc/"+training_data_name, 'rb') #'adjacency_records_GAT_total_synthetic_region1_STnCCC', 'rb')
    row_col, edge_weight = pickle.load(f)

    print("row_col %d"%len(row_col))
    #print(edge_weight)
    ###########


    edge_index = torch.tensor(np.array(row_col), dtype=torch.long).T
    edge_attr = torch.tensor(np.array(edge_weight), dtype=torch.float)
    print('X shape ')
    print(X.shape)
    graph_bags = []
    graph = Data(x=torch.tensor(X, dtype=torch.float), edge_index=edge_index, edge_attr=edge_attr)


    graph_bags.append(graph)
    print('get graph done')
    return graph_bags


class Encoder(nn.Module):
    def __init__(self, in_channels, hidden_channels, heads):
        super(Encoder, self).__init__()
        print('incoming channel %d'%in_channels)
        '''self.conv =  RGATConv(in_channels, hidden_channels, 2, edge_dim=1)
        self.conv_2 =  RGATConv(hidden_channels, hidden_channels, 2, edge_dim=1)
        self.conv_3 =  RGATConv(hidden_channels, hidden_channels, 2, edge_dim=1)
        #self.conv_4 =  RGATConv(hidden_channels, hidden_channels, 2, edge_dim=1)'''
        heads = heads
        self.conv =  GATv2Conv(in_channels, hidden_channels, edge_dim=2, heads=heads)
        self.conv_2 =  GATv2Conv(hidden_channels*heads, hidden_channels, edge_dim=2, heads=1)
#        self.conv_3 =  GATv2Conv(hidden_channels, hidden_channels, edge_dim=2)
        '''self.conv_4 =  GATConv(hidden_channels, hidden_channels, edge_dim=1)'''



        '''self.conv = RGCNConv(in_channels, hidden_channels, 1)
        self.conv_2 = RGCNConv(hidden_channels, hidden_channels, 1)
        self.conv_3 = RGCNConv(hidden_channels, hidden_channels, 1)
        self.conv_4 = RGCNConv(hidden_channels, hidden_channels, 1)'''

        '''self.conv = RGCNConv(in_channels, hidden_channels, 2) #, num_bases=300)
        self.conv_2 = RGCNConv(hidden_channels, hidden_channels, 2) #,6240, num_bases=300)
        self.conv_3 = RGCNConv(hidden_channels, hidden_channels, 2) #6240, num_bases=300)
        self.conv_4 = RGCNConv(hidden_channels, hidden_channels, 2) #6240, num_bases=300)
        '''

	self.attention_scores_mine = 'attention'
        self.attention_scores_mine_unnormalized = 'attention_unnormalized'
        #self.prelu = nn.Tanh(hidden_channels)
        self.prelu = nn.PReLU(hidden_channels)


    def forward(self, data):
        #x, edge_index, edge_attr, edge_type = data.x, data.edge_index, data.edge_attr, data.edge_type
        #x = self.conv(x, edge_index, edge_type = edge_type, edge_attr=edge_attr)
        #print('1st pass')
        #print(x)
        #x = self.conv_2(x, edge_index, edge_type = edge_type, edge_attr=edge_attr)
        #print('2nd pass')
        #print(x)
        #x, attention_scores = self.conv_3(x, edge_index, edge_type = edge_type, edge_attr=edge_attr, return_attention_weights = True)
#        x = self.conv_4(x, edge_index, edge_type = edge_type, edge_attr=edge_weight)
#        x = self.conv_5(x, edge_index, edge_type) #, edge_attr=edge_weight)
#        x = self.conv_6(x, edge_index, edge_type) #, edge_attr=edge_weight)

        x, edge_index, edge_weight = data.x, data.edge_index, data.edge_attr

#        x, attention_scores = self.conv(x, edge_index, edge_attr=edge_weight, return_attention_weights = True)
        x = self.conv(x, edge_index, edge_attr=edge_weight)
        x, attention_scores, attention_scores_unnormalized  = self.conv_2(x, edge_index, edge_attr=edge_weight, return_attention_weights = True)
#        x, attention_scores = self.conv_3(x, edge_index, edge_attr=edge_weight, return_attention_weights = True)
        '''x = self.conv_3(x, edge_index, edge_attr=edge_weight)
        x = self.conv_4(x, edge_index, edge_attr=edge_weight)'''

        x = self.prelu(x)
        self.attention_scores_mine = attention_scores
        self.attention_scores_mine_unnormalized = attention_scores_unnormalized
        return x, attention_scores

class my_data():
    def __init__(self, x, edge_index, edge_attr):
        self.x = x
        self.edge_index = edge_index
        self.edge_attr = edge_attr


def corruption(data):
    #print('inside corruption function')
    x = data.x[torch.randperm(data.x.size(0))]
    return my_data(x, data.edge_index, data.edge_attr)

def train_DGI(args, data_loader, in_channels):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    DGI_model = DeepGraphInfomax(
        hidden_channels=args.hidden,
        encoder=Encoder(in_channels=in_channels, hidden_channels=args.hidden, heads=args.heads),
        summary=lambda z, *args, **kwargs: torch.sigmoid(z.mean(dim=0)),
        corruption=corruption).to(device)
    #print('initialized DGI model')
    #print(DGI_model.encoder.attention_scores_mine)
    DGI_optimizer = torch.optim.Adam(DGI_model.parameters(), lr=1e-5) #6
    DGI_filename = args.model_path+'DGI'+ args.model_name  +'.pth.tar'
    if args.load:
        DGI_model.load_state_dict(torch.load(DGI_filename))
    else:
	import datetime
        start_time = datetime.datetime.now()
        min_loss=10000
        if args.retrain==1:
            DGI_load_path = args.model_load_path+'DGI'+ args.model_name+'.pth.tar'
            DGI_model.load_state_dict(torch.load(DGI_load_path))
        #print('Saving init model state ...')
        torch.save(DGI_model.state_dict(), args.model_path+'DGI_init'+ args.model_name  + '.pth.tar')
        #print('training starts ...')
        for epoch in range(args.num_epoch):
            DGI_model.train()
            DGI_optimizer.zero_grad()
            DGI_all_loss = []

            for data in data_loader:
                data = data.to(device)
                '''x_old, attention_score = DGI_model(data=data)
                print('model output')
                print(x_old)
                pos_z = x_old[0]
                neg_z = x_old[1]
                summary = x_old[2]'''

                pos_z, neg_z, summary = DGI_model(data=data)
                #print('epoch %d '%epoch)
                #print(DGI_model.encoder.attention_scores_mine)
                DGI_loss = DGI_model.loss(pos_z, neg_z, summary)
                DGI_loss.backward()
                DGI_all_loss.append(DGI_loss.item())
                DGI_optimizer.step()

            if ((epoch+1)%100) == 0:
                print('Epoch: {:03d}, Loss: {:.4f}'.format(epoch+1, np.mean(DGI_all_loss)))
                if np.mean(DGI_all_loss)<min_loss:
                    min_loss=np.mean(DGI_all_loss)
                    torch.save(DGI_model.state_dict(), DGI_filename)
                    save_tupple=[pos_z, neg_z, summary]
                    saved_attention = DGI_model.encoder.attention_scores_mine
                    saved_attention_unnormalized = DGI_model.encoder.attention_scores_mine_unnormalized
                    print(DGI_model.encoder.attention_scores_mine[0][0:10])
                    print(DGI_model.encoder.attention_scores_mine[1][0:10])

        end_time = datetime.datetime.now()

#        torch.save(DGI_model.state_dict(), DGI_filename)
        print('Training time in seconds: ', (end_time-start_time).seconds)
        DGI_model.load_state_dict(torch.load(DGI_filename))
        print("debug loss")
        DGI_loss = DGI_model.loss(pos_z, neg_z, summary)
        print("debug loss latest tupple %g"%DGI_loss.item())
        DGI_loss = DGI_model.loss(save_tupple[0], save_tupple[1], save_tupple[2])
        print("debug loss min loss tupple %g"%DGI_loss.item())
        DGI_model.encoder.attention_scores_mine = saved_attention
        DGI_model.encoder.attention_scores_mine_unnormalized = saved_attention_unnormalized
    return DGI_model

def PCA_process(X, nps):
    from sklearn.decomposition import PCA
    print('Shape of data to PCA:', X.shape)
    pca = PCA(n_components=nps)
    X_PC = pca.fit_transform(X)     #等价于pca.fit(X) pca.transform(X)
    print('Shape of data output by PCA:', X_PC.shape)
    print('PCA recover:', pca.explained_variance_ratio_.sum())
    return X_PC

from sklearn.cluster import KMeans, DBSCAN, AffinityPropagation
def Kmeans_cluster(X_embedding, n_clusters, merge=False):
    cluster_model = KMeans(n_clusters=n_clusters, init='k-means++', n_init=100, max_iter=1000, tol=1e-6)
    cluster_labels = cluster_model.fit_predict(X_embedding)

    # merge clusters with less than 3 cells
    if merge:
	cluster_labels = merge_cluser(X_embedding, cluster_labels)

    score = metrics.silhouette_score(X_embedding, cluster_labels, metric='euclidean')

    return cluster_labels, score

def Umap(args, X, label, n_clusters, score):
    import umap
    reducer = umap.UMAP(random_state=42)
    embedding = reducer.fit_transform(X)

    plt.scatter(embedding[:, 0], embedding[:, 1], c=label, cmap='Spectral', s=20)
    plt.gca().set_aspect('equal', 'datalim')
    plt.colorbar(boundaries=np.arange(n_clusters+1)-0.5).set_ticks(np.arange(n_clusters))
    plt.title('UMAP projection')
    if score:
	plt.text(0.0, 0.0, score, fontdict={'size':'16','color':'black'},  transform = plt.gca().transAxes)
    plt.savefig(args.result_path + '/Umap.jpg')
    #plt.show()
    plt.close()




  
