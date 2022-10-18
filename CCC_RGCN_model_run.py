  GNU nano 2.3.1                                                                                                                                                                                            File: CCST_ST_utils.py                                                                                                                                                                                                                                                                                                                                                                                                

##exocrine GCNG with normalized graph matrix
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

from CCC_rgcn import get_graph, train_DGI, PCA_process, Kmeans_cluster

rootPath = os.path.dirname(sys.path[0])
os.chdir(rootPath+'/CCST')

def get_data(args):
    data_file = args.data_path + args.data_name +'/'
    with open(data_file + 'Adjacent', 'rb') as fp:
        adj_0 = pickle.load(fp)

#    with open(data_file + 'features', 'rb') as fp:
#        X_data = pickle.load(fp)

    X_data = np.load(data_file + 'features.npy')

    num_points = X_data.shape[0]
    adj_I = np.eye(num_points)
    adj_I = sparse.csr_matrix(adj_I)

    if args.all_distance != 0:
        print('all distance true')
        adj = adj_0
    else:
	adj = adj_0-adj_I # diagonal becomes zero
        const = args.meu
        print('spatial:', args.meu)
        adj = adj*const + adj_I*args.lambda_I # adj*0.05 + adj_I # 2k

        #adj = adj_0 + adj_I
        #adj = 0.2*adj_0 + args.lambda_I*adj_I #lembda_I=1, mehigh folder
        #adj = (1-args.lambda_I)*adj_0 + args.lambda_I*adj_I # lembda=0.8,.95,.98

#    cell_type_indeces = np.load(data_file + 'cell_types.npy')

    return adj_0, adj, X_data, 5 #CHANGE##cell_type_indeces



def draw_map(args, adj_0, barplot=False):
    data_folder = args.data_path + args.data_name+'/'
    save_path = args.result_path

    f = open(save_path+'/types.txt')
    line = f.readline() # drop the first line
    cell_cluster_type_list = []

    while line:
        tmp = line.split('\t')
        cell_id = int(tmp[0]) # index start is start from 0 here
        #cell_type_index = int(tmp[1])
        cell_cluster_type = int(tmp[1].replace('\n', ''))
        cell_cluster_type_list.append(cell_cluster_type)
        line = f.readline()
    f.close()

    n_clusters = max(cell_cluster_type_list) + 1 # start from 0

    print('n clusters in drwaing:', n_clusters)
    coordinates = np.load(data_folder+'coordinates.npy')

    sc_cluster = plt.scatter(x=coordinates[:,0], y=-coordinates[:,1], s=5, c=cell_cluster_type_list) #, cmap='rainbow')

    plt.legend(handles = sc_cluster.legend_elements(num=n_clusters)[0],labels=np.arange(n_clusters).tolist(), bbox_to_anchor=(1,0.5), loc='center left', prop={'size': 9})

    #cb_cluster = plt.colorbar(sc_cluster, boundaries=np.arange(n_types+1)-0.5).set_ticks(np.arange(n_types))
    plt.xticks([])
    plt.yticks([])
    plt.axis('scaled')
    #plt.xlabel('X')
    #plt.ylabel('Y')
    plt.title('CCST')
    plt.savefig(save_path+'/spatial_'+args.cluster_alg+'.png', dpi=400, bbox_inches='tight')
    plt.clf()


    # draw barplot
    if barplot:
        total_cell_num = len(cell_cluster_type_list)
        barplot = np.zeros([n_clusters, n_clusters], dtype=int)
        source_cluster_type_count = np.zeros(n_clusters, dtype=int)
        p1, p2 = adj_0.nonzero()
        def get_all_index(lst=None, item=''):
            return [i for i in range(len(lst)) if lst[i] == item]

        for i in range(total_cell_num):
            source_cluster_type_index = cell_cluster_type_list[i]
            edge_indeces = get_all_index(p1, item=i)
            paired_vertices = p2[edge_indeces]
            for j in paired_vertices:
                neighbor_type_index = cell_cluster_type_list[j]
                barplot[source_cluster_type_index, neighbor_type_index] += 1
                source_cluster_type_count[source_cluster_type_index] += 1

        np.savetxt(save_path + '/cluster_' + str(n_clusters) + '_barplot.txt', barplot, fmt='%3d', delimiter='\t')
        norm_barplot = barplot/(source_cluster_type_count.reshape(-1, 1))
        np.savetxt(save_path + '/cluster_' + str(n_clusters) + '_barplot_normalize.txt', norm_barplot, fmt='%3f', delimiter='\t')

        for clusters_i in range(n_clusters):
            plt.bar(range(n_clusters), norm_barplot[clusters_i], label='graph '+str(clusters_i))
            plt.xlabel('cell type index')
            plt.ylabel('value')
            plt.title('barplot_'+str(clusters_i))
            plt.savefig(save_path + '/barplot_sub' + str(clusters_i)+ '.jpg')
            plt.clf()

    return




def CCST_on_ST(args):
    lambda_I = args.lambda_I
    # Parameters
    batch_size = 1  # Batch size

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(device)
    adj_0, adj, X_data, cell_type_indeces = get_data(args)


    num_cell = X_data.shape[0]
    num_feature = X_data.shape[1]
    print('Adj:', adj.shape, 'Edges:', len(adj.data))
    print('X:', X_data.shape)

    n_clusters = 5 #CHANGE#max(cell_type_indeces)+1 #num_cell_types, start from 0

    print('n clusters:', n_clusters)


    X = np.ones((num_cell, 200)) #np.eye(num_cell, num_cell)
    #np.random.shuffle(X)
    num_feature = 200 #num_cell
    X_data = X


    if args.DGI and (lambda_I>=0):
        print("-----------Deep Graph Infomax-------------")
        data_list = get_graph(adj, X_data)
        data_loader = DataLoader(data_list, batch_size=batch_size)
        DGI_model = train_DGI(args, data_loader=data_loader, in_channels=num_feature)

        for data in data_loader:
            data.to(device)
            X_embedding, _, _ = DGI_model(data)
            X_embedding = X_embedding.cpu().detach().numpy()
            X_embedding_filename =  args.embedding_data_path + args.model_name + '_Embed_X.npy'
            np.save(X_embedding_filename, X_embedding)
    print("DGI is finished")
    if args.cluster:
        cluster_type = args.cluster_alg # 'leiden' #'kmeans' # 'louvain' # 'leiden'

        print("-----------Clustering-------------")
        X_embedding_filename =  args.embedding_data_path+'Embed_X.npy'
        X_embedding = np.load(X_embedding_filename)
        if cluster_type == 'kmeans':
            X_embedding = PCA_process(X_embedding, nps=30)

            #X_data_PCA = PCA_process(X_data, nps=X_embedding.shape[1])

            # concate
            #X_embedding = np.concatenate((X_embedding, X_data), axis=1)

            print('Shape of data to cluster:', X_embedding.shape)
            cluster_labels, score = Kmeans_cluster(X_embedding, n_clusters)
        else:
            results_file = args.result_path + '/adata.h5ad'
            adata = ad.AnnData(X_embedding)

            sc.tl.pca(adata, n_comps=50, svd_solver='arpack')
            sc.pp.neighbors(adata, knn=False, method='gauss', n_neighbors=15, n_pcs=50) # 20
            eval_resolution = res_search_fixed_clus(cluster_type, adata, n_clusters)
            if cluster_type == 'leiden':
                print('leiden start')
                sc.tl.leiden(adata, key_added="CCST_leiden", resolution=eval_resolution)
                cluster_labels = np.array(adata.obs['leiden'])
                print ('leiden done')
            if cluster_type == 'louvain':
                print('louvain start')
                sc.tl.louvain(adata, key_added="CCST_louvain", resolution=eval_resolution)
                cluster_labels = np.array(adata.obs['louvain'])
                print('louvain dome')

            #sc.tl.umap(adata)
            #sc.pl.umap(adata, color=['leiden'], save='_lambdaI_' + str(lambda_I) + '.png')
            adata.write(results_file)
            cluster_labels = [ int(x) for x in cluster_labels ]
            score = False

        all_data = []
        for index in range(num_cell):
            #all_data.append([index, cell_type_indeces[index], cluster_labels[index]])  # txt: cell_id, gt_labels, cluster type
            all_data.append([index,  cluster_labels[index]])   #txt: cell_id, cluster type
        np.savetxt(args.result_path+'/types.txt', np.array(all_data), fmt='%3d', delimiter='\t')


    if args.draw_map:
        print("-----------Drawing map-------------")
        draw_map(args, adj_0)











