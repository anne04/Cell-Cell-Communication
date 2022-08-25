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
parser.add_argument( '--data_name', type=str, default='V10M25-060_A1_PDA_128033_Pa_R_Spatial10x_new', help="'MERFISH' or 'V1_Breast_Cancer_Block_A_Section_1")
parser.add_argument( '--lambda_I', type=float, default=0.3) #0.8 on MERFISH, 0.3 on ST
parser.add_argument( '--data_path', type=str, default='generated_data_new/', help='data path')
parser.add_argument( '--model_path', type=str, default='model')
parser.add_argument( '--embedding_data_path', type=str, default='Embedding_data')
parser.add_argument( '--result_path', type=str, default='results')
parser.add_argument( '--DGI', type=int, default=0, help='run Deep Graph Infomax(DGI) model, otherwise direct load embeddings, 0 or 1')
parser.add_argument( '--load', type=int, default=0, help='Load pretrained DGI model')
parser.add_argument( '--num_epoch', type=int, default=5000, help='numebr of epoch in training DGI')
parser.add_argument( '--hidden', type=int, default=256, help='hidden channels in DGI')
parser.add_argument( '--PCA', type=int, default=1, help='run PCA or not')
parser.add_argument( '--cluster', type=int, default=1, help='run cluster or not')
parser.add_argument( '--n_clusters', type=int, default=5, help='number of clusters in Kmeans, when ground truth label is not avalible.') #5 on MERFISH, 20 on Breast
parser.add_argument( '--draw_map', type=int, default=1, help='run drawing map')
parser.add_argument( '--diff_gene', type=int, default=0, help='Run differential gene expression analysis')
parser.add_argument( '--cluster_alg', type=str, default='leiden' , help='Run which clustering at the end')
args = parser.parse_args()

args.embedding_data_path = args.embedding_data_path +'/'+ args.data_name +'/'
args.result_path = args.result_path +'/'+ args.data_name +'/'




n_clusters = args.n_clusters

print("-----------Clustering-------------")
lambda_I=args.lambda_I


data_file = args.data_path + args.data_name +'/'
with open(data_file + 'features', 'rb') as fp:
    X_data = pickle.load(fp)

print('X_data shape: ',X_data.shape)


X_embedding_filename =  args.embedding_data_path+'lambdaI' + str(lambda_I) + '_Embed_X.npy'
X_embedding = np.load(X_embedding_filename)
print('X_embedding shape: ',X_embedding.shape)




num_cell=X_embedding.shape[0]
print("numcells %d"%num_cell)

cluster_type=args.cluster_alg
print(cluster_type)
if cluster_type == 'kmeans':
    print('kmeans')
    X_embedding = PCA_process(X_embedding, nps=50)
    #X_data_PCA = PCA_process(X_data, nps=X_embedding.shape[1])
    # concate
    #X_embedding = np.concatenate((X_embedding, X_data), axis=1)
    print('Shape of data to cluster:', X_embedding.shape)
    cluster_labels, score = Kmeans_cluster(X_embedding, n_clusters)
else:
    results_file = args.result_path + '/adata.h5ad'
    if args.DGI==0:
        print('Using only gene expression for clustering')
        adata = ad.AnnData(X_data)
    else:
        adata = ad.AnnData(X_embedding)

#    sc.tl.pca(adata, n_comps=70, svd_solver='arpack')
    sc.pp.neighbors(adata, knn=False, method='gauss', use_rep='X', n_neighbors=15) #, metric='manhattan')
#    sc.pp.neighbors(adata, knn=False, method='gauss', n_neighbors=5, n_pcs=50) # 20 #use_rep='X'
#    sc.pp.neighbors(adata, n_pcs=50) # 20 #use_rep='X'
#    eval_resolution = res_search_fixed_clus(cluster_type, adata, n_clusters)
    if cluster_type == 'leiden':
        print('leiden')
        print(adata.shape)
        sc.tl.leiden(adata, directed=False, resolution=1)
#        sc.tl.leiden(adata, key_added="CCST_leiden", resolution=eval_resolution)
        cluster_labels = np.array(adata.obs['leiden'])
    if cluster_type == 'louvain':
        print('louvain')
        sc.tl.louvain(adata, key_added="CCST_louvain", resolution=eval_resolution)
        cluster_labels = np.array(adata.obs['louvain'])
    #sc.tl.umap(adata)
    #sc.pl.umap(adata, color=['leiden'], save='_lambdaI_' + str(lambda_I) + '.png')
    adata.write(results_file)
    cluster_labels = [ int(x) for x in cluster_labels ]
    score = False

all_data = np.zeros((num_cell,2))

for index in range (0, num_cell):
    all_data[index,0] = index
    all_data[index,1] = cluster_labels[index]   #txt: cell_id, cluster type

#np.savetxt(X_embedding_filename, X_embedding_T, delimiter=",")

np.savetxt(args.result_path+'/cluster_label.csv', all_data, fmt='%3d', delimiter=',')



'''import csv
barcode_file='/cluster/home/t116508uhn/64630/barcodes.tsv'
barcode_info=[]
with open(barcode_file) as file:
    tsv_file = csv.reader(file, delimiter="\t")
    for line in tsv_file:
        barcode_info.append(line)

for index in range (0, num_cell):
    all_data[index,0] = barcode_info[index][0]

np.savetxt(args.result_path+'/barcode_label.csv', all_data, delimiter=',')'''



data_folder = args.data_path + args.data_name+'/'
coordinates = np.load(data_folder+'coordinates.npy')

unique_labels=list(np.unique(cluster_labels))
for label in unique_labels:
    x_index=[]
    y_index=[]
    for i in range (0, len(cluster_labels)):
        if cluster_labels[i]==label:
           x_index.append(coordinates[i,0])
           y_index.append(coordinates[i,1])

    plt.scatter(x=np.array(x_index),y=-np.array(y_index) , label = label, s=5)

plt.legend()

save_path = args.result_path
plt.savefig(save_path+'/spatial_'+cluster_type+'_'+args.data_name+'_test_'+str(args.n_clusters)+'.png', dpi=400)
plt.clf()


os.popen('cp '+save_path+'/spatial_'+cluster_type+'_'+args.data_name+'_test_'+str(args.n_clusters)+'.png /cluster/home/t116508uhn/data/CCST/spatial_'+cluster_type+'_'+args.data_name+'_test_'+str(args.n_clusters)+'.png')
