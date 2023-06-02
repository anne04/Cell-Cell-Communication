import os
#import glob 
import pandas as pd
#import shutil
import copy
import csv
import numpy as np
import sys
import altair as alt
from collections import defaultdict
import stlearn as st
import scanpy as sc
import qnorm
import scipy
import pickle
import gzip
import matplotlib.pyplot as plt
from scipy import sparse
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components
from sklearn.metrics.pairwise import euclidean_distances
from kneed import KneeLocator
import argparse
parser = argparse.ArgumentParser()
parser.add_argument( '--data_path', type=str, default='/cluster/home/t116508uhn/64630/cellrangere/' , help='The path to dataset') #'/cluster/projects/schwartzgroup/fatema/pancreatic_cancer_visium/210827_A00827_0396_BHJLJTDRXY_Notta_Karen/V10M25-61_D1_PDA_64630_Pa_P_Spatial10x_new/outs/'
parser.add_argument( '--data_name', type=str, default='V10M25-61_D1_PDA_64630_Pa_P_Spatial10x_new', help='The name of dataset')
parser.add_argument( '--generated_data_path', type=str, default='generated_data/', help='The folder to store the generated data')
parser.add_argument( '--embedding_data_path', type=str, default='new_alignment/Embedding_data_ccc_rgcn/' , help='The path to attention') #'/cluster/projects/schwartzgroup/fatema/pancreatic_cancer_visium/210827_A00827_0396_BHJLJTDRXY_Notta_Karen/V10M25-61_D1_PDA_64630_Pa_P_Spatial10x_new/outs/'
args = parser.parse_args()


import os
import gc
import ot
import pickle
import anndata
import pandas as pd
from scipy import sparse
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components
from scipy.stats import spearmanr, pearsonr
from scipy.spatial import distance_matrix
import commot as ct

threshold_distance = 2 #2 = path equally spaced
k_nn = 10 # #5 = h
distance_measure = 'knn'  #'threshold_dist' # <-----------
datatype = 'path_mixture_of_distribution' #'path_equally_spaced' #

options = 'dt-path_mixture_of_distribution_lrc8_cp100_noise0_random_overlap_knn_cellCount2534_f_3dim'

gene_vs_cell = pd.read_csv('/cluster/home/t116508uhn/synthetic_gene_vs_cell_type6_f.csv', index_col=0)  
cell_vs_gene = gene_vs_cell.transpose()
df_x=pd.read_csv('/cluster/home/t116508uhn/synthetic_cell_type6_f_x.csv',header=None)
df_y=pd.read_csv('/cluster/home/t116508uhn/synthetic_cell_type6_f_y.csv',header=None)
coordinate_synthetic = np.zeros((cell_vs_gene.shape[0],2))
for i in range (0, len(df_x)):
    coordinate_synthetic[i][0] = df_x[0][i]
    coordinate_synthetic[i][1] = df_y[0][i]

spatial_dict = dict()
spatial_dict['spatial'] = coordinate_synthetic
adata_synthetic = anndata.AnnData(cell_vs_gene, obsm=spatial_dict)

adata_synthetic.var_names_make_unique()
adata_synthetic.raw = adata_synthetic
sc.pp.normalize_total(adata_synthetic, inplace=True)
sc.pp.log1p(adata_synthetic)


lr_db = pd.read_csv("/cluster/home/t116508uhn/synthetic_lr_type6_f.csv")
pathways = ['1','2','3','4','5','6','7','8']
types = ['secreted signaling', 'secreted signaling', 'secreted signaling', 'secreted signaling', 'secreted signaling', 'secreted signaling', 'secreted signaling', 'secreted signaling']
lr_db['pathways'] = pathways
lr_db['type'] = types

ct.tl.spatial_communication(adata_synthetic, database_name='syndb', df_ligrec=lr_db, dis_thr=12, heteromeric=True, pathway_sum=True)
adata_synthetic.write("/cluster/projects/schwartzgroup/fatema/syn_type6_f_commot_adata.h5ad")
adata_synthetic = sc.read_h5ad("/cluster/projects/schwartzgroup/fatema/syn_type6_f_commot_adata.h5ad")
###########################################
'''
options = 'dt-path_equally_spaced_lrc8_cp100_noise0_random_overlap_threshold_dist_cellCount3000_e_3dim'
gene_vs_cell = pd.read_csv('/cluster/home/t116508uhn/synthetic_type4_e_gene_vs_cell.csv', index_col=0)  
df_x=pd.read_csv('/cluster/home/t116508uhn/synthetic_cell_type4_e_x.csv',header=None)
df_y=pd.read_csv('/cluster/home/t116508uhn/synthetic_cell_type4_e_y.csv',header=None)
lr_db = pd.read_csv("/cluster/home/t116508uhn/synthetic_lr_type4_e.csv")
cell_vs_gene = gene_vs_cell.transpose()
coordinate_synthetic = np.zeros((cell_vs_gene.shape[0],2))
for i in range (0, len(df_x)):
    coordinate_synthetic[i][0] = df_x[0][i]
    coordinate_synthetic[i][1] = df_y[0][i]

spatial_dict = dict()
spatial_dict['spatial'] = coordinate_synthetic
adata_synthetic = anndata.AnnData(cell_vs_gene, obsm=spatial_dict)

adata_synthetic.var_names_make_unique()
adata_synthetic.raw = adata_synthetic
sc.pp.normalize_total(adata_synthetic, inplace=True)
sc.pp.log1p(adata_synthetic)

pathways = ['1','2','3','4','5','6','7','8']
types = ['secreted signaling', 'secreted signaling', 'secreted signaling', 'secreted signaling', 'secreted signaling', 'secreted signaling', 'secreted signaling', 'secreted signaling']
lr_db['pathways'] = pathways
lr_db['type'] = types

ct.tl.spatial_communication(adata_synthetic, database_name='syndb', df_ligrec=lr_db, dis_thr=10, heteromeric=True, pathway_sum=True)
adata_synthetic.write("/cluster/projects/schwartzgroup/fatema/syn_type4_e_commot_adata.h5ad")
'''
#########################################################

LR_pairs = ['g0-g8', 'g1-g9', 'g2-g10', 'g3-g11', 'g4-g12', 'g5-g13', 'g6-g14', 'g7-g15']

#################################################################################
attention_scores = []
lig_rec_dict = []
datapoint_size = cell_vs_gene.shape[0]
for i in range (0, datapoint_size):
    attention_scores.append([])   
    lig_rec_dict.append([])   
    for j in range (0, datapoint_size):	
        attention_scores[i].append([])   
        attention_scores[i][j] = []
        lig_rec_dict[i].append([])   
        lig_rec_dict[i][j] = []

distribution = []
for pair_index in range(0, len(LR_pairs)):
    pair = LR_pairs[pair_index]
    key_pair = 'commot-syndb-' + pair
    for i in range (0, datapoint_size):
        for j in range (0, datapoint_size):
            if adata_synthetic.obsp[key_pair][i,j]>0:
                attention_scores[i][j].append(adata_synthetic.obsp[key_pair][i,j])
                lig_rec_dict[i][j].append(pair_index)
                distribution.append(adata_synthetic.obsp[key_pair][i,j])
            
            
with gzip.open("/cluster/projects/schwartzgroup/fatema/find_ccc/" + 'synthetic_data_type6_f_commot_result', 'wb') as fp:
    pickle.dump([attention_scores, lig_rec_dict, distribution], fp)            

with gzip.open("/cluster/projects/schwartzgroup/fatema/find_ccc/" + 'synthetic_data_type4_e_commot_result', 'wb') as fp:
    pickle.dump([attention_scores, lig_rec_dict, distribution], fp)            
    
    
