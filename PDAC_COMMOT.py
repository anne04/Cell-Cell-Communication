# This script will take very high memory to run
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
from sklearn.metrics.pairwise import euclidean_distances
import scanpy as sc
import commot as ct
import gc
import ot
import anndata
import argparse

parser = argparse.ArgumentParser()
parser.add_argument( '--data_path', type=str, default='/cluster/projects/schwartzgroup/fatema/pancreatic_cancer_visium/210827_A00827_0396_BHJLJTDRXY_Notta_Karen/V10M25-61_D1_PDA_64630_Pa_P_Spatial10x_new/outs/' , help='The path to dataset') 
parser.add_argument( '--data_name', type=str, default='PDAC_64630', help='The name of dataset')
args = parser.parse_args()

parser = argparse.ArgumentParser()
parser.add_argument( '--data_path', type=str, default='/cluster/projects/schwartzgroup/fatema/pancreatic_cancer_visium/exp1_C1/outs/' , help='The path to dataset') 
parser.add_argument( '--data_name', type=str, default='PDAC_140694', help='The name of dataset')
args = parser.parse_args()

threshold_distance = 500


adata = st.Read10X(path=args.data_path, count_file='filtered_feature_bc_matrix.h5') 
print(adata)

adata.var_names_make_unique()
adata.raw = adata
sc.pp.normalize_total(adata, inplace=True)
sc.pp.log1p(adata)

df_cellchat = ct.pp.ligand_receptor_database(species='human', signaling_type='Secreted Signaling', database='CellChat')
df_cellchat_filtered = ct.pp.filter_lr_database(df_cellchat, adata, min_cell_pct=0.05)
ct.tl.spatial_communication(adata, database_name='cellchat', df_ligrec=df_cellchat_filtered, dis_thr=threshold_distance, heteromeric=True, pathway_sum=True)
print('data write')
adata.write(args.data_name+"_commot_adata.h5ad")
print('data read')
adata = sc.read_h5ad(args.data_name+"_commot_adata.h5ad")

################################################# retrieve the scores for only Classical regions ########################## 
classical = []
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
    print('%d, size %d, matrix %d'%(pair_index, len(distribution), np.max(adata_synthetic.obsp[key_pair])))
    for i in range (0, datapoint_size):
        if i not in classical:
            continue
        for j in range (0, datapoint_size):
            if j not in classical:
                continue
            if distance_matrix[i,j] > threshold_distance: 
                continue
            if adata_synthetic.obsp[key_pair][i,j]>0:
                attention_scores[i][j].append(adata_synthetic.obsp[key_pair][i,j])
                lig_rec_dict[i][j].append(pair_index)
                distribution.append(adata_synthetic.obsp[key_pair][i,j])
            
            
with gzip.open("/cluster/projects/schwartzgroup/fatema/find_ccc/" + args.data_name + '_classical_commot_result', 'wb') as fp:
    pickle.dump([attention_scores, lig_rec_dict, distribution], fp)            






ct.tl.communication_direction(adata, database_name='cellchat',  lr_pair=['PLXNB2','MET'],  k=5) #pathway_name='CCL',
print('Plot the CCC')
ct.pl.plot_cell_communication(adata, database_name='cellchat', lr_pair=['CCL19','CCR7'], plot_method='grid', background_legend=True,
    scale=0.00003, ndsize=8, grid_density=0.4, summary='sender', background='image', clustering='leiden', cmap='Alphabet',
    normalize_v = True, normalize_v_quantile=0.995, filename='PSAP_cluster.pdf') #pathway_name='CCL', 


