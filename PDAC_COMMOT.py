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
path = '/cluster/projects/schwartzgroup/fatema/CCC_project/'

adata = st.Read10X(path=args.data_path, count_file='filtered_feature_bc_matrix.h5') 
print(data)

cell_barcode = np.array(adata.obs.index)
cell_barcode_index = dict()
for i in range (0, len(cell_barcode)):
    cell_barcode_index[cell_barcode[i]] = i

adata.var_names_make_unique()
adata.raw = adata
sc.pp.normalize_total(adata, inplace=True)
sc.pp.log1p(adata)

df_cellchat = ct.pp.ligand_receptor_database(species='human', signaling_type='Secreted Signaling', database='CellChat')
df_cellchat_filtered = ct.pp.filter_lr_database(df_cellchat, adata, min_cell_pct=0.05)
ct.tl.spatial_communication(adata, database_name='cellchat', df_ligrec=df_cellchat_filtered, dis_thr=threshold_distance, heteromeric=True, pathway_sum=True)
print('data write')
adata.write(path + args.data_name+"_commot_adata.h5ad")
print('data read')
adata = sc.read_h5ad(path + args.data_name+"_commot_adata.h5ad")

################################################# retrieve the scores for only Classical regions ########################## 
cell_vs_cluster = pd.read_csv('/cluster/projects/schwartzgroup/fatema/CCC_project/pdac_64630_niches_seurat_barcode_vs_cluster.csv', index_col=0)
#cell_vs_cluster = pd.read_csv('/cluster/projects/schwartzgroup/fatema/CCC_project/pdac_140694_niches_seurat_barcode_vs_cluster.csv', index_col=0)

# In [33]: cell_vs_cluster['seurat_clusters'][0]
# Out[33]: 5
# In [36]: cell_vs_cluster['seurat_clusters'].index[0]
# Out[36]: 'AAACCGGGTAGGTACC-1'

classical = []
for i in range (0, cell_vs_cluster.shape[0]):
    if cell_vs_cluster['seurat_clusters'][i] == 7:
        classical.append(cell_barcode_index[cell_vs_cluster['seurat_clusters'].index[i]])

attention_scores = []
lig_rec_dict = []
datapoint_size = cell_vs_cluster.shape[0]
for i in range (0, datapoint_size):
    attention_scores.append([])   
    lig_rec_dict.append([])   
    for j in range (0, datapoint_size):	
        attention_scores[i].append([])   
        attention_scores[i][j] = []
        lig_rec_dict[i].append([])   
        lig_rec_dict[i][j] = []

distribution = []
LR_pair = list(adata.obsp.keys())
for pair_index in range(0, len(LR_pair)):
    key_pair = LR_pair[pair_index]
    pairs = key_pair.split('-')[2:]
    if len(pairs) < 2: # it means it is a pathway, not a LR pair
        continue
    print('%d, size %d, matrix %g'%(pair_index, len(distribution), np.max(adata.obsp[key_pair])))
    
    for i in range (0, datapoint_size):
        if i not in classical:
            continue
        for j in range (0, datapoint_size):
            if j not in classical:
                continue
            if adata.obsp[key_pair][i,j]>0:
                attention_scores[i][j].append(adata.obsp[key_pair][i,j])
                lig_rec_dict[i][j].append(pairs[0] + '-' + pairs[1])
                distribution.append(adata.obsp[key_pair][i,j])
            
            
#with gzip.open("/cluster/projects/schwartzgroup/fatema/find_ccc/" + args.data_name + '_classical_commot_result', 'wb') as fp:
#    pickle.dump([attention_scores, lig_rec_dict, distribution], fp)            



top20 = np.percentile(distribution, 80)
top_hist = defaultdict(list)
for i in range (0, datapoint_size):
    if i not in classical:
        continue
    for j in range (0, datapoint_size):
        if j not in classical:
            continue 
        for k in range (0, len(attention_scores[i][j])):
            score = attention_scores[i][j][k]
            if score >= top20:
                top_hist[lig_rec_dict[i][j][k]].append('')
        
for key in top_hist:
    top_hist[key] = len(top_hist[key])


key_distribution = []
same_count = 0
for key in top_hist:
    count = top_hist[key]
    key_distribution.append([key, count]) 

key_distribution = sorted(key_distribution, key = lambda x: x[1], reverse=True) # high to low


data_list=dict()
data_list['X']=[]
data_list['Y']=[] 
for i in range (0, 100): 
    data_list['X'].append(key_distribution[i][0])
    data_list['Y'].append(key_distribution[i][1])
    
data_list_pd = pd.DataFrame({
    'Ligand-Receptor': data_list['X'],
    'Communication (#)': data_list['Y']
})

chart = alt.Chart(data_list_pd).mark_bar().encode(
    x=alt.X("Ligand-Receptor:N", axis=alt.Axis(labelAngle=45), sort='-y'),
    y='Communication (#)'
)

chart.save(output_name + args.data_name +'_commot_classical_ccc.html')
