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
parser.add_argument( '--data_path', type=str, default='/cluster/home/t116508uhn/64630/cellrangere/' , help='The path to dataset') #'/cluster/projects/schwartzgroup/fatema/pancreatic_cancer_visium/210827_A00827_0396_BHJLJTDRXY_Notta_Karen/V10M25-61_D1_PDA_64630_Pa_P_Spatial10x_new/outs/'
parser.add_argument( '--data_name', type=str, default='V10M25-61_D1_PDA_64630_Pa_P_Spatial10x_new', help='The name of dataset')
parser.add_argument( '--generated_data_path', type=str, default='generated_data/', help='The folder to store the generated data')
parser.add_argument( '--embedding_data_path', type=str, default='new_alignment/Embedding_data_ccc_rgcn/' , help='The path to attention') #'/cluster/projects/schwartzgroup/fatema/pancreatic_cancer_visium/210827_A00827_0396_BHJLJTDRXY_Notta_Karen/V10M25-61_D1_PDA_64630_Pa_P_Spatial10x_new/outs/'
args = parser.parse_args()



#from scipy import sparse
#from scipy.sparse import csr_matrix
#from scipy.stats import spearmanr, pearsonr
#from scipy.spatial import distance_matrix


threshold_distance = 4 #2 = path equally spaced
#k_nn = 4 # #5 = h
#distance_measure = 'knn'  #'threshold_dist' # <-----------
#datatype = 'path_mixture_of_distribution' #'path_equally_spaced' #
options =  'dt-path_uniform_distribution_lrc112_cp100_noise0_random_overlap_threshold_dist_cellCount5000_3dim_3patterns_temp'


pathways = [] #['1','2','3','4','5','6','7','8']
types = [] #['secreted signaling', 'secreted signaling', 'secreted signaling', 'secreted signaling', 'secreted signaling', 'secreted signaling', 'secreted signaling', 'secreted signaling']
lr_db = pd.read_csv("/cluster/home/t116508uhn/synthetic_lr_"+options+".csv")
for i in range (0, len(lr_db)):
    types.append('secreted signaling')
    pathways.append(1) 
 


gene_vs_cell = pd.read_csv('/cluster/home/t116508uhn/synthetic_gene_vs_cell_'+options+'.csv', index_col=0)  
cell_vs_gene = gene_vs_cell.transpose()

df_x=pd.read_csv('/cluster/home/t116508uhn/synthetic_cell_'+options+'_x.csv',header=None)
df_y=pd.read_csv('/cluster/home/t116508uhn/synthetic_cell_'+options+'_y.csv',header=None)
coordinate_synthetic = np.zeros((cell_vs_gene.shape[0],2))
for i in range (0, len(df_x)):
    coordinate_synthetic[i][0] = df_x[0][i]
    coordinate_synthetic[i][1] = df_y[0][i]

distance_matrix = euclidean_distances(coordinate_synthetic, coordinate_synthetic)




spatial_dict = dict()
spatial_dict['spatial'] = coordinate_synthetic
adata_synthetic = anndata.AnnData(cell_vs_gene, obsm=spatial_dict)

adata_synthetic.var_names_make_unique()
adata_synthetic.raw = adata_synthetic
#sc.pp.normalize_total(adata_synthetic, inplace=True)
#sc.pp.log1p(adata_synthetic)


   
lr_db['pathways'] = pathways
lr_db['type'] = types

ct.tl.spatial_communication(adata_synthetic, database_name='syndb', df_ligrec=lr_db, dis_thr=threshold_distance, heteromeric=True, pathway_sum=True) #11.05
adata_synthetic.write("/cluster/projects/schwartzgroup/fatema/syn_"+options+"_commot_adata.h5ad")
adata_synthetic = sc.read_h5ad("/cluster/projects/schwartzgroup/fatema/syn_"+options+"_commot_adata.h5ad")
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
LR_pairs = []
for i in range (0, len(lr_db)):
    LR_pairs.append(lr_db['ligand'][i]+'-'+lr_db['receptor'][i])
   
#LR_pairs = ['g0-g8', 'g1-g9', 'g2-g10', 'g3-g11', 'g4-g12', 'g5-g13', 'g6-g14', 'g7-g15']

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
    print('%d, size %d'%(pair_index, len(distribution)))
    for i in range (0, datapoint_size):
        for j in range (0, datapoint_size):
            if distance_matrix[i,j] > threshold_distance:
                continue
            if adata_synthetic.obsp[key_pair][i,j]>0:
                attention_scores[i][j].append(adata_synthetic.obsp[key_pair][i,j])
                lig_rec_dict[i][j].append(pair_index)
                distribution.append(adata_synthetic.obsp[key_pair][i,j])
            
            
with gzip.open("/cluster/projects/schwartzgroup/fatema/find_ccc/" + 'synthetic_data_'+options+'_commot_result2', 'wb') as fp:
    pickle.dump([attention_scores, lig_rec_dict, distribution], fp)            

#with gzip.open("/cluster/projects/schwartzgroup/fatema/find_ccc/" + 'synthetic_data_type4_e_commot_result', 'wb') as fp:
#    pickle.dump([attention_scores, lig_rec_dict, distribution], fp)            

#######################################################################################
import stlearn as st
import argparse
parser = argparse.ArgumentParser()
parser.add_argument( '--data_path', type=str, default='/cluster/projects/schwartzgroup/fatema/data/V1_Human_Lymph_Node_spatial/' , help='The path to dataset') 
parser.add_argument( '--embedding_data_path', type=str, default='new_alignment/Embedding_data_ccc_rgcn/' , help='The path to attention') #'/cluster/projects/schwartzgroup/fatema/pancreatic_cancer_visium/210827_A00827_0396_BHJLJTDRXY_Notta_Karen/V10M25-61_D1_PDA_64630_Pa_P_Spatial10x_new/outs/'
parser.add_argument( '--data_name', type=str, default='V1_Human_Lymph_Node_spatial', help='The name of dataset')
parser.add_argument( '--model_name', type=str, default='gat_r1_2attr', help='model name')
parser.add_argument( '--slice', type=int, default=0, help='starting index of ligand')
args = parser.parse_args()

 
adata = st.Read10X(path=args.data_path, count_file='filtered_feature_bc_matrix.h5') #count_file=args.data_name+'_filtered_feature_bc_matrix.h5' )
print(adata)

adata.var_names_make_unique()
adata.raw = adata
sc.pp.normalize_total(adata, inplace=True)
sc.pp.log1p(adata)

df_cellchat = ct.pp.ligand_receptor_database(species='human', signaling_type='Secreted Signaling', database='CellChat')
df_cellchat_filtered = ct.pp.filter_lr_database(df_cellchat, adata, min_cell_pct=0.05)
ct.tl.spatial_communication(adata, database_name='cellchat', df_ligrec=df_cellchat_filtered, dis_thr=500, heteromeric=True, pathway_sum=True)
#adata.write("/cluster/projects/schwartzgroup/fatema/"+args.data_name+"_commot_adata.h5ad")
adata = sc.read_h5ad("/cluster/projects/schwartzgroup/fatema/"+args.data_name+"_commot_adata.h5ad")


ct.tl.communication_direction(adata, database_name='cellchat',  lr_pair=['CCL19','CCR7'],  k=5) #pathway_name='CCL',
ct.pl.plot_cell_communication(adata, database_name='cellchat', lr_pair=['CCL19','CCR7'], plot_method='grid', background_legend=True,
    scale=0.00003, ndsize=8, grid_density=0.4, summary='sender', background='image', clustering='leiden', cmap='Alphabet',
    normalize_v = True, normalize_v_quantile=0.995, filename='PSAP_cluster.pdf') #pathway_name='CCL', 




#ct.tl.spatial_communication(adata, database_name='syndb', df_ligrec=lr_db, dis_thr=500, heteromeric=True, pathway_sum=True)
#adata_synthetic.write("/cluster/projects/schwartzgroup/fatema/"+args.data_name+"_commot_adata.h5ad")

#adata_synthetic = sc.read_h5ad("/cluster/projects/schwartzgroup/fatema/syn_type6_f_commot_adata.h5ad")


