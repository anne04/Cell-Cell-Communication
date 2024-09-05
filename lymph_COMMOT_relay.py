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
parser.add_argument( '--data_path', type=str, default='/cluster/projects/schwartzgroup/fatema/data/V1_Human_Lymph_Node_spatial/' , help='The path to dataset') 
parser.add_argument( '--data_name', type=str, default='V1_Human_Lymph_Node_spatial', help='The name of dataset')
#parser.add_argument( '--top_count', type=int, default=1300, help='The name of dataset')
parser.add_argument( '--annotation_file_path', type=str, default='NEST_figures_input_human_lymph/V1_Human_Lymph_Node_spatial_annotation.csv', help='Path to load the annotation file in csv format (if available) ') 
args = parser.parse_args()

threshold_distance = 500

adata = st.Read10X(path=args.data_path, count_file='filtered_feature_bc_matrix.h5') 
print(adata)

cell_barcode = np.array(adata.obs.index)
datapoint_size = len(cell_barcode)
if args.annotation_file_path != '':
    annotation_data = pd.read_csv(args.annotation_file_path, sep=",")
    pathologist_label=[]
    for i in range (0, len(annotation_data)):
        pathologist_label.append([annotation_data['Barcode'][i], annotation_data['Type'][i]])

    barcode_type=dict() # record the type (annotation) of each spot (barcode)
    for i in range (0, len(pathologist_label)):
        barcode_type[pathologist_label[i][0]] = pathologist_label[i][1]


###################################################################################
'''
adata.var_names_make_unique()
adata.raw = adata
sc.pp.normalize_total(adata, inplace=True)
sc.pp.log1p(adata)

df_cellchat = ct.pp.ligand_receptor_database(species='human', signaling_type='Secreted Signaling', database='CellChat')
df_cellchat_filtered = ct.pp.filter_lr_database(df_cellchat, adata, min_cell_pct=0.05)
ct.tl.spatial_communication(adata, database_name='cellchat', df_ligrec=df_cellchat_filtered, dis_thr=threshold_distance, heteromeric=True, pathway_sum=True)
print('data write')
adata.write(path + args.data_name+"_commot_adata.h5ad")
'''
print('data read')
adata = sc.read_h5ad('/cluster/home/t116508uhn/NEST_figures_input/NEST_figures_input_human_lymph/' + args.data_name+"_commot_adata.h5ad")

attention_scores = []
lig_rec_dict = []

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
print('total pairs %d'%len(LR_pair))
for pair_index in range(0, len(LR_pair)):
    key_pair = LR_pair[pair_index]
    pairs = key_pair.split('-')[2:]
    if len(pairs) < 2: # it means it is a pathway, not a LR pair
        continue
    print('%d, size %d, matrix %g'%(pair_index, len(distribution), np.max(adata.obsp[key_pair])))
    for i in range (0, datapoint_size):
        for j in range (0, datapoint_size):
            if adata.obsp[key_pair][i,j]>0:
                attention_scores[i][j].append(adata.obsp[key_pair][i,j])
                lig_rec_dict[i][j].append(pairs[0] + '-' + pairs[1])
                distribution.append(adata.obsp[key_pair][i,j])
            

with gzip.open("/cluster/projects/schwartzgroup/fatema/find_ccc/" + args.data_name + '_commot_result', 'wb') as fp:
    pickle.dump([attention_scores, lig_rec_dict, distribution], fp)            

##################################################################################################################
with gzip.open("/cluster/projects/schwartzgroup/fatema/find_ccc/" + args.data_name + '_commot_result', 'rb') as fp:
    attention_scores, lig_rec_dict, distribution = pickle.load(fp)            

csv_record_final = []
# columns are: from_cell, to_cell, ligand_gene, receptor_gene, rank, component, from_id, to_id, attention_score
# keep only top 20% connections
top20 = np.percentile(distribution, 80)
top_hist = defaultdict(list)
for i in range (0, datapoint_size):
    for j in range (0, datapoint_size):
        for k in range (0, len(attention_scores[i][j])):
            score = attention_scores[i][j][k]
            lr = lig_rec_dict[i][j][k]
            ligand = lr.split('-')[0]
            receptor = lr.split('-')[1]
            if ligand == 'total':
                continue
            if score >= top20:
                csv_record_final.append([cell_barcode[i], cell_barcode[j], ligand, receptor, -1, -1, i, j, score])
                
csv_record_final = sorted(csv_record_final, key = lambda x: x[8], reverse=True) # high to low based on 'score'
csv_record_final = csv_record_final[0:args.top_count]
####################### pattern finding ##########################################################################
# make a dictionary to keep record of all the outgoing edges [to_node, ligand, receptor] for each node
each_node_outgoing = defaultdict(list)
for k in range (0, len(csv_record_final)): # last record is a dummy for histogram preparation
    i = csv_record_final[k][6]
    j = csv_record_final[k][7]
    if i == j:
        continue        
    ligand = csv_record_final[k][2]
    receptor = csv_record_final[k][3]
    each_node_outgoing[i].append([j, ligand, receptor, k]) 

# all possible 2-hop pattern count
pattern_distribution = defaultdict(list)
# pattern_distribution['ligand-receptor to ligand-receptor']=[1,1,1,1, ...]
edge_list_2hop = []
#target_relay = 'CCL21-CXCR4 to CCL21-CXCR4'
for i in each_node_outgoing:
    for tupple in each_node_outgoing[i]: # first hop
        j = tupple[0]
        lig_rec_1 = tupple[1]+'-'+tupple[2]
        record_id_1 = tupple[3]
        if j in each_node_outgoing:
            for tupple_next in each_node_outgoing[j]: # second hop
                k = tupple_next[0]
                if k == i or k == j:
                    continue
                lig_rec_2 = tupple_next[1]+'-'+tupple_next[2]
                record_id_2 = tupple_next[3]
                pattern_distribution[lig_rec_1 + ' to ' + lig_rec_2].append(1)
                relay = lig_rec_1 + ' to ' + lig_rec_2
                #if relay == target_relay:
                #    edge_list_2hop.append([record_id_1,record_id_2])



two_hop_pattern_distribution = []
same_count = 0
for key in pattern_distribution:
    count = len(pattern_distribution[key]) 
    two_hop_pattern_distribution.append([key, count]) 
    #if lig_rec_1 == lig_rec_2:
    #    same_count = same_count + 1

two_hop_pattern_distribution = sorted(two_hop_pattern_distribution, key = lambda x: x[1], reverse=True) # high to low

data_list=dict()
data_list['X']=[]
data_list['Y']=[] 
for i in range (0, len(two_hop_pattern_distribution)):
    data_list['X'].append(two_hop_pattern_distribution[i][0])
    data_list['Y'].append(two_hop_pattern_distribution[i][1])
    
data_list_pd = pd.DataFrame({
    'Relay Patterns': data_list['X'],
    'Pattern Abundance (#)': data_list['Y']
})

chart = alt.Chart(data_list_pd).mark_bar().encode(
    x=alt.X("Relay Patterns:N", axis=alt.Axis(labelAngle=45), sort='-y'),
    y='Pattern Abundance (#)'
)

chart.save("/cluster/projects/schwartzgroup/fatema/find_ccc/" + args.data_name +'_COMMOT_pattern_distribution.html')
###############


#############################################################
