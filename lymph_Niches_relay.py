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

######## read the top5 edges (ccc) by Niches ########################################
datapoint_size = 
attention_scores_temp = []
lig_rec_dict_temp = []
datapoint_size = temp_x.shape[0]
for i in range (0, datapoint_size):
    attention_scores_temp.append([])   
    lig_rec_dict_temp.append([])   
    for j in range (0, datapoint_size):	
        attention_scores_temp[i].append([])   
        attention_scores_temp[i][j] = []
        lig_rec_dict_temp[i].append([])   
        lig_rec_dict_temp[i][j] = []
        

marker_list = pd.read_csv('/cluster/home/t116508uhn/niches_output_ccc_lr_pairs_markerList_top5_'+options+'.csv')
marker_list = marker_list.sort_values(by=['myAUC'], ascending=False) #marker_list.sort_values(by=['avg_log2FC'], ascending=False) # high fc to low fc
positive_class_found = 0
distribution_temp = []
total_edge_count = 0
flag_break = 0
for index in range (0, len(marker_list.index)):
    cluster_type = marker_list['cluster'][index]
    pair_type = marker_list['gene'][index]
    ligand_gene = pair_type.split('—')[0]
    receptor_gene = pair_type.split('—')[1]
    ligand_gene = int(ligand_gene.split('g')[1])
    receptor_gene = int(receptor_gene.split('g')[1])
    lr_pair_id = ligand_dict_dataset[ligand_gene][receptor_gene] 

    edge_list = clusterType_edge_dictionary[cluster_type]
    for edge in edge_list:
        ccc_score_scaled = edge_pair_dictionary[edge][lr_pair_id]
        i = int(edge.split('-')[0])
        j = int(edge.split('-')[1])
        total_edge_count = total_edge_count + 1
        if total_edge_count > len(row_col):
            flag_break = 1
            break

        lig_rec_dict_temp[i][j].append(lr_pair_id)
        attention_scores_temp[i][j].append(ccc_score_scaled)
        distribution_temp.append(ccc_score_scaled)
	    
 
    if flag_break == 1:
        break

            
\
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

chart.save(args.data_name +'_COMMOT_pattern_distribution.html')
###############


#############################################################
