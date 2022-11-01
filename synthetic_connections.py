
import numpy as np
import csv
import pickle
from scipy import sparse
import scipy.io as sio
import scanpy as sc
import matplotlib
matplotlib.use('Agg')
import gzip
import matplotlib.pyplot as plt
import stlearn as st
import numpy as np
from matplotlib.colors import LinearSegmentedColormap, to_hex, rgb2hex
from typing import List
import qnorm
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components
from collections import defaultdict
import pandas as pd

import argparse
parser = argparse.ArgumentParser()
parser.add_argument( '--data_path', type=str, default='/cluster/home/t116508uhn/64630/spaceranger_output_new/' , help='The path to dataset') 
parser.add_argument( '--embedding_data_path', type=str, default='new_alignment/Embedding_data_ccc_rgcn/' , help='The path to attention') #'/cluster/projects/schwartzgroup/fatema/pancreatic_cancer_visium/210827_A00827_0396_BHJLJTDRXY_Notta_Karen/V10M25-61_D1_PDA_64630_Pa_P_Spatial10x_new/outs/'
parser.add_argument( '--data_name', type=str, default='V10M25-61_D1_PDA_64630_Pa_P_Spatial10x_new', help='The name of dataset')
parser.add_argument( '--model_name', type=str, default='gat_r1_2attr', help='model name')
args = parser.parse_args()



#############################
with gzip.open("/cluster/projects/schwartzgroup/fatema/find_ccc/" + 'synthetic_ccc_region_1', 'rb') as fp:
    cell_vs_gene, region_list, ligand_list, activated_cell, gene_ids = pickle.load(fp)

#############################
gene_file='/cluster/home/t116508uhn/64630/spaceranger_output_new/unzipped/features.tsv' # 1406
gene_info=dict()
for gene in range gene_ids:
    gene_info[gene]=''
    
#############################
ligand_dict_dataset = defaultdict(list)
ligand_dict_db = defaultdict(list)
cell_chat_file = '/cluster/home/t116508uhn/64630/Human-2020-Jin-LR-pairs_cellchat.csv'
df = pd.read_csv(cell_chat_file)
cell_cell_contact = []
for i in range (0, df["ligand_symbol"].shape[0]):
    ligand = df["ligand_symbol"][i]
    if ligand not in gene_info:
        continue
        
    if df["annotation"][i] == 'ECM-Receptor':    
        continue
        
    receptor_symbol_list = df["receptor_symbol"][i]
    receptor_symbol_list = receptor_symbol_list.split("&")
    for receptor in receptor_symbol_list:
        if receptor in gene_info:
            ligand_dict_dataset[ligand].append(receptor)
            #######
            if df["annotation"][i] == 'Cell-Cell Contact':
                cell_cell_contact.append(receptor)
            #######                
            
print(len(ligand_dict_dataset.keys()))

nichetalk_file = '/cluster/home/t116508uhn/64630/NicheNet-LR-pairs.csv'   
df = pd.read_csv(nichetalk_file)
for i in range (0, df["from"].shape[0]):
    ligand = df["from"][i]
    if ligand not in gene_info:
        continue
    receptor = df["to"][i]
    ligand_dict_dataset[ligand].append(receptor)
    
print(len(ligand_dict_dataset.keys()))

##################################################################
total_relation = 0
l_r_pair = dict()
count = 0
for gene in list(ligand_dict_dataset.keys()): 
    ligand_dict_dataset[gene]=list(set(ligand_dict_dataset[gene]))
    l_r_pair[gene] = dict()
    for receptor_gene in ligand_dict_dataset[gene]:
        l_r_pair[gene][receptor_gene] = -1 #count #
        count = count + 1
##################################################################
print(count)
cells_ligand_vs_receptor = []
for i in range (0, cell_vs_gene.shape[0]):
    cells_ligand_vs_receptor.append([])
 
cell_percentile = []
for i in range (0, cell_vs_gene.shape[0]):
    for j in range (0, cell_vs_gene.shape[0]):
        cells_ligand_vs_receptor[i].append([])
        cells_ligand_vs_receptor[i][j] = []
    
cell_rec_count = np.zeros((cell_vs_gene.shape[0]))
count_total_edges = 0
pair_id = 1
for i in range (0, cell_vs_gene.shape[0]): # ligand
    count_rec = 0
    for gene in ligand_list: 
        if cell_vs_gene_dict[i][gene] >= cell_percentile[i][2]:
            for j in range (0, cell_vs_gene.shape[0]): # receptor
                for gene_rec in ligand_dict_dataset[gene]:
                    if (gene_rec in gene_list) and cell_vs_gene_dict[j][gene_rec] >= cell_percentile[j][2]: #gene_list_percentile[gene_rec][1]: #global_percentile: #
                        if gene_rec in cell_cell_contact and distance_matrix[i,j] > spot_diameter:
                            continue
                        else:
                            if distance_matrix[i,j] > spot_diameter*4:
                                continue
                            communication_score = cell_vs_gene_dict[i][gene] * cell_vs_gene_dict[j][gene_rec]
                            
                            '''if communication_score > max_score:
                                max_score = communication_score
                            if communication_score < min_score:
                                min_score = communication_score ''' 
                                
                            if l_r_pair[gene][gene_rec] == -1: 
                                l_r_pair[gene][gene_rec] = pair_id
                                pair_id = pair_id + 1 
                           
                            relation_id = l_r_pair[gene][gene_rec]
                            cells_ligand_vs_receptor[i][j].append([gene, gene_rec, communication_score, relation_id])
                            count_rec = count_rec + 1
                            count_total_edges = count_total_edges + 1
                            
                            
    cell_rec_count[i] =  count_rec   
    print("%d - %d "%(i, count_rec))
    #print("%d - %d , max %g and min %g "%(i, count_rec, max_score, min_score))
    
print(pair_id)
with gzip.open("/cluster/projects/schwartzgroup/fatema/find_ccc/" + 'synthetic_communication_scores_'+str(len(ligand_list))+'_region_1', 'wb') as fp:
    pickle.dump([cells_ligand_vs_receptor,l_r_pair], fp)


row_col = []
edge_weight = []
for i in range (0, len(cells_ligand_vs_receptor)):
    #ccc_j = []
    for j in range (0, len(cells_ligand_vs_receptor)):
        if distance_matrix[i][j]<300:
            
            #if i==j:
            if len(cells_ligand_vs_receptor[i][j])>0:
                mean_ccc = 0
                for k in range (0, len(cells_ligand_vs_receptor[i][j])):
                    mean_ccc = mean_ccc + cells_ligand_vs_receptor[i][j][k][2]
                mean_ccc = mean_ccc/len(cells_ligand_vs_receptor[i][j])
                row_col.append([i,j])
                edge_weight.append([0.5, mean_ccc])
            elif i==j: # if not onlyccc, then remove the condition i==j
            #else:
                row_col.append([i,j])
                edge_weight.append([0.5, 0])
		
#with gzip.open("/cluster/projects/schwartzgroup/fatema/find_ccc/" + 'adjacency_records_GAT_STnCCC_97', 'wb') as fp:             
with gzip.open("/cluster/projects/schwartzgroup/fatema/find_ccc/" + 'adjacency_records_GAT_synthetic_region1_onlyccc_70', 'wb') as fp:
    pickle.dump([row_col, edge_weight], fp)
	  

##############################
