
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
################################3
spot_diameter = 89.43 #pixels

coordinates = np.load('/cluster/projects/schwartzgroup/fatema/CCST/generated_data_new/V10M25-61_D1_PDA_64630_Pa_P_Spatial10x_new/'+'coordinates.npy')
from sklearn.metrics.pairwise import euclidean_distances
distance_matrix = euclidean_distances(coordinates, coordinates)


barcode_file='/cluster/home/t116508uhn/64630/spaceranger_output_new/unzipped/barcodes.tsv'
barcode_info=[]
#barcode_info.append("")
i=0
with open(barcode_file) as file:
    tsv_file = csv.reader(file, delimiter="\t")
    for line in tsv_file:
        barcode_info.append([line[0], coordinates[i,0],coordinates[i,1],0])
        i=i+1


#############################
with gzip.open("/cluster/projects/schwartzgroup/fatema/find_ccc/" + 'synthetic_ccc_region_1', 'rb') as fp:
    cell_vs_gene, region_list, ligand_list, activated_cell, gene_ids, cell_percentile = pickle.load(fp)

#############################
#gene_file='/cluster/home/t116508uhn/64630/spaceranger_output_new/unzipped/features.tsv' # 1406
gene_info=dict()
for gene in gene_ids:
    gene_info[gene]=''


gene_index=dict()
i = 0
for gene in gene_ids: 
    gene_index[gene] = i
    i = i+1  
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
    if receptor not in gene_info:
        continue
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
 

for i in range (0, cell_vs_gene.shape[0]):
    for j in range (0, cell_vs_gene.shape[0]):
        cells_ligand_vs_receptor[i].append([])
        cells_ligand_vs_receptor[i][j] = []

from sklearn.metrics.pairwise import euclidean_distances
distance_matrix = euclidean_distances(coordinates, coordinates)

cell_rec_count = np.zeros((cell_vs_gene.shape[0]))
count_total_edges = 0
pair_id = 1
activated_cell_index = dict()
for gene in ligand_list: 
    for i in range (0, cell_vs_gene.shape[0]): # ligand
        count_rec = 0    
        if cell_vs_gene[i][gene_index[gene]] > cell_percentile[i][2]:
            for j in range (0, cell_vs_gene.shape[0]): # receptor
                for gene_rec in ligand_dict_dataset[gene]:
                    if cell_vs_gene[j][gene_index[gene_rec]] > cell_percentile[j][2]: #gene_list_percentile[gene_rec][1]: #global_percentile: #
                        if gene_rec in cell_cell_contact and distance_matrix[i,j] > spot_diameter:
                            continue
                        else:
                            if distance_matrix[i,j] > spot_diameter*4:
                                continue
                            communication_score = cell_vs_gene[i][gene_index[gene]] * cell_vs_gene[j][gene_index[gene_rec]]
                            
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
                            activated_cell_index[i] = ''
                            activated_cell_index[j] = ''
                            
                            
        cell_rec_count[i] =  count_rec   
        #print("%d - %d "%(i, count_rec))
        #print("%d - %d , max %g and min %g "%(i, count_rec, max_score, min_score))
    
    print(pair_id)
	
with gzip.open("/cluster/projects/schwartzgroup/fatema/find_ccc/" + 'synthetic_communication_scores_region_1_a', 'wb') as fp:
    pickle.dump([cells_ligand_vs_receptor,l_r_pair,ligand_list,activated_cell_index], fp)

with gzip.open("/cluster/projects/schwartzgroup/fatema/find_ccc/" + 'synthetic_communication_scores_region_1_a', 'rb') as fp:
   cells_ligand_vs_receptor,l_r_pair,ligand_list,activated_cell_index = pickle.load(fp)

ccc_index_dict = dict()
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
                ccc_index_dict[i] = ''
                ccc_index_dict[j] = ''
                edge_weight.append([0.5, mean_ccc])
            #elif i==j: # if not onlyccc, then remove the condition i==j
            else:
                row_col.append([i,j])
                edge_weight.append([0.5, 0])

		
with gzip.open("/cluster/projects/schwartzgroup/fatema/find_ccc/" + 'adjacency_records_GAT_synthetic_region1_STnCCC_70', 'wb') as fp:             
#with gzip.open("/cluster/projects/schwartzgroup/fatema/find_ccc/" + 'adjacency_records_GAT_synthetic_region1_onlyccc_70', 'wb') as fp:
    pickle.dump([row_col, edge_weight], fp)
	  
#############################

ccc_index_dict = dict()
row_col = []
edge_weight = []
edge_type = []
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
		# type 0
                row_col.append([i,j])
                edge_weight.append([0.5])
                edge_type.append(0)
		# type 1
                row_col.append([i,j])
                edge_weight.append([mean_ccc])
                edge_type.append(1)
		
                ccc_index_dict[i] = ''
                ccc_index_dict[j] = ''
		
            #elif i==j: # if not onlyccc, then remove the condition i==j
            else:               
		# type 0
                row_col.append([i,j])
                edge_weight.append([0.5])
                edge_type.append(0)
		# type 1
                row_col.append([i,j])
                edge_weight.append([0])
                edge_type.append(1)

		
with gzip.open("/cluster/projects/schwartzgroup/fatema/find_ccc/" + 'adjacency_records_2RGAT_synthetic_region1_STnCCC_70', 'wb') as fp:             
#with gzip.open("/cluster/projects/schwartzgroup/fatema/find_ccc/" + 'adjacency_records_GAT_synthetic_region1_onlyccc_70', 'wb') as fp:
    pickle.dump([row_col, edge_weight, edge_type], fp)
	  
#############################

########
number = 20
cmap = plt.get_cmap('tab20')
colors = [cmap(i) for i in np.linspace(0, 1, number)]

number = 20
cmap = plt.get_cmap('tab20b')
colors_2 = [cmap(i) for i in np.linspace(0, 1, number)]

colors=colors+colors_2

number = 8
cmap = plt.get_cmap('Set2')
colors_2 = [cmap(i) for i in np.linspace(0, 1, number)]

colors=colors+colors_2

number = 12
cmap = plt.get_cmap('Set3')
colors_2 = [cmap(i) for i in np.linspace(0, 1, number)]

colors=colors+colors_2

number = 20
cmap = plt.get_cmap('tab20c')
colors_2 = [cmap(i) for i in np.linspace(0, 1, number)]

colors=colors+colors_2


x_index=[]
y_index=[]
marker_size = []

x_index_ccc=[]
y_index_ccc=[]
marker_size_ccc = []

for i in range (0, len(barcode_info)):
    if i in ccc_index_dict:
        x_index_ccc.append(barcode_info[i][1])
        y_index_ccc.append(barcode_info[i][2])
        marker_size_ccc.append('^')  
    else:
        x_index.append(barcode_info[i][1])
        y_index.append(barcode_info[i][2])
        marker_size.append('o') 


for i in range (0, len(x_index)):  
    plt.scatter(x=x_index[i], y=-y_index[i], label = 0, color=colors[0], marker=matplotlib.markers.MarkerStyle(marker=marker_size[i]), s=15)   

for i in range (0, len(x_index_ccc)):  
    plt.scatter(x=x_index_ccc[i], y=-y_index_ccc[i], label = 1, color=colors[1], marker=matplotlib.markers.MarkerStyle(marker=marker_size_ccc[i]), s=15)   

      
#plt.legend(fontsize=4,loc='upper right')

save_path = '/cluster/home/t116508uhn/64630/'
plt.savefig(save_path+'toomanycells_PCA_64embedding_pathologist_label_l1mp5_temp_plot.svg', dpi=400)
plt.clf()

