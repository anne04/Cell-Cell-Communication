import numpy as np
import csv
import pickle
from scipy import sparse
import scipy.io as sio
import scanpy as sc
import matplotlib
matplotlib.use('Agg')
#matplotlib.use('TkAgg')
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
import gzip

import argparse
parser = argparse.ArgumentParser()
parser.add_argument( '--data_path', type=str, default='/cluster/home/t116508uhn/64630/spaceranger_output_new/' , help='The path to dataset') 
parser.add_argument( '--embedding_data_path', type=str, default='new_alignment/Embedding_data_ccc_rgcn/' , help='The path to attention') #'/cluster/projects/schwartzgroup/fatema/pancreatic_cancer_visium/210827_A00827_0396_BHJLJTDRXY_Notta_Karen/V10M25-61_D1_PDA_64630_Pa_P_Spatial10x_new/outs/'
parser.add_argument( '--data_name', type=str, default='V10M25-61_D1_PDA_64630_Pa_P_Spatial10x_new', help='The name of dataset')
parser.add_argument( '--model_name', type=str, default='gat_r1_2attr', help='model name')
args = parser.parse_args()



spot_diameter = 1#pixels
############

####### get the gene expressions ######
data_fold = args.data_path #+args.data_name+'/'
print(data_fold)
adata_h5 = st.Read10X(path=data_fold, count_file='filtered_feature_bc_matrix.h5') #count_file=args.data_name+'_filtered_feature_bc_matrix.h5' )
print(adata_h5)
sc.pp.filter_genes(adata_h5, min_cells=1)
print(adata_h5)
gene_ids = list(adata_h5.var_names)
#################### 
temp = qnorm.quantile_normalize(np.transpose(sparse.csr_matrix.toarray(adata_h5.X)))  
adata_X = np.transpose(temp)  
adata_X = sc.pp.scale(adata_X)
cell_vs_gene = adata_X[0:2000]   # rows = cells, columns = genes

####################
cell_percentile = []
for i in range (0, cell_vs_gene.shape[0]):
    cell_percentile.append([np.percentile(sorted(cell_vs_gene[i]), 5), np.percentile(sorted(cell_vs_gene[i]), 50),np.percentile(sorted(cell_vs_gene[i]), 70), np.percentile(sorted(cell_vs_gene[i]), 97)])

####################
'''adata_X = sc.pp.normalize_total(adata_h5, target_sum=1, exclude_highly_expressed=True, inplace=False)['X']
adata_X = sc.pp.scale(adata_X)
#adata_X = sc.pp.pca(adata_X, n_comps=args.Dim_PCA)
features = adata_X'''
####################

#gene_file='/cluster/home/t116508uhn/64630/spaceranger_output_new/unzipped/features.tsv' # 1406

gene_info=dict()
for gene in gene_ids:
    gene_info[gene]=''

gene_index=dict()    
i = 0
for gene in gene_ids: 
    gene_index[gene] = i
    i = i+1
#################################################


ligand_dict_dataset = defaultdict(list)
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

for gene in list(ligand_dict_dataset.keys()): 
    ligand_dict_dataset[gene]=list(set(ligand_dict_dataset[gene]))
    gene_info[gene] = 'included'
    for receptor_gene in ligand_dict_dataset[gene]:
        gene_info[receptor_gene] = 'included'
   
count = 0
for gene in gene_info.keys(): 
    if gene_info[gene] == 'included':
        count = count + 1
print(count)
affected_gene_count = count

#################################################
cell_noise = []
for i in range (0, cell_vs_gene.shape[0]):
    temp = (cell_percentile[i][1] - cell_percentile[i][0]) * np.random.random_sample(size=affected_gene_count) + cell_percentile[i][0]
    cell_noise.append(temp)
   
for i in range (0, cell_vs_gene.shape[0]):
    j = 0
    for gene in gene_info.keys(): 
        if gene_info[gene] == 'included': 
            cell_vs_gene[i, gene_index[gene]] = cell_noise[i][j]
            j = j+1
    #print(j)    
#################################################
cell_expressed = []
for i in range (0, cell_vs_gene.shape[0]):
    temp = (np.max(cell_vs_gene[i]) - cell_percentile[i][2]) * np.random.random_sample(size=affected_gene_count) + cell_percentile[i][2]
    cell_expressed.append(temp)


#################################################
with gzip.open("/cluster/projects/schwartzgroup/fatema/find_ccc/" + 'scRNAseq_spatial_location_synthetic_2', 'rb') as fp:
    temp_x, temp_y = pickle.load(fp)

datapoint_size = temp_x.shape[0]
coordinates = np.zeros((temp_x.shape[0],2))
for i in range (0, datapoint_size):
    coordinates[i][0] = temp_x[i]
    coordinates[i][1] = temp_y[i]
    
from sklearn.metrics.pairwise import euclidean_distances
distance_matrix = euclidean_distances(coordinates, coordinates)

#################################################

	
ligand_list = list(ligand_dict_dataset.keys())  
region_list = [[20, 40, 3, 7], [40, 80, 12, 18]] # [[25, 75, 1, 15], [125, 175, 11, 25]] #[60, 80, 1, 7] 
activated_cell = []
for cell_index in range (0, cell_vs_gene.shape[0]):
    j = 0
    for region in region_list:
        region_x_min = region[0]
        region_x_max = region[1]
        region_y_min = region[2]
        region_y_max = region[3]
        if barcode_info[cell_index][1] > region_x_min and barcode_info[cell_index][1] < region_x_max and barcode_info[cell_index][2] > region_y_min and barcode_info[cell_index][2] < region_y_max:
            for i in range (0, 5): 
                ligand_gene = ligand_list[i]
                recp_list = ligand_dict_dataset[ligand_gene]
                cell_vs_gene[cell_index, gene_index[ligand_gene]] = cell_expressed[cell_index][j]
                j = j+1
                for receptor_gene in recp_list:                    
                    cell_vs_gene[cell_index, gene_index[receptor_gene]] = cell_expressed[cell_index][j] 
                    j = j+1
    if j>0:
        activated_cell.append(1)
    else:
        activated_cell.append(0)
#################################################


with gzip.open("/cluster/projects/schwartzgroup/fatema/find_ccc/" + 'synthetic_ccc_scRNAseq_equallySpaced', 'wb') as fp:
    pickle.dump([cell_vs_gene, region_list, ligand_list[0:5], activated_cell, gene_ids, cell_percentile], fp)

######################################       
'''activated_cell = []
for cell_index in range (0, cell_vs_gene.shape[0]):
    j = 0
    for i in range (0, 5): 
        ligand_gene = ligand_list[i]
        recp_list = ligand_dict_dataset[ligand_gene]
        if cell_vs_gene[cell_index, gene_index[ligand_gene]]>cell_percentile[cell_index][2]:
            j = j+1
        for receptor_gene in recp_list:
            if cell_vs_gene[cell_index, gene_index[receptor_gene]]>cell_percentile[cell_index][2]:
                j = j+1
    print(j)
    if j>0:
        activated_cell.append(1)
    else:
        activated_cell.append(0)
    '''
######################################
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

from sklearn.metrics.pairwise import euclidean_distances
distance_matrix = euclidean_distances(coordinates, coordinates)

cells_ligand_vs_receptor = []
for i in range (0, cell_vs_gene.shape[0]):
    cells_ligand_vs_receptor.append([])

for i in range (0, cell_vs_gene.shape[0]):
    for j in range (0, cell_vs_gene.shape[0]):
        cells_ligand_vs_receptor[i].append([])
        cells_ligand_vs_receptor[i][j] = []
	
cell_rec_count = np.zeros((cell_vs_gene.shape[0]))
count_total_edges = 0
pair_id = 1
activated_cell_index = dict()

for gene in ligand_list[0:5]: 
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
	
with gzip.open("/cluster/projects/schwartzgroup/fatema/find_ccc/" + 'synthetic_communication_scores_scRNAseq_equallySpaced', 'wb') as fp:
    pickle.dump([cells_ligand_vs_receptor,l_r_pair,ligand_list[0:5],activated_cell_index], fp)


ccc_index_dict = dict()
row_col = []
edge_weight = []
for i in range (0, len(cells_ligand_vs_receptor)):
    #ccc_j = []
    for j in range (0, len(cells_ligand_vs_receptor)):
        if distance_matrix[i][j]<spot_diameter*3:
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

		
with gzip.open("/cluster/projects/schwartzgroup/fatema/find_ccc/" + 'adjacency_records_GAT_synthetic_scRNAseq_STnCCC_equallySpaced', 'wb') as fp:             
#with gzip.open("/cluster/projects/schwartzgroup/fatema/find_ccc/" + 'adjacency_records_GAT_synthetic_region1_onlyccc_70', 'wb') as fp:
    pickle.dump([row_col, edge_weight], fp)
############################################################
pathologist_label_file='/cluster/home/t116508uhn/64630/IX_annotation_artifacts.csv' #IX_annotation_artifacts.csv' #
pathologist_label=[]
with open(pathologist_label_file) as file:
    csv_file = csv.reader(file, delimiter=",")
    for line in csv_file:
        pathologist_label.append(line)

barcode_type=dict()
for i in range (1, len(pathologist_label)):
    if pathologist_label[i][1] == 'tumor': #'Tumour':
        barcode_type[pathologist_label[i][0]] = 1
    elif pathologist_label[i][1] =='stroma_deserted':
        barcode_type[pathologist_label[i][0]] = 0
    elif pathologist_label[i][1] =='acinar_reactive':
        barcode_type[pathologist_label[i][0]] = 2
    else:
        barcode_type[pathologist_label[i][0]] = 0
	
attention_scores = np.zeros((len(barcode_info),len(barcode_info)))
distribution = []
for index in range (0, len(row_col)):
    i = row_col[index][0]
    j = row_col[index][1]
    attention_scores[i][j] = edge_weight[index][1]
    distribution.append(attention_scores[i][j])
    

threshold =  np.percentile(sorted(distribution), 90)
connecting_edges = np.zeros((len(barcode_info),len(barcode_info)))

for j in range (0, attention_scores.shape[1]):
    #threshold =  np.percentile(sorted(attention_scores[:,j]), 97) #
    for i in range (0, attention_scores.shape[0]):
        if attention_scores[i][j] > threshold: #np.percentile(sorted(attention_scores[:,i]), 50): #np.percentile(sorted(distribution), 50):
            connecting_edges[i][j] = 1
            
############


graph = csr_matrix(connecting_edges)
n_components, labels = connected_components(csgraph=graph,directed=True, connection = 'weak',  return_labels=True) #
print('number of component %d'%n_components)

count_points_component = np.zeros((n_components))
for i in range (0, len(labels)):
     count_points_component[labels[i]] = count_points_component[labels[i]] + 1
           
print(count_points_component)

id_label = 0  
index_dict = dict()
for i in range (0, count_points_component.shape[0]):
    if count_points_component[i]>1:
        id_label = id_label+1
        index_dict[i] = id_label
print(id_label)
    
     
coordinates = np.load('/cluster/projects/schwartzgroup/fatema/CCST/generated_data_new/V10M25-61_D1_PDA_64630_Pa_P_Spatial10x_new/'+'coordinates.npy')
barcode_file='/cluster/home/t116508uhn/64630/spaceranger_output_new/unzipped/barcodes.tsv'

#coordinates = np.load('/cluster/projects/schwartzgroup/fatema/CCST/generated_data_noPCA_QuantileTransform_wighted_TDistance_2k/V10M25-61_D1_PDA_64630_Pa_P_Spatial10x/'+'coordinates.npy')
#barcode_file='/cluster/home/t116508uhn/64630/barcodes.tsv'
barcode_info=[]
#barcode_info.append("")
i=0
with open(barcode_file) as file:
    tsv_file = csv.reader(file, delimiter="\t")
    for line in tsv_file:
        barcode_info.append([line[0], coordinates[i,0],coordinates[i,1],0])
        i=i+1
 
for i in range (0, len(barcode_info)):
#    if barcode_info[i][0] in barcode_label:
    if count_points_component[labels[i]] > 1:
        barcode_info[i][3] = index_dict[labels[i]]
    
       

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


cell_count_cluster=np.zeros((labels.shape[0]))
filltype='none'
for j in range (0, n_components):
    label_i = j
    x_index=[]
    y_index=[]
    marker_size = []
    #fillstyles_type = []
    for i in range (0, len(barcode_info)):
        if barcode_info[i][3] == label_i:
            x_index.append(barcode_info[i][1])
            y_index.append(barcode_info[i][2])
            cell_count_cluster[j] = cell_count_cluster[j]+1
            spot_color = colors[j]
            if barcode_type[barcode_info[i][0]] == 0:
                marker_size.append('o') 
                #fillstyles_type.append('full') 
            elif barcode_type[barcode_info[i][0]] == 1:
                marker_size.append('^')  
                #fillstyles_type.append('full') 
            else:
                marker_size.append('*') 
                #fillstyles_type.append('full') 
            
            ###############
            
    
    for i in range (0, len(x_index)):  
        plt.scatter(x=x_index[i], y=-y_index[i], label = j, color=colors[j], marker=matplotlib.markers.MarkerStyle(marker=marker_size[i], fillstyle=filltype), s=15)   
    filltype = 'full'
    #plt.scatter(x=np.array(x_index), y=-np.array(y_index), label = j, color=spot_color, marker=marker_size)     
    #plt.scatter(x=np.array(x_index), y=-np.array(y_index), label = j+10)
    
#plt.legend(fontsize=4,loc='upper right')

save_path = '/cluster/home/t116508uhn/64630/'
plt.savefig(save_path+'toomanycells_PCA_64embedding_pathologist_label_l1mp5_temp_plot.svg', dpi=400)
#plt.savefig(save_path+'toomanycells_PCA_64embedding_pathologist_label_l1mp5_temp_plot.svg', dpi=400)
plt.clf()
 
