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
from kneed import KneeLocator
import copy 

import argparse
parser = argparse.ArgumentParser()
parser.add_argument( '--data_path', type=str, default='/cluster/home/t116508uhn/64630/spaceranger_output_new/' , help='The path to dataset') 
parser.add_argument( '--embedding_data_path', type=str, default='new_alignment/Embedding_data_ccc_rgcn/' , help='The path to attention') #'/cluster/projects/schwartzgroup/fatema/pancreatic_cancer_visium/210827_A00827_0396_BHJLJTDRXY_Notta_Karen/V10M25-61_D1_PDA_64630_Pa_P_Spatial10x_new/outs/'
parser.add_argument( '--data_name', type=str, default='V10M25-61_D1_PDA_64630_Pa_P_Spatial10x_new', help='The name of dataset')
parser.add_argument( '--model_name', type=str, default='gat_r1_2attr', help='model name')
parser.add_argument( '--slice', type=int, default=0, help='starting index of ligand')
args = parser.parse_args()



spot_diameter = 89.43 #pixels
############

############
coordinates = np.load('/cluster/projects/schwartzgroup/fatema/CCST/generated_data_new/V10M25-61_D1_PDA_64630_Pa_P_Spatial10x_new/'+'coordinates.npy')
barcode_file='/cluster/home/t116508uhn/64630/spaceranger_output_new/unzipped/barcodes.tsv'
barcode_info=[]
#barcode_info.append("")
i=0
with open(barcode_file) as file:
    tsv_file = csv.reader(file, delimiter="\t")
    for line in tsv_file:
        barcode_info.append([line[0], coordinates[i,0],coordinates[i,1],0])
        i=i+1
 
####### get the gene expressions ######
data_fold = args.data_path #+args.data_name+'/'
print(data_fold)
adata_h5 = st.Read10X(path=data_fold, count_file='filtered_feature_bc_matrix.h5') #count_file=args.data_name+'_filtered_feature_bc_matrix.h5' )
print(adata_h5)
sc.pp.filter_genes(adata_h5, min_cells=1)
#sc.pp.log1p(adata_h5)
#sc.pp.highly_variable_genes(adata_h5) #3952
print(adata_h5)
gene_ids = list(adata_h5.var_names)
coordinates = adata_h5.obsm['spatial']
#################### 
temp = qnorm.quantile_normalize(np.transpose(sparse.csr_matrix.toarray(adata_h5.X)))  
adata_X = np.transpose(temp)  
#adata_X = sc.pp.scale(adata_X)
cell_vs_gene = copy.deepcopy(adata_X)
#cell_vs_gene_scaled = sc.pp.scale(adata_X) # rows = cells, columns = genes

####################
'''
adata_X = sc.pp.normalize_total(adata_h5, target_sum=1, exclude_highly_expressed=True, inplace=False)['X']
#adata_X = sc.pp.scale(adata_X)
#adata_X = sc.pp.pca(adata_X, n_comps=args.Dim_PCA)
cell_vs_gene = sparse.csr_matrix.toarray(adata_X) #adata_X
'''
####################
####################
'''
cell_percentile = []
for i in range (0, cell_vs_gene.shape[0]):
    cell_percentile.append([np.percentile(sorted(cell_vs_gene_scaled[i]), 10), np.percentile(sorted(cell_vs_gene_scaled[i]), 20),np.percentile(sorted(cell_vs_gene_scaled[i]), 70), np.percentile(sorted(cell_vs_gene_scaled[i]), 97)])
'''
cell_percentile = []
for i in range (0, cell_vs_gene.shape[0]):
    y = sorted(cell_vs_gene[i])
    x = range(1, len(y)+1)
    kn = KneeLocator(x, y, curve='convex', direction='increasing')
    kn_value = y[kn.knee-1]
    cell_percentile.append([np.percentile(y, 10), np.percentile(y, 20),np.percentile(y, 70), np.percentile(y, 97), kn_value])


'''
cell_percentile = []
for i in range (0, cell_vs_gene.shape[0]):
    y = np.histogram(cell_vs_gene[i])[0] # density: 
    x = range(1, len(y)+1)
    kn = KneeLocator(x, y, curve='convex', direction='decreasing')
    kn_value = np.histogram(cell_vs_gene[i])[1][kn.knee-1]
    cell_percentile.append([np.percentile(cell_vs_gene[i], 10), np.percentile(cell_vs_gene[i], 20),np.percentile(cell_vs_gene[i], 70), np.percentile(cell_vs_gene[i], 97), kn_value])

#gene_file='/cluster/home/t116508uhn/64630/spaceranger_output_new/unzipped/features.tsv' # 1406

gene_percentile = dict()
for i in range (0, cell_vs_gene.shape[1]):
    y = np.histogram(cell_vs_gene[:,i])[0]
    x = range(1, len(y)+1)
    kn = KneeLocator(x, y, curve='convex', direction='decreasing')
    kn_value = np.histogram(cell_vs_gene[:,i])[1][kn.knee-1]
    gene_percentile[gene_ids[i]] = [np.percentile(cell_vs_gene[:,i], 10), np.percentile(cell_vs_gene[:,i], 50),np.percentile(cell_vs_gene[:,i], 70), np.percentile(cell_vs_gene[:,i], 97), kn_value]
'''
gene_info=dict()
for gene in gene_ids:
    gene_info[gene]=''

gene_index=dict()    
i = 0
for gene in gene_ids: 
    gene_index[gene] = i
    i = i+1
	
'''gene_marker_ids = dict()
gene_marker_file = '/cluster/home/t116508uhn/64630/Geneset_22Sep21_Subtypesonly_edited.csv'
df = pd.read_csv(gene_marker_file)
for i in range (0, df["Name"].shape[0]):
    if df["Name"][i] in gene_info:
        gene_marker_ids[df["Name"][i]] = ''
'''

ligand_dict_dataset = defaultdict(list)
cell_chat_file = '/cluster/home/t116508uhn/64630/Human-2020-Jin-LR-pairs_cellchat.csv'
df = pd.read_csv(cell_chat_file)
cell_cell_contact = []
for i in range (0, df["ligand_symbol"].shape[0]):
    ligand = df["ligand_symbol"][i]
    #if ligand not in gene_marker_ids:
    if ligand not in gene_info:
        continue
        
    if df["annotation"][i] == 'ECM-Receptor':    
        continue
        
    receptor_symbol_list = df["receptor_symbol"][i]
    receptor_symbol_list = receptor_symbol_list.split("&")
    for receptor in receptor_symbol_list:
        if receptor in gene_info:
        #if receptor in gene_marker_ids:
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
    #if ligand not in gene_marker_ids:
    if ligand not in gene_info:
        continue
    receptor = df["to"][i]
    #if receptor not in gene_marker_ids:
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

ligand_list = list(ligand_dict_dataset.keys())  

cells_ligand_vs_receptor = []
for i in range (0, cell_vs_gene.shape[0]):
    cells_ligand_vs_receptor.append([])
 

for i in range (0, cell_vs_gene.shape[0]):
    for j in range (0, cell_vs_gene.shape[0]):
        cells_ligand_vs_receptor[i].append([])
        cells_ligand_vs_receptor[i][j] = []

from sklearn.metrics.pairwise import euclidean_distances
distance_matrix = euclidean_distances(coordinates, coordinates)


dist_X = np.zeros((distance_matrix.shape[0], distance_matrix.shape[1]))

for j in range(0, distance_matrix.shape[1]):
    max_value=np.max(distance_matrix[:,j])
    min_value=np.min(distance_matrix[:,j])
    for i in range(distance_matrix.shape[0]):
        dist_X[i,j] = 1-(distance_matrix[i,j]-min_value)/(max_value-min_value)
        	
    #list_indx = list(np.argsort(dist_X[:,j]))
    #k_higher = list_indx[len(list_indx)-k_nn:len(list_indx)]
    for i in range(0, distance_matrix.shape[0]):
        if distance_matrix[i,j] > spot_diameter*4: #i not in k_higher:
            dist_X[i,j] = 0 #-1
            
cell_rec_count = np.zeros((cell_vs_gene.shape[0]))
count_total_edges = 0
pair_id = 1
activated_cell_index = dict()
#count = 0
#count2 = 0
print('len ligand_list %d'%len(ligand_list))

start_index = args.slice
end_index = min(len(ligand_list), start_index+100)

for gene in ligand_list[start_index: end_index]: #[0:5]: 
    for i in range (0, cell_vs_gene.shape[0]): # ligand
        count_rec = 0    
        if cell_vs_gene[i][gene_index[gene]] > cell_percentile[i][2]:
            for j in range (0, cell_vs_gene.shape[0]): # receptor
                for gene_rec in ligand_dict_dataset[gene]:
                    if cell_vs_gene[j][gene_index[gene_rec]] > cell_percentile[j][2]: #gene_list_percentile[gene_rec][1]: #global_percentile: #
                        if gene_rec in cell_cell_contact and distance_matrix[i,j] > spot_diameter:
                            continue
                        else:
                            '''if gene_rec in cell_cell_contact and distance_matrix[i,j] < spot_diameter:
                                print(gene)'''
                            if distance_matrix[i,j] > spot_diameter*4:
                                continue
                            communication_score = cell_vs_gene[i][gene_index[gene]] * cell_vs_gene[j][gene_index[gene_rec]]
                            '''if gene=='L1CAM':
                                count = count+1
                            elif gene=='LAMC2':
                                count2 = count2+1'''
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
	
with gzip.open("/cluster/projects/schwartzgroup/fatema/find_ccc/" + 'synthetic_communication_scores_selective_lr_STnCCC_c_'+str(args.slice), 'wb') as fp: #b, b_1, a
    pickle.dump([cells_ligand_vs_receptor,l_r_pair,ligand_list[start_index, end_index],activated_cell_index], fp) #a - [0:5]


ccc_index_dict = dict()
row_col = []
edge_weight = []
for i in range (0, len(cells_ligand_vs_receptor)):
    #ccc_j = []
    for j in range (0, len(cells_ligand_vs_receptor)):
        if distance_matrix[i][j] <= spot_diameter*4:
            #if i==j:
            if len(cells_ligand_vs_receptor[i][j])>0:
                mean_ccc = 0
                for k in range (0, len(cells_ligand_vs_receptor[i][j])):
                    mean_ccc = mean_ccc + cells_ligand_vs_receptor[i][j][k][2]
                #mean_ccc = mean_ccc/len(cells_ligand_vs_receptor[i][j])
                row_col.append([i,j])
                ccc_index_dict[i] = ''
                ccc_index_dict[j] = ''
                edge_weight.append([dist_X[i,j], mean_ccc])
            #elif i==j: # if not onlyccc, then remove the condition i==j
            else:
                row_col.append([i,j])
                edge_weight.append([dist_X[i,j], 0])

		
with gzip.open("/cluster/projects/schwartzgroup/fatema/find_ccc/" + 'adjacency_records_GAT_selective_lr_STnCCC_c_'+str(args.slice), 'wb') as fp:  #b, a:[0:5]           
#with gzip.open("/cluster/projects/schwartzgroup/fatema/find_ccc/" + 'adjacency_records_GAT_synthetic_region1_onlyccc_70', 'wb') as fp:
    pickle.dump([row_col, edge_weight], fp)
############################################################
	
    
coordinates = np.load('/cluster/projects/schwartzgroup/fatema/CCST/generated_data_new/V10M25-61_D1_PDA_64630_Pa_P_Spatial10x_new/'+'coordinates.npy')	
from sklearn.metrics.pairwise import euclidean_distances
distance_matrix = euclidean_distances(coordinates, coordinates)

cells_ligand_vs_receptor = []
for i in range (0, cell_vs_gene.shape[0]):
    cells_ligand_vs_receptor.append([])
 

for i in range (0, cell_vs_gene.shape[0]):
    for j in range (0, cell_vs_gene.shape[0]):
        cells_ligand_vs_receptor[i].append([])
        cells_ligand_vs_receptor[i][j] = []

slice = -30
while slice < 544:
    slice = slice + 30
    print('read %d'%slice)
    with gzip.open("/cluster/projects/schwartzgroup/fatema/find_ccc/" + 'synthetic_communication_scores_selective_lr_STnCCC_c_'+str(slice), 'rb') as fp: #b, b_1, a
        cells_ligand_vs_receptor_temp, l_r_pair, ligand_list, activated_cell_index = pickle.load(fp) 
        
    for i in range (0, len(cells_ligand_vs_receptor)):
        for j in range (0, len(cells_ligand_vs_receptor)):
            if len(cells_ligand_vs_receptor_temp[i][j])>0:
                cells_ligand_vs_receptor[i][j].extend(cells_ligand_vs_receptor_temp[i][j]) 
###############################
ccc_index_dict = dict()
row_col = []
edge_weight = []
for i in range (0, len(cells_ligand_vs_receptor)):
    #ccc_j = []
    for j in range (0, len(cells_ligand_vs_receptor)):
        if distance_matrix[i][j] <= spot_diameter*4:
            #if i==j:
            if len(cells_ligand_vs_receptor[i][j])>0:
                mean_ccc = 0
                for k in range (0, len(cells_ligand_vs_receptor[i][j])):
                    mean_ccc = mean_ccc + cells_ligand_vs_receptor[i][j][k][2] #*dist_X[i,j]
                mean_ccc = mean_ccc/len(cells_ligand_vs_receptor[i][j])
                row_col.append([i,j])
                ccc_index_dict[i] = ''
                ccc_index_dict[j] = ''
                edge_weight.append([dist_X[i,j], mean_ccc])
            #elif i==j: # if not onlyccc, then remove the condition i==j
            else:
                row_col.append([i,j])
                edge_weight.append([dist_X[i,j], 0])
                
print('len row col %d'%len(row_col))

#with gzip.open("/cluster/projects/schwartzgroup/fatema/find_ccc/" + 'adjacency_records_GAT_selective_lr_STnCCC_c_'+'all_avg', 'wb') as fp:  #b, a:[0:5]   
with gzip.open("/cluster/projects/schwartzgroup/fatema/find_ccc/" + 'adjacency_records_GAT_selective_lr_STtimesCCC_c_'+'all_avg', 'wb') as fp:  #b, a:[0:5]  
#with gzip.open("/cluster/projects/schwartzgroup/fatema/find_ccc/" + 'adjacency_records_GAT_selective_lr_STnCCC_c_'+'all_sum', 'wb') as fp:  #b, a:[0:5]           
    pickle.dump([row_col, edge_weight], fp)
################################################################################
ccc_index_dict = dict()
row_col = []
edge_weight = []
lig_rec = []
count_edge = 0
max_local = 0
local_list = np.zeros((102))
for i in range (0, len(cells_ligand_vs_receptor)):
    #ccc_j = []
    for j in range (0, len(cells_ligand_vs_receptor)):
        if distance_matrix[i][j] <= spot_diameter*4: 
            count_local = 0
            if len(cells_ligand_vs_receptor[i][j])>0:
                for k in range (0, len(cells_ligand_vs_receptor[i][j])):
                    gene = cells_ligand_vs_receptor[i][j][k][0]
                    gene_rec = cells_ligand_vs_receptor[i][j][k][1]
                    # above 5th percentile only
                    if cell_vs_gene[i][gene_index[gene]] > cell_percentile[i][4] or cell_vs_gene[j][gene_index[gene_rec]] > cell_percentile[j][4]:
                        count_edge = count_edge + 1
                        count_local = count_local + 1
#print(count_edge)                      
                        mean_ccc = cells_ligand_vs_receptor[i][j][k][2]
                        row_col.append([i,j])
                        ccc_index_dict[i] = ''
                        ccc_index_dict[j] = ''
                        edge_weight.append([dist_X[i,j], mean_ccc])
                        lig_rec.append([gene, gene_rec])                      
                
                if max_local < count_local:
                    max_local = count_local
            else:
                row_col.append([i,j])
                edge_weight.append([dist_X[i,j], 0])
                lig_rec.append(['', ''])
            local_list[count_local] = local_list[count_local] + 1

print('len row col %d'%len(row_col))
print('count local %d'%count_local) 
with gzip.open("/cluster/projects/schwartzgroup/fatema/find_ccc/" + 'adjacency_records_GAT_selective_lr_STnCCC_separate_'+'all_kneepoint', 'wb') as fp:  # at least one of lig or rec has exp > respective knee point          
#with gzip.open("/cluster/projects/schwartzgroup/fatema/find_ccc/" + 'adjacency_records_GAT_selective_lr_STnCCC_separate_'+'all_density_kneepoint', 'wb') as fp:  #b, a:[0:5]   
    pickle.dump([row_col, edge_weight, lig_rec], fp)


################################################################################
ccc_index_dict = dict()
row_col = []
edge_weight = []
lig_rec = []
count_edge = 0
max_local = 0
#local_list = np.zeros((20))
for i in range (0, len(cells_ligand_vs_receptor)):
    #ccc_j = []
    for j in range (0, len(cells_ligand_vs_receptor)):
        if distance_matrix[i][j] <= spot_diameter*4: 
            count_local = 0
            if len(cells_ligand_vs_receptor[i][j])>0:
                for k in range (0, len(cells_ligand_vs_receptor[i][j])):
                    gene = cells_ligand_vs_receptor[i][j][k][0]
                    gene_rec = cells_ligand_vs_receptor[i][j][k][1]
                    # above 5th percentile only
                    if cell_vs_gene[i][gene_index[gene]] > gene_percentile[gene][1] and cell_vs_gene[j][gene_index[gene_rec]] > gene_percentile[gene_rec][1]:
                        count_edge = count_edge + 1
                        count_local = count_local + 1
#print(count_edge)                      
                        mean_ccc = cells_ligand_vs_receptor[i][j][k][2]
                        row_col.append([i,j])
                        ccc_index_dict[i] = ''
                        ccc_index_dict[j] = ''
                        edge_weight.append([dist_X[i,j], mean_ccc])
                        lig_rec.append([gene, gene_rec])                      
                
                if max_local < count_local:
                    max_local = count_local
            else:
                row_col.append([i,j])
                edge_weight.append([dist_X[i,j], 0])
                lig_rec.append(['', ''])
            #local_list[count_local] = local_list[count_local] + 1

print('len row col %d'%len(row_col))
print('count local %d'%count_local) 
#with gzip.open("/cluster/projects/schwartzgroup/fatema/find_ccc/" + 'adjacency_records_GAT_selective_lr_STnCCC_separate_'+'all_alonggene_kneepoint', 'wb') as fp:  # at least one of lig or rec has exp > respective knee point          
with gzip.open("/cluster/projects/schwartzgroup/fatema/find_ccc/" + 'adjacency_records_GAT_selective_lr_STnCCC_separate_'+'all_alonggene_density_kneepoint', 'wb') as fp:  #b, a:[0:5]   
    pickle.dump([row_col, edge_weight, lig_rec], fp)


###########################################################Visualization starts ##################
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


coordinates = np.load('/cluster/projects/schwartzgroup/fatema/CCST/generated_data_new/V10M25-61_D1_PDA_64630_Pa_P_Spatial10x_new/'+'coordinates.npy')
barcode_file='/cluster/home/t116508uhn/64630/spaceranger_output_new/unzipped/barcodes.tsv'
barcode_info=[]
#barcode_info.append("")
i=0
with open(barcode_file) as file:
    tsv_file = csv.reader(file, delimiter="\t")
    for line in tsv_file:
        barcode_info.append([line[0], coordinates[i,0],coordinates[i,1],0])
        i=i+1
        
#####
#X_attention_filename = args.embedding_data_path + args.data_name + '/' + 'totalsynccc_gat_r1_2attr_noFeature_selective_lr_STnCCC_c_70_attention.npy' #a
#X_attention_filename = args.embedding_data_path + args.data_name + '/' + 'totalsynccc_gat_r1_2attr_noFeature_selective_lr_STnCCC_c_all_avg_bothlayer_attention_l1.npy' #a
#X_attention_filename = args.embedding_data_path + args.data_name + '/' + 'GAT_selective_lr_STnCCC_separate_all_density_kneepoint_r1_attention_l1.npy' #a
X_attention_filename = args.embedding_data_path + args.data_name + '/' + 'GAT_selective_lr_STnCCC_separate_all_kneepoint_r1_attention_l1.npy' #a
X_attention_bundle = np.load(X_attention_filename, allow_pickle=True) 


attention_scores = []
datapoint_size = len(barcode_info)
for i in range (0, datapoint_size):
    attention_scores.append([])   
    for j in range (0, datapoint_size):	
        attention_scores[i].append([])   
        attention_scores[i][j] = []
	
#attention_scores = np.zeros((len(barcode_info),len(barcode_info)))
distribution = []
for index in range (0, X_attention_bundle[0].shape[1]):
    i = X_attention_bundle[0][0][index]
    j = X_attention_bundle[0][1][index]
    #attention_scores[i][j] = X_attention_bundle[3][index][0] #X_attention_bundle[2][index][0]
    #distribution.append(attention_scores[i][j])
    attention_scores[i][j].append(X_attention_bundle[3][index][0]) #X_attention_bundle[2][index][0]
    distribution.append(X_attention_bundle[3][index][0])
##############
attention_scores_normalized = np.zeros((len(barcode_info),len(barcode_info)))
for index in range (0, X_attention_bundle[0].shape[1]):
    i = X_attention_bundle[0][0][index]
    j = X_attention_bundle[0][1][index]
    attention_scores_normalized [i][j] = X_attention_bundle[1][index][0]
##############
adjacency_matrix = np.zeros((len(barcode_info),len(barcode_info)))
for index in range (0, X_attention_bundle[0].shape[1]):
    i = X_attention_bundle[0][0][index]
    j = X_attention_bundle[0][1][index]
    adjacency_matrix [i][j] = 1


##############
#with gzip.open("/cluster/projects/schwartzgroup/fatema/find_ccc/" + 'adjacency_records_GAT_selective_lr_STnCCC_c_'+'all_avg', 'rb') as fp:  #b, a:[0:5]           
#with gzip.open("/cluster/projects/schwartzgroup/fatema/find_ccc/" + 'adjacency_records_GAT_synthetic_region1_onlyccc_70', 'wb') as fp:
#    row_col, edge_weight = pickle.load(fp)

with gzip.open("/cluster/projects/schwartzgroup/fatema/find_ccc/" + 'adjacency_records_GAT_selective_lr_STnCCC_separate_'+'all_kneepoint', 'rb') as fp:  #b, a:[0:5]   
    row_col, edge_weight, lig_rec = pickle.load(fp) # density_

lig_rec_dict = []
for i in range (0, datapoint_size):
    lig_rec_dict.append([])  
    for j in range (0, datapoint_size):	
        lig_rec_dict[i].append([])   
        lig_rec_dict[i][j] = []
        
total_type = np.zeros((2))        
for index in range (0, len(row_col)):
    i = row_col[index][0]
    j = row_col[index][1]
    lig_rec_dict[i][j].append(lig_rec[index])  
    
    
attention_scores = np.zeros((datapoint_size,datapoint_size))
distribution = []
ccc_index_dict = dict()
for index in range (0, len(row_col)):
    i = row_col[index][0]
    j = row_col[index][1]
    if edge_weight[index][1]>0:
        attention_scores[i][j] = edge_weight[index][1]
        distribution.append(attention_scores[i][j])
        ccc_index_dict[i] = ''
        #ccc_index_dict[j] = ''   
	
###########################
ccc_index_dict = dict()
threshold_down =  np.percentile(sorted(distribution), 95)
threshold_up =  np.percentile(sorted(distribution), 100)
connecting_edges = np.zeros((len(barcode_info),len(barcode_info)))
for j in range (0, datapoint_size):
    #threshold =  np.percentile(sorted(attention_scores[:,j]), 97) #
    for i in range (0, datapoint_size):
        atn_score_list = attention_scores[i][j]
        #print(len(atn_score_list))
        s = min(0,len(atn_score_list)-1)
        for k in range (0, len(atn_score_list)):
            if attention_scores[i][j][k] >= threshold_down and attention_scores[i][j][k] <= threshold_up: #np.percentile(sorted(distribution), 50):
                connecting_edges[i][j] = 1
                ccc_index_dict[i] = ''
                ccc_index_dict[j] = ''
     
	

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
    
 
for i in range (0, len(barcode_info)):
#    if barcode_info[i][0] in barcode_label:
    if count_points_component[labels[i]] > 1:
        barcode_info[i][3] = index_dict[labels[i]] #2
    else: 
        barcode_info[i][3] = 0
       


#############
'''
datapoint_label = []
for i in range (0, datapoint_size):
    if i in ccc_index_dict:
        barcode_info[i][3] = 2
    else:
        barcode_info[i][3] = 0
'''
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


#cell_count_cluster=np.zeros((labels.shape[0]))
filltype='none'

#id_label = [0,2]
#for j in id_label:
for j in range (0, n_components):
    label_i = j
    x_index=[]
    y_index=[]
    marker_size = []
    #fillstyles_type = []
    for i in range (0, len(barcode_info)):
        if barcode_info[i][3] == j:
            x_index.append(barcode_info[i][1])
            y_index.append(barcode_info[i][2])
            #cell_count_cluster[j] = cell_count_cluster[j]+1
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
 
plt.hist(distribution, color = 'blue',bins = int(len(distribution)/5))
save_path = '/cluster/home/t116508uhn/64630/'
plt.savefig(save_path+'toomanycells_PCA_64embedding_pathologist_label_l1mp5_temp_plot.svg', dpi=400)
plt.clf()



ids = []
x_index=[]
y_index=[]
colors_point = []
for i in range (0, len(barcode_info)):    
    ids.append(i)
    x_index.append(barcode_info[i][1])
    y_index.append(barcode_info[i][2])    
    colors_point.append(colors[barcode_info[i][3]]) 
  
max_x = np.max(x_index)
max_y = np.max(y_index)


from pyvis.network import Network
import networkx as nx

    
g = nx.MultiDiGraph(directed=True) #nx.Graph()
for i in range (0, len(barcode_info)):
    if barcode_type[barcode_info[i][0]] == 0:
        marker_size = 'circle'
    elif barcode_type[barcode_info[i][0]] == 1:
        marker_size = 'box'
    else:
        marker_size = 'ellipse'
    g.add_node(int(ids[i]), x=int(x_index[i]), y=int(y_index[i]), label = str(i), physics=False, shape = marker_size, color=matplotlib.colors.rgb2hex(colors_point[i]))
   		
#nx.draw(g, pos= nx.circular_layout(g)  ,with_labels = True, edge_color = 'b', arrowstyle='fancy')
#g.toggle_physics(True)
nt = Network( directed=True) #"500px", "500px",
nt.from_nx(g)
for i in range (0, datapoint_size):
    for j in range (0, datapoint_size):
        atn_score_list = attention_scores[i][j]
        #print(len(atn_score_list))
        
        for k in range (0, min(len(atn_score_list),len(lig_rec_dict[i][j])) ):
            if attention_scores[i][j][k] >= threshold_down:
                #print('hello')
                title_str =  ""+lig_rec_dict[i][j][k][0]+", "+lig_rec_dict[i][j][k][1]
                nt.add_edge(int(i), int(j), title =title_str) #, weight=1, arrowsize=int(20),  arrowstyle='fancy'

nt.show('mygraph.html')


#g.show('mygraph.html')
cp mygraph.html /cluster/home/t116508uhn/64630/mygraph.html

