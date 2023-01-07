import os
#import glob
import pandas as pd
#import shutil
import csv
import numpy as np
import sys
import scikit_posthocs as post
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

import argparse
parser = argparse.ArgumentParser()
parser.add_argument( '--data_path', type=str, default='/cluster/home/t116508uhn/64630/cellrangere/' , help='The path to dataset') #'/cluster/projects/schwartzgroup/fatema/pancreatic_cancer_visium/210827_A00827_0396_BHJLJTDRXY_Notta_Karen/V10M25-61_D1_PDA_64630_Pa_P_Spatial10x_new/outs/'
parser.add_argument( '--data_name', type=str, default='V10M25-61_D1_PDA_64630_Pa_P_Spatial10x_new', help='The name of dataset')
parser.add_argument( '--generated_data_path', type=str, default='generated_data/', help='The folder to store the generated data')
parser.add_argument( '--embedding_data_path', type=str, default='new_alignment/Embedding_data_ccc_rgcn/' , help='The path to attention') #'/cluster/projects/schwartzgroup/fatema/pancreatic_cancer_visium/210827_A00827_0396_BHJLJTDRXY_Notta_Karen/V10M25-61_D1_PDA_64630_Pa_P_Spatial10x_new/outs/'
args = parser.parse_args()
#th_dist = 4
spot_diameter = 89.43 #pixels
threshold_distance = 1.5 #4 : a, b, c
k_nn = 4

data_fold = args.data_path + 'filtered_feature_bc_matrix.h5'
print(data_fold)

#cell_vs_gene = adata_X   # rows = cells, columns = genes

datapoint_size = 2000
x_max = 50 #100
x_min = 0
y_max = 20
y_min = 0
#################################
temp_x = []
temp_y = []
index_dict = defaultdict(dict)
i = x_min
# row major order, bottom up
k = 0
while i < x_max:
    j = y_min
    while j < y_max:
        temp_x.append(i)
        temp_y.append(j)
        index_dict[i][j] = k
        k = k+1
        
        j = j + 1
        
    i = i + 1
    
#0, 2, 4, ...24, 26, 28   
temp_x = np.array(temp_x)
temp_y = np.array(temp_y)

##############################################

print(len(temp_x))
plt.gca().set_aspect(1)	
plt.scatter(x=np.array(temp_x), y=np.array(temp_y), s=1)
save_path = '/cluster/home/t116508uhn/64630/'
plt.savefig(save_path+'synthetic_spatial_plot_equallySpaced_da.svg', dpi=400)
#plt.savefig(save_path+'synthetic_spatial_plot_3.svg', dpi=400)
plt.clf()

#with gzip.open("/cluster/projects/schwartzgroup/fatema/find_ccc/" + 'synthetic_communication_scores_control_model_a_xny', 'wb') as fp:
with gzip.open("/cluster/projects/schwartzgroup/fatema/find_ccc/" + 'synthetic_communication_scores_control_model_d_xny', 'wb') as fp:
    pickle.dump([temp_x, temp_y], fp)

datapoint_size = temp_x.shape[0]
coordinates = np.zeros((temp_x.shape[0],2))
for i in range (0, datapoint_size):
    coordinates[i][0] = temp_x[i]
    coordinates[i][1] = temp_y[i]
    
distance_matrix = euclidean_distances(coordinates, coordinates)

########### No Feature ##########
dist_X = np.zeros((distance_matrix.shape[0], distance_matrix.shape[1]))

for j in range(0, distance_matrix.shape[1]):
    max_value=np.max(distance_matrix[:,j])
    min_value=np.min(distance_matrix[:,j])
    for i in range(distance_matrix.shape[0]):
        dist_X[i,j] = 1-(distance_matrix[i,j]-min_value)/(max_value-min_value)
	
    '''list_indx = list(np.argsort(dist_X[:,j]))
    k_higher = list_indx[len(list_indx)-k_nn:len(list_indx)]
    for i in range(0, distance_matrix.shape[0]):
        if i not in k_higher:
            dist_X[i,j] = -1'''
    for i in range(0, distance_matrix.shape[0]):
        if distance_matrix[i,j] > threshold_distance: #i not in k_higher:
            dist_X[i,j] = 0 #-1
            
# take gene_count normal distributions where each distribution has len(temp_x) datapoints.
# https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.pareto.html
cell_count = len(temp_x)
gene_count = 4
gene_distribution_active = np.zeros((gene_count, cell_count))
gene_distribution_inactive = np.zeros((gene_count, cell_count))

gene_distribution_inactive[0,:] = np.random.normal(loc=2,scale=2,size=len(temp_x)) # L1
gene_distribution_inactive[1,:] = np.random.normal(loc=3,scale=2,size=len(temp_x)) # R1
gene_distribution_inactive[2,:] = np.random.normal(loc=5,scale=2,size=len(temp_x)) # L2
gene_distribution_inactive[3,:] = np.random.normal(loc=6,scale=2,size=len(temp_x)) # R2

gene_distribution_active[0,:] = np.random.normal(loc=30,scale=2,size=len(temp_x)) # L1
gene_distribution_active[1,:] = np.random.normal(loc=35,scale=2,size=len(temp_x)) # R1
gene_distribution_active[2,:] = np.random.normal(loc=40,scale=2,size=len(temp_x)) # L2
gene_distribution_active[3,:] = np.random.normal(loc=45,scale=2,size=len(temp_x)) # R2

# ensure that all distributions start from >= 0 
for i in range (0, gene_count):
    a = np.min(gene_distribution_inactive[i,:])
    if a < 0:
        gene_distribution_inactive[i,:] = gene_distribution_inactive[i,:] - a
    print('gene %d, min: %g, max:%g '%(i, np.min(gene_distribution_inactive[i,:]), np.max(gene_distribution_inactive[i,:]) ))
     
        
for i in range (0, gene_count):
    a = np.min(gene_distribution_active[i,:])
    if a < 0:
        gene_distribution_active[i,:] = gene_distribution_active[i,:] - a
    print('gene %d, min: %g, max:%g '%(i, np.min(gene_distribution_active[i,:]), np.max(gene_distribution_active[i,:]) ))
#################################################
cell_vs_gene = np.zeros((cell_count,gene_count))
# initially all are in inactive state
for i in range (0, gene_count):
    cell_vs_gene[:,i] = gene_distribution_inactive[i,:]
    
# Pick the regions for which L should be high. Increase raw gene counts for them by replacing their values with gene_distribution_active
ligand_1_index_x = [1, 3, 5, 7, 9, 11, 13, 15, 17, 19, 21]
ligand_2_index_x = [1, 3, 5, 7, 9, 11, 13, 15, 17, 19, 21] 
# Pick the regions for which R should be high. Increase raw gene counts for them by adding +15
receptor_1_index_x = [0, 4, 8, 12, 16, 20] 
receptor_2_index_x = [2, 6, 10, 14, 18] 


for i in range (0, cell_count):
    x_index = coordinates[i][0]
    
    if x_index in ligand_1_index_x:
        # increase the ligand expression
        cell_vs_gene[i,0] = gene_distribution_active[0,i]
        
    if x_index in receptor_1_index_x:
        # increase the receptor expression
        cell_vs_gene[i,1] = gene_distribution_active[1,i] 

    if x_index in ligand_2_index_x:
        # increase the ligand expression
        cell_vs_gene[i,2] = gene_distribution_active[2,i]

    if x_index in receptor_2_index_x:
        # increase the receptor expression
        cell_vs_gene[i,3] = gene_distribution_active[3,i]

# take quantile normalization.
#temp = qnorm.quantile_normalize(np.transpose(cell_vs_gene))  
#adata_X = np.transpose(temp)  
#cell_vs_gene = adata_X
with gzip.open("/cluster/projects/schwartzgroup/fatema/find_ccc/" + 'synthetic_cell_vs_gene_control_model_d_a_notQuantileTransformed', 'wb') as fp:
#with gzip.open("/cluster/projects/schwartzgroup/fatema/find_ccc/" + 'synthetic_cell_vs_gene_control_model_c_notQuantileTransformed', 'wb') as fp:
#with gzip.open("/cluster/projects/schwartzgroup/fatema/find_ccc/" + 'synthetic_cell_vs_gene_control_model_b_quantileTransformed', 'wb') as fp:
    pickle.dump(cell_vs_gene, fp)
    
###############
with gzip.open("/cluster/projects/schwartzgroup/fatema/find_ccc/" + 'synthetic_cell_vs_gene_control_model_d_a_notQuantileTransformed', 'rb') as fp:
#with gzip.open("/cluster/projects/schwartzgroup/fatema/find_ccc/" + 'synthetic_cell_vs_gene_control_model_c_notQuantileTransformed', 'rb') as fp:
    cell_vs_gene = pickle.load(fp)
###############
gene_ids = ['L1', 'R1', 'L2', 'R2']

gene_info=dict()
for gene in gene_ids:
    gene_info[gene]=''

gene_index=dict()    
i = 0
for gene in gene_ids: 
    gene_index[gene] = i
    i = i+1
#############
ligand_dict_dataset = defaultdict(list)
ligand_dict_dataset['L1']=['R1']
ligand_dict_dataset['L2']=['R2']
# ready to go
################################################################################################
# do the usual things
ligand_list = list(ligand_dict_dataset.keys())  

cells_ligand_vs_receptor = []
for i in range (0, cell_vs_gene.shape[0]):
    cells_ligand_vs_receptor.append([])
 

for i in range (0, cell_vs_gene.shape[0]):
    for j in range (0, cell_vs_gene.shape[0]):
        cells_ligand_vs_receptor[i].append([])
        cells_ligand_vs_receptor[i][j] = []

activated_cell_index = dict()
for gene in ligand_list:
    for i in range (0, cell_vs_gene.shape[0]): # ligand                 
        for j in range (0, cell_vs_gene.shape[0]): # receptor
            for gene_rec in ligand_dict_dataset[gene]:                
                if distance_matrix[i,j] > threshold_distance:
                    continue
                communication_score = cell_vs_gene[i][gene_index[gene]] * cell_vs_gene[j][gene_index[gene_rec]]               
                relation_id = 0
                cells_ligand_vs_receptor[i][j].append([gene, gene_rec, communication_score, relation_id])              
                activated_cell_index[i] = ''
                activated_cell_index[j] = ''
#with gzip.open("/cluster/projects/schwartzgroup/fatema/find_ccc/" + 'synthetic_communication_scores_control_model_'+'b', 'wb') as fp: #b, b_1, a
#with gzip.open("/cluster/projects/schwartzgroup/fatema/find_ccc/" + 'synthetic_communication_scores_control_model_'+'a', 'wb') as fp: #b, b_1, a
#with gzip.open("/cluster/projects/schwartzgroup/fatema/find_ccc/" + 'synthetic_communication_scores_control_model_'+'c_notQuantileTransformed', 'wb') as fp: #b, b_1, a
with gzip.open("/cluster/projects/schwartzgroup/fatema/find_ccc/" + 'synthetic_communication_scores_control_model_'+'d_a_notQuantileTransformed', 'wb') as fp: #b, b_1, a
    pickle.dump([cells_ligand_vs_receptor,-1,ligand_list,activated_cell_index], fp) #a - [0:5]

'''
with gzip.open("/cluster/projects/schwartzgroup/fatema/find_ccc/" + 'synthetic_communication_scores_control_model_'+'c_notQuantileTransformed', 'rb') as fp: #b, b_1, a
    cells_ligand_vs_receptor,a,ligand_list,activated_cell_index = pickle.load(fp) #a - [0:5]
'''
     
    
lig_rec_dict_TP = []
datapoint_size = temp_x.shape[0]
for i in range (0, datapoint_size): 
    lig_rec_dict_TP.append([])  
    for j in range (0, datapoint_size):	
        lig_rec_dict_TP[i].append([])   
        lig_rec_dict_TP[i][j] = []
        	
ccc_index_dict = dict()
row_col = []
edge_weight = []
lig_rec = []
count_edge = 0
max_local = 0
local_list = np.zeros((20))
for i in range (0, len(cells_ligand_vs_receptor)):
    #ccc_j = []
    for j in range (0, len(cells_ligand_vs_receptor)):
        if distance_matrix[i][j] <= threshold_distance: 
            count_local = 0
            if len(cells_ligand_vs_receptor[i][j])>0:
                for k in range (0, len(cells_ligand_vs_receptor[i][j])):
                    gene = cells_ligand_vs_receptor[i][j][k][0]
                    gene_rec = cells_ligand_vs_receptor[i][j][k][1]
                    count_edge = count_edge + 1
                    count_local = count_local + 1
                    #print(count_edge)                      
                    mean_ccc = cells_ligand_vs_receptor[i][j][k][2]  #*dist_X[i,j]
                    row_col.append([i,j])
                    ccc_index_dict[i] = ''
                    ccc_index_dict[j] = ''
                    edge_weight.append([dist_X[i,j], mean_ccc])
                    lig_rec.append([gene, gene_rec])
                    if (gene_rec=='R1' and temp_x[j] in receptor_1_index_x) and (temp_x[i] in ligand_1_index_x):
                        lig_rec_dict_TP[i][j].append(gene_rec) #append([gene, gene_rec])
                    elif (gene_rec=='R2' and temp_x[j] in receptor_2_index_x) and (temp_x[i] in ligand_2_index_x):
                        lig_rec_dict_TP[i][j].append(gene_rec) #append([gene, gene_rec])
                
                if max_local < count_local:
                    max_local = count_local
            else:
                row_col.append([i,j])
                edge_weight.append([dist_X[i,j], 0])
                lig_rec.append(['', ''])
            local_list[count_local] = local_list[count_local] + 1


		
print('len row col %d'%len(row_col))
print('count local %d'%count_local) 
#with gzip.open("/cluster/projects/schwartzgroup/fatema/find_ccc/" + 'adjacency_records_synthetic_communication_scores_control_model_'+'b', 'wb') as fp:  # at least one of lig or rec has exp > respective knee point          
#with gzip.open("/cluster/projects/schwartzgroup/fatema/find_ccc/" + 'adjacency_records_synthetic_communication_scores_control_model_'+'a', 'wb') as fp:  # at least one of lig or rec has exp > respective knee point          
'''
with gzip.open("/cluster/projects/schwartzgroup/fatema/find_ccc/" + 'adjacency_records_synthetic_communication_scores_control_model_'+'c_notQuantileTransformed', 'wb') as fp:  # at least one of lig or rec has exp > respective knee point          
    pickle.dump([row_col, edge_weight, lig_rec, lig_rec_dict_TP], fp)
with gzip.open("/cluster/projects/schwartzgroup/fatema/find_ccc/" + 'adjacency_records_synthetic_communication_scores_control_model_nn_'+'c_notQuantileTransformed', 'wb') as fp:  # at least one of lig or rec has exp > respective knee point          
    pickle.dump([row_col, edge_weight, lig_rec, lig_rec_dict_TP], fp)
'''
with gzip.open("/cluster/projects/schwartzgroup/fatema/find_ccc/" + 'adjacency_records_synthetic_communication_scores_control_model_'+'d_a_notQuantileTransformed', 'wb') as fp:  # at least one of lig or rec has exp > respective knee point          
    pickle.dump([row_col, edge_weight, lig_rec, lig_rec_dict_TP], fp)

################## introducing pattern ####################
# (x,y) indexes whose R2 gene exp has to be changed. receptor_2_index_x = [2, 6, 10, 14, 18] 
#
#                 o(L2 = 12)
#   o(R2 = 12)    o(L2 = 12)
#
#
#
x_R2 = [2, 10, 18, 2, 10, 18, 2, 10, 18, 2, 10, 18, 2, 10, 18]
y_R2 = [3, 3, 3, 6, 6, 6, 9, 9, 9, 12, 12, 12, 15, 15, 15]
for index in range (0, len(x_R2)):
    index_i = x_R2[index]
    index_j = y_R2[index]
    datapoint_R2 = index_dict[index_i][index_j] 
    datapoint_L2_right = index_dict[index_i+1][index_j] 
    datapoint_L2_diagonal_right = index_dict[index_i+1][index_j+1] 
    
    cell_vs_gene[datapoint_R2,3] = 8
    cell_vs_gene[datapoint_L2_right,2] = 8
    cell_vs_gene[datapoint_L2_diagonal_right,2] = 8

# do the usual things
ligand_list = list(ligand_dict_dataset.keys())  

cells_ligand_vs_receptor = []
for i in range (0, cell_vs_gene.shape[0]):
    cells_ligand_vs_receptor.append([])
 

for i in range (0, cell_vs_gene.shape[0]):
    for j in range (0, cell_vs_gene.shape[0]):
        cells_ligand_vs_receptor[i].append([])
        cells_ligand_vs_receptor[i][j] = []

activated_cell_index = dict()
for gene in ligand_list:
    for i in range (0, cell_vs_gene.shape[0]): # ligand                 
        for j in range (0, cell_vs_gene.shape[0]): # receptor
            for gene_rec in ligand_dict_dataset[gene]:                
                if distance_matrix[i,j] > threshold_distance:
                    continue
                communication_score = cell_vs_gene[i][gene_index[gene]] * cell_vs_gene[j][gene_index[gene_rec]]               
                relation_id = 0
                cells_ligand_vs_receptor[i][j].append([gene, gene_rec, communication_score, relation_id])              
                activated_cell_index[i] = ''
                activated_cell_index[j] = ''
                
#with gzip.open("/cluster/projects/schwartzgroup/fatema/find_ccc/" + 'synthetic_communication_scores_control_model_'+'d_b_notQuantileTransformed', 'wb') as fp: #b, b_1, a
with gzip.open("/cluster/projects/schwartzgroup/fatema/find_ccc/" + 'synthetic_communication_scores_control_model_'+'d_c_notQuantileTransformed', 'wb') as fp: #b, b_1, a
    pickle.dump([cells_ligand_vs_receptor,-1,ligand_list,activated_cell_index], fp) #a - [0:5]
    
with gzip.open("/cluster/projects/schwartzgroup/fatema/find_ccc/" + 'synthetic_communication_scores_control_model_'+'d_c_notQuantileTransformed', 'rb') as fp: #b, b_1, a
    cells_ligand_vs_receptor,a,ligand_list,activated_cell_index = pickle.load(fp) #a - [0:5]

    
lig_rec_dict_TP = []
datapoint_size = temp_x.shape[0]
for i in range (0, datapoint_size): 
    lig_rec_dict_TP.append([])  
    for j in range (0, datapoint_size):	
        lig_rec_dict_TP[i].append([])   
        lig_rec_dict_TP[i][j] = []
        	
ccc_index_dict = dict()
row_col = []
edge_weight = []
lig_rec = []
count_edge = 0
max_local = 0
local_list = np.zeros((20))
for i in range (0, len(cells_ligand_vs_receptor)):
    #ccc_j = []
    for j in range (0, len(cells_ligand_vs_receptor)):
        if distance_matrix[i][j] <= threshold_distance: 
            count_local = 0
            if len(cells_ligand_vs_receptor[i][j])>0:
                for k in range (0, len(cells_ligand_vs_receptor[i][j])):
                    gene = cells_ligand_vs_receptor[i][j][k][0]
                    gene_rec = cells_ligand_vs_receptor[i][j][k][1]
                    count_edge = count_edge + 1
                    count_local = count_local + 1
                    #print(count_edge)                      
                    mean_ccc = cells_ligand_vs_receptor[i][j][k][2] #*dist_X[i,j]
                    row_col.append([i,j])
                    ccc_index_dict[i] = ''
                    ccc_index_dict[j] = ''
                    edge_weight.append([dist_X[i,j], mean_ccc])
                    lig_rec.append([gene, gene_rec])
                    if (gene_rec=='R1' and temp_x[j] in receptor_1_index_x) and (temp_x[i] in ligand_1_index_x):
                        lig_rec_dict_TP[i][j].append(gene_rec) #append([gene, gene_rec])
                    elif (gene_rec=='R2' and temp_x[j] in receptor_2_index_x) and (temp_x[i] in ligand_2_index_x):
                        lig_rec_dict_TP[i][j].append(gene_rec) #append([gene, gene_rec])
                
                if max_local < count_local:
                    max_local = count_local
            else:
                row_col.append([i,j])
                edge_weight.append([dist_X[i,j], 0])
                lig_rec.append(['', ''])
            local_list[count_local] = local_list[count_local] + 1


		
print('len row col %d'%len(row_col))
print('count local %d'%count_local) 

#with gzip.open("/cluster/projects/schwartzgroup/fatema/find_ccc/" + 'adjacency_records_synthetic_communication_scores_control_model_'+'d_b_notQuantileTransformed', 'wb') as fp:  # at least one of lig or rec has exp > respective knee point          
#    pickle.dump([row_col, edge_weight, lig_rec, lig_rec_dict_TP], fp)
with gzip.open("/cluster/projects/schwartzgroup/fatema/find_ccc/" + 'adjacency_records_synthetic_communication_scores_control_model_'+'d_c_notQuantileTransformed', 'wb') as fp:  # at least one of lig or rec has exp > respective knee point          
    pickle.dump([row_col, edge_weight, lig_rec, lig_rec_dict_TP], fp) #nn_
with gzip.open("/cluster/projects/schwartzgroup/fatema/find_ccc/" + 'adjacency_records_synthetic_communication_scores_control_model_nn_'+'d_c_notQuantileTransformed', 'rb') as fp:  # at least one of lig or rec has exp > respective knee point          
    row_col, edge_weight, lig_rec, lig_rec_dict_TP = pickle.load(fp)

###########################################################

'''
2000
gene 0, min: 0, max:13.7769 
gene 1, min: 0, max:13.4894 
gene 2, min: 0, max:13.7656 
gene 3, min: 0, max:13.8944 
gene 0, min: 22.5465, max:37.0469 
gene 1, min: 28.4818, max:41.517 
gene 2, min: 34.072, max:45.9164 
gene 3, min: 38.2734, max:53.4325 
len row col 177016
count local 2
'''
###############################################Visualization starts###################################################################################################
import os
#import glob
import pandas as pd
#import shutil
import csv
import numpy as np
import sys
import scikit_posthocs as post
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

import argparse
parser = argparse.ArgumentParser()
parser.add_argument( '--data_path', type=str, default='/cluster/home/t116508uhn/64630/cellrangere/' , help='The path to dataset') #'/cluster/projects/schwartzgroup/fatema/pancreatic_cancer_visium/210827_A00827_0396_BHJLJTDRXY_Notta_Karen/V10M25-61_D1_PDA_64630_Pa_P_Spatial10x_new/outs/'
parser.add_argument( '--data_name', type=str, default='V10M25-61_D1_PDA_64630_Pa_P_Spatial10x_new', help='The name of dataset')
parser.add_argument( '--generated_data_path', type=str, default='generated_data/', help='The folder to store the generated data')
parser.add_argument( '--embedding_data_path', type=str, default='new_alignment/Embedding_data_ccc_rgcn/' , help='The path to attention') #'/cluster/projects/schwartzgroup/fatema/pancreatic_cancer_visium/210827_A00827_0396_BHJLJTDRXY_Notta_Karen/V10M25-61_D1_PDA_64630_Pa_P_Spatial10x_new/outs/'
args = parser.parse_args()
#th_dist = 4
spot_diameter = 89.43 #pixels
threshold_distance = 1.5 #4 : a, b, c
k_nn = 4


#with gzip.open("/cluster/projects/schwartzgroup/fatema/find_ccc/" + 'synthetic_communication_scores_control_model_a_xny', 'rb') as fp:
with gzip.open("/cluster/projects/schwartzgroup/fatema/find_ccc/" + 'synthetic_communication_scores_control_model_d_xny', 'rb') as fp:
    temp_x, temp_y = pickle.load(fp)

datapoint_size = temp_x.shape[0]

coordinates = np.zeros((temp_x.shape[0],2))
for i in range (0, datapoint_size):
    coordinates[i][0] = temp_x[i]
    coordinates[i][1] = temp_y[i]
    
from sklearn.metrics.pairwise import euclidean_distances
distance_matrix = euclidean_distances(coordinates, coordinates)


'''for i in range (0, distance_matrix.shape[0]):
    if np.sort(distance_matrix[i])[1]<0.1:
        print(np.sort(distance_matrix[i])[0:5])'''

#####################################

#with gzip.open("/cluster/projects/schwartzgroup/fatema/find_ccc/" + 'adjacency_records_GAT_total_synthetic_region1_STnCCC_equallySpaced_data0', 'rb') as fp:             
#with gzip.open("/cluster/projects/schwartzgroup/fatema/find_ccc/" + 'adjacency_records_GAT_total_synthetic_region1_STnCCC_equallySpaced', 'rb') as fp:             
#    row_col, edge_weight = pickle.load(fp)
#with gzip.open("/cluster/projects/schwartzgroup/fatema/find_ccc/" + 'adjacency_records_synthetic_communication_scores_control_model_'+'c_notQuantileTransformed', 'rb') as fp: 
#with gzip.open("/cluster/projects/schwartzgroup/fatema/find_ccc/" + 'adjacency_records_GAT_selective_lr_STnCCC_separate_'+'all_density_kneepoint', 'rb') as fp:  #b, a:[0:5]   
with gzip.open("/cluster/projects/schwartzgroup/fatema/find_ccc/" + 'adjacency_records_synthetic_communication_scores_control_model_'+'d_c_notQuantileTransformed', 'rb') as fp: 
    row_col, edge_weight, lig_rec, lig_rec_dict_TP = pickle.load(fp) 
    
#####################################
keep_i = []
for i in range (0, len(row_col)):
    item_i = row_col[i][0]
    item_j = row_col[i][1]
    if temp_x[item_i]<=21 or temp_x[item_j]<=21: 
        keep_i.append(i)
    
    
#####################################
lig_rec_dict = []
datapoint_size = temp_x.shape[0]
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
    if temp_x[i]<=21 or temp_x[j]<=21:
        if lig_rec[index][1]=='R1':
            total_type[0] = total_type[0]+1
        elif lig_rec[index][1]=='R2':
            total_type[1] = total_type[1]+1
######################################	

attention_scores = []
datapoint_size = temp_x.shape[0]
for i in range (0, datapoint_size):
    attention_scores.append([])   
    for j in range (0, datapoint_size):	
        attention_scores[i].append([])   
        attention_scores[i][j] = []
	    
#attention_scores = np.zeros((datapoint_size,datapoint_size))
distribution = []
ccc_index_dict = dict()
for index in range (0, len(row_col)):
    i = row_col[index][0]
    j = row_col[index][1]
    lig_rec_dict[i][j].append(lig_rec[index])
    #attention_scores[i][j] = edge_weight[index][1]
    attention_scores[i][j].append(edge_weight[index][1])
    distribution.append(edge_weight[index][1])
    if edge_weight[index][1]>0:
        ccc_index_dict[i] = ''
        ccc_index_dict[j] = ''    
	


ccc_index_dict = dict()
threshold_down =  np.percentile(sorted(distribution), 98)
threshold_up =  np.percentile(sorted(distribution), 100)
connecting_edges = np.zeros((temp_x.shape[0],temp_x.shape[0]))
for j in range (0, datapoint_size):
    #threshold =  np.percentile(sorted(attention_scores[:,j]), 97) #
    for i in range (0, datapoint_size):
        atn_score_list = attention_scores[i][j]
        for k in range (0, len(atn_score_list)):   
            if attention_scores[i][j][k] >= threshold_down and attention_scores[i][j][k] <= threshold_up: #np.percentile(sorted(distribution), 50):
                connecting_edges[i][j] = 1
                #lig_rec_dict_filtered[i][j].append(lig_rec_dict[i][j][k][1])
                ccc_index_dict[i] = ''
                ccc_index_dict[j] = ''

################

########
#X_attention_filename = args.embedding_data_path + args.data_name + '/' + 'totalsynccc_gat_r1_2attr_noFeature_STnCCC_equallySpaced_knn_data0_a_attention.npy'
#X_attention_filename = args.embedding_data_path + args.data_name + '/' + 'totalsynccc_gat_r1_2attr_noFeature_STnCCC_equallySpaced_knn_data0_attention.npy'
#X_attention_filename = args.embedding_data_path + args.data_name + '/' + 'synthetic_communication_scores_control_model_a_attention_l1.npy' #a
#X_attention_filename = args.embedding_data_path + args.data_name + '/' + 'synthetic_communication_scores_control_model_b_attention_l1.npy' #a
#X_attention_filename = args.embedding_data_path + args.data_name + '/' + 'synthetic_communication_scores_control_model_a_notQuantileTransformed_attention_l1.npy' #a
#X_attention_filename = args.embedding_data_path + args.data_name + '/' + 'synthetic_communication_scores_control_model_a_notQuantileTransformed_h1024_attention_l1.npy' #a
#X_attention_filename = args.embedding_data_path + args.data_name + '/' + 'synthetic_communication_scores_control_model_c_notQuantileTransformed_h512_attention_l1.npy' #a
#X_attention_filename = args.embedding_data_path + args.data_name + '/' + 'synthetic_communication_scores_control_model_d_b_notQuantileTransformed_h512_attention_l1.npy' #a
#X_attention_filename = args.embedding_data_path + args.data_name + '/' + 'synthetic_communication_scores_control_model_d_b_notQuantileTransformed_h1024_r3_attention_l1.npy' #a
#X_attention_filename = args.embedding_data_path + args.data_name + '/' + 'synthetic_communication_scores_control_model_d_b_notQuantileTransformed_h2048_r3_attention_l1.npy' #a
X_attention_filename = args.embedding_data_path + args.data_name + '/' + 'synthetic_communication_scores_control_model_d_c_notQuantileTransformed_h1024_attention_l1.npy' 
X_attention_bundle = np.load(X_attention_filename, allow_pickle=True) 

attention_scores = []
datapoint_size = temp_x.shape[0]
for i in range (0, datapoint_size):
    attention_scores.append([])   
    for j in range (0, datapoint_size):	
        attention_scores[i].append([])   
        attention_scores[i][j] = []
	
#attention_scores = np.zeros((2000,2000))
distribution = []
for index in range (0, X_attention_bundle[0].shape[1]):
    i = X_attention_bundle[0][0][index]
    j = X_attention_bundle[0][1][index]
    ###################################
    '''if i==row_col[index][0] and j==row_col[index][1]:
        continue
    else:
        print('found mismatch')
        break
    '''
    ###################################
    attention_scores[i][j].append(X_attention_bundle[3][index][0]) #X_attention_bundle[2][index][0]
    distribution.append(X_attention_bundle[3][index][0])
    #attention_scores[i][j] = X_attention_bundle[3][index][0] #X_attention_bundle[2][index][0]
    #distribution.append(attention_scores[i][j])
#######################
'''attention_scores_normalized = np.zeros((temp_x.shape[0],temp_x.shape[0]))
for index in range (0, X_attention_bundle[0].shape[1]):
    i = X_attention_bundle[0][0][index]
    j = X_attention_bundle[0][1][index]
    attention_scores_normalized [i][j] = X_attention_bundle[1][index][0]
    #attention_scores[i][j] =  X_attention_bundle[1][index][0]
    #distribution.append(attention_scores[i][j])
##############
adjacency_matrix = np.zeros((temp_x.shape[0],temp_x.shape[0]))
for index in range (0, X_attention_bundle[0].shape[1]):
    i = X_attention_bundle[0][0][index]
    j = X_attention_bundle[0][1][index]
    adjacency_matrix [i][j] = 1

ccc_index_dict = dict()
threshold_down =  np.percentile(sorted(distribution), 95)
threshold_up =  np.percentile(sorted(distribution), 100)
connecting_edges = np.zeros((temp_x.shape[0],temp_x.shape[0]))

for j in range (0, attention_scores.shape[1]):
    #threshold =  np.percentile(sorted(attention_scores[:,j]), 97) #
    for i in range (0, attention_scores.shape[0]):
        
        if attention_scores[i][j] >= threshold_down and attention_scores[i][j] <= threshold_up: #np.percentile(sorted(distribution), 50):
            connecting_edges[i][j] = 1
            ccc_index_dict[i] = ''
            ccc_index_dict[j] = ''
'''
for percentage_value in [67, 70, 75, 78, 85, 90, 93, 95, 97]:
#for percentage_value in [79, 85, 90, 93, 95, 97]:
    existing_lig_rec_dict = []
    datapoint_size = temp_x.shape[0]
    for i in range (0, datapoint_size):
        existing_lig_rec_dict.append([])   
        for j in range (0, datapoint_size):	
            existing_lig_rec_dict[i].append([])   
            existing_lig_rec_dict[i][j] = []

    ccc_index_dict = dict()
    threshold_down =  np.percentile(sorted(distribution), percentage_value)
    threshold_up =  np.percentile(sorted(distribution), 100)
    connecting_edges = np.zeros((temp_x.shape[0],temp_x.shape[0]))
    rec_dict = defaultdict(dict)
    for i in range (0, datapoint_size):
        for j in range (0, datapoint_size):
            atn_score_list = attention_scores[i][j]
            #print(len(atn_score_list))
            for k in range (0, len(atn_score_list)):
                if attention_scores[i][j][k] >= threshold_down and attention_scores[i][j][k] <= threshold_up: #np.percentile(sorted(distribution), 50):
                    connecting_edges[i][j] = 1
                    ccc_index_dict[i] = ''
                    ccc_index_dict[j] = ''
                    existing_lig_rec_dict[i][j].append(lig_rec_dict[i][j][k][1])


    #############
    num_pairs = 2
    real_count = np.zeros((num_pairs))
    pred_count = np.zeros((num_pairs))
    for i in range (0, datapoint_size):
        for j in range (0, datapoint_size):
            if temp_x[i]<=21 or temp_x[j]<=21: 
                if len(lig_rec_dict_TP[i][j])>0:
                    #print(lig_rec_dict_TP[i][j])
                    for k in range (0, len(lig_rec_dict_TP[i][j])):
                        if lig_rec_dict_TP[i][j][k] == 'R1':
                            real_count[0] = real_count[0] + 1
                            if 'R1' in existing_lig_rec_dict[i][j]:
                                pred_count[0] = pred_count[0] + 1

                        elif lig_rec_dict_TP[i][j][k] == 'R2':
                            real_count[1] = real_count[1] + 1
                            if 'R2' in existing_lig_rec_dict[i][j]:
                                pred_count[1] = pred_count[1] + 1


    model_count = np.zeros((num_pairs))
    real_lr_count = np.zeros((num_pairs))

    for i in range (0, datapoint_size):
        for j in range (0, datapoint_size):
            if temp_x[i]<=21 or temp_x[j]<=21:
                if len(existing_lig_rec_dict[i][j])>0:
                    #print(lig_rec_dict[i][j])
                    for k in range (0, len(existing_lig_rec_dict[i][j])):
                        if existing_lig_rec_dict[i][j][k] == 'R1':
                            model_count[0] = model_count[0] + 1
                            if 'R1' in lig_rec_dict_TP[i][j]:
                                real_lr_count[0] = real_lr_count[0] + 1

                        elif existing_lig_rec_dict[i][j][k] == 'R2':
                            model_count[1] = model_count[1] + 1
                            if 'R2' in lig_rec_dict_TP[i][j]:
                                real_lr_count[1] = real_lr_count[1] + 1	
    '''
    print('real_count',real_count)
    print('pred_count',pred_count)
    print('model_count',model_count )
    print('real_lr_count',real_lr_count)
    '''
    TN = 14820 # 7656 - 4474 
    TN = np.zeros((2))
    TN[0] = total_type[0] - real_count[0]
    TN[1] = total_type[1] - real_count[1]
    '''for i in range (0, num_pairs):
        #print('%d, %d, %d, %d, %d, %g, %g'%(i, real_count[i], pred_count[i], model_count[i], real_lr_count[i], (pred_count[i]/real_count[i]),(model_count[i]-real_lr_count[i])/14820))
        print('%g, %g'%((pred_count[i]/real_count[i]),(model_count[i]-real_lr_count[i])/TN))
    '''
    print('%g, %g, %g, %g'%((pred_count[0]/real_count[0]),(model_count[0]-real_lr_count[0])/TN[0],(pred_count[1]/real_count[1]),(model_count[1]-real_lr_count[1])/TN[1]))

                        
                        
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

datapoint_label = []
node_list = []
for i in range (0, temp_x.shape[0]):
    if count_points_component[labels[i]]>1:
        datapoint_label.append(2) #
        #if coordinates[i][0] <100 and (coordinates[i][1]>150 and coordinates[i][1]<250):
            #print('%d'%i)
            #node_list.append(i)
        #datapoint_label.append(index_dict[labels[i]])
    else:
        datapoint_label.append(0)
	
#############
'''
datapoint_label = []
for i in range (0, temp_x.shape[0]):
    if i in ccc_index_dict:
        datapoint_label.append(2)
    else:
        datapoint_label.append(0)
'''
########
plt.gca().set_aspect(1)	

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

number = 20
cmap = plt.get_cmap('tab20c')
colors_2 = [cmap(i) for i in np.linspace(0, 1, number)]

colors=colors+colors_2
       
id_label = [0,2]
for j in id_label:
#for j in range (0, id_label+1):
    x_index=[]
    y_index=[]
    #fillstyles_type = []
    for i in range (0, temp_x.shape[0]):
        if datapoint_label[i] == j:
            x_index.append(temp_x[i])
            y_index.append(temp_y[i])
    #print(len(x_index))
            
            ##############
    plt.scatter(x=x_index, y=y_index, label=j, color=colors[j], s=1)   
    
plt.legend(fontsize=4,loc='upper right')


save_path = '/cluster/home/t116508uhn/64630/'
plt.savefig(save_path+'toomanycells_PCA_64embedding_pathologist_label_l1mp5_temp_plot.svg', dpi=400)
plt.clf()
 
plt.hist(distribution, color = 'blue',
         bins = int(len(distribution)/5))
save_path = '/cluster/home/t116508uhn/64630/'
plt.savefig(save_path+'toomanycells_PCA_64embedding_pathologist_label_l1mp5_temp_plot.svg', dpi=400)
plt.clf()
####################
ids = []
x_index=[]
y_index=[]
colors_point = []
for i in range (0, len(temp_x)):    
    ids.append(i)
    x_index.append(temp_x[i]*100)
    y_index.append(temp_y[i]*100)    
    colors_point.append(colors[datapoint_label[i]]) 
  
max_x = np.max(x_index)
max_y = np.max(y_index)


from pyvis.network import Network
import networkx as nx
import matplotlib#.colors.rgb2hex as rgb2hex
    
g = nx.MultiDiGraph(directed=True) #nx.Graph() MultiDiGraph
marker_size = 'circle'
for i in range (0, len(temp_x)):
    '''if barcode_type[barcode_info[i][0]] == 0:
        marker_size = 'circle'
    elif barcode_type[barcode_info[i][0]] == 1:
        marker_size = 'box'
    else:
        marker_size = 'ellipse'
    '''
    g.add_node(int(ids[i]), x=int(x_index[i]), y=int(y_index[i]), label = str(i), physics=False, shape = marker_size, color=matplotlib.colors.rgb2hex(colors_point[i]))
   		
#nx.draw(g, pos= nx.circular_layout(g)  ,with_labels = True, edge_color = 'b', arrowstyle='fancy')
#g.toggle_physics(True)
nt = Network( directed=True) #"500px", "500px",
nt.from_nx(g)
for i in range (0, datapoint_size):
    for j in range (0, datapoint_size):
        atn_score_list = attention_scores[i][j]
        #print(len(atn_score_list))
        for k in range (0, len(atn_score_list)):
            if attention_scores[i][j][k] >= threshold_down:
                #print('hello')
                nt.add_edge(int(i), int(j), title = ) #, weight=1, arrowsize=int(20),  arrowstyle='fancy'

nt.show('mygraph.html')


#g.show('mygraph.html')
cp mygraph.html /cluster/home/t116508uhn/64630/mygraph.html
