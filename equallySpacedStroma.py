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
k_nn = 4

data_fold = args.data_path + 'filtered_feature_bc_matrix.h5'
print(data_fold)

#cell_vs_gene = adata_X   # rows = cells, columns = genes

datapoint_size = 2000
x_max = 200
x_min = 0
y_max = 40
y_min = 0
#################################
temp_x = []
temp_y = []
i = x_min
while i < x_max:
    j = y_min
    while j < y_max:
        temp_x.append(i)
        temp_y.append(j)
        j = j + 2
    i = i + 2
    
#0, 2, 4, ...24, 26, 28   
region_list =  [[25, 75, 1, 15], [125, 175, 11, 25]] #[[20, 40, 3, 7], [40, 60, 12, 18]] #[60, 80, 1, 7] 
for region in region_list:
    x_max = region[1]
    x_min = region[0]
    y_max = region[3]
    y_min = region[2]
    i = x_min
    while i < x_max:
        j = y_min
        while j < y_max:
            temp_x.append(i)
            temp_y.append(j)
            j = j + 2
        i = i + 2

temp_x = np.array(temp_x)
temp_y = np.array(temp_y)

###############################################


discard_points = dict()
for i in range (0, temp_x.shape[0]):
    if i not in discard_points:
        for j in range (i+1, temp_x.shape[0]):
            if j not in discard_points:
                if euclidean_distances(np.array([[temp_x[i],temp_y[i]]]), np.array([[temp_x[j],temp_y[j]]]))[0][0] < 0.5 :
                    print('i: %d and j: %d'%(i,j))
                    discard_points[j]=''

coord_x = []
coord_y = []
for i in range (0, temp_x.shape[0]):
    if i not in discard_points:
        coord_x.append(temp_x[i])
        coord_y.append(temp_y[i])

temp_x = np.array(coord_x)
temp_y = np.array(coord_y)

print(len(temp_x))

plt.scatter(x=np.array(temp_x), y=np.array(temp_y), s=1)

save_path = '/cluster/home/t116508uhn/64630/'

plt.savefig(save_path+'synthetic_spatial_plot_equallySpacedStroma_data0.svg', dpi=400)
#plt.savefig(save_path+'synthetic_spatial_plot_3.svg', dpi=400)
plt.clf()

with gzip.open("/cluster/projects/schwartzgroup/fatema/find_ccc/" + 'scRNAseq_spatial_location_synthetic_equallySpacedStroma_data0', 'wb') as fp:
#with gzip.open("/cluster/projects/schwartzgroup/fatema/find_ccc/" + 'scRNAseq_spatial_location_synthetic_2', 'wb') as fp:
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
	
    list_indx = list(np.argsort(dist_X[:,j]))
    k_higher = list_indx[len(list_indx)-k_nn:len(list_indx)]
    for i in range(0, distance_matrix.shape[0]):
        if i not in k_higher:
            dist_X[i,j] = -1
	

region_list =  [[25, 75, 26, 35], [25, 75, 1, 15], [125, 175, 11, 25]] #data0_a # data0:[ [25, 75, 1, 15], [125, 175, 11, 25]]
ccc_scores_count = []
for region in region_list:
    count = 0
    for i in range (0, distance_matrix.shape[0]):
    #ccc_j = []
        for j in range (0, distance_matrix.shape[1]):
            if dist_X[i,j] > -1:  
                region_x_min = region[0]
                region_x_max = region[1]
                region_y_min = region[2]
                region_y_max = region[3]  		
                if temp_x[i] > region_x_min and temp_x[i] < region_x_max and temp_y[i] > region_y_min and temp_y[i] <  region_y_max: 
                    count = count + 1
    ccc_scores_count.append(count)          

		
a = 20
b = +558
ccc_index_dict = dict()
row_col = []
edge_weight = []
for region_index in range (0, len(region_list)):
    region = region_list[region_index]
    ccc_scores = (b - a) * np.random.random_sample(size=ccc_scores_count[region_index]+1) + a
    k=0
    for i in range (0, distance_matrix.shape[0]):
        for j in range (0, distance_matrix.shape[1]):
            if dist_X[i,j] > -1:
                flag = 0          
                region_x_min = region[0]
                region_x_max = region[1]
                region_y_min = region[2]
                region_y_max = region[3]  		
                if temp_x[i] > region_x_min and temp_x[i] < region_x_max and temp_y[i] > region_y_min and temp_y[i] <  region_y_max: 
                    mean_ccc = ccc_scores[k]
                    k = k + 1
                    row_col.append([i,j])
                    ccc_index_dict[i] = ''
                    ccc_index_dict[j] = ''
                    edge_weight.append([dist_X[i,j], mean_ccc])
                    #edge_weight.append([0.5, mean_ccc])
                    #print([0.5, mean_ccc])
                    flag = 1
			    
print("len row_col with ccc %d"%len(row_col))

for i in range (0, distance_matrix.shape[0]):
    for j in range (0, distance_matrix.shape[1]):
        if dist_X[i,j] > -1:
            if i not in ccc_index_dict and j not in ccc_index_dict:
                row_col.append([i,j])
		#edge_weight.append([sp_weight[i][j], 0])
                edge_weight.append([dist_X[i,j], 0])
                #edge_weight.append([0.5, 0])


with gzip.open("/cluster/projects/schwartzgroup/fatema/find_ccc/" + 'adjacency_records_GAT_total_synthetic_region1_STnCCC_equallySpacedStroma_data0_a', 'wb') as fp:             		 		
#with gzip.open("/cluster/projects/schwartzgroup/fatema/find_ccc/" + 'adjacency_records_GAT_total_synthetic_region1_STnCCC_equallySpacedStroma_data0', 'wb') as fp:             		  
#with gzip.open("/cluster/projects/schwartzgroup/fatema/find_ccc/" + 'adjacency_records_GAT_total_synthetic_region1_STnCCC_equallySpacedStroma', 'wb') as fp:             
#with gzip.open("/cluster/projects/schwartzgroup/fatema/find_ccc/" + 'adjacency_records_GAT_total_synthetic_region1_STnCCC_equallySpaced', 'wb') as fp:             
#with gzip.open("/cluster/projects/schwartzgroup/fatema/find_ccc/" + 'adjacency_records_GAT_synthetic_region1_onlyccc_70', 'wb') as fp:
    pickle.dump([row_col, edge_weight], fp)
		  
print(len(row_col))
print(len(temp_x))

###############################################Visualization starts###################################################################################################


with gzip.open("/cluster/projects/schwartzgroup/fatema/find_ccc/" + 'scRNAseq_spatial_location_synthetic_equallySpacedStroma_data0', 'rb') as fp:
#with gzip.open("/cluster/projects/schwartzgroup/fatema/find_ccc/" + 'scRNAseq_spatial_location_synthetic_2', 'rb') as fp:
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

with gzip.open("/cluster/projects/schwartzgroup/fatema/find_ccc/" + 'adjacency_records_GAT_total_synthetic_region1_STnCCC_equallySpacedStroma_data1', 'rb') as fp:             
#with gzip.open("/cluster/projects/schwartzgroup/fatema/find_ccc/" + 'adjacency_records_GAT_total_synthetic_region1_STnCCC_equallySpaced', 'rb') as fp:             
    row_col, edge_weight = pickle.load(fp)

attention_scores = np.zeros((datapoint_size,datapoint_size))
distribution = []
ccc_index_dict = dict()
for index in range (0, len(row_col)):
    i = row_col[index][0]
    j = row_col[index][1]
    
    attention_scores[i][j] = edge_weight[index][1]
    distribution.append(attention_scores[i][j])
    if edge_weight[index][1]>0:
        ccc_index_dict[i] = ''
        ccc_index_dict[j] = ''    
	
ccc_index_dict = dict()
threshold_down =  np.percentile(sorted(distribution), 80)
threshold_up =  np.percentile(sorted(distribution), 100)
connecting_edges = np.zeros((temp_x.shape[0],temp_x.shape[0]))
for j in range (0, attention_scores.shape[1]):
    #threshold =  np.percentile(sorted(attention_scores[:,j]), 97) #
    for i in range (0, attention_scores.shape[0]):
        if attention_scores[i][j] >= threshold_down and attention_scores[i][j] <= threshold_up: #np.percentile(sorted(distribution), 50):
            connecting_edges[i][j] = 1
            ccc_index_dict[i] = ''
            ccc_index_dict[j] = ''

################

########
X_attention_filename = args.embedding_data_path + args.data_name + '/' + 'totalsynccc_gat_r1_2attr_noFeature_STnCCC_equallySpacedStroma_knn_data0_a_attention.npy'
X_attention_bundle = np.load(X_attention_filename, allow_pickle=True) 

attention_scores = np.zeros((temp_x.shape[0],temp_x.shape[0]))
distribution = []
for index in range (0, X_attention_bundle[0].shape[1]):
    i = X_attention_bundle[0][0][index]
    j = X_attention_bundle[0][1][index]
    attention_scores[i][j] = X_attention_bundle[2][index][0]
    if attention_scores[i][j]<-.25:
        attention_scores[i][j] = (attention_scores[i][j]+0.25) * (-1)

    distribution.append(attention_scores[i][j])
##############
attention_scores_normalized = np.zeros((temp_x.shape[0],temp_x.shape[0]))
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
threshold_down =  np.percentile(sorted(distribution), 70)
threshold_up =  np.percentile(sorted(distribution), 100)
connecting_edges = np.zeros((temp_x.shape[0],temp_x.shape[0]))

for j in range (0, attention_scores.shape[1]):
    #threshold =  np.percentile(sorted(attention_scores[:,j]), 97) #
    for i in range (0, attention_scores.shape[0]):
        if attention_scores[i][j] >= threshold_down and attention_scores[i][j] <= threshold_up: #np.percentile(sorted(distribution), 50):
            connecting_edges[i][j] = 1
            ccc_index_dict[i] = ''
            ccc_index_dict[j] = ''


#############


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
