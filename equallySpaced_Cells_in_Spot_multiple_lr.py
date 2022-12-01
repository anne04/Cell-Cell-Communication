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
import random

import argparse
parser = argparse.ArgumentParser()
parser.add_argument( '--data_path', type=str, default='/cluster/home/t116508uhn/64630/cellrangere/' , help='The path to dataset') #'/cluster/projects/schwartzgroup/fatema/pancreatic_cancer_visium/210827_A00827_0396_BHJLJTDRXY_Notta_Karen/V10M25-61_D1_PDA_64630_Pa_P_Spatial10x_new/outs/'
parser.add_argument( '--data_name', type=str, default='V10M25-61_D1_PDA_64630_Pa_P_Spatial10x_new', help='The name of dataset')
parser.add_argument( '--generated_data_path', type=str, default='generated_data/', help='The folder to store the generated data')
parser.add_argument( '--embedding_data_path', type=str, default='new_alignment/Embedding_data_ccc_rgcn/' , help='The path to attention') #'/cluster/projects/schwartzgroup/fatema/pancreatic_cancer_visium/210827_A00827_0396_BHJLJTDRXY_Notta_Karen/V10M25-61_D1_PDA_64630_Pa_P_Spatial10x_new/outs/'
args = parser.parse_args()
#th_dist = 4
spot_diameter = 89.43 #pixels
k_nn = 5
lr_pair_type = 2
data_fold = args.data_path + 'filtered_feature_bc_matrix.h5'
print(data_fold)

#cell_vs_gene = adata_X   # rows = cells, columns = genes

datapoint_size = 3000
x_max = 200
x_min = 0
y_max = 40
y_min = 0
#################################

cell_count_list = []
temp_x = []
temp_y = []
i = x_min
while i <= x_max:
    j = y_min
    while j <= y_max:
        cell_count = 1 #random.sample(range(1, 4), 1)[0]
        for k in range (0, cell_count):	#each spot have 3 cells   	
            temp_x.append(i)
            temp_y.append(j)
        cell_count_list.append(cell_count)
        j = j + 2
    i = i + 2
    
#0, 2, 4, ...24, 26, 28   


temp_x = np.array(temp_x)
temp_y = np.array(temp_y)

##############################################

print(len(temp_x))

plt.scatter(x=np.array(temp_x), y=np.array(temp_y), s=1)

save_path = '/cluster/home/t116508uhn/64630/'

plt.savefig(save_path+'synthetic_spatial_plot_equallySpaced_data0.svg', dpi=400)
#plt.savefig(save_path+'synthetic_spatial_plot_3.svg', dpi=400)
plt.clf()

#with gzip.open("/cluster/projects/schwartzgroup/fatema/find_ccc/" + 'scRNAseq_spatial_location_synthetic_equallySpaced_cell_spot_multiple_lr_data0', 'wb') as fp:
with gzip.open("/cluster/projects/schwartzgroup/fatema/find_ccc/" + 'scRNAseq_spatial_location_synthetic_equallySpaced_multiple_lr_data0', 'wb') as fp:
    pickle.dump([temp_x, temp_y, cell_count, k_nn, lr_pair_type], fp)


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
	
region_list = [[50, 150, 8, 20]] #[[20, 40, 3, 7], [40, 80, 12, 18]] #[60, 80, 1, 7] 
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

num1 = np.zeros((1,2)) 
num_center = np.zeros((2,2))
# c1
num_center[0][0]=100
num_center[0][1]= 20
# c2
num_center[1][0]=100
num_center[1][1]= 8


a = 20
b = +558
limit_list =[[300,500],[100,200]] #data2: [[20,50],[300,500],[100,200]] #data=1:[[200,500],[20,50],[20,50]]
ccc_index_dict = dict()
row_col = []
edge_weight = []
ccc_score_max = -1
ccc_score_min = 1000
for region_index in range (0, len(region_list)):
    region = region_list[region_index]
    a = limit_list[region_index][0]
    b = limit_list[region_index][1]
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
                    # point i
                    num1[0][0]=temp_x[i]
                    num1[0][1]=temp_y[i]
                    # c1
                    mean_ccc = euclidean_distances(num1, np.array([[num_center[0,0], num_center[0,1]]]))[0][0]           	    
                    row_col.append([i,j])                    
                    edge_weight.append([dist_X[i,j], mean_ccc])
                    # c2              
                    mean_ccc = euclidean_distances(num1, np.array([[num_center[1,0], num_center[1,1]]]))[0][0]                    
                    row_col.append([i,j])                    
                    edge_weight.append([dist_X[i,j], mean_ccc])
                    if  mean_ccc > ccc_score_max:
                        ccc_score_max = mean_ccc
                    if  mean_ccc < ccc_score_min:
                        ccc_score_min = mean_ccc                       
                    #mean_ccc = ccc_scores[k]
                    ccc_index_dict[i] = ''
                    ccc_index_dict[j] = ''             
                    flag = 1
		    
print("len row_col with ccc %d, min_ccc_score %g, max_ccc_score %g"%(len(row_col), ccc_score_min, ccc_score_max))

for i in range (0, distance_matrix.shape[0]):
    for j in range (0, distance_matrix.shape[1]):
        if dist_X[i,j] > -1:
            if i not in ccc_index_dict and j not in ccc_index_dict:
                row_col.append([i,j])
                edge_weight.append([dist_X[i,j], 0])
                #edge_weight.append([0.5, 0])

with gzip.open("/cluster/projects/schwartzgroup/fatema/find_ccc/" + 'adjacency_records_GAT_total_synthetic_region1_STnCCC_equallySpaced_multiple_lr_data0', 'wb') as fp:
#with gzip.open("/cluster/projects/schwartzgroup/fatema/find_ccc/" + 'adjacency_records_GAT_total_synthetic_region1_STnCCC_equallySpaced_cell_spot_multiple_lr_data0', 'wb') as fp:
    pickle.dump([row_col, edge_weight], fp)
		  
print(len(row_col))
print(len(temp_x))

###############################################Visualization starts###################################################################################################


#with gzip.open("/cluster/projects/schwartzgroup/fatema/find_ccc/" + 'scRNAseq_spatial_location_synthetic_equallySpaced_cell_spot_multiple_lr_data0', 'rb') as fp:
with gzip.open("/cluster/projects/schwartzgroup/fatema/find_ccc/" + 'scRNAseq_spatial_location_synthetic_equallySpaced_multiple_lr_data0', 'rb') as fp:
    temp_x, temp_y, cell_count_list, k_nn, lr_pair_type = pickle.load(fp)

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

with gzip.open("/cluster/projects/schwartzgroup/fatema/find_ccc/" + 'adjacency_records_GAT_total_synthetic_region1_STnCCC_equallySpaced_cell_spot_data0', 'rb') as fp:             
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
X_attention_filename = args.embedding_data_path + args.data_name + '/' + 'totalsynccc_gat_r1_2attr_noFeature_STnCCC_equallySpaced_multiple_lr_knn_data0_attention.npy'
#X_attention_filename = args.embedding_data_path + args.data_name + '/' + 'totalsynccc_gat_r1_2attr_noFeature_STnCCC_equallySpaced_cell_spot_multiple_lr_knn_data0_attention.npy'
X_attention_bundle = np.load(X_attention_filename, allow_pickle=True) 

attention_scores = []
datapoint_size = temp_x.shape[0]
for i in range (0, datapoint_size):
    attention_scores.append([])   
    for j in range (0, datapoint_size):	
        attention_scores[i].append([])   
        attention_scores[i][j] = []

distribution = []
for index in range (0, X_attention_bundle[0].shape[1]):
    i = X_attention_bundle[0][0][index]
    j = X_attention_bundle[0][1][index]
    attention_scores[i][j].append(X_attention_bundle[2][index][0])
    '''if attention_scores[i][j]<-.25:
        attention_scores[i][j] = (attention_scores[i][j]+0.25) * (-1)
    '''
    distribution.append(X_attention_bundle[2][index][0])


ccc_index_dict = dict()
threshold_down =  np.percentile(sorted(distribution), 80)
threshold_up =  np.percentile(sorted(distribution), 100)
connecting_edges = np.zeros((temp_x.shape[0],temp_x.shape[0]))
c1_list = []
c2_list = []
for j in range (0, datapoint_size):
    for i in range (0, datapoint_size):
        atn_score_list = attention_scores[i][j]
        for k in range (0, len(atn_score_list)):
            if attention_scores[i][j][k] >= threshold_down and attention_scores[i][j][k] <= threshold_up: #np.percentile(sorted(distribution), 50):
                connecting_edges[i][j] = 1
                ccc_index_dict[i] = ''
                ccc_index_dict[j] = ''
                if k==0:
                    c1_list.append([temp_x[i],temp_y[i]])	
                elif k==1:
                    c2_list.append([temp_x[i],temp_y[i]])

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
