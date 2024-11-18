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
from scipy import sparse
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components
from sklearn.metrics.pairwise import euclidean_distances




import argparse
parser = argparse.ArgumentParser()
parser.add_argument( '--embedding_data_path', type=str, default='new_alignment/Embedding_data_ccc_rgcn/' , help='The path to attention') #'/cluster/projects/schwartzgroup/fatema/pancreatic_cancer_visium/210827_A00827_0396_BHJLJTDRXY_Notta_Karen/V10M25-61_D1_PDA_64630_Pa_P_Spatial10x_new/outs/'
args = parser.parse_args()



########################################################################################
noise_type = 'no_noise' #'high_noise' #'low_noise' #
with gzip.open("/cluster/projects/schwartzgroup/fatema/find_ccc/synthetic_data/type_equidistant/"+ noise_type +"/equidistant_" + noise_type + "_coordinate" , 'rb') as fp: #datatype
#with gzip.open("/cluster/projects/schwartzgroup/fatema/find_ccc/" + 'synthetic_data_ccc_roc_control_model_'+ options  +'_xny', 'rb') as fp: #datatype
    temp_x, temp_y , ccc_region = pickle.load(fp) #

datapoint_size = temp_x.shape[0]

coordinates = np.zeros((temp_x.shape[0],2))
for i in range (0, datapoint_size):
    coordinates[i][0] = temp_x[i]
    coordinates[i][1] = temp_y[i]
    
from sklearn.metrics.pairwise import euclidean_distances
distance_matrix = euclidean_distances(coordinates, coordinates)

with gzip.open("/cluster/projects/schwartzgroup/fatema/find_ccc/synthetic_data/type_equidistant/"+ noise_type +"/equidistant_"+noise_type+"_ground_truth_ccc" , 'rb') as fp:            
    lr_database, lig_rec_dict_TP, random_activation = pickle.load( fp)


ligand_dict_dataset = defaultdict(dict)
for i in range (0, len(lr_database)):
    ligand_dict_dataset[lr_database[i][0]][lr_database[i][1]] = i
    
ligand_list = list(ligand_dict_dataset.keys())  

with gzip.open("/cluster/projects/schwartzgroup/fatema/find_ccc/synthetic_data/type_equidistant/"+noise_type+"/equidistant_"+noise_type+"_input_graph" , 'rb') as fp:  # +'_'+'notQuantileTransformed'at least one of lig or rec has exp > respective knee point          
    row_col, edge_weight, lig_rec  = pickle.load(fp)  #, lr_database, lig_rec_dict_TP, random_activation
    

	
max_tp_distance = 0
datapoint_size = temp_x.shape[0]              
total_type = np.zeros((len(lr_database)))
for i in range (0, datapoint_size):
    for j in range (0, datapoint_size):
        if i==j: 
            continue
        if i in lig_rec_dict_TP and j in lig_rec_dict_TP[i] and len(lig_rec_dict_TP[i][j]) > 0:
            for k in range (0, len(lig_rec_dict_TP[i][j])):
                total_type[lig_rec_dict_TP[i][j][k]] = total_type[lig_rec_dict_TP[i][j][k]] + 1
                if max_tp_distance<distance_matrix[i,j]:
                    max_tp_distance = distance_matrix[i,j]
count = 0
for index in range (0, len(row_col)):
    i = row_col[index][0]
    j = row_col[index][1]
    if i!=j:
	    count = count +1     
		
positive_class = np.sum(total_type)
negative_class = count - positive_class           
############# draw the points which are participating in positive classes  ######################
ccc_index_dict = dict()     
for i in lig_rec_dict_TP:
    ccc_index_dict[i] = ''
    for j in lig_rec_dict_TP[i]:
        ccc_index_dict[j] = ''   
######################################	

attention_scores = []
lig_rec_dict = []
datapoint_size = temp_x.shape[0]
for i in range (0, datapoint_size):
    attention_scores.append([])   
    lig_rec_dict.append([])   
    for j in range (0, datapoint_size):	
        attention_scores[i].append([])   
        attention_scores[i][j] = []
        lig_rec_dict[i].append([])   
        lig_rec_dict[i][j] = []
        
distribution = []
for index in range (0, len(row_col)):
    i = row_col[index][0]
    j = row_col[index][1]
    lig_rec_dict[i][j].append(lig_rec[index])


#####################################################################################

options = ''
path = '/cluster/projects/schwartzgroup/fatema/CCC_project/niches_output/' #'/cluster/home/t116508uhn/
# get all the edges and their scaled scores that they use for plotting the heatmap
df_pair_vs_cells = pd.read_csv(path + 'niches_output_pair_vs_cells_'+options+'.csv')

edge_pair_dictionary = defaultdict(dict) # edge_pair_dictionary[edge[pair]]=score
coexpression_scores = []
lig_rec_dict_all = []
datapoint_size = temp_x.shape[0]
for i in range (0, datapoint_size):
    coexpression_scores.append([])   
    lig_rec_dict_all.append([])   
    for j in range (0, datapoint_size):	
        coexpression_scores[i].append([])   
        coexpression_scores[i][j] = []
        lig_rec_dict_all[i].append([])   
        lig_rec_dict_all[i][j] = []

distribution_all = []
for col in range (1, len(df_pair_vs_cells.columns)):
    col_name = df_pair_vs_cells.columns[col]
    l_c = df_pair_vs_cells.columns[col].split("—")[0]
    r_c = df_pair_vs_cells.columns[col].split("—")[1]
    l_c = l_c.split('.')[1]
    r_c = r_c.split('.')[1]
    i = int(l_c)
    j = int(r_c)
    
    for index in range (0, len(df_pair_vs_cells.index)):
        lig_rec_dict_all[i][j].append(df_pair_vs_cells.index[index])
        coexpression_scores[i][j].append(df_pair_vs_cells[col_name][df_pair_vs_cells.index[index]])
        distribution_all.append(df_pair_vs_cells[col_name][df_pair_vs_cells.index[index]])
        edge_pair_dictionary[str(i)+'-'+str(j)][df_pair_vs_cells.index[index]]=df_pair_vs_cells[col_name][df_pair_vs_cells.index[index]]


######### read which edge belongs to which cluster type #############################
vector_type = pd.read_csv(path + 'niches_VectorType_'+options+'.csv')
clusterType_edge_dictionary = defaultdict(list)
for index in range (0, len(vector_type.index)):
    cell_cell_pair = vector_type['Unnamed: 0'][index]
    l_c = cell_cell_pair.split("—")[0]
    r_c = cell_cell_pair.split("—")[1]
    l_c = l_c.split('.')[1]
    r_c = r_c.split('.')[1]
    i = int(l_c)
    j = int(r_c)

    cluster_type = vector_type['VectorType'][index]
    clusterType_edge_dictionary[cluster_type].append(str(i)+'-'+str(j))
    
######## read the top5 edges (ccc) by Niches ########################################
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
        

marker_list = pd.read_csv(path + 'niches_output_ccc_lr_pairs_markerList_top5_'+options+'.csv')
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
    #if lr_pair_id>12: 
    #    continue
    edge_list = clusterType_edge_dictionary[cluster_type]
    for edge in edge_list:
        if lr_pair_id not in edge_pair_dictionary[edge]:
            continue
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
	    
        if i in lig_rec_dict_TP and j in lig_rec_dict_TP[i] and lr_pair_id in lig_rec_dict_TP[i][j]:
            positive_class_found = positive_class_found + 1
	
    if flag_break == 1:
        break

print('positive_class_found %d'%positive_class_found)    
lig_rec_dict = lig_rec_dict_temp
attention_scores = attention_scores_temp
distribution = distribution_temp
negative_class = len(distribution) - positive_class_found
###################
ccc_csv_record = []
ccc_csv_record.append(['from', 'to', 'lr', 'score'])
for i in range (0, datapoint_size):
    for j in range (0, datapoint_size):
        if len(lig_rec_dict[i][j])>0:
            for k in range (0, len(lig_rec_dict[i][j])):
                ccc_csv_record.append([i, j, lig_rec_dict[i][j][k], attention_scores[i][j][k]])

df = pd.DataFrame(ccc_csv_record) # output 4
df.to_csv('/cluster/projects/schwartzgroup/fatema/find_ccc/ccc_list_all_'+options+'_Niches.csv', index=False, header=False)

##################
plot_dict = defaultdict(list)
percentage_value = 100
while percentage_value > 0:
    percentage_value = percentage_value - 10
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
    total_edges_count = 0
    for i in range (0, datapoint_size):
        for j in range (0, datapoint_size):
            #if i==j: 
            #    continue
            atn_score_list = attention_scores[i][j]
            #print(len(atn_score_list))
            
            for k in range (0, len(atn_score_list)):
                if attention_scores[i][j][k] >= threshold_down and attention_scores[i][j][k] <= threshold_up: #np.percentile(sorted(distribution), 50):
                    connecting_edges[i][j] = 1
                    ccc_index_dict[i] = ''
                    ccc_index_dict[j] = ''
                    existing_lig_rec_dict[i][j].append(lig_rec_dict[i][j][k])
                    total_edges_count = total_edges_count + 1
                    


    ############# 
    print('total edges %d'%total_edges_count)
    #negative_class = 0
    confusion_matrix = np.zeros((2,2))
    for i in range (0, datapoint_size):
        for j in range (0, datapoint_size):

            #if i==j: 
            #    continue
            ''' 
            if i in lig_rec_dict_TP and j in lig_rec_dict_TP[i]:
                for k in range (0, len(lig_rec_dict_TP[i][j])):
                    if lig_rec_dict_TP[i][j][k] in existing_lig_rec_dict[i][j]: #
                        confusion_matrix[0][0] = confusion_matrix[0][0] + 1
                    else:
                        confusion_matrix[0][1] = confusion_matrix[0][1] + 1 

            '''
            if len(existing_lig_rec_dict[i][j])>0:
                for k in existing_lig_rec_dict[i][j]:   
                    if i in lig_rec_dict_TP and j in lig_rec_dict_TP[i] and k in lig_rec_dict_TP[i][j]:
                        #print("i=%d j=%d k=%d"%(i, j, k))
                        confusion_matrix[0][0] = confusion_matrix[0][0] + 1
                    else:
                        confusion_matrix[1][0] = confusion_matrix[1][0] + 1                 
             
    print('%d, %g, %g'%(percentage_value,  (confusion_matrix[1][0]/negative_class)*100, (confusion_matrix[0][0]/positive_class)*100))    
    FPR_value = (confusion_matrix[1][0]/negative_class)#*100
    TPR_value = (confusion_matrix[0][0]/positive_class)#*100
    plot_dict['FPR'].append(FPR_value)
    plot_dict['TPR'].append(TPR_value)
    plot_dict['Type'].append('Niches') #_lowNoise


with gzip.open("/cluster/projects/schwartzgroup/fatema/find_ccc/" + options +'_'+'Niches', 'wb') as fp: #b, b_1, a  11to20runs
    pickle.dump(plot_dict, fp) #a - [0:5]
