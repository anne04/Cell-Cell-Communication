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


model_type = 'alternative_cutoff'


import argparse
parser = argparse.ArgumentParser()
parser.add_argument( '--embedding_data_path', type=str, default='new_alignment/Embedding_data_ccc_rgcn/' , help='The path to attention') #'/cluster/projects/schwartzgroup/fatema/pancreatic_cancer_visium/210827_A00827_0396_BHJLJTDRXY_Notta_Karen/V10M25-61_D1_PDA_64630_Pa_P_Spatial10x_new/outs/'
args = parser.parse_args()



########################################################################################
old_dataname = ['noise0',  'noise30_lowNoise' ,'noise30_heavyNoise']
noise_type = ['no_noise', 'low_noise', 'high_noise']
nest_model_noise_type = ['temp', 'lowNoise_temp_v2','heavyNoise_temp_v2']
# ls /cluster/projects/schwartzgroup/fatema/find_ccc/*synthetic_data_ccc_roc_control_model_dt-path*uni*random_overlap_knn10*

for sample_type in range (2, len(noise_type)):
    with gzip.open("/cluster/projects/schwartzgroup/fatema/find_ccc/synthetic_data_ccc_roc_control_model_dt-path_uniform_distribution_lrc112_cp100_"+old_dataname[sample_type]+"_random_overlap_knn10_cellCount5000_3dim_3patterns_temp_v2_xny" , 'rb') as fp: #datatype
        temp_x, temp_y , ccc_region = pickle.load(fp) #
    
    datapoint_size = temp_x.shape[0]
    
    coordinates = np.zeros((temp_x.shape[0],2))
    for i in range (0, datapoint_size):
        coordinates[i][0] = temp_x[i]
        coordinates[i][1] = temp_y[i]
        
    from sklearn.metrics.pairwise import euclidean_distances
    distance_matrix = euclidean_distances(coordinates, coordinates)
    
    with gzip.open("/cluster/projects/schwartzgroup/fatema/find_ccc/Tclass_synthetic_data_ccc_roc_control_model_dt-path_uniform_distribution_lrc112_cp100_"+old_dataname[sample_type]+"_random_overlap_knn10_cellCount5000_3dim_3patterns_temp_v2" , 'rb') as fp:            
        lr_database, lig_rec_dict_TP, random_activation = pickle.load( fp)
    
    
    ligand_dict_dataset = defaultdict(dict)
    for i in range (0, len(lr_database)):
        ligand_dict_dataset[lr_database[i][0]][lr_database[i][1]] = i
        
    ligand_list = list(ligand_dict_dataset.keys())  
    
    with gzip.open("/cluster/projects/schwartzgroup/fatema/find_ccc/adjacency_records_synthetic_data_ccc_roc_control_model_dt-path_uniform_distribution_lrc112_cp100_"+ old_dataname[sample_type] +"_random_overlap_knn10_cellCount5000_3dim_3patterns_temp_v2" , 'rb') as fp:  # +'_'+'notQuantileTransformed'at least one of lig or rec has exp > respective knee point          
        row_col, edge_weight, lig_rec  = pickle.load(fp)  #, lr_database, lig_rec_dict_TP, random_activation
        
    
    print('data read done %d'%sample_type)
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
    
    print('read nest output %d'%sample_type)

    
    filename = ["r1", "r2", "r3", "r4", "r5"] #, "r6", "r7", "r8", "r9", "r10"]
    total_runs = 5
    plot_dict = defaultdict(list)
    distribution_rank = []
    all_edge_sorted_by_avgrank = []
    for layer in range (0, 2):
        distribution_rank.append([])
        all_edge_sorted_by_avgrank.append([])
    
    layer = -1
    percentage_value = 0
    
    for l in [2, 3]: # 2 = layer 2, 3 = layer 1
        layer = layer + 1
        csv_record_dict = defaultdict(list)
        for run_time in range (0,total_runs):
            run = run_time
          
            X_attention_filename = "/cluster/projects/schwartzgroup/fatema/CCC_project/new_alignment/Embedding_data_ccc_rgcn/V10M25-61_D1_PDA_64630_Pa_P_Spatial10x_new/synthetic_data_ccc_roc_control_model_uniform_path_knn10_lrc112_cell5000_tanh_3d_"+ nest_model_noise_type[sample_type] +"_"+ filename[run]+"_attention_l1.npy"
            
            #synthetic_data_ccc_roc_control_model_equiDistant_path_knn10_lrc1467_cell5000_tanh_3d_"+ nest_model_noise_type[sample_type] +"temp_"+filename[run]+"_attention_l1.npy"   
            X_attention_bundle = np.load(X_attention_filename, allow_pickle=True) # f_
    
            distribution = []
            for index in range (0, X_attention_bundle[0].shape[1]):
                i = X_attention_bundle[0][0][index]
                j = X_attention_bundle[0][1][index]
                distribution.append(X_attention_bundle[l][index][0])
    
            max_value = np.max(distribution)
            min_value = np.min(distribution)
            
            
                
            attention_scores = []
            datapoint_size = temp_x.shape[0]
            for i in range (0, datapoint_size):
                attention_scores.append([])   
                for j in range (0, datapoint_size):	
                    attention_scores[i].append([])   
                    attention_scores[i][j] = []
                    
            min_attention_score = max_value
            distribution = []
            for index in range (0, X_attention_bundle[0].shape[1]):
                i = X_attention_bundle[0][0][index]
                j = X_attention_bundle[0][1][index]
                #if barcode_type[barcode_info[i][0]] != 1 or barcode_type[barcode_info[j][0]] != 1:
                #    continue
                scaled_score = (X_attention_bundle[l][index][0]-min_value)/(max_value-min_value)
                attention_scores[i][j].append(scaled_score) #X_attention_bundle[2][index][0]
                if min_attention_score > scaled_score:
                    min_attention_score = scaled_score
                distribution.append(scaled_score)
                
            #print('min attention score with scaling %g'%min_attention_score)
    
    
    
            

    
            datapoint_size = temp_x.shape[0]
    
            count = 0
            existing_lig_rec_dict = []
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
                    if i==j: 
                        continue
                    atn_score_list = attention_scores[i][j]
                    #print(len(atn_score_list))
                    for k in range (0, len(atn_score_list)):
                        if attention_scores[i][j][k] >= threshold_down and attention_scores[i][j][k] <= threshold_up: #np.percentile(sorted(distribution), 50):
                            connecting_edges[i][j] = 1
                            ccc_index_dict[i] = ''
                            ccc_index_dict[j] = ''
                            existing_lig_rec_dict[i][j].append(lig_rec_dict[i][j][k])
                            key_value = str(i) +'-'+ str(j) + '-' + str(lig_rec_dict[i][j][k])
                            csv_record_dict[key_value].append([attention_scores[i][j][k], run])
                            count = count + 1
                            #distribution_partial.append(attention_scores[i][j][k])
    
    
    ############### merge multiple runs ##################
        for key_value in csv_record_dict.keys():
            run_dict = defaultdict(list)
            for scores in csv_record_dict[key_value]:
                run_dict[scores[1]].append(scores[0])
    
            for runs in run_dict.keys():
                run_dict[runs] = np.mean(run_dict[runs])
    
    
            csv_record_dict[key_value] = []
            for runs in run_dict.keys():
                csv_record_dict[key_value].append([run_dict[runs],runs])
    
    
        
        #######################################
        
        all_edge_list = []
        for key_value in csv_record_dict.keys():
            edge_score_runs = []
            edge_score_runs.append(key_value)
            for runs in csv_record_dict[key_value]:
                edge_score_runs.append(runs[0]) # 
                
            all_edge_list.append(edge_score_runs)
    
        ## Find the rank of product
        edge_rank_dictionary = defaultdict(list)
        # sort the all_edge_list by runs and record the rank 
        print('total runs %d'%total_runs)
        for runs in range (0, total_runs):
            sorted_list_temp = sorted(all_edge_list, key = lambda x: x[runs+1], reverse=True) # sort based on score by current run and large to small
            for rank in range (0, len(sorted_list_temp)):
                edge_rank_dictionary[sorted_list_temp[rank][0]].append(rank+1) # small rank being high attention
    
        all_edge_avg_rank = []
        for key_val in edge_rank_dictionary.keys():
            rank_product = 1
            for i in range (0, len(edge_rank_dictionary[key_val])):
                rank_product = rank_product * edge_rank_dictionary[key_val][i]
                
            all_edge_avg_rank.append([key_val, rank_product**(1/total_runs)])  # small rank being high attention
            distribution_rank[layer].append(rank_product**(1/total_runs))
            
        all_edge_sorted_by_avgrank[layer] = sorted(all_edge_avg_rank, key = lambda x: x[1]) # small rank being high attention 
    
    # now you can start roc curve by selecting top 90%, 80%, 70% edges ...so on
    
    percentage_value = 100  # when creating ccc list
    
    csv_record_intersect_dict = defaultdict(list)
    for layer in range (0, 2):
        threshold_up = np.percentile(distribution_rank[layer], percentage_value)
        for i in range (0, len(all_edge_sorted_by_avgrank[layer])):
            if all_edge_sorted_by_avgrank[layer][i][1] <= threshold_up:
                csv_record_intersect_dict[all_edge_sorted_by_avgrank[layer][i][0]].append(all_edge_sorted_by_avgrank[layer][i][1])
    '''
    threshold_up = np.percentile(distribution_rank_layer2, percentage_value)
    for i in range (0, len(all_edge_sorted_by_avgrank_layer2)):
        if all_edge_sorted_by_avgrank_layer2[i][1] <= threshold_up:
            csv_record_intersect_dict[all_edge_sorted_by_avgrank_layer2[i][0]].append(all_edge_sorted_by_avgrank_layer2[i][1])
    '''
    ###### this small block does not have any impact now ###########
    for key_value in csv_record_intersect_dict.keys():  
        if len(csv_record_intersect_dict[key_value])>1:
            csv_record_intersect_dict[key_value] = [np.min(csv_record_intersect_dict[key_value])]
    #######################################################
    
    existing_lig_rec_dict = []
    for i in range (0, datapoint_size):
        existing_lig_rec_dict.append([])   
        for j in range (0, datapoint_size):	
            existing_lig_rec_dict[i].append([])   
            existing_lig_rec_dict[i][j] = []    
    
    ccc_csv_record = []
    ccc_csv_record.append(['from', 'to', 'lr pair', 'rank'])

    for key_value in csv_record_intersect_dict.keys():
        item = key_value.split('-')
        i = int(item[0])
        j = int(item[1])
        LR_pair_id = int(item[2])
        existing_lig_rec_dict[i][j].append(LR_pair_id)
        ccc_csv_record.append([i, j, LR_pair_id, csv_record_intersect_dict[key_value][0]])
#######################################
    df = pd.DataFrame(ccc_csv_record) # output 4
    df.to_csv('/cluster/projects/schwartzgroup/fatema/find_ccc/synthetic_data/type_uniform_distribution/'+ noise_type[sample_type] +'/ccc_list_all_'+ model_type +'.csv', index=False, header=False)
        
