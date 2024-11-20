import os
import pandas as pd
import copy
import csv
import numpy as np
import sys
from collections import defaultdict
import pickle
import gzip
import matplotlib.pyplot as plt
import altairThemes
import altair as alt

alt.themes.register("publishTheme", altairThemes.publishTheme)
# enable the newly registered theme
alt.themes.enable("publishTheme")

dataType = ['equidistant','equidistant','equidistant','uniform_distribution','uniform_distribution','uniform_distribution','mixed_distribution', 'mixed_distribution', 'mixed_distribution']
noise_type = ['no_noise', 'low_noise', 'high_noise', 'no_noise', 'low_noise', 'high_noise', 'no_noise', 'low_noise', 'high_noise']
commotResult_name = ['equidistant', 'equidistant', 'equidistant', 'uniform', 'uniform', 'uniform', 'mixture', 'mixture', 'mixture' ]
noise_level= ['noise0', 'noise30level1', 'noise30level2', 'noise0', 'noise30level1', 'noise30level2', 'noise0', 'noise30level1', 'noise30level2']
for index in range (6, len(dataType)):
    print(index)
    with gzip.open("/cluster/projects/schwartzgroup/fatema/find_ccc/synthetic_data/type_" +  dataType[index]  +"/"+ noise_type[index] + "/" + dataType[index] +"_"+noise_type[index]+ "_coordinate", 'rb') as fp: #datatype
        x_index, y_index , no_need = pickle.load(fp) #
    
    with gzip.open("/cluster/projects/schwartzgroup/fatema/find_ccc/synthetic_data/type_"+  dataType[index]  +"/"+ noise_type[index] + "/" + dataType[index] +"_"+noise_type[index]+"_ground_truth_ccc" , 'rb') as fp:  
        lr_database, lig_rec_dict_TP, random_activation = pickle.load( fp)
    
    # lig_rec_dict_TP has the true positive edge list. lig_rec_dict_TP[i][j] is a list of lr pairs between cell i and cell j
    # find the count of true positives
    datapoint_size = x_index.shape[0]    # total number of cells or datapoints          
    tp = 0 # true positives
    for i in lig_rec_dict_TP.keys():
        for j in lig_rec_dict_TP[i].keys():
            tp = tp + len(lig_rec_dict_TP[i][j])
    
    positive_class = tp # WE NEED THIS TO CALCULATE 'TRUE POSITIVE RATE'
    
    with gzip.open("/cluster/projects/schwartzgroup/fatema/find_ccc/synthetic_data/type_"+  dataType[index] +'/'+ noise_type[index] +"/"+ dataType[index] +"_"+noise_type[index]+"_input_graph" , 'rb') as fp:           
        row_col, edge_weight, lig_rec  = pickle.load(fp) 
    
    ######################### COMMOT ###############################################################################################################
    with gzip.open("/cluster/projects/schwartzgroup/fatema/CCC_project/commot_result/synthetic_data_" + commotResult_name[index] + "_"+ noise_level[index]  +'_commot_result', 'rb') as fp:
        attention_scores, lig_rec_dict, distribution = pickle.load(fp)            
    
    # lig_rec_dict[i][j]=[...] # is a list of lr pairs (edges) between cell i and cell j 
    # attention_scores[i][j]=[...] # is a list of COMMOT assigned scores of the lr pairs (edges) between cell i and cell j
    # distribution=[...] is a combined list of COMMOT assigned scores of all edges. 
    
    # TP = 2800, NEST selected total edge = 21,659. But COMMOT reports total edge = 6,634,880. 
    # Since there is a big imbalance between total edge, it also causes a big  
    # imbalance between FP by NEST and COMMOT. To keep them compatible, we keep highly 
    # scored top 21,659 edges by COMMOT. 
    distribution = sorted(distribution, reverse=True) # large to small
    distribution = distribution[0:len(row_col)] # keep top 21,659 edges by COMMOT. Ignore the rest.
    min_limit =  distribution[len(distribution)-1] # min score to be considered
    
    distribution = sorted(distribution, reverse=True)
    distribution = distribution[0:len(row_col)] # len(distribution) = 6634880, len(row_col)=21659
    #########################################################################################################################################################################
    ccc_csv_record = []
    ccc_csv_record.append(['from', 'to', 'lr', 'score'])
    for i in range (0, datapoint_size):
        for j in range (0, datapoint_size):
            if len(attention_scores[i][j])>0:
                for k in range (0, len(attention_scores[i][j])):
                    if attention_scores[i][j][k] >= distribution[len(distribution)-1]:
                        ccc_csv_record.append([i, j, lig_rec_dict[i][j][k], attention_scores[i][j][k]])
    
    df = pd.DataFrame(ccc_csv_record) # output 4
    df.to_csv('/cluster/projects/schwartzgroup/fatema/find_ccc/synthetic_data/type_'+ dataType[index] +'/'+ noise_type[index] +'/ccc_list_all_COMMOT.csv', index=False, header=False)
########################################################################################################################################################################


