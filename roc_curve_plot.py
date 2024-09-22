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


with gzip.open("/cluster/projects/schwartzgroup/fatema/find_ccc/synthetic_data/type_gaussian_distribution/no_noise/uniform_distribution_coordinate", 'rb') as fp: #datatype
    x_index, y_index , no_need = pickle.load(fp) #

with gzip.open("/cluster/projects/schwartzgroup/fatema/find_ccc/synthetic_data/type_gaussian_distribution/no_noise/uniform_distribution_ground_truth_ccc" , 'rb') as fp:  
    lr_database, lig_rec_dict_TP, random_activation = pickle.load( fp)

# lig_rec_dict_TP has the true positive edge list. lig_rec_dict_TP[i][j] is a list of lr pairs between cell i and cell j
# find the count of true positives
datapoint_size = x_index.shape[0]    # total number of cells or datapoints          
tp = 0 # true positives
for i in lig_rec_dict_TP.keys():
    for j in lig_rec_dict_TP[i].keys():
        tp = tp + len(lig_rec_dict_TP[i][j])

positive_class = tp # WE NEED THIS TO CALCULATE 'TRUE POSITIVE RATE'

#with gzip.open("/cluster/projects/schwartzgroup/fatema/find_ccc/synthetic_data/type_gaussian_distribution/no_noise/uniform_distribution_input_graph" , 'rb') as fp:           
#    row_col, edge_weight, lig_rec  = pickle.load(fp) 

######################### COMMOT ###############################################################################################################
options = 'uniform_distribution_no_noise'
with gzip.open("/cluster/projects/schwartzgroup/fatema/find_ccc/synthetic_data/type_gaussian_distribution/no_noise/" + options+'_commot_result', 'rb') as fp:
    attention_scores, lig_rec_dict, distribution = pickle.load(fp)            

# lig_rec_dict[i][j]=[...] # is a list of lr pairs (edges) between cell i and cell j 
# attention_scores[i][j]=[...] # is a list of attention scores of the lr pairs (edges) between cell i and cell j
# distribution=[...] is a combined list of attention scores of all edges. 

# let's first find the total count of negative classes in the COMMOT result. 
# We calculate the tp detected by COMMOT and then deduct it from total detection by COMMOT to get the negative classes. 
confusion_matrix = np.zeros((2,2))
for i in range (0, datapoint_size):
    for j in range (0, datapoint_size):
        lr_pair_list = lig_rec_dict[i][j]
        if len(lr_pair_list)>0:
            for k in lr_pair_list:   
                if i in lig_rec_dict_TP and j in lig_rec_dict_TP[i] and k in lig_rec_dict_TP[i][j]:
                    #print("i=%d j=%d k=%d"%(i, j, k))
                    confusion_matrix[0][0] = confusion_matrix[0][0] + 1 # detected true positive by COMMOT
                else:
                    confusion_matrix[1][0] = confusion_matrix[1][0] + 1 #  detected false positive by COMMOT            
    
negative_class = len(distribution) - confusion_matrix[0][0] # WE NEED THIS TO CALCULATE 'FALSE POSITIVE RATE'

distribution = sorted(distribution, reverse=True) 

# start roc plot here. select top 10% (90th), 20% (80th), 30% (70th), ... ccc and calculate TPR and FPR 
plot_dict = defaultdict(list)
for percentile_value in [90, 80, 70, 60, 50, 40, 30, 20, 10, 0]:
    threshold_percentile =  np.percentile(distribution, percentile_value)
    existing_lig_rec_dict = [] # record COMMOT detected edges that are above the threshold percentile attention score
    for i in range (0, datapoint_size):
        existing_lig_rec_dict.append([])   
        for j in range (0, datapoint_size):	
            existing_lig_rec_dict[i].append([])   
            existing_lig_rec_dict[i][j] = []

    # connecting_edges = np.zeros((datapoint_size, datapoint_size))
    
    total_edges_count = 0
    for i in range (0, datapoint_size):
        for j in range (0, datapoint_size):
            atn_score_list = attention_scores[i][j]
            for k in range (0, len(atn_score_list)):
                if attention_scores[i][j][k] >= threshold_percentile: 
                    # connecting_edges[i][j] = 1
                    existing_lig_rec_dict[i][j].append(lig_rec_dict[i][j][k])
                    total_edges_count = total_edges_count + 1
                    


    ############# 
    #print('total edges %d'%total_edges_count)
    #negative_class = 0
    confusion_matrix = np.zeros((2,2))
    for i in range (0, datapoint_size):
        for j in range (0, datapoint_size):
            if len(existing_lig_rec_dict[i][j])>0:
                for k in existing_lig_rec_dict[i][j]:   
                    if i in lig_rec_dict_TP and j in lig_rec_dict_TP[i] and k in lig_rec_dict_TP[i][j]:
                        #print("i=%d j=%d k=%d"%(i, j, k))
                        confusion_matrix[0][0] = confusion_matrix[0][0] + 1
                    else:
                        confusion_matrix[1][0] = confusion_matrix[1][0] + 1                 
             
    print('%d, %g, %g'%(percentile_value,  (confusion_matrix[1][0]/negative_class)*100, (confusion_matrix[0][0]/positive_class)*100))    
    FPR_value = (confusion_matrix[1][0]/negative_class)#*100
    TPR_value = (confusion_matrix[0][0]/positive_class)#*100
    plot_dict['FPR'].append(FPR_value)
    plot_dict['TPR'].append(TPR_value)
    plot_dict['Type'].append('COMMOT') # no noise

with gzip.open("/cluster/projects/schwartzgroup/fatema/find_ccc/synthetic_data/type_gaussian_distribution/no_noise/" + options +'_COMMOT_roc', 'wb') as fp: #b, b_1, a  11to20runs
    pickle.dump(plot_dict, fp) #a - [0:5]

data_list_pd = pd.DataFrame(plot_dict)    
chart = alt.Chart(data_list_pd).mark_line().encode(
    x='FPR:Q',
    y='TPR:Q',
    color='Type:N',
)	
save_path = '/cluster/home/t116508uhn/'
chart.save(save_path+options+'_COMMOT_roc_plot.html')

