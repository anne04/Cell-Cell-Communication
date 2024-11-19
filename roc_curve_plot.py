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


with gzip.open("/cluster/projects/schwartzgroup/fatema/find_ccc/synthetic_data/type_" +  dataType[index]  +"/"+ noise_type[index] + "/" + dataType[index] +"_"+noise_type[]+ "_coordinate", 'rb') as fp: #datatype
    x_index, y_index , no_need = pickle.load(fp) #

with gzip.open("/cluster/projects/schwartzgroup/fatema/find_ccc/synthetic_data/type_"+  dataType[index]  +"/"+ noise_type[index] + "/" + dataType[index] +"_"+noise_type[]+"_ground_truth_ccc" , 'rb') as fp:  
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
options = 'uniform_distribution_no_noise'
with gzip.open("/cluster/projects/schwartzgroup/fatema/find_ccc/synthetic_data/type_uniform_distribution/no_noise/" + options+'_commot_result', 'rb') as fp:
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
                #if attention_scores[i][j][k] >= distribution[len(distribution)-1]:
                    ccc_csv_record.append([i, j, lig_rec_dict[i][j][k], attention_scores[i][j][k]])

df = pd.DataFrame(ccc_csv_record) # output 4
df.to_csv('/cluster/projects/schwartzgroup/fatema/find_ccc/synthetic_data/type_'+ dataType[index] +'/'+ noise_type[index] +'/ccc_list_all_COMMOT.csv', index=False, header=False)
########################################################################################################################################################################




# Find the total count of negative classes in the COMMOT result. 
# We calculate the tp detected by COMMOT and then deduct it from total detection by COMMOT to get the negative classes. 
detected_TP = 0
for i in range (0, datapoint_size):
    for j in range (0, datapoint_size):
        lr_pair_list = lig_rec_dict[i][j]
        if len(lr_pair_list)>0:
            for k in lr_pair_list:  
                if attention_scores[i][j][k] < min_limit:
                    continue # ignore                
                if i in lig_rec_dict_TP and j in lig_rec_dict_TP[i] and k in lig_rec_dict_TP[i][j]:
                    #print("i=%d j=%d k=%d"%(i, j, k))
                    detected_TP = detected_TP + 1 # detected true positive by COMMOT
    
negative_class = len(distribution) - detected_TP # WE NEED THIS TO CALCULATE 'FALSE POSITIVE RATE'

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
                if attention_scores[i][j][k] >= threshold_percentile and attention_scores[i][j][k] <= max_limit: 
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

with gzip.open("/cluster/projects/schwartzgroup/fatema/find_ccc/synthetic_data/type_uniform_distribution/no_noise/" + options +'_COMMOT_roc', 'wb') as fp: #b, b_1, a  11to20runs
    pickle.dump(plot_dict, fp) #a - [0:5]

data_list_pd = pd.DataFrame(plot_dict)    
chart = alt.Chart(data_list_pd).mark_line().encode(
    x='FPR:Q',
    y='TPR:Q',
    color='Type:N',
)	
save_path = '/cluster/home/t116508uhn/'
chart.save(save_path+options+'_COMMOT_roc_plot.html')

