import gzip
import pandas as pd
import pickle
from collections import defaultdict
import numpy as np
import altair as alt


datapoint_size_list = [3000, 3000, 3000, 5000, 5000, 5000, 5000, 5000, 5000, 3000, 5000, 5000]
options_list = ['dt-path_equally_spaced_lrc1467_cp100_noise0_random_overlap_threshold_dist_cellCount3000_3dim_3patterns_temp',
               'dt-path_equally_spaced_lrc1467_cp100_noise30_lowNoise_random_overlap_threshold_dist_cellCount3000_3dim_3patterns_temp',
               'dt-path_equally_spaced_lrc1467_cp100_noise30_heavyNoise_random_overlap_threshold_dist_cellCount3000_3dim_3patterns_temp',
               
               'dt-path_uniform_distribution_lrc112_cp100_noise0_random_overlap_threshold_dist_cellCount5000_3dim_3patterns_temp',
               'dt-path_uniform_distribution_lrc112_cp100_noise30_lowNoise_random_overlap_threshold_dist_cellCount5000_3dim_3patterns_temp',
               'dt-path_uniform_distribution_lrc112_cp100_noise30_heavyNoise_random_overlap_threshold_dist_cellCount5000_3dim_3patterns_temp_v2',
               
               'dt-path_mixture_of_distribution_lrc112_cp100_noise0_random_overlap_knn_cellCount5000_3dim_3patterns_temp',
                'dt-path_mixture_of_distribution_lrc112_cp100_noise30_lowNoise_random_overlap_knn_cellCount5000_3dim_3patterns_temp',
                'dt-path_mixture_of_distribution_lrc112_cp100_noise30_heavyNoise_random_overlap_knn_cellCount5000_3dim_3patterns_temp',
                # -- done -- #
                'dt-randomCCC_equally_spaced_lrc105_cp100_noise0_threshold_dist_cellCount3000',
                'dt-randomCCC_uniform_distribution_lrc105_cp100_noise0_threshold_dist_cellCount5000',
                'dt-randomCCC_mix_distribution_lrc105_cp100_noise0_knn_cellCount5000'
               ] 
sample_type = ['equidistant_path', 'equidistant_path_lowNoise', 'equidistant_path_highNoise', 'uniform', 'uniform_lowNoise', 'uniform_highNoise', 'mixed', 'mixed_low', 'mixed_high', 'random_equi', 'random_unifrom', 'random_mixed']
noise_type = ['no_noise', 'low_noise', 'high_noise', 'no_noise', 'low_noise', 'high_noise', 'no_noise', 'low_noise', 'high_noise' ]
location = ['equidistant/','equidistant/','equidistant/', 'uniform_distribution/', 'uniform_distribution/', 'uniform_distribution/', 'mixed_distribution/', 'mixed_distribution/', 'mixed_distribution/']
for op_index in [1, 2, 6, 7, 8]: #len(options_list)):
    print('%d'%op_index)
  
    options = options_list[op_index]

    fp = gzip.open("/cluster/projects/schwartzgroup/fatema/find_ccc/" + 'Tclass_synthetic_data_ccc_roc_control_model_'+ options, 'rb')  # at least one of lig or rec has exp > respective knee point          
    lr_database, lig_rec_dict_TP, random_activation = pickle.load(fp)
      
    datapoint_size = datapoint_size_list[op_index]
    total_type = np.zeros((len(lr_database)))
    count = 0
    for i in range (0, datapoint_size):
        for j in range (0, datapoint_size):
            if i==j: 
                continue
            if i in lig_rec_dict_TP and j in lig_rec_dict_TP[i] and len(lig_rec_dict_TP[i][j]) > 0:
                count = count + 1   
                for k in range (0, len(lig_rec_dict_TP[i][j])):
                    total_type[lig_rec_dict_TP[i][j][k]] = total_type[lig_rec_dict_TP[i][j][k]] + 1

    with gzip.open("/cluster/projects/schwartzgroup/fatema/find_ccc/" + 'adjacency_records_synthetic_data_ccc_roc_control_model_'+ options , 'rb') as fp:  # +'_'+'notQuantileTransformed'at least one of lig or rec has exp > respective knee point          
        row_col, edge_weight, lig_rec  = pickle.load(fp)  #, lr_database, lig_rec_dict_TP, random_activation
        
  
    count = 0
    for index in range (0, len(row_col)):
        i = row_col[index][0]
        j = row_col[index][1]
        if i!=j:
            count = count +1     
            
    positive_class = np.sum(total_type)
    #negative_class = count - positive_class 
  ######################################################## cytosignal #########################################################################
    # get all the edges and their scaled scores that they use for plotting the heatmap
    #list_ccc = pd.read_csv('/cluster/projects/schwartzgroup/fatema/cytosignal/sender_vs_rec_'+options+'.csv') 
    cell_vs_cell = pd.read_csv('/cluster/projects/schwartzgroup/fatema/cytosignal/cell_cell_score_'+ options +'.csv', index_col=0)
    ccc_list = pd.read_csv('/cluster/projects/schwartzgroup/fatema/cytosignal/ccc_name_'+ options +'.csv', index_col=0)
    ccc_list = list(ccc_list['x'])
  
    # rows = senders
    # cols = receivers
    '''
    In [6]: list_ccc
    Out[6]: 
          i   j     score   ccc
    0     1   1  0.000179    10
    1     2   1  0.031757    10
    2     3   1  0.892900    10
    '''
    #negative_class=len(distribution)-confusion_matrix[0][0]
    attention_scores= []
    lig_rec_dict = []
    
    for i in range (0, datapoint_size):
        attention_scores.append([])   
        lig_rec_dict.append([])   
        for j in range (0, datapoint_size):	
            attention_scores[i].append([])   
            attention_scores[i][j] = []
            lig_rec_dict[i].append([])   
            lig_rec_dict[i][j] = []
            
    
    ccc_csv_record = []
    ccc_csv_record.append(['from', 'to', 'lr', 'score'])

    distribution = []
    for i in range (0, datapoint_size):
        for j in range (0, datapoint_size):	 
            if i==j:
                continue
            if cell_vs_cell['a'+str(j)][i+1] > 0:
                attention_scores[i][j].append(cell_vs_cell['a'+str(j)][i+1])
                lig_rec_dict[i][j].append(1)
                distribution.append(cell_vs_cell['a'+str(j)][i+1])
                ccc_csv_record.append([i, j, 1, cell_vs_cell['a'+str(j)][i+1]])


    df = pd.DataFrame(ccc_csv_record) # output 4
    df.to_csv('/cluster/projects/schwartzgroup/fatema/find_ccc/synthetic_data/type_'+location[op_index]+noise_type[op_index]+'/ccc_list_all_cytoSignal.csv', index=False, header=False)
    

  
    percentage_value = 10
    while percentage_value > 0:
        percentage_value = percentage_value - 10
   ###########################  
        #percentage_value = 0
        existing_lig_rec_dict = []
        for i in range (0, datapoint_size):
            existing_lig_rec_dict.append([])   
            for j in range (0, datapoint_size):	
                existing_lig_rec_dict[i].append([])   
                existing_lig_rec_dict[i][j] = []
    
        ccc_index_dict = dict()
        threshold_down =  np.percentile(sorted(distribution), percentage_value)
        threshold_up =  np.percentile(sorted(distribution), 100)
        connecting_edges = np.zeros((datapoint_size, datapoint_size))
        rec_dict = defaultdict(dict)
        total_edges_count = 0
        for i in range (0, datapoint_size):
            for j in range (0, datapoint_size):
                if i==j: 
                    continue
                atn_score_list = attention_scores[i][j]
                if len(atn_score_list)==0:
                    continue
                for k in range (0, 1):
                    if attention_scores[i][j][k] >= threshold_down and attention_scores[i][j][k] <= threshold_up: #np.percentile(sorted(distribution), 50):
                        connecting_edges[i][j] = 1
                        ccc_index_dict[i] = ''
                        ccc_index_dict[j] = ''
                        existing_lig_rec_dict[i][j].append(1) #lig_rec_dict[i][j][k]
                        total_edges_count = total_edges_count + 1
                        
    
    
        ############# 
        print('total edges %d'%total_edges_count)
        #negative_class = 0
        confusion_matrix = np.zeros((2,2))
        for i in range (0, datapoint_size):
            for j in range (0, datapoint_size):
                if i==j: 
                    continue
                if len(existing_lig_rec_dict[i][j])>0:
                    for k in existing_lig_rec_dict[i][j]:   
                        if i in lig_rec_dict_TP and j in lig_rec_dict_TP[i] and len(lig_rec_dict_TP[i][j])>0:
                            
                            found_ccc = 0
                            for ccc in lig_rec_dict_TP[i][j]:
                                if ccc in ccc_list:
                                    found_ccc = 1
                                    break
    
                            if found_ccc == 1:                       
                            #print("i=%d j=%d k=%d"%(i, j, k))
                            
                                confusion_matrix[0][0] = confusion_matrix[0][0] + 1
                        else:
                            confusion_matrix[1][0] = confusion_matrix[1][0] + 1                 
    


    negative_class=len(distribution)-confusion_matrix[0][0]
    plot_dict = defaultdict(list)
    percentage_value = 100
    while percentage_value > 0:
        percentage_value = percentage_value - 10
   ###########################  
        #percentage_value = 0
        existing_lig_rec_dict = []
        for i in range (0, datapoint_size):
            existing_lig_rec_dict.append([])   
            for j in range (0, datapoint_size):	
                existing_lig_rec_dict[i].append([])   
                existing_lig_rec_dict[i][j] = []
    
        ccc_index_dict = dict()
        threshold_down =  np.percentile(sorted(distribution), percentage_value)
        threshold_up =  np.percentile(sorted(distribution), 100)
        connecting_edges = np.zeros((datapoint_size, datapoint_size))
        rec_dict = defaultdict(dict)
        total_edges_count = 0
        for i in range (0, datapoint_size):
            for j in range (0, datapoint_size):
                if i==j: 
                    continue
                atn_score_list = attention_scores[i][j]
                if len(atn_score_list)==0:
                    continue
                for k in range (0, 1):
                    if attention_scores[i][j][k] >= threshold_down and attention_scores[i][j][k] <= threshold_up: #np.percentile(sorted(distribution), 50):
                        connecting_edges[i][j] = 1
                        ccc_index_dict[i] = ''
                        ccc_index_dict[j] = ''
                        existing_lig_rec_dict[i][j].append(1) #lig_rec_dict[i][j][k]
                        total_edges_count = total_edges_count + 1
                        
    
    
        ############# 
        print('total edges %d'%total_edges_count)
        #negative_class = 0
        confusion_matrix = np.zeros((2,2))
        for i in range (0, datapoint_size):
            for j in range (0, datapoint_size):
                if i==j: 
                    continue
                if len(existing_lig_rec_dict[i][j])>0:
                    for k in existing_lig_rec_dict[i][j]:   
                        if i in lig_rec_dict_TP and j in lig_rec_dict_TP[i] and len(lig_rec_dict_TP[i][j])>0:
                            
                            found_ccc = 0
                            for ccc in lig_rec_dict_TP[i][j]:
                                if ccc in ccc_list:
                                    found_ccc = 1
                                    break
    
                            if found_ccc == 1:                       
                            #print("i=%d j=%d k=%d"%(i, j, k))
                            
                                confusion_matrix[0][0] = confusion_matrix[0][0] + 1
                        else:
                            confusion_matrix[1][0] = confusion_matrix[1][0] + 1                 
    
    ##########################
        print('%d, %g, %g'%(percentage_value,  (confusion_matrix[1][0]/negative_class)*100, (confusion_matrix[0][0]/positive_class)*100))    
        FPR_value = (confusion_matrix[1][0]/negative_class)#*100
        TPR_value = (confusion_matrix[0][0]/positive_class)#*100
        plot_dict['FPR'].append(FPR_value)
        plot_dict['TPR'].append(TPR_value)
        plot_dict['Type'].append('CytoSignal') #_lowNoise
    
    
    with gzip.open("/cluster/projects/schwartzgroup/fatema/find_ccc/" + options +'_'+'CytoSignal', 'wb') as fp: #b, b_1, a  11to20runs
        pickle.dump(plot_dict, fp) #a - [0:5]


###########################################################

for t in range (0, len(options_list)):
    print('%d'%t)
  
    options = options_list[t]

    fp = gzip.open("/cluster/projects/schwartzgroup/fatema/find_ccc/" + options +'_'+'CytoSignal', 'rb') 
    plot_dict_temp = pickle.load(fp)

  
    plot_dict = defaultdict(list)
    plot_dict['FPR'].append(0)
    plot_dict['TPR'].append(0)
    plot_dict['Type'].append("cytosignal_"+sample_type[t]) #(plot_dict_temp['Type'][0])
    for i in range (0, len(plot_dict_temp['Type'])):
        plot_dict['FPR'].append(plot_dict_temp['FPR'][i])
        plot_dict['TPR'].append(plot_dict_temp['TPR'][i])
        plot_dict['Type'].append("cytosignal_"+sample_type[t]) #(plot_dict_temp['Type'][i])
    
    data_list_pd = pd.DataFrame(plot_dict)    
    chart = alt.Chart(data_list_pd).mark_line().encode(
        x='FPR:Q',
        y='TPR:Q',
        color='Type:N',
    )	
    chart.save('/cluster/projects/schwartzgroup/fatema/cytosignal/'+"cytosignal_"+sample_type[t]+'.html')
# op = 3,  1163.0, op 4 = 1400, op 5 = 
