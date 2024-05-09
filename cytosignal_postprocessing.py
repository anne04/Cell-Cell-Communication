import gzip
import pandas as pd
import pickle
from collections import defaultdict
import numpy as np

options_list = ['dt-path_equally_spaced_lrc1467_cp100_noise0_random_overlap_threshold_dist_cellCount3000_3dim_3patterns_temp',
               'dt-path_equally_spaced_lrc1467_cp100_noise30_lowNoise_random_overlap_threshold_dist_cellCount3000_3dim_3patterns_temp',
               'dt-path_equally_spaced_lrc1467_cp100_noise30_heavyNoise_random_overlap_threshold_dist_cellCount3000_3dim_3patterns_temp',
               
               'dt-path_uniform_distribution_lrc112_cp100_noise0_random_overlap_threshold_dist_cellCount5000_3dim_3patterns_temp',
               'dt-path_uniform_distribution_lrc112_cp100_noise30_lowNoise_random_overlap_threshold_dist_cellCount5000_3dim_3patterns_temp',
               'dt-path_uniform_distribution_lrc112_cp100_noise30_heavyNoise_random_overlap_threshold_dist_cellCount5000_3dim_3patterns_temp_v2',
               
               'dt-path_mixture_of_distribution_lrc112_cp100_noise0_random_overlap_knn_cellCount5000_3dim_3patterns_temp',
                'dt-path_mixture_of_distribution_lrc112_cp100_noise30_lowNoise_random_overlap_knn_cellCount5000_3dim_3patterns_temp',
                'dt-path_mixture_of_distribution_lrc112_cp100_noise30_heavyNoise_random_overlap_knn_cellCount5000_3dim_3patterns_temp',

                'dt-randomCCC_equally_spaced_lrc105_cp100_noise0_threshold_dist_cellCount3000',
                'dt-randomCCC_uniform_distribution_lrc105_cp100_noise0_threshold_dist_cellCount5000',
                'dt-randomCCC_mix_distribution_lrc105_cp100_noise0_knn_cellCount5000'
               ] 

for op_index in range (0, len(options_list)):
    print('%d'%op_index)
    options = options_list[op_index]

    fp = gzip.open("/cluster/projects/schwartzgroup/fatema/find_ccc/" + 'Tclass_synthetic_data_ccc_roc_control_model_'+ options, 'rb')  # at least one of lig or rec has exp > respective knee point          
    lr_database, lig_rec_dict_TP, random_activation = pickle.load(fp)

######################################################## cytosignal #########################################################################
    # get all the edges and their scaled scores that they use for plotting the heatmap
    list_ccc = pd.read_csv('/cluster/projects/schwartzgroup/fatema/cytosignal/sender_vs_rec_'+options+'.csv')
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
    datapoint_size = 3000
    for i in range (0, datapoint_size):
        attention_scores.append([])   
        lig_rec_dict.append([])   
        for j in range (0, datapoint_size):	
            attention_scores[i].append([])   
            attention_scores[i][j] = []
            lig_rec_dict[i].append([])   
            lig_rec_dict[i][j] = []
    
    distribution = []
    for index in range (0, len(list_ccc.index)):
        i = int(list_ccc['i'][index])
        j = int(list_ccc['j'][index])
        score = list_ccc['score'][index]
        ccc_id = int(list_ccc['ccc'][index])
        attention_scores[i][j].append(score)
        lig_rec_dict[i][j].append(ccc_id)
        distribution.append(score)


    plot_dict = defaultdict(list)
    percentage_value = 100
    while percentage_value > 0:
        percentage_value = percentage_value - 10
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
        plot_dict['Type'].append('CytoSignal') #_lowNoise
    
    
    with gzip.open("/cluster/projects/schwartzgroup/fatema/find_ccc/" + options +'_'+'CytoSignal', 'wb') as fp: #b, b_1, a  11to20runs
        pickle.dump(plot_dict, fp) #a - [0:5]
