import gzip
import pandas as pd
import pickle
from collections import defaultdict

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
    with gzip.open("/cluster/projects/schwartzgroup/fatema/find_ccc/" + 'synthetic_data_ccc_roc_control_model_'+ options +'_'+'_cellvsgene_'+ 'not_quantileTransformed', 'rb') as fp:
    #with gzip.open("/cluster/projects/schwartzgroup/fatema/find_ccc/" + 'synthetic_data_ccc_roc_control_model_'+ options +'_cellvsgene', 'rb') as fp: #'not_quantileTransformed'
        cell_vs_gene = pickle.load(fp)
    
    with gzip.open("/cluster/projects/schwartzgroup/fatema/find_ccc/" + 'synthetic_data_ccc_roc_control_model_'+ options +'_xny', 'rb') as fp:
        temp_x, temp_y, ccc_region  = pickle.load(fp)
    
    data_list = defaultdict(list)
    min_value = np.min(cell_vs_gene)
    max_value = np.max(cell_vs_gene)
    for i in range (0, cell_vs_gene.shape[0]):
        for j in range (0, cell_vs_gene.shape[1]):
            data_list['a'+str(i)].append((cell_vs_gene[i][j]-min_value)/(max_value-min_value))
            
    cell_id = list(data_list.keys())    
    spatial_dict = {' ': cell_id, 'x': list(temp_x), 'y': list(temp_y)} 
    spatial_dict = pd.DataFrame(spatial_dict)  
    spatial_dict = spatial_dict.set_index(' ')  
    spatial_dict.to_csv('/cluster/projects/schwartzgroup/fatema/find_ccc/synthetic_cell_'+options+'_spatial.csv')
  
    data_list_pd = pd.DataFrame(data_list)    
    gene_name = []
    for i in range (0, cell_vs_gene.shape[1]):
        gene_name.append('g'+str(i))
        
    data_list_pd[' ']=gene_name   
    data_list_pd = data_list_pd.set_index(' ')    
    data_list_pd.to_csv('/cluster/projects/schwartzgroup/fatema/find_ccc/synthetic_raw_gene_vs_cell_'+options+'.csv')
    '''
    fp = gzip.open("/cluster/projects/schwartzgroup/fatema/find_ccc/" + 'Tclass_synthetic_data_ccc_roc_control_model_'+ options, 'rb')  # at least one of lig or rec has exp > respective knee point          
    lr_database, lig_rec_dict_TP, random_activation = pickle.load(fp)
    
    data_list=dict()
    data_list['ligand']=[]
    data_list['receptor']=[]
    for i in range (0, len(lr_database)):
        data_list['ligand'].append('g'+str(lr_database[i][0]))
        data_list['receptor'].append('g'+str(lr_database[i][1]))
        
    data_list_pd = pd.DataFrame(data_list)        
    data_list_pd.to_csv('/cluster/projects/schwartzgroup/fatema/find_ccc/synthetic_lr_'+options+'.csv', index=False)
    '''
