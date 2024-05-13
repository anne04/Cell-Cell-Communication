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
#    with gzip.open("/cluster/projects/schwartzgroup/fatema/find_ccc/" + 'synthetic_data_ccc_roc_control_model_'+ options +'_'+'_cellvsgene_'+ 'not_quantileTransformed', 'rb') as fp:
    with gzip.open("/cluster/projects/schwartzgroup/fatema/find_ccc/" + 'synthetic_data_ccc_roc_control_model_'+ options +'_cellvsgene', 'rb') as fp: #'not_quantileTransformed'
        cell_vs_gene = pickle.load(fp)

    #cell_vs_gene = cell_vs_gene*100
    min_gene_expr = np.min(cell_vs_gene)
    min_gene_expr = min_gene_expr*(-1)
    # scale the gene values from 0 to 1    
    data_list = defaultdict(list)
    for i in range (0, cell_vs_gene.shape[0]):
        for j in range (0, cell_vs_gene.shape[1]):
            #data_list['a'+str(i)].append(int(np.round(min_gene_expr+cell_vs_gene[i][j])))
            data_list['a'+str(i)].append(int(np.round(cell_vs_gene[i][j])))
    
    data_list_pd = pd.DataFrame(data_list)        
    gene_name = []
    for i in range (0, cell_vs_gene.shape[1]):
        gene_name.append('G'+str(i))
        
    data_list_pd[' ']=gene_name   
    data_list_pd = data_list_pd.set_index(' ')    
    data_list_pd.to_csv('/cluster/projects/schwartzgroup/fatema/find_ccc/synthetic_raw_gene_vs_cell_'+options+'.csv')

    ########################################################################
    with gzip.open("/cluster/projects/schwartzgroup/fatema/find_ccc/" + 'synthetic_data_ccc_roc_control_model_'+ options +'_xny', 'rb') as fp:
        temp_x, temp_y, ccc_region  = pickle.load(fp)
    
    cell_id = list(data_list.keys())    
    spatial_dict = {' ': cell_id, 'x': list(temp_x), 'y': list(temp_y)} 
    spatial_dict = pd.DataFrame(spatial_dict)  
    spatial_dict = spatial_dict.set_index(' ')  
    spatial_dict.to_csv('/cluster/projects/schwartzgroup/fatema/find_ccc/synthetic_cell_'+options+'_spatial.csv')
  
    ########################################################################
    fp = gzip.open("/cluster/projects/schwartzgroup/fatema/find_ccc/" + 'Tclass_synthetic_data_ccc_roc_control_model_'+ options, 'rb')  # at least one of lig or rec has exp > respective knee point          
    lr_database, lig_rec_dict_TP, random_activation = pickle.load(fp)

    unique_gene_name = dict()
    for i in range (0, len(lr_database)):
        unique_gene_name['G'+str(lr_database[i][0])] = ''
        unique_gene_name['G'+str(lr_database[i][1])] = ''


    data_list=dict()
    data_list['gene_name']=[]
    data_list['uniprot']=[]
    data_list['hgnc_symbol']=[]
    for gene in unique_gene_name.keys():
        data_list['gene_name'].append(gene)
        data_list['uniprot'].append(gene)
        data_list['hgnc_symbol'].append(gene)

    data_list_pd = pd.DataFrame(data_list)   
    data_list_pd.to_csv('/cluster/home/t116508uhn/cytosignal/gene_to_u_'+options+'.csv', index=False) 


    data_list=dict()
    data_list['id_cp_interaction']=[] # this is also used as index
    data_list['partner_a']=[]
    data_list['partner_b']=[]
    data_list['protein_name_a']=[]
    data_list['protein_name_b']=[]
    data_list['annotation_strategy']=[]
    data_list['source']=[]
    data_list['partner_a_is_recep']=[] #TRUE or FALSE
    data_list['partner_b_is_recep']=[] #TRUE or FALSE
    data_list['recep_flag']=[] # = 1 if one of a or b is receptor. if both, then 2. if none then 0.
    data_list['order_flag']=[] # =0 if a is receptor, else 1
    

    
    for i in range (0, len(lr_database)):
        data_list['id_cp_interaction'].append(str(i)) # this is also used as index
        data_list['partner_a'].append('G'+str(lr_database[i][0]))
        data_list['partner_b'].append('G'+str(lr_database[i][1]))
        data_list['protein_name_a'].append('G'+str(lr_database[i][0]))
        data_list['protein_name_b'].append('G'+str(lr_database[i][1]))
        data_list['annotation_strategy'].append('curated')
        data_list['source'].append('synthetic')
        data_list['partner_a_is_recep'].append(False)
        data_list['partner_b_is_recep'].append(True)
        data_list['recep_flag'].append(1) # = 1 if one of a or b is receptor. if both, then 2. if none then 0.
        data_list['order_flag'].append(1) # =0 if a is receptor, else 1
          
    data_list_pd = pd.DataFrame(data_list)   
    data_list_pd[' '] = data_list['id_cp_interaction']
    data_list_pd = data_list_pd.set_index(' ')  
    data_list_pd.to_csv('/cluster/home/t116508uhn/cytosignal/inter.index_'+options+'.csv') #, index=False 

    protein_name_ligand = []
    interaction_id_ligand = []
    protein_name_receptor = []
    interaction_id_receptor = []

    for i in range (0, len(lr_database)):
        protein_name_ligand.append('G'+str(lr_database[i][0]))
        interaction_id_ligand.append(data_list['id_cp_interaction'][i])
        protein_name_receptor.append('G'+str(lr_database[i][1]))
        interaction_id_receptor.append(data_list['id_cp_interaction'][i])

    list_pd = pd.DataFrame(protein_name_ligand) 
    list_pd.to_csv('/cluster/home/t116508uhn/cytosignal/protein_name_ligand_'+options+'.csv', header=False, index=False) 

    list_pd = pd.DataFrame(interaction_id_ligand) 
    list_pd.to_csv('/cluster/home/t116508uhn/cytosignal/interaction_id_ligand_'+options+'.csv', header=False, index=False) 

    list_pd = pd.DataFrame(protein_name_receptor) 
    list_pd.to_csv('/cluster/home/t116508uhn/cytosignal/protein_name_receptor_'+options+'.csv', header=False, index=False) 

    list_pd = pd.DataFrame(interaction_id_receptor) 
    list_pd.to_csv('/cluster/home/t116508uhn/cytosignal/interaction_id_receptor_'+options+'.csv', header=False, index=False) 

