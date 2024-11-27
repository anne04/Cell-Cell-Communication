import gzip
import pandas as pd
import pickle
from collections import defaultdict
import numpy as np

options_list = ['', '', ''
               ] 
dirType = ['type_equidistant_mechanistic/','type_equidistant_mechanistic/','type_equidistant_mechanistic/', 'type_uniform_distribution_mechanistic/','type_uniform_distribution_mechanistic/','type_uniform_distribution_mechanistic/', 'type_mixed_distribution_mechanistic/','type_mixed_distribution_mechanistic/','type_mixed_distribution_mechanistic/' ]
noise_dir = ['no_noise/', 'lowNoise/', 'highNoise/', 'no_noise/', 'lowNoise/', 'highNoise/', 'no_noise/', 'lowNoise/', 'highNoise/']
datatype = ['equidistant_mechanistic','equidistant_mechanistic','equidistant_mechanistic', 'uniform_mechanistic',  'uniform_mechanistic', 'uniform_mechanistic','mixture_mechanistic', 'mixture_mechanistic', 'mixture_mechanistic']
noisetype = ['noise0', 'noise30level1', 'noise30level2','noise0', 'noise30level1', 'noise30level2','noise0', 'noise30level1', 'noise30level2']

for op_index in range (0, len(datatype)):
    print('%d'%op_index)
    options = datatype[op_index] + '_' + noisetype[op_index]
    sample_type = op_index  
    with gzip.open("/cluster/projects/schwartzgroup/fatema/find_ccc/synthetic_data/"+ dirType[sample_type] + noise_dir[sample_type]+ datatype[sample_type] + '_' + noisetype[sample_type] +"_cellvsgene_not_normalized", 'rb') as fp: #'not_quantileTransformed'
        cell_vs_gene = pickle.load(fp)
      
    cell_vs_gene = cell_vs_gene*100
    print(np.min(cell_vs_gene))
    min_gene_expr = np.min(cell_vs_gene)
    toggle = 0
    if min_gene_expr<0:
        toggle = 1
        min_gene_expr = min_gene_expr*(-1)
    # scale the gene values from 0 to 1    
    data_list = defaultdict(list)
    for i in range (0, cell_vs_gene.shape[0]):
        for j in range (0, cell_vs_gene.shape[1]):
            if toggle == 1:
                cell_vs_gene[i][j] = min_gene_expr+cell_vs_gene[i][j]
            
            data_list['a'+str(i)].append(int(np.round(cell_vs_gene[i][j])))
          
    print('min %d and max %d'%(np.min(cell_vs_gene), np.max(cell_vs_gene)))
    data_list_pd = pd.DataFrame(data_list)        
    gene_name = []
    for i in range (0, cell_vs_gene.shape[1]):
        gene_name.append('G'+str(i))
        
    data_list_pd[' ']=gene_name   
    data_list_pd = data_list_pd.set_index(' ')    
    data_list_pd.to_csv('/cluster/projects/schwartzgroup/fatema/find_ccc/cytosignal_metadata/synthetic_raw_gene_vs_cell_'+options+'.csv')
    #######################################################################
    data_list = defaultdict(list)
    for i in range (0, cell_vs_gene.shape[0]):
        for j in range (0, cell_vs_gene.shape[1]):
            if toggle == 1:
                cell_vs_gene[i][j] = min_gene_expr+cell_vs_gene[i][j]
            
            data_list['a-'+str(i)].append(int(np.round(cell_vs_gene[i][j])))
          
    print('min %d and max %d'%(np.min(cell_vs_gene), np.max(cell_vs_gene)))
    data_list_pd = pd.DataFrame(data_list)        
    gene_name = []
    for i in range (0, cell_vs_gene.shape[1]):
        gene_name.append('g'+str(i))
        
    data_list_pd[' ']=gene_name   
    data_list_pd = data_list_pd.set_index(' ')    
    data_list_pd.to_csv('/cluster/home/t116508uhn/synthetic_gene_vs_cell_'+options+'_shifted.csv')


    ########################################################################
    with gzip.open("/cluster/projects/schwartzgroup/fatema/find_ccc/synthetic_data/"+ dirType[sample_type] + noise_dir[sample_type]+ datatype[sample_type] + '_' + noisetype[sample_type] +"_coordinate", 'rb') as fp:
        temp_x, temp_y, ccc_region  = pickle.load(fp)
    
    cell_id = list(data_list.keys())    
    spatial_dict = {' ': cell_id, 'x': list(temp_x), 'y': list(temp_y)} 
    spatial_dict = pd.DataFrame(spatial_dict)  
    spatial_dict = spatial_dict.set_index(' ')  
    spatial_dict.to_csv('/cluster/projects/schwartzgroup/fatema/find_ccc/cytosignal_metadata/synthetic_cell_'+options+'_spatial.csv')
  
    ########################################################################
    fp = gzip.open("/cluster/projects/schwartzgroup/fatema/find_ccc/synthetic_data/"+ dirType[sample_type] + noise_dir[sample_type]+ datatype[sample_type] + '_' + noisetype[sample_type] +"_ground_truth", 'rb')  # _ccc at least one of lig or rec has exp > respective knee point          
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
    data_list_pd.to_csv('/cluster/projects/schwartzgroup/fatema/find_ccc/cytosignal_metadata/gene_to_u_'+options+'.csv', index=False) 


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
    data_list_pd.to_csv('/cluster/projects/schwartzgroup/fatema/find_ccc/cytosignal_metadata/inter.index_'+options+'.csv') #, index=False 

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
    list_pd.to_csv('/cluster/projects/schwartzgroup/fatema/find_ccc/cytosignal_metadata/protein_name_ligand_'+options+'.csv', header=False, index=False) 

    list_pd = pd.DataFrame(interaction_id_ligand) 
    list_pd.to_csv('/cluster/projects/schwartzgroup/fatema/find_ccc/cytosignal_metadata/interaction_id_ligand_'+options+'.csv', header=False, index=False) 

    list_pd = pd.DataFrame(protein_name_receptor) 
    list_pd.to_csv('/cluster/projects/schwartzgroup/fatema/find_ccc/cytosignal_metadata/protein_name_receptor_'+options+'.csv', header=False, index=False) 

    list_pd = pd.DataFrame(interaction_id_receptor) 
    list_pd.to_csv('/cluster/projects/schwartzgroup/fatema/find_ccc/cytosignal_metadata/interaction_id_receptor_'+options+'.csv', header=False, index=False) 
'''
0
-1514.8639982391362

min 0 and max 5966
1
-1588.0357282745742
min 0 and max 6047
2
-1659.604418395068
min 0 and max 6148
3
131.64238378638817
min 131 and max 4319
4
104.6523318111215
min 104 and max 4321
5
-31.63807988355923
min 0 and max 4422
6
131.3582469862863
min 131 and max 4262
7
77.6526719037674
min 77 and max 4271
8
-32.888527193529086
min 0 and max 4369
9
67.88465167205061
min 67 and max 627
10
194.55597458895164
min 194 and max 1747
11
116.20666755039852
min 116 and max 947
'''
