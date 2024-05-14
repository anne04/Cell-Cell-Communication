import gzip
import pandas as pd
import pickle
from collections import defaultdict
import numpy as np
import scanpy as sc
from scipy import sparse



adata_h5 = sc.read_visium(path='/cluster/projects/schwartzgroup/fatema/data/V1_Human_Lymph_Node_spatial/', count_file='filtered_feature_bc_matrix.h5')
print('input data read done')
adata_h5.var_names_make_unique()
gene_count_before = len(list(adata_h5.var_names) )    
sc.pp.filter_genes(adata_h5, min_cells=1)
gene_count_after = len(list(adata_h5.var_names) )  
print('Gene filtering done. Number of genes reduced from %d to %d'%(gene_count_before, gene_count_after))
gene_ids = list(adata_h5.var_names)
coordinates = adata_h5.obsm['spatial']
cell_barcode = np.array(adata_h5.obs.index)
cell_vs_gene = sparse.csr_matrix.toarray(adata_h5.X)

for i in range (0, len(cell_barcode)):
    cell_barcode[i] = cell_barcode[i].replace('-','.')


data_list = defaultdict(list)
for i in range (0, cell_vs_gene.shape[0]):
    for j in range (0, cell_vs_gene.shape[1]): 
        data_list[cell_barcode[i]].append(cell_vs_gene[i][j])

data_list_pd = pd.DataFrame(data_list)        

gene_name = []
for i in range (0, cell_vs_gene.shape[1]):
    gene_name.append(gene_ids[i])
    
data_list_pd[' ']=gene_name   
data_list_pd = data_list_pd.set_index(' ')    
data_list_pd.to_csv('/cluster/projects/schwartzgroup/fatema/find_ccc/cytosignal_metadata/human_lymph_raw_gene_vs_cell.csv')

cell_id = list(data_list.keys())    
spatial_dict = {' ': cell_id, 'x': list(coordinates[:,0]), 'y': list(coordinates[:,1])} 
spatial_dict = pd.DataFrame(spatial_dict)  
spatial_dict = spatial_dict.set_index(' ')  
spatial_dict.to_csv('/cluster/projects/schwartzgroup/fatema/find_ccc/cytosignal_metadata/human_lymph_spatial.csv')

#######################################################################################################
lr_database = pd.read_csv("/cluster/projects/schwartzgroup/fatema/NEST/database/NEST_database.csv")  # at least one of lig or rec has exp > respective knee point          


unique_gene_name = dict()
for i in range (0, len(lr_database)):
    unique_gene_name[lr_database['Ligand'][i]] = ''
    unique_gene_name[lr_database['Receptor'][i]] = ''


data_list=dict()
data_list['gene_name']=[]
data_list['uniprot']=[]
data_list['hgnc_symbol']=[]
for gene in unique_gene_name.keys():
    data_list['gene_name'].append(gene)
    data_list['uniprot'].append(gene)
    data_list['hgnc_symbol'].append(gene)

data_list_pd = pd.DataFrame(data_list)   
data_list_pd.to_csv('/cluster/projects/schwartzgroup/fatema/find_ccc/cytosignal_metadata/NEST_db_gene_to_u.csv', index=False) 


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
    data_list['id_cp_interaction'].append('CPI-'+str(i)) # this is also used as index
    data_list['partner_a'].append(lr_database['Ligand'][i])
    data_list['partner_b'].append(lr_database['Receptor'][i])
    data_list['protein_name_a'].append(lr_database['Ligand'][i])
    data_list['protein_name_b'].append(lr_database['Receptor'][i]) 
    data_list['annotation_strategy'].append('database')
    data_list['source'].append(lr_database['Reference'][i])
    data_list['partner_a_is_recep'].append('FALSE')
    data_list['partner_b_is_recep'].append('TRUE')
    data_list['recep_flag'].append(1) # = 1 if one of a or b is receptor. if both, then 2. if none then 0.
    data_list['order_flag'].append(1) # =0 if a is receptor, else 1
      
data_list_pd = pd.DataFrame(data_list)   
data_list_pd[' '] = data_list['id_cp_interaction']
data_list_pd = data_list_pd.set_index(' ')  
data_list_pd.to_csv('/cluster/projects/schwartzgroup/fatema/find_ccc/cytosignal_metadata/NEST_db_inter.index.csv') #, index=False 
#########################################################################################################################################
protein_name_ligand = []
interaction_id_ligand = []
protein_name_receptor = []
interaction_id_receptor = []

for i in range (0, len(lr_database)):
    if lr_database['Annotation'][i]=='Secreted Signaling':
        protein_name_ligand.append(lr_database['Ligand'][i])
        interaction_id_ligand.append(data_list['id_cp_interaction'][i])
        protein_name_receptor.append(lr_database['Receptor'][i])
        interaction_id_receptor.append(data_list['id_cp_interaction'][i])

list_pd = pd.DataFrame(protein_name_ligand) 
list_pd.to_csv('/cluster/projects/schwartzgroup/fatema/find_ccc/cytosignal_metadata/NEST_db_diff_protein_name_ligand.csv', header=False, index=False) 

list_pd = pd.DataFrame(interaction_id_ligand) 
list_pd.to_csv('/cluster/projects/schwartzgroup/fatema/find_ccc/cytosignal_metadata/NEST_db_diff_interaction_id_ligand.csv', header=False, index=False) 

list_pd = pd.DataFrame(protein_name_receptor) 
list_pd.to_csv('/cluster/projects/schwartzgroup/fatema/find_ccc/cytosignal_metadata/NEST_db_diff_protein_name_receptor.csv', header=False, index=False) 

list_pd = pd.DataFrame(interaction_id_receptor) 
list_pd.to_csv('/cluster/projects/schwartzgroup/fatema/find_ccc/cytosignal_metadata/NEST_db_diff_interaction_id_receptor.csv', header=False, index=False) 

#########################################################################################################################################
protein_name_ligand = []
interaction_id_ligand = []
protein_name_receptor = []
interaction_id_receptor = []

for i in range (0, len(lr_database)):
    if lr_database['Annotation'][i]=='Cell-Cell Contact':
        protein_name_ligand.append(lr_database['Ligand'][i])
        interaction_id_ligand.append(data_list['id_cp_interaction'][i])
        protein_name_receptor.append(lr_database['Receptor'][i])
        interaction_id_receptor.append(data_list['id_cp_interaction'][i])

list_pd = pd.DataFrame(protein_name_ligand) 
list_pd.to_csv('/cluster/projects/schwartzgroup/fatema/find_ccc/cytosignal_metadata/NEST_db_cont_protein_name_ligand.csv', header=False, index=False) 

list_pd = pd.DataFrame(interaction_id_ligand) 
list_pd.to_csv('/cluster/projects/schwartzgroup/fatema/find_ccc/cytosignal_metadata/NEST_db_cont_interaction_id_ligand.csv', header=False, index=False) 

list_pd = pd.DataFrame(protein_name_receptor) 
list_pd.to_csv('/cluster/projects/schwartzgroup/fatema/find_ccc/cytosignal_metadata/NEST_db_cont_protein_name_receptor.csv', header=False, index=False) 

list_pd = pd.DataFrame(interaction_id_receptor) 
list_pd.to_csv('/cluster/projects/schwartzgroup/fatema/find_ccc/cytosignal_metadata/NEST_db_cont_interaction_id_receptor.csv', header=False, index=False) 



