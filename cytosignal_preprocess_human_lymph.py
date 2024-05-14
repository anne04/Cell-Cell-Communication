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

