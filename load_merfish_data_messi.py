%matplotlib inline

import copy
import os
import pickle
from collections import defaultdict
import itertools
import gzip

import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
import scipy as sp
from sklearn.cluster import AgglomerativeClustering


import messi
from messi.data_processing import *
from messi.hme import hme
from messi.gridSearch import gridSearch

input_path = 'input/'
output_path = 'output/'
data_type = 'merfish'
sex = 'Female'
behavior = 'Parenting'
behavior_no_space = behavior.replace(" ", "_")
current_cell_type = 'Excitatory'
current_cell_type_no_space = current_cell_type.replace(" ", "_")

grid_search = True #False #
n_sets = 5  # for example usage only; we recommend 5

n_classes_0 = 1
n_classes_1 = 5
n_epochs = 20  # for example usage only; we recommend using the default 20 n_epochs 

preprocess = 'neighbor_cat'
top_k_response = None #20  # for example usage only; we recommend use all responses (i.e. None)
top_k_regulator = None
response_type = 'original'  # use raw values to fit the model
condition = f"response_{top_k_response}_l1_{n_classes_0}_l2_{n_classes_1}"

if grid_search:
    condition = f"response_{top_k_response}_l1_{n_classes_0}_l2_grid_search"
else:
    condition = f"response_{top_k_response}_l1_{n_classes_0}_l2_{n_classes_1}"

read_in_functions = {'merfish': [read_meta_merfish, read_merfish_data, get_idx_per_dataset_merfish],
                    'merfish_cell_line': [read_meta_merfish_cell_line, read_merfish_cell_line_data, get_idx_per_dataset_merfish_cell_line],
                    'starmap': [read_meta_starmap_combinatorial, read_starmap_combinatorial, get_idx_per_dataset_starmap_combinatorial]}

# set data reading functions corresponding to the data type
if data_type in ['merfish', 'merfish_cell_line', 'starmap']:
    read_meta = read_in_functions[data_type][0]
    read_data = read_in_functions[data_type][1]
    get_idx_per_dataset = read_in_functions[data_type][2]
else:
    raise NotImplementedError(f"Now only support processing 'merfish', 'merfish_cell_line' or 'starmap'")

# read in ligand and receptor lists
#l_u, r_u = get_lr_pairs(input_path='../messi/input/')  # may need to change to the default value

lr_pairs = pd.read_html(os.path.join('input/','ligand_receptor_pairs2.txt'), header=None)[0] #pd.read_table(os.path.join(input_path, filename), header=None)
lr_pairs.columns = ['ligand','receptor']
lr_pairs[['ligand','receptor']] = lr_pairs['receptor'].str.split('\t',expand=True)
lr_pairs['ligand'] = lr_pairs['ligand'].apply(lambda x: x.upper())
lr_pairs['receptor'] = lr_pairs['receptor'].apply(lambda x: x.upper())


l_u_p = set([l.upper() for l in lr_pairs['ligand']])
r_u_p = set([g.upper() for g in lr_pairs['receptor']])
l_u_search = [] # set(['CBLN1', 'CXCL14', 'CBLN2', 'VGF','SCG2','CARTPT','TAC2'])
r_u_search = [] # set(['CRHBP', 'GABRA1', 'GPR165', 'GLRA3', 'GABRG1', 'ADORA2A'])
l_u = l_u_p.union(l_u_search)
r_u = r_u_p.union(r_u_search)


# read in meta information about the dataset # meta_all = cell x metadata
meta_all, meta_all_columns, cell_types_dict, genes_list, genes_list_u, \
response_list_prior, regulator_list_prior = read_meta('input/', behavior_no_space, sex, l_u, r_u)  # TO BE MODIFIED: number of responses


#genes_list_u = genes_list_us_messi
   


# get all available animals/samples -- get unique IDs
all_animals = list(set(meta_all[:, meta_all_columns['Animal_ID']])) # 16, 17, 18, 19

test_animal  = 16
test_animals = [test_animal]
samples_test = np.array(test_animals)
samples_train = np.array(list(set(all_animals))) #np.array(list(set(all_animals)-set(test_animals)))
print(f"Test set is {samples_test}")
print(f"Training set is {samples_train}")
bregma = None
idx_train, idx_test, idx_train_in_general, \
idx_test_in_general, idx_train_in_dataset, \
idx_test_in_dataset, meta_per_dataset_train, \
meta_per_dataset_test = find_idx_for_train_test(samples_train, samples_test, 
                                                meta_all, meta_all_columns, data_type, 
                                                current_cell_type, get_idx_per_dataset,
                                                return_in_general = False, 
                                                bregma=bregma)
##################################################################
data_sets = []
data_sets_gatconv = []
i = 0
for animal_id, bregma in meta_per_dataset_train:
    hp, hp_cor, hp_genes = read_data('input/', bregma, animal_id, genes_list, genes_list_u)
    

    if hp is not None:
        hp_columns = dict(zip(hp.columns, range(0, len(hp.columns))))
        hp_np = hp.to_numpy()
    else:
        hp_columns = None
        hp_np = None
        
        
    hp_cor_columns = dict(zip(hp_cor.columns, range(0, len(hp_cor.columns))))
    hp_genes_columns = dict(zip(hp_genes.columns, range(0, len(hp_genes.columns))))
    data_sets.append([hp_np, hp_columns, hp_cor.to_numpy(), hp_cor_columns,
                      hp_genes.to_numpy(), hp_genes_columns])
        
    cell_barcodes = data_sets[i][0][:,0]
    coordinates = data_sets[i][0][:,5:7]
    cell_vs_gene  = data_sets[i][4]
    cell_vs_animal_id_sex_behavior_bregma = data_sets[i][0][:,1:5]
    cell_vs_class_neuron_cluster_id = data_sets[i][0][:,7:]
    gene_ids = data_sets[i][5]
    data_sets_gatconv.append([cell_barcodes, coordinates, cell_vs_gene, gene_ids, cell_vs_animal_id_sex_behavior_bregma, cell_vs_class_neuron_cluster_id])
    i = i + 1
    del hp, hp_cor, hp_genes


with gzip.open("/cluster/projects/schwartzgroup/fatema/find_ccc/merfish_mouse_cortex/" + 'messi_merfish_data', 'wb') as fp:  #b, a:[0:5]           
#with gzip.open("/cluster/projects/schwartzgroup/fatema/find_ccc/" + 'adjacency_records_GAT_synthetic_region1_onlyccc_70', 'wb') as fp:
    pickle.dump([data_sets_gatconv, lr_pairs], fp) 
    
    
