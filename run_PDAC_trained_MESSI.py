%matplotlib inline
import copy
import os
import pickle
from collections import defaultdict
import itertools

import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd

import messi
from messi.data_processing import *
from messi.hme import hme
from messi.gridSearch import gridSearch

import scipy as sp
from sklearn.cluster import AgglomerativeClustering
import stlearn as st
import scanpy as sc
from scipy import sparse
import csv
import argparse
parser = argparse.ArgumentParser()
parser.add_argument( '--data_path', type=str, default='/cluster/home/t116508uhn/64630/spaceranger_output_new/' , help='The path to dataset') 
args = parser.parse_args()

####
input_path = '../../input/merfish/'
output_path = '../output/'
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
l_u_search = set(['CBLN1', 'CXCL14', 'CBLN2', 'VGF','SCG2','CARTPT','TAC2'])
r_u_search = set(['CRHBP', 'GABRA1', 'GPR165', 'GLRA3', 'GABRG1', 'ADORA2A'])
l_u = l_u_p.union(l_u_search)
r_u = r_u_p.union(r_u_search)



with gzip.open("/cluster/projects/schwartzgroup/fatema/find_ccc/" + 'gene_ids_messi_us', 'rb') as fp: #b, b_1, a
    genes_list_us_messi = pickle.load(fp) 

genes_list_us_messi_list = list(genes_list_us_messi)
genes_list_us_messi_dict = dict()
for gene in genes_list_us_messi_list:
    genes_list_us_messi_dict[gene] = ''
    
us_messi_r = genes_list_us_messi.intersection(r_u) 
us_messi_r_dict = dict()
for gene in us_messi_r:
    us_messi_r_dict[gene] = ''
    
# pick 'N'(=5) l_r 

r_chosen_list = []
r_chosen_dict = dict()
for i in range (0, len(lr_pairs)):
    if lr_pairs['receptor'][i] in us_messi_r_dict:
        # see if it's ligand exist in other pairs as well
        my_ligand = lr_pairs['ligand'][i]
        if my_ligand not in genes_list_us_messi_dict:
            continue
        
        for j in range (0, len(lr_pairs)):
            if lr_pairs['ligand'][j] == my_ligand:
                r_chosen_dict[lr_pairs['receptor'][i]] = '' # rec pairs to keep 
                r_chosen_list.append(i) # index to keep
                break
        
        if len(r_chosen_list) == 5:
            break
            
chosen_ligand_rec_pair = []
for index in r_chosen_list:
    chosen_ligand_rec_pair.append([lr_pairs['ligand'][index], lr_pairs['receptor'][index]])

# find rows to remove - remove all those receptors 
r_remove_list = []
for i in range (0, len(lr_pairs)):
    if lr_pairs['receptor'][i] in r_chosen_dict:
        r_remove_list.append(i)
        
filtered_lr_pairs = lr_pairs.drop(r_remove_list)  

lr_pairs = filtered_lr_pairs
      
l_u_p = set([l.upper() for l in lr_pairs['ligand']])
r_u_p = set([g.upper() for g in lr_pairs['receptor']])
l_u_search = set(['CBLN1', 'CXCL14', 'CBLN2', 'VGF','SCG2','CARTPT','TAC2'])
r_u_search = set(['CRHBP', 'GABRA1', 'GPR165', 'GLRA3', 'GABRG1', 'ADORA2A'])
l_u = l_u_p.union(l_u_search)
r_u = r_u_p.union(r_u_search)



#################### coordinate #############################################
coordinates = np.load('/cluster/projects/schwartzgroup/fatema/CCST/generated_data_new/V10M25-61_D1_PDA_64630_Pa_P_Spatial10x_new/'+'coordinates.npy')
       
hp_cor = np.zeros((coordinates.shape[0],2))
for i in range (0, coordinates.shape[0]):
    hp_cor[i][0] = coordinates[i][0]
    hp_cor[i][1] = coordinates[i][1]

hp_cor_columns = dict()
hp_cor_columns['Centroid_X'] = 0
hp_cor_columns['Centroid_Y'] = 1

##################### gene ########################################    
data_fold = args.data_path #+args.data_name+'/'
adata_h5 = st.Read10X(path=data_fold, count_file='filtered_feature_bc_matrix.h5') #count_file=args.data_name+'_filtered_feature_bc_matrix.h5' )
print(adata_h5)
sc.pp.filter_genes(adata_h5, min_cells=1)
print(adata_h5)
gene_ids = list(adata_h5.var_names)
cell_vs_gene = sparse.csr_matrix.toarray(adata_h5.X)
###########################
hp_columns = dict() #set(['Cell_ID','Cell_class','Animal_ID','Bregma','ID_in_dataset'])
# EDIT
hp_columns['Cell_ID'] = 0
hp_columns['Animal_ID'] = 1
hp_columns['Animal_sex'] = 2
hp_columns['Behavior'] = 3
hp_columns['Bregma'] = 4
hp_columns['Cell_class'] = 5
hp_columns['ID_in_dataset'] = 6

#############################################################
barcode_file='/cluster/home/t116508uhn/64630/spaceranger_output_new/unzipped/barcodes.tsv'
barcode_info=[]
#barcode_info.append("")
i=0
with open(barcode_file) as file:
    tsv_file = csv.reader(file, delimiter="\t")
    for line in tsv_file:
        barcode_info.append([])
        barcode_info[i].append(line[0])
        barcode_info[i].append(1)
        barcode_info[i].append('Female')
        barcode_info[i].append('Parenting')
        barcode_info[i].append(.26)      
        barcode_info[i].append('Excitatory')
        barcode_info[i].append(i)
        i=i+1
              
barcode_type = pd.DataFrame(barcode_info)
hp_np = barcode_type.to_numpy()        

#####################################
with gzip.open("/cluster/projects/schwartzgroup/fatema/find_ccc/" + 'gene_ids_messi_us', 'rb') as fp: #b, b_1, a
    genes_list_us_messi = pickle.load(fp) 

index_genes = [] # to keep
for gene in genes_list_us_messi:
    for i in range (0, len(gene_ids)):
        if gene_ids[i] == gene:
            index_genes.append(i)
            break
              

cell_vs_gene = cell_vs_gene[:,index_genes]
#gene_ids = genes_list_us_messi
genes_list_u = genes_list_us_messi

#print(gene_ids)
#################
hp_genes = cell_vs_gene
hp_genes_columns = dict()
j = 0
for gene in genes_list_us_messi:
   hp_genes_columns[gene] = j
   j = j+1

############################################
data_sets=[]
data_sets.append([hp_np, hp_columns, hp_cor, hp_cor_columns, hp_genes, hp_genes_columns])
datasets_test = data_sets

## training data #####
data_sets=[]
total_spots = len(barcode_info)
j = len(barcode_info)
for tr_index in [0, 2, 3]:
    barcode_info_next=[]
    for i in range (0, total_spots):
        barcode_info_next.append([])
        barcode_info_next[i].append(i)
        barcode_info_next[i].append(tr_index)
        barcode_info_next[i].append('Female')
        barcode_info_next[i].append('Parenting')
        barcode_info_next[i].append(.26)      
        barcode_info_next[i].append('Excitatory')
        barcode_info_next[i].append(i)
              
        # ########################   
        barcode_info.append([])
        barcode_info[j].append(i)
        barcode_info[j].append(tr_index)
        barcode_info[j].append('Female')
        barcode_info[j].append('Parenting')
        barcode_info[j].append(.26)      
        barcode_info[j].append('Excitatory')
        barcode_info[j].append(i)
        j = j+1
        ##############      
    barcode_type_next = pd.DataFrame(barcode_info_next)
    hp_np = barcode_type_next.to_numpy()    
    print(hp_np.shape)
    ## random noise to gene exp ##
    random_noise = np.random.uniform(tr_index*1,(tr_index+1)*2, [hp_genes.shape[0],hp_genes.shape[1]])
    data_sets.append([hp_np, hp_columns, hp_cor, hp_cor_columns, hp_genes+random_noise, hp_genes_columns])
     
datasets_train = data_sets
####

####
response_list_prior = regulator_list_prior = None
barcode_type = pd.DataFrame(barcode_info)
meta_all = barcode_type.to_numpy()
a, meta_all_columns, cell_types_dict, a, a, response_list_prior, regulator_list_prior = read_meta('input/', behavior_no_space, sex, l_u, r_u)  # TO BE MODIFIED: number of responses

# get all available animals/samples -- get unique IDs
all_animals = list(set(meta_all[:, meta_all_columns['Animal_ID']])) # 16, 17, 18, 19
# test animal is 16. and all others are train. Then get the index of 16 and train animals separately. 
test_animal  = 1
test_animals = [test_animal]
samples_test = np.array(test_animals)
samples_train = np.array(list(set(all_animals)-set(test_animals)))
samples_train = np.array([0, 2, 3])
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


#############
if data_type == 'merfish_rna_seq':
    neighbors_train = None
    neighbors_test = None
else: 
    if data_type == 'merfish':
        dis_filter = 300
    else:
        dis_filter = 1e9  
        
    neighbors_train = get_neighbors_datasets(datasets_train, "Del", k=5, dis_filter=dis_filter, include_self = False)
    neighbors_test = get_neighbors_datasets(datasets_test, "Del", k=5, dis_filter=dis_filter, include_self = False)

lig_n =  {'name':'regulators_neighbor','helper':preprocess_X_neighbor_per_cell, 
                      'feature_list_type': 'regulator_neighbor', 'per_cell':True, 'baseline':False, 
                      'standardize': True, 'log':True, 'poly':False}
rec_s = {'name':'regulators_self','helper':preprocess_X_self_per_cell, 
                      'feature_list_type': 'regulator_self', 'per_cell':True, 'baseline':False, 
                      'standardize': True, 'log':True, 'poly':False}
lig_s = {'name':'regulators_neighbor_self','helper':preprocess_X_self_per_cell, 
                      'feature_list_type':'regulator_neighbor', 'per_cell':True, 'baseline':False, 
                      'standardize': True, 'log':True, 'poly':False}
type_n =  {'name': 'neighbor_type','helper':preprocess_X_neighbor_type_per_dataset, 
                      'feature_list_type':None,'per_cell':False, 'baseline':False, 
                      'standardize': True, 'log':False, 'poly':False}
base_s = {'name':'baseline','helper':preprocess_X_baseline_per_dataset,'feature_list_type':None, 
                      'per_cell':False, 'baseline':True, 'standardize': True, 'log':False, 'poly':False}


if data_type == 'merfish_cell_line':
    feature_types = [lig_n, rec_s, base_s, lig_s]
    
else:
    feature_types = [lig_n, rec_s, type_n , base_s, lig_s]

X_trains, X_tests, regulator_list_neighbor, regulator_list_self  = prepare_features(data_type, datasets_train, datasets_test, meta_per_dataset_train, meta_per_dataset_test, 
                     idx_train, idx_test, idx_train_in_dataset, idx_test_in_dataset, neighbors_train, neighbors_test,
                    feature_types, regulator_list_prior, top_k_regulator, genes_list_u, l_u, r_u, cell_types_dict)



total_regulators = regulator_list_neighbor + regulator_list_self
log_response = True  # take log transformation of the response genes
Y_train, Y_train_true, Y_test, Y_test_true, response_list = prepare_responses(data_type, datasets_train,
                                                                                  datasets_test, idx_train_in_general,
                                                                                  idx_test_in_general,
                                                                                  idx_train_in_dataset,
                                                                                  idx_test_in_dataset, neighbors_train,
                                                                                  neighbors_test,
                                                                                  response_type, log_response,
                                                                                  response_list_prior, top_k_response,
                                                                                  genes_list_u, l_u, r_u)
if grid_search:
    X_trains_gs = copy.deepcopy(X_trains)
    Y_train_gs = copy.copy(Y_train)
    
# transform features
transform_features(X_trains, X_tests, feature_types)
print(f"Minimum value after transformation can below 0: {np.min(X_trains['regulators_self'])}")

if data_type == 'merfish':
    num_coordinates = 3
elif data_type == 'starmap' or data_type == 'merfish_cell_line':
    num_coordinates = 2
else:
    num_coordinates = None

if np.ndim(X_trains['baseline']) > 1 and np.ndim(X_tests['baseline']) > 1:
    X_train, X_train_clf_1, X_train_clf_2 = combine_features(X_trains, preprocess, num_coordinates)
    X_test, X_test_clf_1, X_test_clf_2 = combine_features(X_tests, preprocess, num_coordinates)
elif np.ndim(X_trains['baseline']) > 1:
    X_train, X_train_clf_1, X_train_clf_2 = combine_features(X_trains, preprocess, num_coordinates)

print(f"Dimension of X train is: {X_train.shape}")
print(f"Dimension of Y train is: {Y_train.shape}")

#Construct and train MESSI model

# ------ set parameters ------
model_name_gates = 'logistic'
model_name_experts = 'mrots'
num_response = Y_train.shape[1]

# default values 
soft_weights = True
partial_fit_expert = True

# specify default parameters for MESSI
model_params = {'n_classes_0': n_classes_0,
                'n_classes_1': n_classes_1,
                'model_name_gates': model_name_gates,
                'model_name_experts': model_name_experts,
                'num_responses': Y_train.shape[1],
                'soft_weights': soft_weights,
                'partial_fit_expert': partial_fit_expert,
                'n_epochs': n_epochs,
                'tolerance': 3}
# set up directory for saving the model
sub_condition = f"{condition}_{model_name_gates}_{model_name_experts}"
sub_dir = f"{data_type}/{behavior_no_space}/{sex}/{current_cell_type_no_space}/{preprocess}/{sub_condition}"
current_dir = os.path.join(output_path, sub_dir)



print(f"Model and validation results (if appliable) saved to: {current_dir}")

suffix = f"_{test_animal}"
filename = 'hme_model_16.pickle' 
# search range for number of experts; for example usage only, we recommend 4
search_range_dict = {'Excitatory': range(7, 9), 'U-2_OS': range(1,3), \
                        'STARmap_excitatory': range(1,3)}  


#################

saved_model = pickle.load(open(os.path.join(current_dir, filename), 'rb'))
Y_hat_final = saved_model.predict(X_test, X_test_clf_1, X_test_clf_2)
print(f"Mean absolute value : {(abs(Y_test - Y_hat_final).mean(axis=1)).mean()}")

# get full list of signaling genes 
regulator_list_neighbor_c = [g.capitalize() for g in regulator_list_neighbor]
response_list_c = [g.capitalize() for g in response_list]
total_regulators_c = [g.capitalize() for g in total_regulators]

neighbor_ligands = [r + '_neighbor' for r in regulator_list_neighbor]
total_regulators_neighbor = total_regulators + neighbor_ligands
total_regulators_neighbor_c = [g.capitalize() for g in total_regulators_neighbor]

sns.set_context("paper", font_scale=1.2) 
_expert = [0,6]

if "None" in sub_condition:
    # dispersion = Y_train.var(axis=0) / abs(Y_train_raw.mean(axis=0)+1e-6)
    dispersion = Y_train.var(axis=0)
    idx_dispersion = np.argsort(-dispersion, axis=0)[:97]
else:
    idx_dispersion = range(0, len(response_list))

response_list_dispersion = np.array(response_list_c)[idx_dispersion]

# model_experts is a dictionary index by 1st layer class and 2nd layer class
_weights = saved_model.model_experts[_expert[0]][_expert[1]].W[:,idx_dispersion]

df_plot = pd.DataFrame(_weights)
df_plot.index = total_regulators_neighbor_c
df_plot.columns = response_list_dispersion
plt.figure(figsize=(16,18))
sns.heatmap(df_plot, xticklabels=True, yticklabels=True, center = 0, cmap = "RdBu_r")
plt.xticks(rotation=90)
plt.ylabel("Features")
plt.xlabel("Response variables")
plt.title(f"Coefficients of expert {_expert[1]}")

save_path = '/cluster/home/t116508uhn/64630/'
plt.savefig(save_path+'toomanycells_PCA_64embedding_pathologist_label_l1mp5_temp_plot.svg', dpi=400)
plt.clf()

plt.close()

