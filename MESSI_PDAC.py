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

grid_search = True
n_sets = 2  # for example usage only; we recommend 5

n_classes_0 = 1
n_classes_1 = 5
n_epochs = 5  # for example usage only; we recommend using the default 20 n_epochs 

preprocess = 'neighbor_cat'
top_k_response = 20  # for example usage only; we recommend use all responses (i.e. None)
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
tlr_pairs['receptor'] = lr_pairs['receptor'].apply(lambda x: x.upper())
l_u_p = set([l.upper() for l in lr_pairs['ligand']])
r_u_p = set([g.upper() for g in lr_pairs['receptor']])

l_u_search = set(['CBLN1', 'CXCL14', 'CBLN2', 'VGF','SCG2','CARTPT','TAC2'])
r_u_search = set(['CRHBP', 'GABRA1', 'GPR165', 'GLRA3', 'GABRG1', 'ADORA2A'])

l_u = l_u_p.union(l_u_search)
r_u = r_u_p.union(r_u_search)


# read in meta information about the dataset # meta_all = cell x metadata
meta_all, meta_all_columns, cell_types_dict, genes_list, genes_list_u, \
response_list_prior, regulator_list_prior = read_meta('input/', behavior_no_space, sex, l_u, r_u)  # TO BE MODIFIED: number of responses

# get all available animals/samples -- get unique IDs
all_animals = list(set(meta_all[:, meta_all_columns['Animal_ID']])) # 16, 17, 18, 19

test_animal  = 16
test_animals = [test_animal]
samples_test = np.array(test_animals)
samples_train = np.array(list(set(all_animals)-set(test_animals)))
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
#hp_hp=barcode,spot type
#hp_columns=set(['Cell_ID','cell_type'])
#hp_cor = [cell_count x 2] #numpy array
#hp_cor_columns = {'Centroid_X': 0, 'Centroid_Y': 1}
#hp_genes = [cell_count x gene_count] #numpy array
#hp_genes_columns = set of gene names and their index


pathologist_label_file='/cluster/home/t116508uhn/64630/IX_annotation_artifacts.csv' #IX_annotation_artifacts.csv' #
pathologist_label=[]
with open(pathologist_label_file) as file:
    csv_file = csv.reader(file, delimiter=",")
    for line in csv_file:
        pathologist_label.append(line)

barcode_type=dict()
for i in range (1, len(pathologist_label)):
    if pathologist_label[i][1] == 'tumor': #'Tumour':
        barcode_type[pathologist_label[i][0]] = 1
    elif pathologist_label[i][1] =='stroma_deserted':
        barcode_type[pathologist_label[i][0]] = 0
    elif pathologist_label[i][1] =='acinar_reactive':
        barcode_type[pathologist_label[i][0]] = 2
    else:
        barcode_type[pathologist_label[i][0]] = 0
        

        
#################################################################


#################################################################
data_sets = []
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
    del hp, hp_cor, hp_genes

datasets_train = data_sets
################
data_sets = []

for animal_id, bregma in meta_per_dataset_test:
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
    del hp, hp_cor, hp_genes

datasets_test = data_sets

del data_sets

#############
for data_tr in datasets_train:
    # keep only barcode and type, nothing else
    for i in range (0,len(data_tr[0])):
        for j in [1, 2, 3, 4, 5, 6, 8]:
            data_tr[0][i][j] = 0

#############
if data_type == 'merfish_rna_seq':
    neighbors_train = None
    neighbors_test = None
else: 
    if data_type == 'merfish':
        dis_filter = 100
    else:
        dis_filter = 1e9  
        
    neighbors_train = get_neighbors_datasets(datasets_train, "Del", k=10, dis_filter=dis_filter, include_self = False)
    neighbors_test = get_neighbors_datasets(datasets_test, "Del", k=10, dis_filter=dis_filter, include_self = False)

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
                     idx_train, idx_test, idx_train_in_dataset, idx_test_in_dataset,neighbors_train, neighbors_test,
                    feature_types, regulator_list_prior, top_k_regulator, 
                     genes_list_u, l_u, r_u,cell_types_dict)



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

if not os.path.exists(current_dir):
    os.makedirs(current_dir)

print(f"Model and validation results (if appliable) saved to: {current_dir}")

suffix = f"_{test_animal}"


# search range for number of experts; for example usage only, we recommend 4
search_range_dict = {'Excitatory': range(7, 9), 'U-2_OS': range(1,3), \
                        'STARmap_excitatory': range(1,3)}  

if grid_search:
    # prepare input meta data
    if data_type == 'merfish':
        meta_per_part = [tuple(i) for i in meta_per_dataset_train]
        meta_idx = meta2idx(idx_train_in_dataset, meta_per_part)
    else:
        meta_per_part, meta_idx = combineParts(samples_train, datasets_train, idx_train_in_dataset)

    # prepare parameters list to be tuned
    if data_type == 'merfish_cell_line':
        current_cell_type_data = 'U-2_OS'
    elif data_type == 'starmap':
        current_cell_type_data = 'STARmap_excitatory'
    else:
        current_cell_type_data = current_cell_type

    params = {'n_classes_1': list(search_range_dict[current_cell_type_data]), 'soft_weights': [True, False],
              'partial_fit_expert': [True, False]}

    keys, values = zip(*params.items())
    params_list = [dict(zip(keys, v)) for v in itertools.product(*values)]

    new_params_list = []
    for d in params_list:
        if d['n_classes_1'] == 1:
            if d['soft_weights'] and d['partial_fit_expert']:
                # n_expert = 1, soft or hard are equivalent
                new_params_list.append(d)
        else:
            if d['soft_weights'] == d['partial_fit_expert']:
                new_params_list.append(d)
    ratio = 0.2

    # initialize with default values
    model_params_val = model_params.copy()
    model_params_val['n_epochs'] = 1  # increase for validation models to converge
    model_params_val['tolerance'] = 0
    print(f"Default model parameters for validation {model_params_val}")
    model = hme(**model_params_val)

    gs = gridSearch(params, model, ratio, n_sets, new_params_list)
    gs.generate_val_sets(samples_train, meta_per_part)
    gs.runCV(X_trains_gs, Y_train_gs, meta_per_part, meta_idx, feature_types, data_type,
             preprocess)
    gs.get_best_parameter()
    print(f"Best params from grid search: {gs.best_params}")

    # modify the parameter setting
    for key, value in gs.best_params.items():
        model_params[key] = value

    print(f"Model parameters for training after grid search {model_params}")

    filename = f"validation_results{suffix}.pickle"
    pickle.dump(gs, open(os.path.join(current_dir, filename), 'wb'))


if grid_search and 'n_classes_1' in params:
    model = AgglomerativeClustering(n_clusters=gs.best_params['n_classes_1'])
else:
    model = AgglomerativeClustering(n_classes_1)

model = model.fit(Y_train)
hier_labels = [model.labels_]
model_params['init_labels_1'] = hier_labels

# ------ construct MESSI  ------
model = hme(**model_params)

# train
model.train(X_train, X_train_clf_1, X_train_clf_2, Y_train)


filename = f"hme_model{suffix}.pickle"
pickle.dump(model, open(os.path.join(current_dir, filename), 'wb'))

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

