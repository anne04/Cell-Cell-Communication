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
'''parser.add_argument( '--embedding_data_path', type=str, default='new_alignment/Embedding_data_ccc_rgcn/' , help='The path to attention') #'/cluster/projects/schwartzgroup/fatema/pancreatic_cancer_visium/210827_A00827_0396_BHJLJTDRXY_Notta_Karen/V10M25-61_D1_PDA_64630_Pa_P_Spatial10x_new/outs/'
parser.add_argument( '--data_name', type=str, default='V10M25-61_D1_PDA_64630_Pa_P_Spatial10x_new', help='The name of dataset')
parser.add_argument( '--model_name', type=str, default='gat_r1_2attr', help='model name')
parser.add_argument( '--slice', type=int, default=0, help='starting index of ligand')
'''
args = parser.parse_args()

####
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
'''
temp = qnorm.quantile_normalize(np.transpose(sparse.csr_matrix.toarray(adata_h5.X)))  
adata_X = np.transpose(temp)  
#adata_X = sc.pp.scale(adata_X)
cell_vs_gene = adata_X #sparse.csr_matrix.toarray(adata_X)   # rows = cells, columns = genes
'''
cell_vs_gene = sparse.csr_matrix.toarray(adata_h5.X)
#########################################################################
#########################################################################
'''gene_info=dict()
for gene in gene_ids:
    gene_info[gene]=''

ligand_dict_dataset = defaultdict(list)
cell_chat_file = '/cluster/home/t116508uhn/64630/Human-2020-Jin-LR-pairs_cellchat.csv'
df = pd.read_csv(cell_chat_file)
cell_cell_contact = []
for i in range (0, df["ligand_symbol"].shape[0]):
    ligand = df["ligand_symbol"][i]
    #if ligand not in gene_marker_ids:
    #if ligand not in gene_info:
    #    continue
        
    if df["annotation"][i] == 'ECM-Receptor':    
        continue
        
    receptor_symbol_list = df["receptor_symbol"][i]
    receptor_symbol_list = receptor_symbol_list.split("&")
    for receptor in receptor_symbol_list:
        #if receptor in gene_info:
        #if receptor in gene_marker_ids:
            ligand_dict_dataset[ligand].append(receptor)
            #######
            if df["annotation"][i] == 'Cell-Cell Contact':
                cell_cell_contact.append(receptor)
            #######                
            
print(len(ligand_dict_dataset.keys()))

nichetalk_file = '/cluster/home/t116508uhn/64630/NicheNet-LR-pairs.csv'   
df = pd.read_csv(nichetalk_file)
for i in range (0, df["from"].shape[0]):
    ligand = df["from"][i]
    #if ligand not in gene_marker_ids:
    #if ligand not in gene_info:
    #    continue
    receptor = df["to"][i]
    #if receptor not in gene_marker_ids:
    #if receptor not in gene_info:
    #    continue
    ligand_dict_dataset[ligand].append(receptor)
    
print(len(ligand_dict_dataset.keys()))

l_u = []
r_u = []
for gene in list(ligand_dict_dataset.keys()): 
    ligand_dict_dataset[gene]=list(set(ligand_dict_dataset[gene]))
    #gene_info[gene] = 'included'
    for receptor_gene in ligand_dict_dataset[gene]:
        #gene_info[receptor_gene] = 'included'
        l_u.append(gene)
        r_u.append(receptor_gene)
   
l_u_p = set(l_u)
r_u_p = set(r_u)
##############


#l_u_search = set(['CBLN1', 'CXCL14', 'CBLN2', 'VGF','SCG2','CARTPT','TAC2'])
#r_u_search = set(['CRHBP', 'GABRA1', 'GPR165', 'GLRA3', 'GABRG1', 'ADORA2A'])

l_u = l_u_p#.union(l_u_search)
r_u = r_u_p#.union(r_u_search)
'''
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
##### keep selective genes ####
'''selective_genes = ['L1CAM','LAMC2','ITGA2']
non_cancer_ccc =  (l_u.union(r_u)).intersection(set(gene_ids)) - set(selective_genes)
selective_genes.append(list(non_cancer_ccc)[0])
only_spot = (set(gene_ids)-l_u) - r_u
selective_genes.append(list(only_spot)[0])
selective_genes.append(list(only_spot)[1])
index_genes = []
for gene in selective_genes:
    for i in range (0, len(gene_ids)):
        if gene_ids[i] == gene:
            index_genes.append(i)
            break
            '''
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
for i in range (0, len(gene_ids)):
    if gene_ids[i] in genes_list_us_messi:
       hp_genes_columns[gene_ids[i]] = j
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


'''input_path = 'input/'
output_path = 'output/'
data_type = 'merfish'
sex = 'Female'
behavior = 'Parenting'
behavior_no_space = behavior.replace(" ", "_")
current_cell_type = 'Excitatory'
current_cell_type_no_space = current_cell_type.replace(" ", "_")

grid_search = False #True #
n_sets = 2  # for example usage only; we recommend 5

n_classes_0 = 1
n_classes_1 = 5
n_epochs = 8  # for example usage only; we recommend using the default 20 n_epochs 

preprocess = 'neighbor_sum' #'neighbor_cat'
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
'''
# read in ligand and receptor lists
#l_u, r_u = get_lr_pairs(input_path='../messi/input/')  # may need to change to the default value

'''
lr_pairs = pd.read_html(os.path.join('input/','ligand_receptor_pairs2.txt'), header=None)[0] #pd.read_table(os.path.join(input_path, filename), header=None)
lr_pairs.columns = ['ligand','receptor']
lr_pairs[['ligand','receptor']] = lr_pairs['receptor'].str.split('\t',expand=True)
lr_pairs['ligand'] = lr_pairs['ligand'].apply(lambda x: x.upper())
lr_pairs['receptor'] = lr_pairs['receptor'].apply(lambda x: x.upper())
l_u_p = set([l.upper() for l in lr_pairs['ligand']])
r_u_p = set([g.upper() for g in lr_pairs['receptor']])
'''

# read in meta information about the dataset # meta_all = cell x metadata
'''
meta_all, meta_all_columns, cell_types_dict, genes_list, genes_list_u, \
response_list_prior, regulator_list_prior = read_meta('input/', behavior_no_space, sex, l_u, r_u)  # TO BE MODIFIED: number of responses
'''
response_list_prior = regulator_list_prior = None
barcode_type = pd.DataFrame(barcode_info)
meta_all = barcode_type.to_numpy()

'''meta_all_columns = dict()
meta_all_columns['Cell_ID'] = 0
meta_all_columns['Cell_class'] = 1
meta_all_columns['Animal_ID'] = 2
meta_all_columns['Bregma'] = 3
meta_all_columns['ID_in_dataset'] = 4

cell_types_dict = dict()
cell_types_dict['Excitatory']=0
gene_ids = genes_list 
'''

#genes_list_u = 

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
################## metadata ################################################
#hp_hp=barcode,spot type
#hp_columns=set(['Cell_ID','cell_type'])

#hp_cor = [cell_count x 2] #numpy array
#hp_cor_columns = {'Centroid_X': 0, 'Centroid_Y': 1}

#hp_genes = [cell_count x gene_count] #numpy array
#hp_genes_columns = set of gene names and their index

#################################################################
#################################################################
'''
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
'''
#############
'''for data_tr in datasets_train:
    # keep only barcode and type, nothing else
    for i in range (0,len(data_tr[0])):
        for j in [1, 2, 3, 4, 5, 6, 8]:
            data_tr[0][i][j] = 0
'''
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

