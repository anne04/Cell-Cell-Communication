import numpy as np
import csv
import pickle
from scipy import sparse
import scipy.io as sio
import scanpy as sc
import matplotlib
matplotlib.use('Agg')
#matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import stlearn as st
import numpy as np
from matplotlib.colors import LinearSegmentedColormap, to_hex, rgb2hex
from typing import List
import qnorm
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components
from collections import defaultdict
import pandas as pd

import argparse
parser = argparse.ArgumentParser()
parser.add_argument( '--data_path', type=str, default='/cluster/home/t116508uhn/64630/spaceranger_output_new/' , help='The path to dataset') 
parser.add_argument( '--embedding_data_path', type=str, default='new_alignment/Embedding_data_ccc_rgcn/' , help='The path to attention') #'/cluster/projects/schwartzgroup/fatema/pancreatic_cancer_visium/210827_A00827_0396_BHJLJTDRXY_Notta_Karen/V10M25-61_D1_PDA_64630_Pa_P_Spatial10x_new/outs/'
parser.add_argument( '--data_name', type=str, default='V10M25-61_D1_PDA_64630_Pa_P_Spatial10x_new', help='The name of dataset')
parser.add_argument( '--model_name', type=str, default='gat_r1_2attr', help='model name')
args = parser.parse_args()


def rgb_to_hex(rgb):
    return '%02x%02x%02x' % rgb

def get_colour_scheme(palette_name: str, num_colours: int) -> List[str]:
    """Extend a colour scheme using colour interpolation.
    Parameters
    ----------
    palette_name: The matplotlib colour scheme name that will be extended.
    num_colours: The number of colours in the output colour scheme.
    Returns
    -------
    New colour scheme containing 'num_colours' of colours. Each colour is a hex
    colour code.
    """
    scheme = [rgb2hex(c) for c in plt.get_cmap(palette_name).colors]
    if len(scheme) >= num_colours:
        return scheme[:num_colours]
    else:
        cmap = LinearSegmentedColormap.from_list("cmap", scheme)
        extended_scheme = cmap(np.linspace(0, 1, num_colours))
        return [to_hex(c, keep_alpha=False) for c in extended_scheme]
    
    
    
############
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
    
 
############
coordinates = np.load('/cluster/projects/schwartzgroup/fatema/CCST/generated_data_new/V10M25-61_D1_PDA_64630_Pa_P_Spatial10x_new/'+'coordinates.npy')
barcode_file='/cluster/home/t116508uhn/64630/spaceranger_output_new/unzipped/barcodes.tsv'

#coordinates = np.load('/cluster/projects/schwartzgroup/fatema/CCST/generated_data_noPCA_QuantileTransform_wighted_TDistance_2k/V10M25-61_D1_PDA_64630_Pa_P_Spatial10x/'+'coordinates.npy')
#barcode_file='/cluster/home/t116508uhn/64630/barcodes.tsv'
barcode_info=[]
#barcode_info.append("")
i=0
with open(barcode_file) as file:
    tsv_file = csv.reader(file, delimiter="\t")
    for line in tsv_file:
        barcode_info.append([line[0], coordinates[i,0],coordinates[i,1],0])
        i=i+1
 
####### get the gene expressions ######
data_fold = args.data_path #+args.data_name+'/'
print(data_fold)
adata_h5 = st.Read10X(path=data_fold, count_file='filtered_feature_bc_matrix.h5') #count_file=args.data_name+'_filtered_feature_bc_matrix.h5' )
print(adata_h5)
sc.pp.filter_genes(adata_h5, min_cells=1)
print(adata_h5)
gene_ids = list(adata_h5.var_names)
coordinates = adata_h5.obsm['spatial']
#################### 
temp = qnorm.quantile_normalize(np.transpose(sparse.csr_matrix.toarray(adata_h5.X)))  
adata_X = np.transpose(temp)  
adata_X = sc.pp.scale(adata_X)
cell_vs_gene = adata_X   # rows = cells, columns = genes
####################
cell_percentile = []
for i in range (0, cell_vs_gene.shape[0]):
    cell_percentile.append([np.percentile(sorted(cell_vs_gene[i]), 5), np.percentile(sorted(cell_vs_gene[i]), 50),np.percentile(sorted(cell_vs_gene[i]), 70), np.percentile(sorted(cell_vs_gene[i]), 97)])


####################
'''adata_X = sc.pp.normalize_total(adata_h5, target_sum=1, exclude_highly_expressed=True, inplace=False)['X']
adata_X = sc.pp.scale(adata_X)
#adata_X = sc.pp.pca(adata_X, n_comps=args.Dim_PCA)
features = adata_X'''
####################

gene_file='/cluster/home/t116508uhn/64630/spaceranger_output_new/unzipped/features.tsv' # 1406
gene_info=dict()
#barcode_info.append("")
i=0
with open(gene_file) as file:
    tsv_file = csv.reader(file, delimiter="\t")
    for line in tsv_file:
        gene_info[line[1]]=''

ligand_dict_dataset = defaultdict(list)
cell_chat_file = '/cluster/home/t116508uhn/64630/Human-2020-Jin-LR-pairs_cellchat.csv'
df = pd.read_csv(cell_chat_file)
cell_cell_contact = []
for i in range (0, df["ligand_symbol"].shape[0]):
    ligand = df["ligand_symbol"][i]
    if ligand not in gene_info:
        continue
        
    if df["annotation"][i] == 'ECM-Receptor':    
        continue
        
    receptor_symbol_list = df["receptor_symbol"][i]
    receptor_symbol_list = receptor_symbol_list.split("&")
    for receptor in receptor_symbol_list:
        if receptor in gene_info:
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
    if ligand not in gene_info:
        continue
    receptor = df["to"][i]
    ligand_dict_dataset[ligand].append(receptor)
    
print(len(ligand_dict_dataset.keys()))

for gene in list(ligand_dict_dataset.keys()): 
    ligand_dict_dataset[gene]=list(set(ligand_dict_dataset[gene]))
    gene_info[gene] = 'included'
    for receptor_gene in ligand_dict_dataset[gene]:
        gene_info[receptor_gene] = 'included'
   
count = 0
for gene in gene_info.keys(): 
    if gene_info[gene] == 'included':
        count = count + 1
print(count)
affected_gene_count = count
#################################################
gene_index = dict()
for gene in gene_info.keys(): 
    gene_index[gene] = i

#################################################
cell_noise = []
for i in range (0, cell_vs_gene.shape[0]):
    temp = (cell_percentile[i][1] - cell_percentile[i][0]) * np.random.random_sample(size=affected_gene_count) + cell_percentile[i][0]
    cell_noise.append(temp)
    
    
'''max_expressions = np.max(cell_vs_gene)
min_expressions = np.min(cell_vs_gene)
random_noise = np.random.random_sample(size=cell_vs_gene.shape[0]*affected_gene_count)'''


for i in range (0, cell_vs_gene.shape[0]):
    j = 0
    for gene in gene_info.keys(): 
        if gene_info[gene] == 'included': 
            cell_vs_gene[i, gene_index[gene]] = cell_noise[i][j]
            j = j+1
        
#################################################
cell_expressed = []
for i in range (0, cell_vs_gene.shape[0]):
    temp = (np.max(cell_vs_gene[i]) - cell_percentile[i][2]) * np.random.random_sample(size=affected_gene_count) + cell_percentile[i][2]
    cell_expressed.append(temp)
    
ligand_list = list(ligand_dict_dataset.keys())  
region_list = [[7000, 9000, 10000, 13000]]
activated_cell = []
for cell_index in range (0, cell_vs_gene.shape[0]):
    j = 0
    for region in region_list:
        region_x_min = region[0]
        region_x_max = region[1]
        region_y_min = region[2]
        region_y_max = region[3]
        if barcode_info[cell_index][1] > region_x_min and barcode_info[cell_index][1] < region_x_max and barcode_info[cell_index][2] > region_y_min and barcode_info[cell_index][2] < region_y_max:
            for i in range (0, 5): 
                ligand_gene = ligand_list[i]
                recp_list = ligand_dict_dataset[ligand_gene]
                cell_vs_gene[cell_index, gene_index[ligand_gene]] = cell_expressed[cell_index][j]
                j = j+1
                for receptor_gene in recp_list:                    
                    cell_vs_gene[cell_index, gene_index[receptor_gene]] = cell_expressed[cell_index][j] 
                    j = j+1
    if j>0:
        activated_cell.append(1)
    else:
        activated_cell.append(0)
#################################################


with gzip.open("/cluster/projects/schwartzgroup/fatema/find_ccc/" + 'synthetic_ccc_region_1', 'wb') as fp:
    pickle.dump([cell_vs_gene, region_list, ligand_list[0:5], activated_cell], fp)


      
      

    
