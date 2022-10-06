orimport os
#import glob
import pandas as pd
#import shutil
import numpy as np
import sys
import scikit_posthocs as post
import altair as alt
from collections import defaultdict
import argparse
parser = argparse.ArgumentParser()
parser.add_argument( '--data_path', type=str, default='/cluster/home/t116508uhn/64630/spaceranger_output_new/' , help='The path to dataset') #'/cluster/projects/schwartzgroup/fatema/pancreatic_cancer_visium/210827_A00827_0396_BHJLJTDRXY_Notta_Karen/V10M25-61_D1_PDA_64630_Pa_P_Spatial10x_new/outs/'
parser.add_argument( '--data_name', type=str, default='V10M25-61_D1_PDA_64630_Pa_P_Spatial10x_new', help='The name of dataset')
parser.add_argument( '--generated_data_path', type=str, default='generated_data/', help='The folder to store the generated data')
args = parser.parse_args()



def read_h5(f, i=0):
    print("hello world! read_h5")
    for k in f.keys():
        if isinstance(f[k], Group):
            print('Group', f[k])
            print('-'*(10-5*i))
            read_h5(f[k], i=i+1)
            print('-'*(10-5*i))
        elif isinstance(f[k], Dataset):
            print('Dataset', f[k])
            print(f[k][()])
        else:
            print('Name', f[k].name)
    print("read_h5_done")


gene_file='/cluster/home/t116508uhn/64630/spaceranger_output_new/unzipped/features.tsv' # 1406
gene_info=dict()
#barcode_info.append("")
i=0
with open(gene_file) as file:
    tsv_file = csv.reader(file, delimiter="\t")
    for line in tsv_file:
        gene_info[line[1]]=''

ligand_dict_dataset = defaultdict(list)


ligand_dict_db = defaultdict(list)
cell_chat_file = '/cluster/home/t116508uhn/64630/Human-2020-Jin-LR-pairs_cellchat.csv'

'''df = pd.read_csv(cell_chat_file)
for i in range (0, df["ligand_symbol"].shape[0]):
    ligand = df["ligand_symbol"][i]
    receptor_symbol_list = df["receptor_symbol"][i]
    receptor_symbol_list = receptor_symbol_list.split("&")
    for receptor in receptor_symbol_list:
        ligand_dict_db[ligand].append(receptor)'''

df = pd.read_csv(cell_chat_file)
for i in range (0, df["ligand_symbol"].shape[0]):
    ligand = df["ligand_symbol"][i]
    if ligand not in gene_info:
        continue
    receptor_symbol_list = df["receptor_symbol"][i]
    receptor_symbol_list = receptor_symbol_list.split("&")
    for receptor in receptor_symbol_list:
        if receptor in gene_info:
            ligand_dict_dataset[ligand].append(receptor)
            
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

##################################################################

data_fold = args.data_path #+args.data_name+'/'
print(data_fold)

adata_h5 = st.Read10X(path=data_fold, count_file='filtered_feature_bc_matrix.h5') #count_file=args.data_name+'_filtered_feature_bc_matrix.h5' )
#print(adata_h5)
sc.pp.filter_genes(adata_h5, min_cells=1)
#print(adata_h5)
####################   
gene_ids = list(adata_h5.var_names)
temp = qnorm.quantile_normalize(np.transpose(scipy.sparse.csr_matrix.toarray(adata_h5.X)))  #quantile_transform(scipy.sparse.csr_matrix.toarray(adata_h5.X), copy=True)
adata_X = np.transpose(temp)    
cell_vs_gene = adata_X   # rows = cells, columns = genes
#####################

gene_list = defaultdict(list)
for cell_index in range (0, cell_vs_gene.shape[0]):
    gene_list = cell_vs_gene[cell_index]
    for gene_i in range (0, len(gene_list)):
        gene_list[gene_ids[gene_i]].append(gene)
        
        
gene_list_percentile = defaultdict(list)
for gene in gene_ids:
    gene_list_percentile[gene].append(np.percentile(gene_list[gene], 50))
    gene_list_percentile[gene].append(np.percentile(gene_list[gene], 70))   


    





    
            
            
