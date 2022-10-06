orimport os
#import glob
import pandas as pd
#import shutil
import numpy as np
import sys
import scikit_posthocs as post
import altair as alt
from collections import defaultdict

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
            
            
            
