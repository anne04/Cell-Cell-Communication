# Written By 
# Fatema Tuz Zohora


print('package loading')
import numpy as np
import pickle
from collections import defaultdict
import pandas as pd
import gzip
import math
import argparse
from random import randint 
import random
import glob


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    # ================ Specify data type firstly ===============
    parser.add_argument( '--file_name', type=str, default='uniprotkb_reviewed_true_AND_proteome_up_2025_02_27.tsv', help='The name of DB')
    parser.add_argument( '--database_path', type=str, default='database/combined_stringent.txt', help='The name of DB')
    parser.add_argument( '--result_path', type=str, default='result/')
    args = parser.parse_args()

    df = pd.read_csv(args.file_name, sep="\t")
    dict_uniprotID_genes = defaultdict(list)
    for i in range (0, df['Sequence'].shape[0]):
        uniprotID = df['Entry'][i]
        if isinstance(df['Gene Names'][i], str):          
            gene_list = (df['Gene Names'][i]).split(' ')
            dict_uniprotID_genes[uniprotID] = gene_list      
            

    ############################
    

    #############################  #####################################
    df = pd.read_csv(args.database_path, sep="\t", header=None)
    lr_unique = defaultdict(dict)
    ligand_list = []
    receptor_list = []
    for i in range (0, df[0].shape[0]):
        ligand_uniprot = df[0][i]
        receptor_uniprot = df[1][i]
        ligand_gene_list = dict_uniprotID_genes[ligand_uniprot]
        rec_gene_list = dict_uniprotID_genes[receptor_uniprot]
        for gene in ligand_gene_list:
            ligand_list.append(gene)

        for gene in rec_gene_list:
            receptor_list.append(gene)


    ligand_list = np.unique(ligand_list)
    receptor_list = np.unique(receptor_list)
  
    

        
