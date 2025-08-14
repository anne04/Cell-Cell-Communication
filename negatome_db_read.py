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
    parser.add_argument( '--negatome_database_path', type=str, default='database/combined_stringent.txt', help='The name of DB')
    parser.add_argument( '--database_path', type=str, default='database/NEST_database.csv', help='The name of DB')
    parser.add_argument( '--database_path_omnipath', type=str, default='database/omnipath_lr_db.csv', help='The name of DB')
    parser.add_argument( '--result_path', type=str, default='result/')
    parser.add_argument( '--get_negatome_pair_only', type=int, default=1)
    args = parser.parse_args()

    df = pd.read_csv(args.file_name, sep="\t")
    dict_uniprotID_genes = defaultdict(list)
    for i in range (0, df['Sequence'].shape[0]):
        uniprotID = df['Entry'][i]
        if isinstance(df['Gene Names'][i], str):          
            gene_list = (df['Gene Names'][i]).split(' ')
            dict_uniprotID_genes[uniprotID] = gene_list      
            


    if args.get_negatome_pair_only == 1:
        df = pd.read_csv(args.negatome_database_path, sep="\t", header=None)
        lr_unique = dict()
        negatome_gene = []
        for i in range (0, df[0].shape[0]):
            ligand_uniprot = df[0][i]
            receptor_uniprot = df[1][i]
            chain_A_list = dict_uniprotID_genes[ligand_uniprot]
            chain_B_list = dict_uniprotID_genes[receptor_uniprot]
    
            for gene_a in chain_A_list:
                for gene_b in chain_B_list:
                    if gene_a == gene_b:
                        continue
                    negatome_gene.append(gene_a)
                    negatome_gene.append(gene_b)
                    lr_unique[gene_a + '_with_' + gene_b] = 0    

        negatome_gene = list(np.unique(negatome_gene))
        print('len of unique pairs: %d, unique genes %d'%(len(lr_unique.keys()), ))
        
        with gzip.open('database/negatome_gene_complex_set', 'wb') as fp:  
        	pickle.dump([negatome_gene, lr_unique], fp)

        
        exit(0)
    ##############################
    df = pd.read_csv(args.database_path_omnipath, sep=",")
    #lr_unique = defaultdict(dict)
    ligand_list = []
    receptor_list = []
    for i in range (0, df['genesymbol_intercell_source'].shape[0]):
        ligand = df['genesymbol_intercell_source'][i]
        receptor = df['genesymbol_intercell_target'][i]
        if (str(df['secreted_intercell_source'][i]) == 'True' or ) and str(df['is_inhibition'][i]) =='False':                
            #lr_unique[ligand][receptor] = 1
            ligand_list.append(ligand)
            receptor_list.append(receptor)
        

    ##################################################################
    df = pd.read_csv(args.negatome_database_path, sep="\t", header=None)
    lr_unique = defaultdict(dict)
    negatome_ligand_list = []
    negatome_receptor_list = []
    for i in range (0, df[0].shape[0]):
        ligand_uniprot = df[0][i]
        receptor_uniprot = df[1][i]
        chain_A_list = dict_uniprotID_genes[ligand_uniprot]
        chain_B_list = dict_uniprotID_genes[receptor_uniprot]

        for gene_a in chain_A_list:
            for gene_b in chain_B_list:
                if gene_a in ligand_list and gene_b in receptor_list:
                    negatome_ligand_list.append(gene_a)
                    negatome_receptor_list.append(gene_b)
                    lr_unique[gene_a][gene_b] = 1

                if gene_b in ligand_list and gene_a in receptor_list:
                    negatome_ligand_list.append(gene_b)
                    negatome_receptor_list.append(gene_a)
                    lr_unique[gene_b][gene_a] = 1
        

    ############################
    df = pd.read_csv(args.database_path, sep=",")
    ligand_list = []
    receptor_list = []
    for i in range (0, df["Ligand"].shape[0]):
        ligand = df["Ligand"][i]
        receptor = df["Receptor"][i]
        #lr_unique[ligand][receptor] = 1
        ligand_list.append(ligand)
        receptor_list.append(receptor)


    df = pd.read_csv(args.negatome_database_path, sep="\t", header=None)
    for i in range (0, df[0].shape[0]):
        ligand_uniprot = df[0][i]
        receptor_uniprot = df[1][i]
        chain_A_list = dict_uniprotID_genes[ligand_uniprot]
        chain_B_list = dict_uniprotID_genes[receptor_uniprot]

        for gene_a in chain_A_list:
            for gene_b in chain_B_list:
                if gene_a in ligand_list and gene_b in receptor_list:
                    negatome_ligand_list.append(gene_a)
                    negatome_receptor_list.append(gene_b)
                    lr_unique[gene_a][gene_b] = 1

                if gene_b in ligand_list and gene_a in receptor_list:
                    negatome_ligand_list.append(gene_b)
                    negatome_receptor_list.append(gene_a)
                    lr_unique[gene_b][gene_a] = 1
        

    
    
    
    negatome_ligand_list = list(np.unique(negatome_ligand_list))
    negatome_receptor_list = list(np.unique(negatome_receptor_list))
    print('There are %d ligand gene and %d receptor gene from negatome database'%(len(negatome_ligand_list), len(negatome_receptor_list)))

    with gzip.open('database/negatome_ligand_receptor_set', 'wb') as fp:  
    	pickle.dump([negatome_ligand_list, negatome_receptor_list, lr_unique], fp)

####
i = 0
for ligand_gene in lr_unique:
    for rec_gene in lr_unique[ligand_gene]:
        print(str(i)+' - '+ligand_gene + '_to_' + rec_gene)
        i = i+1

        
        
    
#### There are 110 ligand gene and 121 receptor gene from negatome database
#### There are 106 ligand gene and 92 receptor gene from negatome database
        
