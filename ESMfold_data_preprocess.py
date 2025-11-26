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
    parser.add_argument( '--database_path', type=str, default='database/CellNEST_database_no_predictedPPI.csv', help='The name of DB')
    parser.add_argument( '--database_path_omnipath', type=str, default='database/omnipath_lr_db.csv', help='The name of DB')
    parser.add_argument( '--negatome_database_path', type=str, default='database/combined_stringent.txt', help='The name of DB')
    parser.add_argument( '--result_path', type=str, default='result/')
    args = parser.parse_args()


    df = pd.read_csv(args.file_name, sep="\t")
    dict_gene_name_vs_seq = dict()
    dict_uniprotID_genes = defaultdict(list)
    for i in range (0, df['Sequence'].shape[0]):
        uniprotID = df['Entry'][i]
        if not isinstance(df['Gene Names'][i], str):
            continue   

        gene_list = (df['Gene Names'][i]).split(' ')
        for gene in gene_list:
            dict_gene_name_vs_seq[gene] = df['Sequence'][i]
        
        dict_uniprotID_genes[uniprotID] = gene_list   

    ############################
    
    df = pd.read_csv(args.database_path, sep=",")
    gene_of_interest = dict()
    for i in range (0, df["Ligand"].shape[0]):
        ligand = df["Ligand"][i]
        ligand_list = ligand.split('&')
        receptor = df["Receptor"][i]
        for ligand in ligand_list:
            gene_of_interest[ligand] = ''
            gene_of_interest[receptor] = ''

    print('num of keys based on manular LRP %d'%len(gene_of_interest.keys()))
    ##############################
    df = pd.read_csv(args.database_path_omnipath, sep=",")
    for i in range (0, df['genesymbol_intercell_source'].shape[0]):
        ligand = df['genesymbol_intercell_source'][i]
        receptor = df['genesymbol_intercell_target'][i]
        if str(df['is_inhibition'][i]) =='False':                
            gene_of_interest[ligand] = ''
            gene_of_interest[receptor] = ''

    
    ##############################
    df = pd.read_csv(args.negatome_database_path, sep="\t", header=None)
    negatome_gene = []
    for i in range (0, df[0].shape[0]):
        ligand_uniprot = df[0][i]
        receptor_uniprot = df[1][i]
        chain_A_list = dict_uniprotID_genes[ligand_uniprot]
        chain_B_list = dict_uniprotID_genes[receptor_uniprot]

        for gene_a in chain_A_list:
            for gene_b in chain_B_list:
                #if gene_a == gene_b:
                #    continue

                gene_of_interest[gene_a] = ''
                gene_of_interest[gene_b] = ''
                negatome_gene.append(gene_a)
                negatome_gene.append(gene_b)
                    

    negatome_gene = list(np.unique(negatome_gene))
    print('len of unique negatome genes %d'%( len(negatome_gene)))


    #################################

    f = open("genes_of_interest.fasta", "w")
    count = 0
    for gene in gene_of_interest.keys():
        if gene not in dict_gene_name_vs_seq:
            continue
        gene_seq = dict_gene_name_vs_seq[gene]
        f.write(">")
        f.write(gene)
        f.write("\n")
        f.write(gene_seq)
        f.write("\n")
        count = count + 1

    f.close()
    print('file write done: total seq %d'%count)
    ################################################
    
