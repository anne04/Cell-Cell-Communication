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

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    # ================ Specify data type firstly ===============
    parser.add_argument( '--file_name', type=str, default='uniprotkb_reviewed_true_AND_proteome_up_2025_02_27.tsv', help='The name of DB')
    parser.add_argument( '--database_path', type=str, default='database/NEST_database_no_predictedPPI.csv', help='The name of DB')
    parser.add_argument( '--result_path', type=str, default='result/')
    args = parser.parse_args()

    df = pd.read_csv(args.file_name, sep="\t")
    dict_gene_seq = dict()
    for i in range (0, df['Sequence'].shape[0]):
        if not isinstance(df['Gene Names'][i], str):
            continue            
        dict_gene_seq[(df['Gene Names'][i]).split(' ')[0]] = df['Sequence'][i]
        

    
    lrp_list = [['CCL21', 'CXCR4'], ['HLA-C', 'CXCR4'], ['HLA-A', 'CXCR4'], 
                ['HLA-DRA', 'PTPRC'], ['PTPRC','CD74'], ['APOE', 'CXCR4'], 
                ['HSP90B1', 'CXCR4'], ['HLA-F', 'CXCR4'], ['CCL19', 'CXCR4'], 
                ['HLA-DRA', 'CD74']]

    
    lrp_list =[['CCL21', 'CCR7'], ['RPS19','C5AR1'], ['HLA-A', 'CD8A'],
               ['HLA-B','CCR7'], ['CCL21','CD8A'], ['RPS19','CCR7'],
               ['HLA-DRA', 'CD4'], ['HLA-DRA','CCR7'],['CCL21', 'RPSA'],
               ['HLA-B', 'C5AR1']
        
    ]
    for pair in lrp_list:
        ligand = pair[0]
        receptor = pair[1]
        lig_seq = dict_gene_seq[ligand]
        rec_seq = dict_gene_seq[receptor]
        f = open('dgi_'+ligand+'_'+receptor+".fasta", "w")
        f.write(">")
        f.write(ligand)
        f.write("\n")
        f.write(lig_seq)
        f.write("\n")
        f.write(">")
        f.write(receptor)
        f.write("\n")
        f.write(rec_seq)
        f.close()

    df = pd.read_csv(args.database_path, sep=",")
    random_selection = [randint(0, df['Ligand'].shape[0]) for p in range(0, 10)] 
    for i in random_selection:
        ligand = df['Ligand'][i]
        receptor = df['Receptor'][i]
        lig_seq = dict_gene_seq[ligand]
        rec_seq = dict_gene_seq[receptor]
        f = open('manualLRP_'+ligand+'_'+receptor+".fasta", "w")
        f.write(">")
        f.write(ligand)
        f.write("\n")
        f.write(lig_seq)
        f.write("\n")
        f.write(">")
        f.write(receptor)
        f.write("\n")
        f.write(rec_seq)
        f.close()
        
        
