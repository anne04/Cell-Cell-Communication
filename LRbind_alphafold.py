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

def write_file(file_name, gene_name, seq):
    f = open('manualLRP_'+ligand+'_'+receptor+".fasta", "w")
    f.write(">")
    f.write(gene_name)
    f.write("\n")
    f.write(seq)
    f.write("\n")
    f.close()


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
        

    ############################
    
    lrp_list = [['CCL21', 'CXCR4'], ['HLA-C', 'CXCR4'], ['HLA-A', 'CXCR4'], 
                ['HLA-DRA', 'PTPRC'], ['PTPRC','CD74'], ['APOE', 'CXCR4'], 
                ['HSP90B1', 'CXCR4'], ['HLA-F', 'CXCR4'], ['CCL19', 'CXCR4'], 
                ['HLA-DRA', 'CD74']]

    
    lrp_list =[['CCL21', 'CCR7'], ['RPS19','C5AR1'], ['HLA-A', 'CD8A'],
               ['HLA-B','CCR7'], ['CCL21','CD8A'], ['RPS19','CCR7'],
               ['HLA-DRA', 'CD4'], ['HLA-DRA','CCR7'],['CCL21', 'RPSA'],
               ['HLA-B', 'C5AR1']]
    lrp_list =[['HLA-C', 'CD8A']]


    lrp_list_LUAD = [['COL1A1','NCL'], ['SERPING1','NCL'], ['A2M','NCL'], \
                ['COL1A1', 'CDH1'], ['HSP90B1', 'NCL'], ['SERPING1', 'ITGB2'], \
                ['COL1A1', 'SDC4'], ['SERPING1', 'ITGB1'], ['SERPING1', 'SDC4']]
    
    lrp_list_BRCA_blockA_sec1 = [['CDH1','DDR1'],['CDH1','NCL'],
                                  ['FN1', 'DDR1'],\
                                 ['FN1', 'NCL'], ['MDK', 'DDR1'], ['ITGB1', 'DDR1'], ['ITGB1', 'NCL']]

    lrp_list_PDAC = [['GPI','CD74'], ['ITGB1','ITGA3'], ['COL1A1','RPSA'], ['MDK','ITGB4'],\
                     ['MDK', 'ITGA3'], ['COL1A1', 'ITGB4'], \
                     ['GRN', 'CD74'], ['CDH1', 'ITGA3']]

    lrp_list = lrp_list_LUAD + lrp_list_BRCA_blockA_sec1 + lrp_list_PDAC
    #lrp_list = set(lrp_list)
    
    path_to = '/cluster/projects/schwartzgroup/fatema/LRbind/alphafold_input/'
    for pair in lrp_list:
        ligand = pair[0]
        receptor = pair[1]
        lig_seq = dict_gene_seq[ligand]
        rec_seq = dict_gene_seq[receptor]
        f = open(path_to + 'lrbind_'+ligand+'_'+receptor+".fasta", "w")
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
        fasta = '/cluster/projects/schwartzgroup/fatema/LRbind/alphafold_input/lrbind_'+ligand+'_'+receptor+'.fasta'
        
        print('bash run_alphafold.sh -d ${DOWNLOAD_DIR} -o output -m model_1_multimer_v3  -p multimer -i ' + fasta + ' -t 2022-01-01 -r \'none\' -c reduced_dbs')
        print('mv output/lrbind_'+ligand+'_'+receptor+'/'+'ranking_debug.json '+ 'output/lrbind_'+ ligand +'_'+ receptor +'_score.json')
        print('rm -r output/lrbind_'+ ligand+'_'+receptor)
    ##################################################################
    df = pd.read_csv(args.database_path, sep=",")
    lr_unique = defaultdict(dict)
    for i in range (0, df["Ligand"].shape[0]):
        ligand = df["Ligand"][i]
        receptor = df["Receptor"][i]
        lr_unique[ligand][receptor] = 1

    
    count = 0
    list_of_fastas =''
    for ligand in lr_unique:
        for receptor in lr_unique[ligand]:    
            if ligand not in dict_gene_seq or receptor not in dict_gene_seq:
                continue
            lig_seq = dict_gene_seq[ligand]
            rec_seq = dict_gene_seq[receptor]
            f = open('alphafold_input/manualLRP_'+ligand+'_'+receptor+".fasta", "w")
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
            count = count + 1
            list_of_fastas = list_of_fastas + 


    
    print('total count %d'%count)

################################################################
    for ligand in lr_unique:
        for receptor in lr_unique[ligand]:    
            if ligand not in dict_gene_seq or receptor not in dict_gene_seq:
                continue

            fasta = '/cluster/projects/schwartzgroup/fatema/LRbind/alphafold_input/manualLRP_'+ligand+'_'+receptor+'.fasta'

            print('bash run_alphafold.sh -d ${DOWNLOAD_DIR} -o output -m model_1_multimer_v3  -p multimer -i ' + fasta + ' -t 2022-01-01 -r \'none\' -c reduced_dbs')
            print('mv output/manualLRP_'+ligand+'_'+receptor+'/'+'ranking_debug.json '+ 'output/manualLRP_'+ ligand +'_'+ receptor +'_score.json')
            print('rm -r output/manualLRP_'+ ligand+'_'+receptor)
    #####################
    df = pd.read_csv(args.database_path, sep=",")
    manual_lrp_genes = dict()
    for i in range (0, df["Ligand"].shape[0]):
        ligand = df["Ligand"][i]
        receptor = df["Receptor"][i]
        manual_lrp_genes[ligand] = ''
        manual_lrp_genes[receptor] = ''
    
    print('num of keys %d'%len(manual_lrp_genes.keys()))
    for gene in manual_lrp_genes.keys():
        gene_seq = dict_gene_seq[gene]
        write_file(gene + ".fasta", gene, gene_seq)
    ################################################
    
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
        
        
