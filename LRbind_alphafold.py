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

def write_file(file_name, gene_name, seq):
    f = open('manualLRP_'+ligand+'_'+receptor+".fasta", "w")
    f.write(">")
    f.write(gene_name)
    f.write("\n")
    f.write(seq)
    f.write("\n")
    f.close()

def print_command(lrp_list, dict_gene_seq, prefix, path_to):
    for pair in lrp_list:
        ligand = pair[0]
        receptor = pair[1]
        lig_seq = dict_gene_seq[ligand]
        rec_seq = dict_gene_seq[receptor]
        f = open(path_to + prefix + ligand +'_'+receptor+".fasta", "w")
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
        fasta = '/cluster/projects/schwartzgroup/fatema/LRbind/alphafold_input/'+ prefix + ligand + '_' + receptor + '.fasta'
        
        print('bash run_alphafold.sh -d ${DOWNLOAD_DIR} -o output -m model_1_multimer_v3  -p multimer -i ' + fasta + ' -t 2022-01-01 -r \'none\' -c reduced_dbs')
        print('mv output/' + prefix + ligand+'_'+receptor+'/'+'ranking_debug.json '+ 'output/' + prefix + ligand +'_'+ receptor +'_score.json')
        print('rm -r output/' + prefix + ligand+'_'+receptor)


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
    lrp_list = [['HLA-B','CD74'], ['HLA-A','CD74'], ['CD24', 'CD74'], ['CD46', 'CD74'],\
               ['FN1','SDC1'], ['FN1', 'ITGB1'], ['PSAP','SDC4'], ['A2M', 'SDC4']]

    lrp_list = [['HLA-B','HLA-B'], ['CD74','CD74'], ['CD24', 'CD24'], ['SDC4', 'SDC4'],\
               ['FN1','FN1'], ['ITGB1', 'ITGB1'], ['PSAP','PSAP'], ['A2M', 'A2M'], \
                ['MDK', 'MDK'], ['GRN', 'GRN'] ]

    lrp_list = [['HLA-B','HLA-B'], ['CD74','CD74'], ['CD24', 'CD24'], ['SDC4', 'SDC4'],\
               ['FN1','FN1'], ['ITGB1', 'ITGB1'], ['PSAP','PSAP'], ['A2M', 'A2M'], \
                ['MDK', 'MDK'], ['GRN', 'GRN'] ]
    
    lrp_list = [['TPH1','INSR'], ['TPH2','INSR'], ['GLS', 'IL2RA'], ['BDNF', 'ADRB2'],\
               ['OXT','EGFR'], ['TNF', 'HTR2A'], ['AGT','GRM5'], ['INS', 'EGFR'], \
                ['IL2', 'GABRA1'], ['HDC', 'GRIN1'] ]

    
    path_to = '/cluster/projects/schwartzgroup/fatema/LRbind/alphafold_input/'
    prefix = 'false_' # 'lrbind_'
    print_command(lrp_list, dict_gene_seq, prefix, path_to)
    
    ##################################################################
    df = pd.read_csv(args.database_path, sep=",")
    lr_unique = defaultdict(dict)
    ligand_list = []
    receptor_list = []
    for i in range (0, df["Ligand"].shape[0]):
        ligand = df["Ligand"][i]
        receptor = df["Receptor"][i]
        lr_unique[ligand][receptor] = 1
        ligand_list.append(ligand)
        receptor_list.append(receptor)

    
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
            #list_of_fastas = list_of_fastas + 


    
    print('total count %d'%count)

    ############## make random pairs ############################
    
    ligand_list = np.unique(ligand_list)
    receptor_list = np.unique(receptor_list)
    
    probable_pairs = []
    for ligand in ligand_list:
        for receptor in receptor_list:
            if ligand not in lr_unique or receptor not in lr_unique[ligand]:
                if ligand in dict_gene_seq and receptor in dict_gene_seq:
                    probable_pairs.append([ligand, receptor])

    random.shuffle(probable_pairs)
    lrp_list = probable_pairs[0:100]
    path_to = '/cluster/projects/schwartzgroup/fatema/LRbind/alphafold_input/'
    prefix = 'random_'
    print_command(lrp_list, dict_gene_seq, prefix, path_to)


########### predicted LRbind ##########
    for i in range (0, len(lrp_list)):
        lrp_list[i] = lrp_list[i][0]+'+'+lrp_list[i][1]

    top_N = 100
    ligand_list = []
    receptor_list = [] #'LRbind_LUAD_1D_manualDB_geneCorrP7KNN_bidir', 
    data_name = ['LRbind_PDAC64630_1D_manualDB_geneCorrKNN_bidir',\
                'LRbind_V1_Breast_Cancer_Block_A_Section_1_spatial_1D_manualDB_geneCorrKNN_bidir']
    j = 0 # 'model_LRbind_LUAD_1D_manualDB_geneCorrP7KNN_bidir_3L',\
    for model_name in ['model_LRbind_PDAC64630_1D_manualDB_geneCorrKNN_bidir_3L',\
                      'model_LRbind_V1_Breast_Cancer_Block_A_Section_1_spatial_1D_manualDB_geneCorrKNN_bidir_3L']:
        df = pd.read_csv('/cluster/home/t116508uhn/LRbind_output/'+ data_name[j] + '/' +model_name+'_down_up_deg_novel_lr_list_sortedBy_totalScore_top'+str(top_N)+'_novelsOutOfallLR.csv', sep=",")
            
        for i in range (0, 60):
            if df["Ligand-Receptor Pairs"][i] in lrp_list:
                continue
                
            ligand = df["Ligand-Receptor Pairs"][i].split('+')[0]
            receptor = df["Ligand-Receptor Pairs"][i].split('+')[1]       
            ligand_list.append(ligand)
            receptor_list.append(receptor)
        j = j + 1


        
        
    probable_pairs = []
    for ligand in ligand_list:
        for receptor in receptor_list:
            
            probable_pairs.append([ligand, receptor])


    
    lrp_list = probable_pairs[0:100]
    
    path_to = '/cluster/projects/schwartzgroup/fatema/LRbind/alphafold_input/'
    prefix = 'lrbind_'
    print_command(lrp_list, dict_gene_seq, prefix, path_to)

    
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
        
        
