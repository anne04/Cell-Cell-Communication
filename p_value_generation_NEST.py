print('package loading')
import numpy as np
import csv
import pickle
import statistics
from scipy import sparse
import scipy.io as sio
import scanpy as sc 
import matplotlib
matplotlib.use('Agg')
#matplotlib.use('TkAgg') 
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap, to_hex, rgb2hex
#from typing import List
import qnorm
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components
from scipy.stats import median_abs_deviation
from scipy.stats import skew
from collections import defaultdict
import pandas as pd
import gzip
#from kneed import KneeLocator
import copy 
import argparse
import gc
import os


# load the NEST detected results
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument( '--data_name', type=str, help='The name of dataset', required=True) # 
    parser.add_argument( '--model_name', type=str, help='Name of the trained model', required=True)
    parser.add_argument( '--top_edge_count', type=int, default=1500 ,help='Number of the top communications to plot. To plot all insert -1') # 
    parser.add_argument( '--top_percent', type=int, default=20, help='Top N percentage communications to pick')    
    parser.add_argument( '--metadata_from', type=str, default='metadata/', help='Path to grab the metadata') 
    parser.add_argument( '--output_path', type=str, default='output/', help='Path to save the visualization results, e.g., histograms, graph etc.')
    parser.add_argument( '--barcode_info_file', type=str, default='', help='Path to load the barcode information file produced during data preprocessing step')
    parser.add_argument( '--annotation_file_path', type=str, default='', help='Path to load the annotation file in csv format (if available) ')
    parser.add_argument( '--selfloop_info_file', type=str, default='', help='Path to load the selfloop information file produced during data preprocessing step')
    parser.add_argument( '--top_ccc_file', type=str, default='', help='Path to load the selected top CCC file produced during data postprocessing step')
    parser.add_argument( '--output_name', type=str, default='', help='Output file name prefix according to user\'s choice')
    parser.add_argument( '--filter', type=int, default=0, help='Set --filter=-1 if you want to filter the CCC')
    parser.add_argument( '--filter_by_ligand_receptor', type=str, default='', help='Set ligand-receptor pair, e.g., --filter_by_ligand_receptor="CCL19-CCR7" if you want to filter the CCC by LR pair')
    parser.add_argument( '--filter_by_annotation', type=str, default='', help='Set cell or spot type, e.g., --filter_by_annotation="T-cell" if you want to filter the CCC')
    parser.add_argument( '--filter_by_component', type=int, default=-1, help='Set component id, e.g., --filter_by_component=9 if you want to filter by component id')
    parser.add_argument( '--histogram_attention_score', type=int, default=-1, help='Set --histogram_attention_score=1 if you want to sort the histograms of CCC by attention score')
    parser.add_argument( '--mad_score', type=float, default=-1, help='Set --mad_score to filter out only ccc that has deviation from median attention score = mad_score')    
    args = parser.parse_args()

    ######################### read the NEST output in csv format ####################################################
    if args.top_ccc_file == '':
        inFile = args.output_path + args.model_name+'_top' + str(args.top_percent) + 'percent.csv'
        df = pd.read_csv(inFile, sep=",")
    else: 
        inFile = args.top_ccc_file
        df = pd.read_csv(inFile, sep=",")

    csv_record = df
    # columns are: from_cell, to_cell, ligand_gene, receptor_gene, rank, component, from_id, to_id,  attention_score 
    for record in range (1, len(csv_record)-1):
        i = csv_record[record][6]
        j = csv_record[record][7]
        ligand_gene = csv_record[record][2]
        receptor_gene = csv_record[record][3]
        lr_pair_id = 


    
    ####################################################################
    # ligand - receptor database 
    print('ligand-receptor database reading.')
    df = pd.read_csv(args.database_path, sep=",")
    
    '''
            Ligand   Receptor          Annotation           Reference
    0        TGFB1     TGFBR1  Secreted Signaling      KEGG: hsa04350
    1        TGFB1     TGFBR2  Secreted Signaling      KEGG: hsa04350
    '''
    print('ligand-receptor database reading done.')
    print('Preprocess start.')
    ligand_dict_dataset = defaultdict(list)
    cell_cell_contact = dict() 
    count_pair = 0
    for i in range (0, df["Ligand"].shape[0]):
        ligand = df["Ligand"][i]
        if ligand not in gene_info: # not found in the dataset
            continue    
            
        receptor = df["Receptor"][i]
        if receptor not in gene_info: # not found in the dataset
            continue   
            
        ligand_dict_dataset[ligand].append(receptor)
        gene_info[ligand] = 'included'
        gene_info[receptor] = 'included'
        count_pair = count_pair + 1
        
        if df["Annotation"][i] == 'Cell-Cell Contact':
            cell_cell_contact[receptor] = '' # keep track of which ccc are labeled as cell-cell-contact
    
    
    print('number of ligand-receptor pairs in this dataset %d '%count_pair) 
    print('number of ligands %d '%len(ligand_dict_dataset.keys()))
    
    included_gene=[]
    for gene in gene_info.keys(): 
        if gene_info[gene] == 'included':
            included_gene.append(gene)
            
    print('Total genes in this dataset: %d, number of genes working as ligand and/or receptor: %d '%(len(gene_ids),len(included_gene)))
    
    # assign id to each entry in the ligand-receptor database
    l_r_pair = dict()
    lr_id = 0
    for gene in list(ligand_dict_dataset.keys()): 
        ligand_dict_dataset[gene]=list(set(ligand_dict_dataset[gene]))
        l_r_pair[gene] = dict()
        for receptor_gene in ligand_dict_dataset[gene]:
            l_r_pair[gene][receptor_gene] = lr_id 
            lr_id  = lr_id  + 1
        
