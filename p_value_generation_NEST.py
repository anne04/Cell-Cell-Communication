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
    args.metadata_from = args.metadata_from + args.data_name + '/'
    args.data_from = args.data_from + args.data_name + '/'
    args.embedding_path  = args.embedding_path + args.data_name + '/'
    args.output_path = args.output_path + args.data_name + '/'
    if not os.path.exists(args.output_path):
        os.makedirs(args.output_path)

##################### get metadata: input_graph ################################## 

    
    with gzip.open(args.data_from + args.data_name + '_adjacency_records', 'rb') as fp:  #b, a:[0:5]  _filtered 
        row_col, edge_weight, lig_rec, total_num_cell = pickle.load(fp)
    
    lig_rec_db = defaultdict(dict)
    for i in range (0, len(edge_weight)):
        lig_rec_db[lig_rec[i][0]][lig_rec[i][1]] =  edge_weight[i][2]   

    


    ##########################################
    if args.top_ccc_file == '':
        inFile = args.output_path + args.model_name+'_top' + str(args.top_percent) + 'percent.csv'
        df_org = pd.read_csv(inFile, sep=",")
    else: 
        inFile = args.top_ccc_file
        df_org = pd.read_csv(inFile, sep=",")

    csv_record = df_org
    # columns are: from_cell, to_cell, ligand_gene, receptor_gene, rank, component, from_id, to_id,  attention_score 
    cell_cell_lr_score = defaultdict(dict)
    for record in range (1, len(csv_record)-1):
        i = csv_record[record][6]
        j = csv_record[record][7]
        ligand_gene = csv_record[record][2]
        receptor_gene = csv_record[record][3]
        lr_pair_id = lig_rec_db[ligand_gene][receptor_gene]
        if i in cell_cell_lr_score:
            if j in cell_cell_lr_score[i]: 
                cell_cell_lr_score[i][j][lr_pair_id] = csv_record[record][8]
            else:
                cell_cell_lr_score[i][j] = dict()
                cell_cell_lr_score[i][j][lr_pair_id] = csv_record[record][8]
        else:
            cell_cell_lr_score[i][j] = dict()
            cell_cell_lr_score[i][j][lr_pair_id] = csv_record[record][8]
            
    ################## N time ##########################################
    cell_cell_lr_score_shuffled = defaultdict(dict)
    for shuffle_time in range (0, args.N):
        ## permutate edge feature vector
        edge_weight_temp = copy.deepcopy(edge_weight)
        random.shuffle(edge_weight_temp)
        # reassign the first 2 dimensions = distance and coexpression. Not changing the 3rd dimension (lr pair) because we may not get any value
        # if that is not found between these two cells during edge shuffling
        for i in range (0, len(row_col)):
            edge_weight[i][0] = edge_weight_temp[i][0]
            edge_weight[i][1] = edge_weight_temp[i][1]
        # save edge_weight as a temp
         with gzip.open(args.data_from + args.data_name + '_adjacency_records_shuffled', 'wb') as fp:  #b, a:[0:5]  _filtered 
            pickle.dump([row_col, edge_weight, lig_rec, total_num_cell],fp)
        # run the model 3 times
       
        # postprocess results

        # read it and get the values
        inFile = args.output_path + args.model_name+'_top' + str(args.top_percent) + 'percent_temporary.csv'
        df = pd.read_csv(inFile, sep=",")    
        csv_record = df
        for record in range (1, len(csv_record)-1):
            i = csv_record[record][6]
            j = csv_record[record][7]
            ligand_gene = csv_record[record][2]
            receptor_gene = csv_record[record][3]
            lr_pair_id = lig_rec_db[ligand_gene][receptor_gene]
            if i in cell_cell_lr_score_shuffled:
                if j in cell_cell_lr_score_shuffled[i]: 
                    cell_cell_lr_score_shuffled[i][j][lr_pair_id] = csv_record[record][8]
                else:
                    cell_cell_lr_score_shuffled[i][j] = deafultdict(list)
                    cell_cell_lr_score_shuffled[i][j][lr_pair_id].append(csv_record[record][8])
            else:
                cell_cell_lr_score_shuffled[i][j] = defeaultdict(list)
                cell_cell_lr_score_shuffled[i][j][lr_pair_id].append(csv_record[record][8])
           
    ######################## N times done. Now assign P values ##############################
    # for each i and j cells, for each k lr_pair, find how many times the attention score was 
    # above the original attention score recorded in cell_cell_lr_score
    for i in cell_cell_lr_score:
        for j in cell_cell_lr_score[i]:
            for lr_pair in cell_cell_lr_score[i][j]:
                original_score = cell_cell_lr_score[i][j][lr_pair]
                # how many times higher
                count_higher = 0
                for atn_score in cell_cell_lr_score_shuffled[i][j][lr_pair]:
                    if atn_score > original_score:
                        count_higher = count_higher + 1
                        
                cell_cell_lr_score[i][j][lr_pair] = count_higher/N  # p-value
        
    #########################################################################  
    csv_record = df_org
    # columns are: from_cell, to_cell, ligand_gene, receptor_gene, rank, component, from_id, to_id,  attention_score 
    cell_cell_lr_score = defaultdict(dict)
    csv_record[0].append('p-value')
    for record in range (1, len(csv_record)-1):
        i = csv_record[record][6]
        j = csv_record[record][7]
        ligand_gene = csv_record[record][2]
        receptor_gene = csv_record[record][3]
        lr_pair_id = lig_rec_db[ligand_gene][receptor_gene]
        csv_record[0].append(cell_cell_lr_score[i][j][lr_pair_id])
    
    df = pd.DataFrame(csv_record) 
    df.to_csv(args.output_path + args.model_name+'_ccc_pvalue.csv', index=False, header=False)
