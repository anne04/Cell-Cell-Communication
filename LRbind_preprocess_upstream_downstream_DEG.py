# Written By 
# Fatema Tuz Zohora
print('package loading')
import numpy as np
import csv
import pickle
import statistics
from scipy import sparse
from scipy import stats 
import scipy.io as sio
import scanpy as sc 
import matplotlib
matplotlib.use('Agg') 
#matplotlib.use('TkAgg') 
import matplotlib.pyplot as plt
import numpy as np
#from matplotlib.colors import LinearSegmentedColormap, to_hex, rgb2hex
#from typing import List
import qnorm
from scipy.sparse import csr_matrix
from scipy.spatial import distance
from scipy.sparse.csgraph import connected_components
from scipy.stats import median_abs_deviation
from scipy.stats import skew
from collections import defaultdict
import pandas as pd
import gzip
from kneed import KneeLocator
import copy 
import argparse
import gc
import os
import altair as alt
import altairThemes # assuming you have altairThemes.py at your current directoy or your system knows the path of this altairThemes.py.
alt.themes.register("publishTheme", altairThemes.publishTheme)
# enable the newly registered theme
alt.themes.enable("publishTheme")
import warnings
warnings.filterwarnings('ignore')
import anndata



##########################################################
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument( '--database_path', type=str, default='database/NEST_database.csv' , help='Provide your desired ligand-receptor database path here. Default database is a combination of CellChat and NicheNet database.')    
    parser.add_argument( '--data_name', type=str, default='LRbind_LUAD_1D_manualDB_geneCorrP7KNN_bidir', help='The name of dataset') #, required=True) # default='',
    #_geneCorr_remFromDB
    #LRbind_GSM6177599_NYU_BRCA0_Vis_processed_1D_manualDB_geneCorr_bidir #LGALS1, PTPRC
    #LRbind_V1_Human_Lymph_Node_spatial_1D_manualDB_geneCorr_bidir
    #LRbind_CID44971_1D_manualDB_geneCorr_bidir, CXCL10-CXCR3
    #LRbind_LUAD_1D_manualDB_geneCorr_signaling_bidir
    #'LRbind_LUAD_1D_manualDB_geneCorrKNN_bidir
    #'LRbind_V1_Breast_Cancer_Block_A_Section_1_spatial_1D_manualDB_geneCorrKNN_bidir'
    parser.add_argument( '--total_runs', type=int, default=3, help='How many runs for ensemble (at least 2 are preferred)') #, required=True) 
    #######################################################################################################
    parser.add_argument( '--embedding_path', type=str, default='embedding_data/', help='Path to grab the attention scores from')
    parser.add_argument( '--metadata_from', type=str, default='metadata/', help='Path to grab the metadata') 
    parser.add_argument( '--data_from', type=str, default='input_graph/', help='Path to grab the input graph from (to be passed to GAT)')
    parser.add_argument( '--output_path', type=str, default='/cluster/home/t116508uhn/LRbind_output/', help='Path to save the visualization results, e.g., histograms, graph etc.')
    parser.add_argument( '--target_ligand', type=str, default='TGFB1', help='') #
    parser.add_argument( '--target_receptor', type=str, default='ACVRL1', help='')
    args = parser.parse_args()

    args.metadata_from = args.metadata_from + args.data_name + '/'
    args.data_from = args.data_from + args.data_name + '/'
    args.embedding_path  = args.embedding_path + args.data_name + '/'
    args.output_path = args.output_path + args.data_name + '/'
    if not os.path.exists(args.output_path):
        os.makedirs(args.output_path)
        
    
##################### get metadata: barcode_info ###################################
    print("data: "+ args.data_name)
    with gzip.open(args.metadata_from +args.data_name+'_barcode_info', 'rb') as fp:  #b, a:[0:5]   _filtered
        barcode_info = pickle.load(fp) 

    barcode_index = dict()
    for i in range (0, len(barcode_info)):
        barcode_index[barcode_info[i][0]] = i
        
    
    with gzip.open(args.metadata_from +args.data_name+'_barcode_info_gene', 'rb') as fp:  #b, a:[0:5]   _filtered
        barcode_info_gene, ligand_list, receptor_list, gene_node_list_per_spot, dist_X, l_r_pair, gene_node_index_active, ligand_active, receptor_active = pickle.load(fp)

    gene_node_to_cell_index = dict()
    for gene_node_info in  barcode_info_gene:
        cell_barcode = gene_node_info[0]
        gene_index = gene_node_info[4]
        gene_node_to_cell_index[gene_index] = barcode_index[cell_barcode]

    
    
    with gzip.open(args.metadata_from + args.data_name +'_test_set', 'rb') as fp:  
        target_LR_index, target_cell_pair = pickle.load(fp)

    with gzip.open(args.data_to + args.data_name + '_adjacency_gene_records', 'rb') as fp:  
        row_col_gene, edge_weight, lig_rec, gene_node_type, gene_node_expression, total_num_gene_node, start_of_intra_edge = pickle.load(fp)


    for i in range(0, len(row_col_gene)):
        row_col = row_col_gene[i] 
        sender_gene = row_col[0]
        rcvr_gene = row_col[1]
        # get the identity of that sender and rcvr cells
        sender_cell_index = gene_node_to_cell_index[sender_gene]
        rcvr_cell_index = gene_node_to_cell_index[rcvr_gene]
        key_list[lig_rec[i][0]+'+'+lig_rec[i][1]].append([sender_cell_index, rcvr_cell_index])
        
    #####################################################################################
    
    with gzip.open(args.data_from + args.data_name + '_cell_vs_gene_quantile_transformed', 'rb') as fp:
        cell_vs_gene = pickle.load(fp)

    with gzip.open(args.data_from + args.data_name + '_gene_index', 'rb') as fp:
        gene_index, gene_names, cell_barcodes = pickle.load(fp)

    
    adata = anndata.AnnData(cell_vs_gene)
    adata.obs_names = cell_barcodes 
    adata.var_names = gene_names
    adata.var_names_make_unique()
    #log transform it
    sc.pp.log1p(adata)

    # Set threshold gene percentile
    threshold_gene_exp = 80
    cell_percentile = []
    for i in range (0, cell_vs_gene.shape[0]):
        y = sorted(cell_vs_gene[i]) # sort each row/cell in ascending order of gene expressions
        ## inter ##
        active_cutoff = np.percentile(y, threshold_gene_exp)
        if active_cutoff == min(cell_vs_gene[i][:]):
            times = 1
            while active_cutoff == min(cell_vs_gene[i][:]):
                new_threshold = threshold_gene_exp + 5 * times
                if new_threshold >= 100:
                    active_cutoff = max(cell_vs_gene[i][:])  
                    break
                active_cutoff = np.percentile(y, new_threshold)
                times = times + 1 

        cell_percentile.append(active_cutoff) 


    
    with gzip.open(args.metadata_from+args.data_name+'_receptor_intra_KG.pkl', 'rb') as fp:
        receptor_intraNW = pickle.load(fp)

    for receptor in receptor_intraNW:
        target_list = []
        for rows in receptor_intraNW[receptor]:
            target_list.append(rows[0])

        receptor_intraNW[receptor] = np.unique(target_list)
        '''
        if len(target_list)!=0:
            receptor_intraNW[receptor] = np.unique(target_list)
        else:
            receptor_intraNW.pop(receptor)
        ''' 
    with gzip.open(args.metadata_from+args.data_name+'_ligand_intra_KG.pkl', 'rb') as fp:
        ligand_intraNW = pickle.load(fp)

    for ligand in ligand_intraNW:
        target_list = []
        
        for rows in ligand_intraNW[ligand]:
            target_list.append(rows[0])

        ligand_intraNW[ligand] = np.unique(target_list)
            
    ############# load output graph #################################################

    for lr_pair in key_list:
        #print(lr_pair)
        ligand = lr_pair.split('+')[0]
        receptor = lr_pair.split('+')[1]

        #ligand = 'TGFB1'
        #receptor = 'ACVRL1'

        list_cell_pairs = lr_dict[ligand + '+' + receptor]
        receptor_cell_list = []
        for pair in list_cell_pairs:
            receptor_cell_list.append(pair[2])

        receptor_cell_list = np.unique(receptor_cell_list)
        if receptor not in receptor_intraNW:
            lr_dict.pop(ligand + '+' + receptor)
            continue
            
        target_list = receptor_intraNW[receptor]
        # what percent of them has the target genes expressed
    

        count = 0
        keep_receptor = dict()
        for cell in receptor_cell_list:
            found = 0
            for gene in target_list:
                if cell_vs_gene[cell][gene_index[gene]] >= cell_percentile[cell]:
                    found = found + 1
                    
                    
            if len(target_list)>0 and found/len(target_list) >= 0.5:
                count = count+1
                keep_receptor[cell] = 1

        filtered_pairs = []
        for pair in list_cell_pairs:
            if pair[2] in keep_receptor:
                filtered_pairs.append(pair)

        #if len(lr_dict[ligand + '+' + receptor]) > len(filtered_pairs):
            #print('list updated: '+ ligand + '+' + receptor)

        if len(filtered_pairs)==0:
            lr_dict.pop(ligand + '+' + receptor)
        else:
            lr_dict[ligand + '+' + receptor] = filtered_pairs
        
    print('After postprocess len %d'%len(lr_dict.keys()))

    #before post process len 112929
    #After postprocess len 40829
    save_lr_dict2 = copy.deepcopy(lr_dict)
    ############################
    lr_dict = copy.deepcopy(save_lr_dict2)   
    ############ Differentially Expressed genes filtering ################
    key_list = list(lr_dict.keys())
    pvals_lr = dict()
    for lr_pair in key_list:
        #print(lr_pair)
        ligand = lr_pair.split('+')[0]
        receptor = lr_pair.split('+')[1]
        
        list_cell_pairs = lr_dict[ligand + '+' + receptor]
        receptor_cell_list = []
        for pair in list_cell_pairs:
            receptor_cell_list.append(pair[2])

        receptor_cell_list = np.unique(receptor_cell_list)
        
        if len(receptor_cell_list) == 1 :
            lr_dict.pop(ligand + '+' + receptor)
            continue
            
        target_list = receptor_intraNW[receptor]
        if receptor not in receptor_intraNW:
            lr_dict.pop(ligand + '+' + receptor)
            continue
            
        # how well the target_list genes are differentially expressed in 
        # receptor_cell_list vs the rest
        index_receptor = []
        for cell_idx in receptor_cell_list: 
            index_receptor.append(cell_barcodes[cell_idx])

        # cells in keep_receptor have differentially-expressed target genes?
        # Let's say your selected M cells have indices stored in a list called `m_cells`
        # We'll make a new column to label your M cells
        
        adata.obs['group'] = 'other'
        adata.obs.loc[index_receptor, 'group'] = 'target'
        adata_temp = adata[:, target_list]
        sc.tl.rank_genes_groups(adata_temp, groupby='group', groups=['target'], reference='other', method='t-test') #, pts = True)
        # Get the result as a dataframe
        # Top genes ranked by p-value or log-fold change
        result = adata_temp.uns['rank_genes_groups']
        df = pd.DataFrame({
        gene: result[gene]['target'] for gene in ['names', 'pvals_adj', 'logfoldchanges']
        })
        found = 0 
        avg_pvals = 0
        for i in range (0, len(df)):
            if df['pvals_adj'][i]<0.05 and df['logfoldchanges'][i]>0:
                found = found+1
                avg_pvals = avg_pvals + df['pvals_adj'][i]
                
        
        
        if len(target_list)>0 and found/len(target_list) >= 0.10:
            avg_pvals = avg_pvals/len(target_list)
            pvals_lr[ligand + '+' + receptor] = avg_pvals
            
        else:
            lr_dict.pop(ligand + '+' + receptor)
            
        
    print('After DEG len %d'%len(lr_dict.keys()))

    #After DEG len 10082
    
#            save_lr_dict2 = copy.deepcopy(lr_dict)
    ############################
#            lr_dict = copy.deepcopy(save_lr_dict2)           
    ############################################# upstream #############################
    key_list = list(lr_dict.keys())
    for lr_pair in key_list:
        #print(lr_pair)
        ligand = lr_pair.split('+')[0]
        receptor = lr_pair.split('+')[1]

        #ligand = 'TGFB1'
        #receptor = 'ACVRL1'
        list_cell_pairs = lr_dict[ligand + '+' + receptor]
        ligand_cell_list = []
        for pair in list_cell_pairs:
            ligand_cell_list.append(pair[1])

        ligand_cell_list = np.unique(ligand_cell_list)    
        if ligand not in ligand_intraNW:
            lr_dict.pop(ligand + '+' + receptor)
            continue

        
        source_list = ligand_intraNW[ligand]
        
        count = 0
        keep_ligand = dict()
        for cell in ligand_cell_list:
            found = 0
            for gene in source_list:
                if cell_vs_gene[cell][gene_index[gene]] >= cell_percentile[cell]:
                    found = found + 1
                    
                    
            if len(source_list)>0 and found/len(source_list) >= 0.5:
                count = count+1
                keep_ligand[cell] = 1

        filtered_pairs = []
        for pair in list_cell_pairs:
            if pair[1] in keep_ligand:
                filtered_pairs.append(pair)

        #if len(lr_dict[ligand + '+' + receptor]) > len(filtered_pairs):
            #print('list updated: '+ ligand + '+' + receptor)

        if len(filtered_pairs)==0:
            lr_dict.pop(ligand + '+' + receptor)
        else:
            lr_dict[ligand + '+' + receptor] = filtered_pairs
            
        # what percent of them are expressed
        
    print('After postprocess len %d'%len(lr_dict.keys()))            

    #After postprocess len 3513
    ############ Differentially Expressed genes filtering ################
    key_list = list(lr_dict.keys())
    #pvals_lr = dict()
    for lr_pair in key_list:
        #print(lr_pair)
        ligand = lr_pair.split('+')[0]
        receptor = lr_pair.split('+')[1]
        
        list_cell_pairs = lr_dict[ligand + '+' + receptor]
        ligand_cell_list = []
        for pair in list_cell_pairs:
            ligand_cell_list.append(pair[1])

        ligand_cell_list = np.unique(ligand_cell_list)
        
        if len(ligand_cell_list) == 1 :
            lr_dict.pop(ligand + '+' + receptor)
            continue
            
        if ligand not in ligand_intraNW:
            lr_dict.pop(ligand + '+' + receptor)
            continue
            
        target_list = ligand_intraNW[ligand]
            
        # how well the target_list genes are differentially expressed in 
        # receptor_cell_list vs the rest
        index_ligand = []
        for cell_idx in ligand_cell_list: 
            index_ligand.append(cell_barcodes[cell_idx])

        # cells in keep_receptor have differentially-expressed target genes?
        # Let's say your selected M cells have indices stored in a list called `m_cells`
        # We'll make a new column to label your M cells
        adata.obs['group'] = 'other'
        adata.obs.loc[index_ligand, 'group'] = 'target'
        adata_temp = adata[:, target_list]
        sc.tl.rank_genes_groups(adata_temp, groupby='group', groups=['target'], reference='other', method='t-test') #, pts = True)
        # Get the result as a dataframe
        # Top genes ranked by p-value or log-fold change
        result = adata_temp.uns['rank_genes_groups']
        df = pd.DataFrame({
        gene: result[gene]['target'] for gene in ['names', 'pvals_adj', 'logfoldchanges']
        })
        found = 0 
        avg_pvals = 0
        for i in range (0, len(df)):
            if df['pvals_adj'][i]<0.05 and df['logfoldchanges'][i]>0:
                found = found+1
                avg_pvals = avg_pvals + df['pvals_adj'][i]
        
        if len(target_list)>0 and found/len(target_list) >= 0.10:
            avg_pvals = avg_pvals/found
            if ligand + '+' + receptor in pvals_lr:
                pvals_lr[ligand + '+' + receptor] = (pvals_lr[ligand + '+' + receptor] + avg_pvals)/2
            else:
                pvals_lr[ligand + '+' + receptor] = avg_pvals
                                
        else:
            lr_dict.pop(ligand + '+' + receptor)

    print('After DEG len %d'%len(lr_dict.keys()))

    #############################################################      
    with gzip.open(args.output_path +model_name+'_top'+str(top_N)+'_lr_dict_after_postprocess.pkl', 'wb') as fp:  
        pickle.dump([lr_dict, pvals_lr], fp)

    
