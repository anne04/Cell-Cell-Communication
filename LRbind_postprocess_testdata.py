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


data_names = ['LRbind_V1_Human_Lymph_Node_spatial_1D_manualDB_geneCorrKNN_bidir',
               'LRbind_V1_Human_Lymph_Node_spatial_1D_manualDB_geneCorrKNN_bidir_prefiltered',
               'LRbind_V1_Human_Lymph_Node_spatial_1D_manualDB_geneLocalCorrKNN_bidir',
               'LRbind_V1_Human_Lymph_Node_spatial_1D_manualDB_geneLocalCorrKNN_bidir_prefiltered',
               'LRbind_V1_Human_Lymph_Node_spatial_1D_manualDB_geneLocalCorrKNN_bidir_prefiltered',
               'LRbind_V1_Human_Lymph_Node_spatial_1D_manualDB_geneLocalCorrKNN_bidir_prefiltered_negatome',
               
               'LRbind_LUAD_1D_manualDB_geneCorrP7KNN_bidir',
               'LRbind_LUAD_1D_manualDB_geneCorrP7KNN_bidir',
               'LRbind_LUAD_1D_manualDB_geneCorrKNN_bidir_prefiltered',
               'LRbind_LUAD_1D_manualDB_geneLocalCorrKNN_bidir',
               'LRbind_LUAD_1D_manualDB_geneLocalCorrKNN_bidir_prefiltered',
               'LRbind_LUAD_1D_manualDB_geneLocalCorrKNN_bidir_prefiltered',
              
               'LRbind_LUAD_1D_manualDB_geneLocalCorrKNN_bidir_prefiltered',              
               'LRbind_LUAD_1D_manualDB_geneLocalCorrKNN_bidir_negatome',
               'LRbind_LUAD_1D_manualDB_geneLocalCorrKNN_bidir_Allnegatome',

               'LRbind_PDAC64630_1D_manualDB_geneCorrKNN_bidir',
               #'LRbind_PDAC64630_1D_manualDB_geneCorrKNN_bidir_prefiltered',
               'LRbind_PDAC64630_1D_manualDB_geneLocalCorrKNN_bidir',
               'LRbind_PDAC64630_1D_manualDB_geneLocalCorrKNN_bidir',
               #'LRbind_PDAC64630_1D_manualDB_geneLocalCorrKNN_bidir_prefiltered',
               
               'LRbind_V1_Breast_Cancer_Block_A_Section_1_spatial_1D_manualDB_geneCorrKNN_bidir',
               'LRbind_V1_Breast_Cancer_Block_A_Section_1_spatial_1D_manualDB_geneCorrKNN_bidir_prefiltered',
               'LRbind_V1_Breast_Cancer_Block_A_Section_1_spatial_1D_manualDB_geneLocalCorrKNN_bidir',
               'LRbind_V1_Breast_Cancer_Block_A_Section_1_spatial_1D_manualDB_geneLocalCorrKNN_bidir_prefiltered',

               'LRbind_Xenium_FFPE_Human_Breast_Cancer_Rep1_manualDB_geneLocalCorrKNN_bidir_full',
               'LRbind_Xenium_FFPE_Human_Breast_Cancer_Rep1_manualDB_geneLocalCorrKNN_bidir_removedLR'
                 ]

model_names = ['model_LRbind_V1_Human_Lymph_Node_spatial_1D_manualDB_geneCorrKNN_bidir_3L',
               'model_LRbind_V1_Human_Lymph_Node_spatial_1D_manualDB_geneCorrKNN_bidir_3L_prefiltered',
               'model_LRbind_V1_Human_Lymph_Node_spatial_1D_manualDB_geneLocalCorrKNN_bidir_3L',
               'model_LRbind_V1_Human_Lymph_Node_spatial_1D_manualDB_geneLocalCorrKNN_bidir_3L_prefiltered',
               'model_LRbind_V1_Human_Lymph_Node_spatial_1D_manualDB_geneLocalCorrKNN_bidir_3L_prefiltered_tanh',
               'model_LRbind_V1_Human_Lymph_Node_spatial_1D_manualDB_geneLocalCorrKNN_bidir_3L_prefiltered_negatome',
               
               'model_LRbind_LUAD_1D_manualDB_geneCorrP7KNN_bidir_3L',
               'model_LRbind_LUAD_1D_manualDB_geneCorrP7KNN_bidir_3L_tanh',
               'model_LRbind_LUAD_1D_manualDB_geneCorrKNN_bidir_3L_prefiltered',
               'model_LRbind_LUAD_1D_manualDB_geneLocalCorrKNN_bidir_3L',
               'model_LRbind_LUAD_1D_manualDB_geneLocalCorrKNN_bidir_3L_prefiltered',
               'model_LRbind_LUAD_1D_manualDB_geneLocalCorrKNN_bidir_3L_prefiltered_tanh',
               
               'model_LRbind_LUAD_1D_manualDB_geneLocalCorrKNN_bidir_3L_prefiltered_negatome',
               'model_LRbind_LUAD_1D_manualDB_geneLocalCorrKNN_bidir_3L_negatome',
               'model_LRbind_LUAD_1D_manualDB_geneLocalCorrKNN_bidir_3L_Allnegatome',
               
               'model_LRbind_PDAC64630_1D_manualDB_geneCorrKNN_bidir_3L',
               #'model_LRbind_PDAC64630_1D_manualDB_geneCorrKNN_bidir_3L_prefiltered',
               'model_LRbind_PDAC64630_1D_manualDB_geneLocalCorrKNN_bidir_3L',
               'model_LRbind_PDAC64630_1D_manualDB_geneLocalCorrKNN_bidir',
               #'model_LRbind_PDAC64630_1D_manualDB_geneLocalCorrKNN_bidir_3L_prefiltered',
               
               'model_LRbind_V1_Breast_Cancer_Block_A_Section_1_spatial_1D_manualDB_geneCorrKNN_bidir_3L',
               'model_LRbind_V1_Breast_Cancer_Block_A_Section_1_spatial_1D_manualDB_geneCorrKNN_bidir_3L_prefiltered',
               'model_LRbind_V1_Breast_Cancer_Block_A_Section_1_spatial_1D_manualDB_geneLocalCorrKNN_bidir_3L',
               'model_LRbind_V1_Breast_Cancer_Block_A_Section_1_spatial_1D_manualDB_geneLocalCorrKNN_bidir_3L_prefiltered',
               
               'model_LRbind_Xenium_FFPE_Human_Breast_Cancer_Rep1_manualDB_geneLocalCorrKNN_bidir_full',
               'model_LRbind_Xenium_FFPE_Human_Breast_Cancer_Rep1_manualDB_geneLocalCorrKNN_bidir_removedLR'                
          ]
target_ligands = ['CCL19', 'CCL19', 'CCL19', 'CCL19', 'CCL19', 'CCL19',  
                  'TGFB1','TGFB1','TGFB1','TGFB1', 'TGFB1','TGFB1',
                  'TGFB1', 'TGFB1','TGFB1',
                  'TGFB1','TGFB1', 'TGFB1',#'TGFB1',
                 'TGFB1','TGFB1','TGFB1','TGFB1',
                 'TGFB1',
                 'CXCL12'
                 ]
target_receptors = ['CCR7', 'CCR7', 'CCR7', 'CCR7', 'CCR7', 'CCR7',  
                    'ACVRL1','ACVRL1','ACVRL1','ACVRL1', 'ACVRL1','ACVRL1',
                    'ACVRL1', 'ACVRL1','ACVRL1',
                   'ACVRL1','ACVRL1', 'ACVRL1',#'ACVRL1',
                   'ACVRL1','ACVRL1','ACVRL1','ACVRL1',
                   'ACVRL1',
                   'CXCR4'
                   ]

if __name__ == "__main__":
    elbow_cut_flag = 0 #1 #0 #histogram
    knee_flag = 0 #1 #0 # pairwise
    file_name_suffix = "100" #'_elbow_' #'100_woHistElbowCut' # '_elbow' #'100' 
    ##########################################################
    # 4, 13
    for data_index in [23]: #range(0, len(data_names)):
        parser = argparse.ArgumentParser()
        parser.add_argument( '--database_path', type=str, default='database/NEST_database.csv' , help='Provide your desired ligand-receptor database path here. Default database is a combination of CellChat and NicheNet database.')    
        parser.add_argument( '--data_name', type=str, default='', help='The name of dataset') #, required=True) # default='',

        parser.add_argument( '--total_runs', type=int, default=3, help='How many runs for ensemble (at least 2 are preferred)') #, required=True) 
        #######################################################################################################
        parser.add_argument( '--embedding_path', type=str, default='embedding_data/', help='Path to grab the attention scores from')
        parser.add_argument( '--metadata_from', type=str, default='metadata/', help='Path to grab the metadata') 
        parser.add_argument( '--data_from', type=str, default='input_graph/', help='Path to grab the input graph from (to be passed to GAT)')
        parser.add_argument( '--output_path', type=str, default='/cluster/home/t116508uhn/LRbind_output/', help='Path to save the visualization results, e.g., histograms, graph etc.') #
        parser.add_argument( '--target_ligand', type=str, default='CCL19', help='') #
        parser.add_argument( '--target_receptor', type=str, default='CCR7', help='')
        parser.add_argument( '--use_attn', type=int, default=1, help='')
        parser.add_argument( '--use_embFusion', type=int, default=1, help='')
        parser.add_argument( '--prediction_threshold', type=float, default=0.7, help='')
        args = parser.parse_args()
        ##############
        if elbow_cut_flag==0:
            args.output_path = '/cluster/home/t116508uhn/LRbind_output/without_elbow_cut/'
        args.data_name = data_names[data_index]
        args.target_ligand = target_ligands[data_index]
        args.target_receptor = target_receptors[data_index] 
        ##############
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
    
        '''
        Tcell_zone = []
        node_type = dict()
        df = pd.read_csv("../NEST/data/V1_Human_Lymph_Node_spatial_annotation.csv", sep=",")
        for i in range (0, df["Barcode"].shape[0]):
            if df["Type"][i] == 'T-cell':
                Tcell_zone.append(barcode_index[df["Barcode"][i]])
                
            node_type[df["Barcode"][i]] = df["Type"][i]
        '''
       
            
        
        with gzip.open(args.metadata_from +args.data_name+'_barcode_info_gene', 'rb') as fp:  #b, a:[0:5]   _filtered
            barcode_info_gene, ligand_list, receptor_list, gene_node_list_per_spot, dist_X, l_r_pair, gene_node_index_active, ligand_active, receptor_active = pickle.load(fp)

        print('total gene node %d'%len(list(gene_node_index_active.keys())))
      
        with gzip.open(args.metadata_from + args.data_name +'_test_set', 'rb') as fp:  
            target_LR_index, target_cell_pair = pickle.load(fp)


        ######################### LR database ###############################################
        df = pd.read_csv(args.database_path, sep=",")
        db_gene_nodes = dict()
        for i in range (0, df["Ligand"].shape[0]):
            ligand = df["Ligand"][i]
            receptor = df["Receptor"][i]
            db_gene_nodes[ligand] = '1'
            db_gene_nodes[receptor] = '1'


        #####################################################################################
        with gzip.open('database/negatome_gene_complex_set', 'rb') as fp:  
            negatome_gene, negatome_lr_unique = pickle.load(fp)
          
        count = 0
        negatome_unique_pair = dict()
        for i in range (0, len(barcode_info)):
            ligand_node_index_intra = []
            for gene in gene_node_list_per_spot[i]:
                if gene in ligand_list and gene in negatome_gene:
                    ligand_node_index_intra.append([gene_node_list_per_spot[i][gene], gene])
            
            receptor_node_index_intra = []
           
            for gene in gene_node_list_per_spot[i]:
                if gene in receptor_list and gene in negatome_gene:
                    receptor_node_index_intra.append([gene_node_list_per_spot[i][gene], gene])

            
            for i_gene in ligand_node_index_intra:  
                for j_gene in receptor_node_index_intra:
                    if i_gene[1]==j_gene[1]:
                        continue
                  
                    if i_gene[1]+'_with_'+j_gene[1] in negatome_lr_unique:
                        #temp = distance.euclidean(X_embedding[i_gene[0]], X_embedding[j_gene[0]]) # 
                        #dot_prod_list_negatome_intra.append([temp, i, i, i_gene[1], j_gene[1], i_gene[0], j_gene[0]])
                        count = count+1
                        negatome_unique_pair[i_gene[1]+'_with_'+j_gene[1]] = 1
                                            

        print('unique pairs found %d, and count %d'%(len(list(negatome_unique_pair.keys())),count))
        #####################
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
    
    
                        
    
        print('****' + args.data_name + '*********')
        print(model_names[data_index])
        print(args.target_ligand + '-' + args.target_receptor)
        
        for model_name in [model_names[data_index]]:
            args.model_name = model_name
            args.model_name = args.model_name + '_r1'
            X_embedding_filename =  args.embedding_path + args.model_name + '_Embed_X' 
            print("\n\n"+ X_embedding_filename)
            with gzip.open(X_embedding_filename, 'rb') as fp:  
                X_embedding = pickle.load(fp)
    
            
            for i in range (0, X_embedding.shape[0]):
                total_score_per_row = np.sum(X_embedding[i][:])
                X_embedding[i] = X_embedding[i]/total_score_per_row



            ############ attention scores ##############################
            
            layer = 3
          
            distribution = []
            X_attention_filename = args.embedding_path +  args.model_name + '_attention' #.npy
            print(X_attention_filename)
            fp = gzip.open(X_attention_filename, 'rb')  
            X_attention_bundle = pickle.load(fp) # 0 = index, 1 - 3 = layer 1 - 3 
            
            for index in range (0, X_attention_bundle[0].shape[1]):
                i = X_attention_bundle[0][0][index]
                j = X_attention_bundle[0][1][index]
                distribution.append(X_attention_bundle[layer][index][0])
                


            min_value = min(distribution)
            max_value = max(distribution)
            distribution = [] 
            for index in range (0, X_attention_bundle[0].shape[1]):
              i = X_attention_bundle[0][0][index]
              j = X_attention_bundle[0][1][index]
              scaled_score = (X_attention_bundle[layer][index][0]-min_value)/(max_value-min_value) # scaled from 0 to 1
              distribution.append(scaled_score)
             
            percentage_value = 80
            th_80th = np.percentile(sorted(distribution), percentage_value) # higher attention score means stronger connection
            # Now keep only 
            attention_scores = defaultdict(dict)  
            for index in range (0, X_attention_bundle[0].shape[1]):
              i = X_attention_bundle[0][0][index]
              j = X_attention_bundle[0][1][index]
              scaled_score = (X_attention_bundle[layer][index][0]-min_value)/(max_value-min_value)
              if scaled_score >= 0.7: #th_80th:
                  attention_scores[i][j] = scaled_score

            
            ########################################################################
            '''
            In [9]: sort_lr_list[282]
            Out[9]: ['TGFB1+RPSA', 87.83863203846781, 0.04901709377146641, 1792]
            
            In [10]: sort_lr_list[369]
            Out[10]: ['TGFB1+TGFBR1', 60.0198603777533, 0.3390952563714876, 177]
            '''
            #########################################################################

            
    
            ########## all ############################################# 
    #        top_lrp_count = 1000
            
            break_flag = 0
            test_mode = 1
            all_ccc_pairs = defaultdict(list)
            all_negatome_pairs = defaultdict(list)
            # all_negatome_pairs_intra = defaultdict(list)
          
            for top_N in [100]: #, 30, 10]:
                print(top_N)
                if break_flag == 1:  
                    break
                if knee_flag == 1:
                    top_N = 0
                    break_flag = 1
                lr_dict = defaultdict(list)
                Tcell_zone_lr_dict = defaultdict(list)
                target_ligand = args.target_ligand
                target_receptor = args.target_receptor
                found_list = defaultdict(list)
                
                for i in dist_X: 
                    # from i to j 
                    ligand_node_index = []
                    for gene in gene_node_list_per_spot[i]:
                        if gene in ligand_list:
                            if gene in db_gene_nodes:
                                ligand_node_index.append([gene_node_list_per_spot[i][gene], gene])

                    #dot_prod_list = []
                    #product_only = []

                    for j in dist_X[i]:
                        if i==j :
                            continue

                        receptor_node_index = []
                        for gene in gene_node_list_per_spot[j]:
                            if gene in receptor_list and gene in db_gene_nodes: 
                                # it must present in LR db to be considered as "inter"
                                receptor_node_index.append([gene_node_list_per_spot[j][gene], gene])

                      
                        # from i to j == total attention score
                        if args.use_attn == 1:
                            total_attention_score = 0 
                            total_connection = 0
                            for i_gene in ligand_node_index:  
                                for j_gene in receptor_node_index:
                                    #if i_gene[1]+'_with_'+j_gene[1] in negatome_lr_unique:
                                    #    continue
                                  
                                    if i_gene[0] in attention_scores and j_gene[0] in attention_scores[i_gene[0]]:
                                        total_attention_score = total_attention_score + attention_scores[i_gene[0]][j_gene[0]]
                                        total_connection = total_connection + 1

                            if total_connection != 0:
                                total_attention_score = total_attention_score/total_connection

                        #if args.use_attn == 1:
                        #    if total_attention_score == 0:
                        #        # means it is below threshold
                        #        continue
                            
                        dot_prod_list = []
                        dot_prod_list_negatome_inter = []
                   
                        product_only = []
                        #product_only_layer1 = []
                        start_index = 0
                        for i_gene in ligand_node_index:  
                            for j_gene in receptor_node_index:
                                if i_gene[1]==j_gene[1]:
                                    continue
                              
                                temp = distance.euclidean(X_embedding[i_gene[0]], X_embedding[j_gene[0]]) # 

                                #if i_gene[1]+'_with_'+j_gene[1] in negatome_lr_unique: 
                                #    dot_prod_list_negatome_inter.append([temp, i, j, i_gene[1], j_gene[1], i_gene[0], j_gene[0]])
                                    #continue
                                  
                                # distance.euclidean(X_embedding[i_gene[0]], X_embedding[j_gene[0]]) 
                                # (X_embedding[i_gene[0]], X_embedding[j_gene[0]])
                                if args.use_attn == 1:
                                    dot_prod_list.append([temp, i, j, i_gene[1], j_gene[1], i_gene[0], j_gene[0], total_attention_score])    
                                else:
                                    dot_prod_list.append([temp, i, j, i_gene[1], j_gene[1], i_gene[0], j_gene[0]]) #, total_attention_score]) #, temp_layer1])
                                product_only.append(temp)
                                #product_only_layer1.append(temp_layer1)
        
                        ###############################################
                                
                      
                      
                        if len(dot_prod_list) == 0:
                            continue
                            
                        # flip so that high score means high probability
                        if len(dot_prod_list) > 1:
                            max_value = max(product_only)
                            min_value = min(product_only)
                            
                            # max_score_layer1 = max(product_only_layer1)
                            for item_idx in range (0, len(dot_prod_list)):
                                #scaled_prod = (dot_prod_list[item_idx][0]-min_value)/(max_value-min_value) # scaled from 0 to 1
                                #scaled_prod = 1 - scaled_prod # flipped
                                scaled_prod = max_value - dot_prod_list[item_idx][0]
                                
                                dot_prod_list[item_idx][0] = scaled_prod
                                #scaled_prod = max_score_layer1 - dot_prod_list[item_idx][5]
                                #dot_prod_list[item_idx][5] = scaled_prod 
    
                              
                            dot_prod_list = sorted(dot_prod_list, key = lambda x: x[0], reverse=True) # high to low
                        if knee_flag == 0:                       
                            dot_prod_list = dot_prod_list[0:top_N]
                        else:
                            ########## knee find ###########
                            x = []
                            score_list = []
                            for score_index in range (0, len(dot_prod_list)):
                                score_list.append(dot_prod_list[score_index][0])
                                x.append(score_index)

                            if len(dot_prod_list)>1:                                
                                kn = KneeLocator(x, score_list, direction='decreasing', curve="convex")
                                dot_prod_list = dot_prod_list[0:kn.knee]
                        ###########################
                        for item in dot_prod_list_negatome_inter:
                            all_negatome_pairs['from_cell'].append(barcode_info[item[1]][0])
                            all_negatome_pairs['to_cell'].append(barcode_info[item[2]][0])
                            all_negatome_pairs['from_gene_node'].append(item[5])
                            all_negatome_pairs['to_gene_node'].append(item[6])
                            all_negatome_pairs['ligand_gene'].append(item[3])
                            all_negatome_pairs['rec_gene'].append(item[4])
                            all_negatome_pairs['score'].append(item[0])


                      
                        for item in dot_prod_list:
                            all_ccc_pairs['from_cell'].append(barcode_info[item[1]][0])
                            all_ccc_pairs['to_cell'].append(barcode_info[item[2]][0])
                            all_ccc_pairs['from_gene_node'].append(item[5])
                            all_ccc_pairs['to_gene_node'].append(item[6])
                            all_ccc_pairs['ligand_gene'].append(item[3])
                            all_ccc_pairs['rec_gene'].append(item[4])
                            all_ccc_pairs['score'].append(item[0])
                            all_ccc_pairs['from_cell_index'].append(item[1])
                            all_ccc_pairs['to_cell_index'].append(item[2])
                            all_ccc_pairs['attention_score'].append(item[7])
                          
                            lr_dict[item[3]+'+'+item[4]].append([item[0], item[1], item[2]])                          
                            #if i in Tcell_zone and j in Tcell_zone:
                            #    Tcell_zone_lr_dict[item[3]+'+'+item[4]].append([item[0], item[1], item[2]])
                                
                            if test_mode == 1 and item[3] == target_ligand and item[4] == target_receptor:
                                found_list[i].append(item[0]) #= 1
                                found_list[j].append(item[0])
                                #break
                                                
                    '''          
                    for item in dot_prod_list_negatome_intra:
                        all_negatome_pairs_intra['from_cell'].append(barcode_info[item[1]][0])
                        all_negatome_pairs_intra['to_cell'].append(barcode_info[item[2]][0])
                        all_negatome_pairs_intra['from_gene_node'].append(item[5])
                        all_negatome_pairs_intra['to_gene_node'].append(item[6])
                        all_negatome_pairs_intra['ligand_gene'].append(item[3])
                        all_negatome_pairs_intra['rec_gene'].append(item[4])
                        all_negatome_pairs_intra['score'].append(item[0])
                        all_negatome_pairs_intra['from_cell_index'].append(item[1])
                        all_negatome_pairs_intra['to_cell_index'].append(item[2])
                    '''    
                        ####################################################
                
                # plot found_list
                print("positive: %d, total pairs %d"%(len(found_list), len(lr_dict.keys())))
                
                data_list_pd = pd.DataFrame({
                    'from_cell': all_ccc_pairs['from_cell'],
                    'to_cell': all_ccc_pairs['to_cell'],
                    'from_gene_node': all_ccc_pairs['from_gene_node'],
                    'to_gene_node': all_ccc_pairs['to_gene_node'],
                    'ligand_gene': all_ccc_pairs['ligand_gene'],
                    'rec_gene': all_ccc_pairs['rec_gene'],
                    'score': all_ccc_pairs['score'],
                    'from_cell_index': all_ccc_pairs['from_cell_index'], 
                    'to_cell_index': all_ccc_pairs['to_cell_index'], 
                    'attention_score': all_ccc_pairs['attention_score']
            
                })
                data_list_pd.to_csv(args.output_path +model_name+'_allLR_nodeInfo.csv.gz', index=False, compression='gzip') #_negatome
                print(len(data_list_pd))
                data_list_pd = pd.DataFrame({
                    'from_cell': all_negatome_pairs['from_cell'],
                    'to_cell': all_negatome_pairs['to_cell'],
                    'from_gene_node': all_negatome_pairs['from_gene_node'],
                    'to_gene_node': all_negatome_pairs['to_gene_node'],
                    'ligand_gene': all_negatome_pairs['ligand_gene'],
                    'rec_gene': all_negatome_pairs['rec_gene'],
                    'score': all_negatome_pairs['score'],
                    'from_cell_index': all_negatome_pairs['from_cell_index'],
                    'to_cell_index': all_negatome_pairs['to_cell_index']
                })
                data_list_pd.to_csv(args.output_path +model_name+'_negatomeLR_nodeInfo_inter.csv.gz', index=False, compression='gzip') #_negatome
                """
                print(len(data_list_pd))
                # you can double check if all sender and rcvr cells are the same one
                data_list_pd = pd.DataFrame({
                    'from_cell': all_negatome_pairs_intra['from_cell'],
                    'to_cell': all_negatome_pairs_intra['to_cell'],
                    'from_gene_node': all_negatome_pairs_intra['from_gene_node'],
                    'to_gene_node': all_negatome_pairs_intra['to_gene_node'],
                    'ligand_gene': all_negatome_pairs_intra['ligand_gene'],
                    'rec_gene': all_negatome_pairs_intra['rec_gene'],
                    'score': all_negatome_pairs_intra['score'],
                    'from_cell_index': all_negatome_pairs_intra['from_cell_index'],
                    'to_cell_index': all_negatome_pairs_intra['to_cell_index']

                    
                })
                data_list_pd.to_csv(args.output_path +model_name+'_negatomeLR_nodeInfo_intra.csv.gz', index=False, compression='gzip') #_negatome
                print(len(data_list_pd))
                negatome_unique_pair = dict()
                for idx in range (0, len(all_negatome_pairs['ligand_gene'])):
                    negatome_unique_pair[all_negatome_pairs['ligand_gene'][idx] + '_with_' + all_negatome_pairs['rec_gene'][idx]] = 1
                
                for idx in range (0, len(all_negatome_pairs_intra['ligand_gene'])):
                    negatome_unique_pair[all_negatome_pairs_intra['ligand_gene'][idx] + '_with_' + all_negatome_pairs_intra['rec_gene'][idx]] = 1

                print('unique pairs found %d'%len(list(negatome_unique_pair.keys())))

                """
                
                ccc_pairs = pd.read_csv(args.output_path +model_name+'_allLR_nodeInfo_LUAD_LYMPH_top20.csv.gz') #LUAD_LYMPH, LUAD_LYMPH_top20, LUADtraining_woNegatome
                #ccc_pairs = pd.read_csv('/cluster/home/t116508uhn/LRbind_output/without_elbow_cut/LRbind_LUAD_1D_manualDB_geneLocalCorrKNN_bidir_negatome/model_LRbind_LUAD_1D_manualDB_geneLocalCorrKNN_bidir_3L_negatome_allLR_nodeInfo_LUADtraining_interNegatome.csv.gz', sep=",")
                ccc_pairs['score'] = all_ccc_pairs['score']
                ccc_pairs['from_cell_index'] = all_ccc_pairs['from_cell_index']
                ccc_pairs['to_cell_index'] = all_ccc_pairs['to_cell_index']
                ccc_pairs['attention_score'] = all_ccc_pairs['attention_score']
                
                lr_dict = defaultdict(list)
                for i in range(0, len(ccc_pairs['attention_score'])):
                    if ccc_pairs['pred_score'][i] <= 0: #< 0.7:
                        continue
                    
                    if ccc_pairs['attention_score'][i] < 0.7:
                        continue

                    #lr_dict[ccc_pairs['ligand_gene'][i]+'+'+ccc_pairs['rec_gene'][i]].append([ccc_pairs['score'][i], ccc_pairs['from_cell_index'], ccc_pairs['to_cell_index'], ccc_pairs['pred_score'][i], ccc_pairs['attention_score']])  # score, cell ids, gene_node ids   
                    lr_dict[ccc_pairs['ligand_gene'][i]+'+'+ccc_pairs['rec_gene'][i]].append([ccc_pairs['score'][i], ccc_pairs['from_cell_index'], ccc_pairs['to_cell_index'], ccc_pairs['attention_score'], -1])  # score, cell ids, gene_node ids   

                    
                # plot input_cell_pair_list  
                '''
                ######### plot output #############################
                # UPDATE # annottaion
                data_list=dict()
                data_list['X']=[]
                data_list['Y']=[]   
                data_list['total_dot']=[] 
                data_list['prediction'] = []
                #data_list['label'] = []
                for i in range (0, len(barcode_info)):
                    data_list['X'].append(barcode_info[i][1])
                    data_list['Y'].append(-barcode_info[i][2])
                    if i in found_list:
                        data_list['total_dot'].append(np.sum(found_list[i]))
                        data_list['prediction'].append('positive')
                    else:
                        data_list['total_dot'].append(0)
                        data_list['prediction'].append('negative')
                    
                        
                    #data_list['label'].append(node_type[barcode_info[i][0]])
                    
                source= pd.DataFrame(data_list)
                
                chart = alt.Chart(source).mark_point(filled=True).encode(
                    alt.X('X', scale=alt.Scale(zero=False)),
                    alt.Y('Y', scale=alt.Scale(zero=False)),
                    color=alt.Color('total_dot:Q', scale=alt.Scale(scheme='magma')),
                    #shape = alt.Shape('label:N')
                )
                chart.save(args.output_path + model_name + '_output_' + target_ligand + '-' + target_receptor +'_top'+ file_name_suffix  + '_wholeTissue_allLR.html')
                #print(args.output_path + args.model_name + '_output_' + target_ligand + '-' + target_receptor +'_top'+ file_name_suffix  + '_wholeTissue_allLR.html') 
                '''
                ####### sort the pairs based on total score ####################
                sort_lr_list = []
                for lr_pair in lr_dict:
                    sum = 0
                    sum_pred = 0
                    #sum_layer1 = 0
                    attention_score_sum = 0
                    cell_pair_list = lr_dict[lr_pair]
                    weighted_sum = 0
                    for item in cell_pair_list:
                        sum = sum + item[0]  
                        #sum_layer1 = sum_layer1 + item[3]
                        #attention_score_sum = attention_score_sum + item[3] 
                        #weighted_sum = weighted_sum + item[0] * item[3] 
                        #sum_pred = sum_pred + item[3]
                        
                    #sum = sum/len(cell_pair_list)
                    sort_lr_list.append([lr_pair, sum, sum/len(cell_pair_list), len(cell_pair_list),  -1, -1, -1, -1]) #, sum_layer1, sum_layer1/len(cell_pair_list)])

#                    sort_lr_list.append([lr_pair, sum, sum/len(cell_pair_list), len(cell_pair_list),  sum_pred, sum_pred/len(cell_pair_list), attention_score_sum, weighted_sum]) #, sum_layer1, sum_layer1/len(cell_pair_list)])
                    
              
                sort_lr_list = sorted(sort_lr_list, key = lambda x: x[1], reverse=True)
                
                ### now remove the LR pairs which are below the elbow point
                if elbow_cut_flag==1:
                    X_axis = []
                    Y_axis = []
                    for i in range (0, len(sort_lr_list)):
                        X_axis.append(i)
                        Y_axis.append(sort_lr_list[i][1])
        
                    kn = KneeLocator(X_axis, Y_axis, direction='decreasing', curve="convex")
                    #kn_value_dec = Y_axis[kn.knee-1]            
                    sort_lr_list = sort_lr_list[0: kn.knee]
        
                    ##################### now keep only those LR ####################
                    keep_pair = dict()
                    for i in range (0, len(sort_lr_list)):
                        pair = sort_lr_list[i][0]
                        keep_pair[pair] = 1
        
                    pair_list = list(lr_dict.keys())
                    for pair in pair_list:
                        if pair not in keep_pair:
                            lr_dict.pop(pair)
                            
                    print('Top %d records are kept'%kn.knee)
                    
                ############### record the top hit without postprocessing ####################3
                
                #with gzip.open(args.output_path +model_name+'_top'+file_name_suffix+'_lr_dict_before_postprocess.pkl', 'wb') as fp:  
                #    pickle.dump(lr_dict, fp)

                # now plot the histograms where X axis will show the name or LR pair and Y axis will show the score.
                data_list=defaultdict(list)
                max_rows = len(sort_lr_list)

                
                for i in range (0, max_rows): #1000): #:
                    ligand = sort_lr_list[i][0].split('+')[0]
                    receptor = sort_lr_list[i][0].split('+')[1]

                    if sort_lr_list[i][3] < 10: # or sort_lr_list[i][6]<1.0:
                        continue
                    
                    if ligand in l_r_pair and receptor in l_r_pair[ligand]:
                        data_list['type'].append('From DB')
                    
                    #elif ligand+'_with_'+receptor in negatome_lr_unique: 
                    #    data_list['type'].append('From negatome')
                    else:
                        #continue
                        data_list['type'].append('Predicted')
                    

                    
                    data_list['X'].append(ligand + '_to_' + receptor)
                    data_list['Y'].append(sort_lr_list[i][1])
                    ligand = sort_lr_list[i][0].split('+')[0]
                    receptor = sort_lr_list[i][0].split('+')[1]
                    data_list['score_avg'].append(sort_lr_list[i][2])
                    data_list['pair_count'].append(sort_lr_list[i][3]) 
                    data_list['total_pred_score'].append(sort_lr_list[i][4])
                    data_list['avg_pred'].append(sort_lr_list[i][5])                    

                    #data_list['score_sum_layer1'].append(sort_lr_list[i][4])
                    #data_list['score_avg_layer1'].append(sort_lr_list[i][5])
                    data_list['total_attention_score'].append(sort_lr_list[i][6])
                    data_list['weighted_sum'].append(sort_lr_list[i][7])                    

                ########################################
                data_list_pd = pd.DataFrame({
                    'Ligand-Receptor Pairs': data_list['X'],
                    'Score_sum': data_list['Y'],
                    'Score_avg': data_list['score_avg'],
                    'Type': data_list['type'],
                    'Pair_count': data_list['pair_count'],
                    'total_pred_score': data_list['total_pred_score'],
                    'avg_pred': data_list['avg_pred']  , 
                    
                    'Total attention score': data_list['total_attention_score'],
                    'Weighted Sum': data_list['weighted_sum']                  
                    #'Score_sum_layer1': data_list['score_sum_layer1'],
                    #'Score_avg_layer1': data_list['score_avg_layer1']
                })
                #data_list_pd.to_csv(args.output_path +model_name+'_lr_list_sortedBy_totalScore_top'+ file_name_suffix+'_allLR_wofilter.csv', index=False) #_negatome
                data_list_pd.to_csv(args.output_path +model_name+'_lr_list_sortedBy_totalScore_top'+ file_name_suffix+'_allLR_predClass_LUAD_LYMPH_top20.csv', index=False) #_top20, LUAD_LYMPH, LUADtraining_woNegatome, LUADtraining_interNegatome



