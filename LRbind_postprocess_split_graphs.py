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

               'LRbind_Xenium_FFPE_Human_Breast_Cancer_Rep1_manualDB_geneLocalCorrKNN_bidir'
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
               
               'model_LRbind_Xenium_FFPE_Human_Breast_Cancer_Rep1_manualDB_geneLocalCorrKNN_bidir'
                               
          ]
target_ligands = ['CCL19', 'CCL19', 'CCL19', 'CCL19', 'CCL19', 'CCL19',  
                  'TGFB1','TGFB1','TGFB1','TGFB1', 'TGFB1','TGFB1',
                  'TGFB1', 'TGFB1','TGFB1',
                  'TGFB1','TGFB1', 'TGFB1',#'TGFB1',
                 'TGFB1','TGFB1','TGFB1','TGFB1',
                 'TGFB1'
                 ]
target_receptors = ['CCR7', 'CCR7', 'CCR7', 'CCR7', 'CCR7', 'CCR7',  
                    'ACVRL1','ACVRL1','ACVRL1','ACVRL1', 'ACVRL1','ACVRL1',
                    'ACVRL1', 'ACVRL1','ACVRL1',
                   'ACVRL1','ACVRL1', 'ACVRL1',#'ACVRL1',
                   'ACVRL1','ACVRL1','ACVRL1','ACVRL1',
                   'TGFBR2'
                   ]

if __name__ == "__main__":
    elbow_cut_flag = 0 #1 #0 #histogram
    knee_flag = 0 #1 #0 # pairwise
    file_name_suffix = "100" #'_elbow_' #'100_woHistElbowCut' # '_elbow' #'100' 
    ##########################################################
    # 4, 13
    for data_index in [22]: #range(0, len(data_names)):
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
    
        
        with gzip.open(args.metadata_from +args.data_name+'_barcode_info_gene', 'rb') as fp:  #b, a:[0:5]   _filtered
            barcode_info_gene, ligand_list, receptor_list, gene_node_list_per_spot, dist_X, l_r_pair, gene_node_index_active, ligand_active, receptor_active = pickle.load(fp)

        print('total gene node %d'%len(list(gene_node_index_active.keys())))
     
    
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
              if scaled_score >= th_80th:
                  attention_scores[i][j] = scaled_score
