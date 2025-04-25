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



##########################################################
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    #parser.add_argument( '--database_path', type=str, default='database/NEST_database.csv' , help='Provide your desired ligand-receptor database path here. Default database is a combination of CellChat and NicheNet database.')    
    parser.add_argument( '--data_name', type=str, default='LRbind_V1_Human_Lymph_Node_spatial_1D_manualDB_geneCorr_bidir', help='The name of dataset') #, required=True) # default='',
    parser.add_argument( '--model_name', type=str, default='model_LRbind_V1_Human_Lymph_Node_spatial_1D_manualDB', help='Name of the trained model') #, required=True) ''
    #_geneCorr_remFromDB
    
    parser.add_argument( '--total_runs', type=int, default=3, help='How many runs for ensemble (at least 2 are preferred)') #, required=True)
    #######################################################################################################
    parser.add_argument( '--embedding_path', type=str, default='embedding_data/', help='Path to grab the attention scores from')
    parser.add_argument( '--metadata_from', type=str, default='metadata/', help='Path to grab the metadata') 
    parser.add_argument( '--data_from', type=str, default='input_graph/', help='Path to grab the input graph from (to be passed to GAT)')
    parser.add_argument( '--output_path', type=str, default='/cluster/home/t116508uhn/LRbind_output/', help='Path to save the visualization results, e.g., histograms, graph etc.')
    parser.add_argument( '--target_ligand', type=str, default='CXCL10', help='') 
    parser.add_argument( '--target_receptor', type=str, default='CXCR3', help='')
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

    Tcell_zone = []
    node_type = dict()
    df = pd.read_csv("../NEST/data/V1_Human_Lymph_Node_spatial_annotation.csv", sep=",")
    for i in range (0, df["Barcode"].shape[0]):
        if df["Type"][i] == 'T-cell':
            Tcell_zone.append(barcode_index[df["Barcode"][i]])
            
        node_type[df["Barcode"][i]] = df["Type"][i]

   
        
    
    with gzip.open(args.metadata_from +args.data_name+'_barcode_info_gene', 'rb') as fp:  #b, a:[0:5]   _filtered
        barcode_info_gene, ligand_list, receptor_list, gene_node_list_per_spot, dist_X, l_r_pair, node_active_index, ligand_active_count, rec_active_count = pickle.load(fp)

    count = 0
    for key in node_active_index:
        if barcode_info_gene[key][5]==args.target_ligand:
            count = count+1

        
    index_vs_gene_node_info = defaultdict(list)
    for item in barcode_info_gene:
        gene_node_index = item[4]
        index_vs_gene_node_info[gene_node_index] = item
    
    with gzip.open(args.metadata_from + args.data_name +'_test_set', 'rb') as fp:  
        target_LR_index, target_cell_pair = pickle.load(fp)

    ############# load output graph #################################################
    model_names = [#'model_LRbind_V1_Human_Lymph_Node_spatial_1D_manualDB_geneCorr',
                   #'model_LRbind_V1_Human_Lymph_Node_spatial_1D_manualDB_geneCorr_vgae',
                   #'model_LRbind_V1_Human_Lymph_Node_spatial_1D_manualDB_vgae',
                   #'model_LRbind_V1_Human_Lymph_Node_spatial_1D_manualDB_vgae_gat',
                   #'model_LRbind_V1_Human_Lymph_Node_spatial_1D_manualDB_vgae_gat_wbce', 
                   # 'LRbind_model_V1_Human_Lymph_Node_spatial_1D_manualDB',
                   #'model_LRbind_V1_Human_Lymph_Node_spatial_1D_manualDB_bidir',
                   #'model_LRbind_V1_Human_Lymph_Node_spatial_1D_manualDB_bidir_3L',
                    'model_LRbind_V1_Human_Lymph_Node_spatial_1D_manualDB_geneCorr_bidir_3L'
                   
              ]
    for model_name in model_names:
        args.model_name = model_name
        args.model_name = args.model_name + '_r1'
        X_embedding_filename =  args.embedding_path + args.model_name + '_Embed_X' #.npy
        print("\n\n"+ X_embedding_filename)
        with gzip.open(X_embedding_filename, 'rb') as fp:  
            X_embedding = pickle.load(fp)

        
        for i in range (0, X_embedding.shape[0]):
            total_score_per_row = np.sum(X_embedding[i][:])
            X_embedding[i] = X_embedding[i]/total_score_per_row

        # apply PCA
        X_PCA = sc.pp.pca(X_embedding, n_comps=2) #args.pca
        # plot those on two dimensional plane
        data_list=dict()
        data_list['X']=[]
        data_list['Y']=[]   
        data_list['Type']=[]   
        lig_count = 0
        rec_count = 0
        for i in range (0, X_PCA.shape[0]):
            #if i in node_active_index:
            #    continue
                
            if index_vs_gene_node_info[i][5] == args.target_ligand: 
                data_list['Type'].append(index_vs_gene_node_info[i][5])
                lig_count = lig_count + 1
            elif index_vs_gene_node_info[i][5] == args.target_receptor:
                data_list['Type'].append(index_vs_gene_node_info[i][5])
                rec_count = rec_count + 1 
            else:
                continue
                #data_list['Type'].append('Other')


            data_list['X'].append(X_PCA[i][0])
            data_list['Y'].append(X_PCA[i][1])
        
        source= pd.DataFrame(data_list)
      
        chart = alt.Chart(source).mark_circle(size=5).encode(
            x = 'X',
            y = 'Y',
            color='Type',
            #shape = alt.Shape('Type:N')
        )
        chart.save(args.output_path + args.model_name + '_output_' + args.target_ligand + '-' + args.target_receptor + '_PCA.html')
        print(args.output_path + args.model_name + '_output_' + args.target_ligand + '-' + args.target_receptor + '_PCA.html')
            
##############
        data_list=dict()
        data_list['X']=[]
        data_list['Y']=[]   
        data_list['Type']=[]   
        lig_count = 0
        rec_count = 0
        for i in range (0, X_PCA.shape[0]):
            if i in node_active_index:
                continue
                
            if index_vs_gene_node_info[i][5] == args.target_ligand: 
                data_list['Type'].append(index_vs_gene_node_info[i][5])
                lig_count = lig_count + 1
            elif index_vs_gene_node_info[i][5] == args.target_receptor:
                data_list['Type'].append(index_vs_gene_node_info[i][5])
                rec_count = rec_count + 1 
            else:
                #continue
                data_list['Type'].append('Other')


            data_list['X'].append(X_PCA[i][0])
            data_list['Y'].append(X_PCA[i][1])
        
        source= pd.DataFrame(data_list)
      
        chart = alt.Chart(source).mark_circle(size=5).encode(
            x = 'X',
            y = 'Y',
            color='Type',
            #shape = alt.Shape('Type:N')
        )
        chart.save(args.output_path + args.model_name + '_output_' + target_ligand + '-' + target_receptor + '_allInactive_PCA.html')
        print(args.output_path + args.model_name + '_output_' + target_ligand + '-' + target_receptor + '_allInactive_PCA.html')
         
