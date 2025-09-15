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


model_names = ['model_LRbind_LUAD_lymph_1D_manualDB_geneLocalCorrKNN_bidir_3L_negatome'                               
               ]
data_names = [ 'LRbind_LUAD_1D_manualDB_geneLocalCorrKNN_bidir',
              'LRbind_V1_Human_Lymph_Node_spatial_1D_manualDB_geneLocalCorrKNN_bidir'
              ]

target_ligands = ['TGFB1', 'CCL19']
target_receptors = ['ACVRL1', 'CCR7']
sample_type = ['LUAD', 'LYMPH']

if __name__ == "__main__":
    elbow_cut_flag = 0 #1 #0 #histogram
    knee_flag = 0 #1 #0 # pairwise
    file_name_suffix = "100" #'_elbow_' #'100_woHistElbowCut' # '_elbow' #'100' 
    ##########################################################

    for model_index in [0]: #range(0, len(data_names)):
        parser = argparse.ArgumentParser()
        parser.add_argument( '--database_path', type=str, default='database/NEST_database.csv' , help='Provide your desired ligand-receptor database path here. Default database is a combination of CellChat and NicheNet database.')    
        parser.add_argument( '--data_name', type=str, default=data_names[model_index], help='The name of dataset') 
        parser.add_argument( '--model_name', type=str, default=model_names[model_index], help='The name of dataset')
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

        args.output_path = args.output_path + args.model_name + '/'
        if not os.path.exists(args.output_path):
            os.makedirs(args.output_path)

        all_ccc_pairs = defaultdict(list)
        all_negatome_pairs = defaultdict(list)
        all_negatome_pairs_intra = defaultdict(list)
        
        for sample_idx in range(0, 1): #len(sample_type)):
            print('****' + sample_type[sample_idx] + '****')
            args.metadata_from = 'metadata/'
            args.data_from = 'input_graph/'
            # args.embedding_path  = 'embedding_data/'
            args.data_name = data_names[sample_idx]
            args.target_ligand = target_ligands[sample_idx]
            args.target_receptor = target_receptors[sample_idx] 
            ##############
            args.metadata_from = args.metadata_from + args.data_name + '/'
            args.data_from = args.data_from + args.data_name + '/'
            args.embedding_path  = args.embedding_path + args.data_name + '/'
                
            
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
                                                

            print('negatome intra unique pairs found %d, and count %d'%(len(list(negatome_unique_pair.keys())),count))
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
            print('****' + args.data_name + '*********')
            print(model_names[model_index])
            print(args.target_ligand + '-' + args.target_receptor)
        

            args.model_name = model_names[model_index] 
            args.model_name = args.model_name + '_r1' + '_' + sample_type[sample_idx]
            X_embedding_filename =  args.embedding_path + args.model_name + '_Embed_X' 
            print("\n\n"+ X_embedding_filename)
            with gzip.open(X_embedding_filename, 'rb') as fp:  
                X_embedding = pickle.load(fp)
    
            
            for i in range (0, X_embedding.shape[0]):
                total_score_per_row = np.sum(X_embedding[i][:])
                X_embedding[i] = X_embedding[i]/total_score_per_row

            '''
            X_embedding_filename =  args.embedding_path + args.model_name + '_Embed_X_layer1' 
            print("\n\n"+ X_embedding_filename)
            with gzip.open(X_embedding_filename, 'rb') as fp:  
                X_embedding_layer1 = pickle.load(fp)
    
            
            for i in range (0, X_embedding_layer1.shape[0]):
                total_score_per_row = np.sum(X_embedding_layer1[i][:])
                X_embedding_layer1[i] = X_embedding_layer1[i]/total_score_per_row
            '''



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

            
            
            break_flag = 0
            test_mode = 1

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
                
                for i in range (0, len(barcode_info)):
                    #if node_type[barcode_info[i][0]] != 'T-cell':
                    #    continue
                    #print("i: %d"%i)
                    #print("found list: %d"%len(found_list))
                    # from i to j 
                    ligand_node_index = []
                    ligand_node_index_intra = []
                    for gene in gene_node_list_per_spot[i]:
                        if gene in ligand_list:
                            if gene in db_gene_nodes:
                                ligand_node_index.append([gene_node_list_per_spot[i][gene], gene])
                            if gene in negatome_gene: # it is coming from negatome
                                ligand_node_index_intra.append([gene_node_list_per_spot[i][gene], gene])
                    
                    receptor_node_index_intra = []
                    for gene in gene_node_list_per_spot[i]:
                        if gene in receptor_list and gene in negatome_gene:
                            receptor_node_index_intra.append([gene_node_list_per_spot[i][gene], gene])

                    dot_prod_list = []
                    product_only = []
                    for j in range (0, len(barcode_info)):
    
                        if dist_X[i][j]==0 or i==j :
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
                                    if i_gene[1]+'_with_'+j_gene[1] in negatome_lr_unique:
                                        continue
                                  
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

                                if i_gene[1]+'_with_'+j_gene[1] in negatome_lr_unique: 
                                    dot_prod_list_negatome_inter.append([temp, i, j, i_gene[1], j_gene[1], i_gene[0], j_gene[0]])
                                    continue
                                  
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
                            all_negatome_pairs['from_cell_index'].append(item[1])
                            all_negatome_pairs['to_cell_index'].append(item[2])


                      
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
                  
                    ## get the list of dot_prod_list_negatome_intra
                    dot_prod_list_negatome_intra = []
                    for i_gene in ligand_node_index_intra:  
                        for j_gene in receptor_node_index_intra:
                            if i_gene[1]==j_gene[1]:
                                continue
                          
                            if i_gene[1]+'_with_'+j_gene[1] in negatome_lr_unique:
                                temp = distance.euclidean(X_embedding[i_gene[0]], X_embedding[j_gene[0]]) # 
                                dot_prod_list_negatome_intra.append([temp, i, i, i_gene[1], j_gene[1], i_gene[0], j_gene[0]])
                                
                              
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
                        
                        ####################################################
                
                # plot found_list
                print("positive: %d, total pairs %d"%(len(found_list), len(lr_dict.keys())))

        """
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
        data_list_pd.to_csv(args.output_path +args.model_name+'_allLR_nodeInfo.csv.gz', index=False, compression='gzip') #_negatome
        print(len(data_list_pd))
        data_list_pd = pd.DataFrame({
            'from_cell': all_negatome_pairs['from_cell'],
            'to_cell': all_negatome_pairs['to_cell'],
            'from_gene_node': all_negatome_pairs['from_gene_node'],
            'to_gene_node': all_negatome_pairs['to_gene_node'],
            'ligand_gene': all_negatome_pairs['ligand_gene'],
            'rec_gene': all_negatome_pairs['rec_gene'],
            'score': all_negatome_pairs['score'],
            'from_cell_index': all_negatome_pairs['from_cell_index'].append(item[1]),
            'to_cell_index': all_negatome_pairs['to_cell_index'].append(item[2])

            
        })
        data_list_pd.to_csv(args.output_path +model_name+'_negatomeLR_nodeInfo_inter.csv', index=False) #_negatome
        print(len(data_list_pd))
        # you can double check if all sender and rcvr cells are the same one
        data_list_pd = pd.DataFrame({
            'from_cell': all_negatome_pairs_intra['from_cell'],
            'to_cell': all_negatome_pairs_intra['to_cell'],
            'from_gene_node': all_negatome_pairs_intra['from_gene_node'],
            'to_gene_node': all_negatome_pairs_intra['to_gene_node'],
            'ligand_gene': all_negatome_pairs_intra['ligand_gene'],
            'rec_gene': all_negatome_pairs_intra['rec_gene'],
            'score': all_negatome_pairs_intra['score']
            'from_cell_index': all_negatome_pairs_intra['from_cell_index'].append(item[1]),
            'to_cell_index': all_negatome_pairs_intra['to_cell_index'].append(item[2])
            
        })
        data_list_pd.to_csv(args.output_path +model_name+'_negatomeLR_nodeInfo_intra.csv', index=False) #_negatome
        print(len(data_list_pd))
        negatome_unique_pair = dict()

        for idx in range (0, len(all_negatome_pairs['ligand_gene'])):
            negatome_unique_pair[all_negatome_pairs['ligand_gene'][idx] + '_with_' + all_negatome_pairs['rec_gene'][idx]] = 1
        
        for idx in range (0, len(all_negatome_pairs_intra['ligand_gene'])):
            negatome_unique_pair[all_negatome_pairs_intra['ligand_gene'][idx] + '_with_' + all_negatome_pairs_intra['rec_gene'][idx]] = 1

        print('unique pairs found %d'%len(list(negatome_unique_pair.keys())))

        """
        
        ccc_pairs = pd.read_csv(args.output_path +args.model_name +'_allLR_nodeInfo.csv.gz') #_negatome
        ccc_pairs['score'] = all_ccc_pairs['score']
        ccc_pairs['from_cell_index'] = all_ccc_pairs['from_cell_index']
        ccc_pairs['to_cell_index'] = all_ccc_pairs['to_cell_index']
        ccc_pairs['attention_score'] = all_ccc_pairs['attention_score']
        
        lr_dict = defaultdict(list)
        for i in range(0, len(ccc_pairs)):
            #if ccc_pairs['pred_score'][i] <= 0: #< 0.7:
            #    continue
            
            #if ccc_pairs['attention_score'][i] < 0.7:
            #    continue

            #lr_dict[ccc_pairs['ligand_gene'][i]+'+'+ccc_pairs['rec_gene'][i]].append([ccc_pairs['score'][i], ccc_pairs['from_cell_index'], ccc_pairs['to_cell_index'], ccc_pairs['pred_score'][i], ccc_pairs['attention_score']])  # score, cell ids, gene_node ids   
            lr_dict[ccc_pairs['ligand_gene'][i]+'+'+ccc_pairs['rec_gene'][i]].append([ccc_pairs['score'][i], ccc_pairs['from_cell_index'], ccc_pairs['to_cell_index'], -1, ccc_pairs['attention_score']])  # score, cell ids, gene_node ids   

            
        # plot input_cell_pair_list  
        #  vs 88497
        0.005 - 5
        0.0005 - 10
        0.00005 -15
        
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
                attention_score_sum = attention_score_sum + item[3] 
                weighted_sum = weighted_sum + item[0] * item[3] 
                sum_pred = sum_pred + item[3]
                
            #sum = sum/len(cell_pair_list)
            sort_lr_list.append([lr_pair, sum, sum/len(cell_pair_list), len(cell_pair_list),  sum_pred, sum_pred/len(cell_pair_list), attention_score_sum, weighted_sum]) #, sum_layer1, sum_layer1/len(cell_pair_list)])
            
        
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
            
            elif ligand+'_with_'+receptor in negatome_lr_unique: 
                data_list['type'].append('From negatome')
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
        #data_list_pd.to_csv(args.output_path +model_name+'_lr_list_sortedBy_totalScore_top'+ file_name_suffix+'_allLR_predScore.csv', index=False) #_negatome
        data_list_pd.to_csv(args.output_path +args.model_name+'_lr_list_sortedBy_totalScore_top'+ file_name_suffix+'_allLR_predClass.csv', index=False) #_negatome

#####################################################################################################################################################################
                
                ######
                data_list=dict()
                data_list['X']=[]
                data_list['Y']=[] 
                max_rows = min(500, len(sort_lr_list))
                for i in range (0, max_rows): #1000): #:
                    ligand = sort_lr_list[i][0].split('+')[0]
                    receptor = sort_lr_list[i][0].split('+')[1]
                    data_list['X'].append(ligand + '_to_' + receptor)
                    data_list['Y'].append(sort_lr_list[i][1])
                    
                data_list_pd = pd.DataFrame({
                    'Ligand-Receptor Pairs': data_list['X'],
                    'Score': data_list['Y']
                })
                
                chart = alt.Chart(data_list_pd).mark_bar().encode(
                    x=alt.X("Ligand-Receptor Pairs:N", axis=alt.Axis(labelAngle=45), sort='-y'),
                    y='Score'
                )
            
                chart.save(args.output_path +model_name+'_lr_list_sortedBy_totalScore_top'+file_name_suffix+'_histogramsallLR.html')
                

    
                #################################
                save_lr_dict = copy.deepcopy(lr_dict)
                ############################
                lr_dict = copy.deepcopy(save_lr_dict)
                print('before post process len %d'%len(lr_dict.keys()))
                #####################
                if 'prefiltered' not in model_name:
                    key_list = list(lr_dict.keys())
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
                                if gene not in gene_index:
                                    continue
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
    
    
                    if receptor not in receptor_intraNW or len(receptor_intraNW[receptor])==0:
                        lr_dict.pop(ligand + '+' + receptor)
                        continue
                        
                    target_list = receptor_intraNW[receptor]
                    temp_target_list = []
                    for gene in target_list:
                        if gene in gene_index:
                            temp_target_list.append(gene)

                    target_list = temp_target_list
    
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
                    sc.tl.rank_genes_groups(adata_temp, groupby='group', groups=['target'], reference='other', method='wilcoxon') #, pts = True)
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
                if 'prefiltered' not in model_name:
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
                                if gene not in gene_index:
                                    continue
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
                        
                    if ligand not in ligand_intraNW or len(ligand_intraNW[ligand])==0:
                        lr_dict.pop(ligand + '+' + receptor)
                        continue
                        
                    target_list = ligand_intraNW[ligand]
                    temp_target_list = []
                    for gene in target_list:
                        if gene in gene_index:
                            temp_target_list.append(gene)

                    target_list = temp_target_list
                    
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
                    sc.tl.rank_genes_groups(adata_temp, groupby='group', groups=['target'], reference='other', method='wilcoxon') #, pts = True)
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
                #with gzip.open(args.output_path +model_name+'_top'+file_name_suffix+'_lr_dict_after_postprocess.pkl', 'wb') as fp:  
                #	pickle.dump([lr_dict, pvals_lr], fp)
    
                
                ########## take top hits #################################### 
                sort_lr_list = []
                for lr_pair in lr_dict:
                    sum = 0
                    cell_pair_list = lr_dict[lr_pair]
                    for item in cell_pair_list:
                        sum = sum + item[0]  
    
                    #sum = sum/len(cell_pair_list)
                    sort_lr_list.append([lr_pair, sum, sum/len(cell_pair_list), pvals_lr[lr_pair], len(cell_pair_list)])
                    
                sort_lr_list = sorted(sort_lr_list, key = lambda x: x[1], reverse=True)
                #sort_lr_list = sorted(sort_lr_list, key = lambda x: x[2])

               # now plot the histograms where X axis will show the name or LR pair and Y axis will show the score.
                data_list=dict()
                data_list['X']=[]
                data_list['Y']=[] 
                data_list['type']=[]
                #data_list['score_sum'] =[]
                data_list['score_avg'] = []
                data_list['avg_pvals_adj'] = []
                data_list['pair_count'] = []
                max_rows = len(sort_lr_list)
                for i in range (0, max_rows): #1000): #:
                    ligand = sort_lr_list[i][0].split('+')[0]
                    receptor = sort_lr_list[i][0].split('+')[1]
                    data_list['X'].append(ligand + '_to_' + receptor)
                    data_list['Y'].append(sort_lr_list[i][1])
                    ligand = sort_lr_list[i][0].split('+')[0]
                    receptor = sort_lr_list[i][0].split('+')[1]
                    data_list['score_avg'].append(sort_lr_list[i][2])
                    data_list['avg_pvals_adj'].append(sort_lr_list[i][3]) 
                    data_list['pair_count'].append(sort_lr_list[i][4])
                    if ligand in l_r_pair and receptor in l_r_pair[ligand]:
                        data_list['type'].append('From DB')
                    else:
                        data_list['type'].append('Predicted')
                        
                data_list_pd = pd.DataFrame({
                    'Ligand-Receptor Pairs': data_list['X'],
                    'Score_sum': data_list['Y'],
                    'Score_avg': data_list['score_avg'],
                    'Avg_pvals_adj': data_list['avg_pvals_adj'],
                    'Type': data_list['type'],
                    'Pair_count': data_list['pair_count']
                })
                data_list_pd.to_csv(args.output_path +model_name+'_down_up_deg_lr_list_sortedBy_totalScore_top'+'_elbow'+'_allLR.csv', index=False)
                
                # now plot the top max_rows histograms where X axis will show the name or LR pair and Y axis will show the score.
                data_list=dict()
                data_list['X']=[]
                data_list['Y']=[] 
                max_rows = min(500, len(sort_lr_list))
                for i in range (0, max_rows): #1000): #:
                    ligand = sort_lr_list[i][0].split('+')[0]
                    receptor = sort_lr_list[i][0].split('+')[1]
                    data_list['X'].append(ligand + '_to_' + receptor)
                    data_list['Y'].append(sort_lr_list[i][1])
                    
                data_list_pd = pd.DataFrame({
                    'Ligand-Receptor Pairs': data_list['X'],
                    'Score': data_list['Y']
                })
                
                chart = alt.Chart(data_list_pd).mark_bar().encode(
                    x=alt.X("Ligand-Receptor Pairs:N", axis=alt.Axis(labelAngle=45), sort='-y'),
                    y='Score'
                )
            
                chart.save(args.output_path +model_name+'_down_up_deg_lr_list_sortedBy_totalScore_top'+file_name_suffix+'_histogramsallLR.html')
                #print(args.output_path +args.model_name+'_novel_lr_list_sortedBy_totalScore_top'+file_name_suffix+'_histogramsallLR.html')   
                #if target_ligand +'+'+ target_receptor in list(data_list_pd['Ligand-Receptor Pairs']):
                #    print("found %d"%top_hit_lrp_dict[target_ligand +'+'+ target_receptor])
                ############################### novel only out of all LR ################
                sort_lr_list_temp = []
                i = 0
                for pair in sort_lr_list:                
                    ligand = pair[0].split('+')[0]
                    receptor = pair[0].split('+')[1]
                    if ligand in l_r_pair and receptor in l_r_pair[ligand]:
                        #if i<15:
                        #    print(i)
                        i=i+1
                        continue
                    i = i + 1
                        
                    sort_lr_list_temp.append(pair) 
    
                data_list=dict()
                data_list['X']=[]
                data_list['Y']=[] 
            
                max_rows = len(sort_lr_list_temp)
                for i in range (0, max_rows): 
                    data_list['X'].append(sort_lr_list_temp[i][0])
                    data_list['Y'].append(sort_lr_list_temp[i][1])
                    
                data_list_pd = pd.DataFrame({
                    'Ligand-Receptor Pairs': data_list['X'],
                    'Score': data_list['Y']
                })
                data_list_pd.to_csv(args.output_path +model_name+'_down_up_deg_novel_lr_list_sortedBy_totalScore_top'+file_name_suffix+'_novelsOutOfallLR.csv', index=False)
                #print('novel LRP length %d out of top %d LRP'%(len(sort_lr_list_temp), top_lrp_count))
                # now plot the top max_rows histograms where X axis will show the name or LR pair and Y axis will show the score.
                data_list=dict()
                data_list['X']=[]
                data_list['Y']=[] 
            
                max_rows = min(500, len(sort_lr_list_temp))
                for i in range (0, max_rows): 
                    data_list['X'].append(sort_lr_list_temp[i][0])
                    data_list['Y'].append(sort_lr_list_temp[i][1])
                    
                data_list_pd = pd.DataFrame({
                    'Ligand-Receptor Pairs': data_list['X'],
                    'Score': data_list['Y']
                })
                
                chart = alt.Chart(data_list_pd).mark_bar().encode(
                    x=alt.X("Ligand-Receptor Pairs:N", axis=alt.Axis(labelAngle=45), sort='-y'),
                    y='Score'
                )
            
                chart.save(args.output_path +model_name+'_down_up_deg_novel_lr_list_sortedBy_totalScore_top'+file_name_suffix+'_histograms_novelsOutOfallLR.html')
                #print(args.output_path +args.model_name+'_novel_lr_list_sortedBy_totalScore_top'+file_name_suffix+'_histograms_novelsOutOfallLR.html')   
                ################################# when not remFromDB ##########################################################################################################
                
                set_LRbind_novel = []
                list_size = min(4000, len(sort_lr_list_temp))
                for i in range (0, list_size):
                    set_LRbind_novel.append(sort_lr_list_temp[i][0])
            
                #print('ligand-receptor database reading.')
                df = pd.read_csv(args.database_path, sep=",")
                set_nichenet_novel = []
                for i in range (0, df["Ligand"].shape[0]):
                    ligand = df["Ligand"][i] 
                    receptor = df["Receptor"][i]
                    if ligand in ligand_list and receptor in receptor_list and 'ppi' in df["Reference"][i]:
                        set_nichenet_novel.append(ligand + '+' + receptor)
            
                set_nichenet_novel = np.unique(set_nichenet_novel)
                common_lr = list(set(set_LRbind_novel) & set(set_nichenet_novel))
                print('top_N:%d, Only LRbind %d, only nichenet %d, common %d'%(top_N, len(set_LRbind_novel)-len(common_lr), len(set_nichenet_novel)-len(common_lr), len(common_lr)))
                pd.DataFrame(common_lr).to_csv(args.output_path +args.model_name+'_down_up_deg_novel_lr_list_sortedBy_totalScore_top'+file_name_suffix+'_common_with_nichenet.csv', index=False)
                print('end\n')
                #print(args.output_path +args.model_name+'_novel_lr_list_sortedBy_totalScore_top'+file_name_suffix+'novelsOutOfallLR.csv') 
                # top_N:100, Only LRbind 3833, only nichenet 4010, common 167
                ##################################################################
                '''
                df = pd.read_csv("../NEST_experimental/output/V1_Human_Lymph_Node_spatial/CellNEST_V1_Human_Lymph_Node_spatial_top20percent.csv", sep=",")
                set_nichenet_novel = []
                for i in range (0, df["ligand"].shape[0]):
                    ligand = df["ligand"][i] 
                    receptor = df["receptor"][i]
                    if (ligand==target_ligand and receptor in receptor_list) or (receptor == target_receptor and ligand in ligand_list):# and ('ppi' not in df["Reference"][i]):
                        set_nichenet_novel.append(ligand + '+' + receptor)
            
                set_nichenet_novel = np.unique(set_nichenet_novel)
                common_lr = list(set(set_LRbind_novel) & set(set_nichenet_novel))
                #print('Only LRbind %d, only manual %d, common %d'%(len(set_LRbind_novel), len(set_nichenet_novel)-len(common_lr), len(common_lr)))
                '''
                ########## Plot an LR pair location ################
                #with gzip.open(args.output_path +model_name+'_top'+file_name_suffix+'_lr_dict_after_postprocess.pkl', 'rb') as fp:  
                #	lr_dict, pvals_lr = pickle.load(fp) #
                '''
                ligand = 'ITGB1'
                receptor = 'ITGA3'
                found_list = defaultdict(list)
                for pairs in lr_dict[ligand + '+' + receptor]:
                    found_list[pairs[1]].append(pairs[0])
                    found_list[pairs[2]].append(pairs[0])
    
                
                data_list=dict()
                data_list['X']=[]
                data_list['Y']=[]   
                data_list['total_dot']=[] 
                data_list['prediction'] = []
                #data_list['label'] = []
                for i in range (0, len(barcode_info)):
                    #if barcode_info[i][1] < 5000 or barcode_info[i][2] > 5000:
                    #    continue
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
                chart.save(args.output_path + args.model_name + '_after_postprocess_spatial_location_' + ligand + '-' + receptor +'_top'+ file_name_suffix  + '.html')
                print(args.output_path + args.model_name + '_spatial_location_' + ligand + '-' + receptor +'_top'+ file_name_suffix  + '.html') 
    
                '''
                
                 ############ only Tcell Zone plot ##############################################################################################################################
                '''
                Tcell_zone_lr_dict = defaultdict(list)
                for lrp in lr_dict:
                    cell_pair_list = lr_dict[lrp]
                    for pair in cell_pair_list:
                        i = pair[1]
                        j = pair[2]
                        if i in Tcell_zone and j in Tcell_zone:
                            Tcell_zone_lr_dict[lrp].append([item[0], item[1], item[2]])
                               
                Tcell_zone_sort_lr_list = []
                for lr_pair in Tcell_zone_lr_dict:
                    #if lr_pair not in top_hit_lrp_dict:
                    #    continue
                    sum = 0
                    cell_pair_list = Tcell_zone_lr_dict[lr_pair]
                    for item in cell_pair_list:
                        sum = sum + item[0] # 
    
                    #sum = sum/len(cell_pair_list) 
                    Tcell_zone_sort_lr_list.append([lr_pair, sum, sum/len(cell_pair_list)])
            
                Tcell_zone_sort_lr_list = sorted(Tcell_zone_sort_lr_list, key = lambda x: x[1], reverse=True)
                
                # now plot the histograms where X axis will show the name or LR pair and Y axis will show the score.
                data_list=dict()
                data_list['X']=[]
                data_list['Y']=[] 
                max_rows = len(Tcell_zone_sort_lr_list) #min(500, len(Tcell_zone_sort_lr_list))
                for i in range (0, max_rows): #1000): #:
                    data_list['X'].append(Tcell_zone_sort_lr_list[i][0])
                    data_list['Y'].append(Tcell_zone_sort_lr_list[i][1])
                    if Tcell_zone_sort_lr_list[i][0]=='CCL19+CCR7':
                        print("Tcell: found CCL19-CCR7: %d"%i)
                    
                data_list_pd = pd.DataFrame({
                    'Ligand-Receptor Pairs': data_list['X'],
                    'Score': data_list['Y']
                })
                #if 'CCL19+CCR7' in list(data_list_pd['Ligand-Receptor Pairs']):
                #    print("found CCL19-CCR7")
                
                data_list_pd.to_csv(args.output_path +args.model_name+'_sortedBy_totalScore_top'+file_name_suffix+'Tcell_zone_allLR.csv', index=False)
                #print(args.output_path +args.model_name+'_novel_lr_list_sortedBy_totalScore_top'+file_name_suffix+'Tcell_zone_allLR.csv')    
                # same as histogram plots
                chart = alt.Chart(data_list_pd).mark_bar().encode(
                    x=alt.X("Ligand-Receptor Pairs:N", axis=alt.Axis(labelAngle=45), sort='-y'),
                    y='Score'
                )
            
                chart.save(args.output_path +args.model_name+'_sortedBy_totalScore_top'+file_name_suffix+'Tcell_zone_histogramsallLR.html')
                print(args.output_path +args.model_name+'_sortedBy_totalScore_top'+file_name_suffix+'Tcell_zone_histogramsallLR.html')  
                '''

######
'''
model_names = [#'model_LRbind_V1_Human_Lymph_Node_spatial_1D_manualDB_geneCorr',
               #'model_LRbind_V1_Human_Lymph_Node_spatial_1D_manualDB_geneCorr_vgae',
               #'model_LRbind_V1_Human_Lymph_Node_spatial_1D_manualDB_vgae',
               #'model_LRbind_V1_Human_Lymph_Node_spatial_1D_manualDB_vgae_gat',
               #'model_LRbind_V1_Human_Lymph_Node_spatial_1D_manualDB_vgae_gat_wbce',
               #'LRbind_model_V1_Human_Lymph_Node_spatial_1D_manualDB',
               #'model_LRbind_V1_Human_Lymph_Node_spatial_1D_manualDB_bidir',
               #'model_LRbind_V1_Human_Lymph_Node_spatial_1D_manualDB_bidir_3L',
               #'model_LRbind_V1_Human_Lymph_Node_spatial_1D_manualDB_geneCorr_bidir_3L',
               # 'model_LRbind_V1_Human_Lymph_Node_spatial_1D_manualDB_geneCorrKNN_bidir_3L',
               #'model_LRbind_V1_Human_Lymph_Node_spatial_1D_manualDB_geneCorrKNN_bidir_3L',
               #'model_LRbind_GSM6177599_NYU_BRCA0_Vis_processed_1D_manualDB_geneCorr_bidir_3L'
               #'model_LRbind_CID44971_1D_manualDB_geneCorr_bidir_3L',
               #'model_LRbind_CID44971_1D_manualDB_geneCorrKNN_bidir_3L'
               #'model_LRbind_LUAD_1D_manualDB_geneCorr_bidir_3L'
               #'model_LRbind_LUAD_1D_manualDB_geneCorr_signaling_bidir_3L'
               #'model_LRbind_LUAD_1D_manualDB_geneCorrKNN_bidir_3L'
               #'model_LRbind_LUAD_1D_manualDB_geneCorrP7KNN_bidir_3L'
               #'model_LRbind_LUAD_1D_manualDB_geneCorrKNN_bidir_3L_prefiltered'
               #'model_LRbind_PDAC64630_1D_manualDB_geneCorrKNN_bidir_3L'
               #'model_LRbind_V1_Breast_Cancer_Block_A_Section_1_spatial_1D_manualDB_geneCorrKNN_bidir_3L',
               'model_LRbind_V1_Breast_Cancer_Block_A_Section_1_spatial_1D_manualDB_geneCorrKNN_bidir_3L',
               #'model_LRbind_V1_Breast_Cancer_Block_A_Section_1_spatial_1D_manualDB_geneCorrKNN_bidir_3L_prefiltered'
                
          ]
'''

   
