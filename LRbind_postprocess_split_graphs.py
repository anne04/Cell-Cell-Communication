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



def get_dataset(
    ccc_pairs: pd.DataFrame,
    cell_vs_gene_emb: defaultdict(dict),
    gene_node_list_per_spot: defaultdict(dict),
    X_protein_embedding: dict()
):
    """
    Return a dictionary as: [sender_cell][recvr_cell] = [(ligand gene, receptor gene, attention score), ...]
    for each pair of cells based on CellNEST detection. And a dictionary with cell_vs_index mapping.
    """
    """
    Parameters:
    ccc_pairs:  ccc_pairs = ['from_cell', 'to_cell', 'from_gene_node', 'to_gene_node', 'ligand_gene', 'rec_gene'] 
    representing cell_barcode_sender, cell_barcode_receiver, ligand gene, receptor gene, 
    edge_rank, component_label, index_sender, index_receiver, attention_score
    barcode_info: list of [cell_barcode, coordinate_x, coordinates_y, -1]
    """
    dataset = []
    # each sample has [sender set, receiver set, score]
    print('len ccc pairs: %d'%len(ccc_pairs))
    record_index = []
    for i in range (0, len(ccc_pairs)):
        #print(i)
        sender_cell_barcode = str(ccc_pairs['from_cell'][i])
        rcv_cell_barcode = str(ccc_pairs['to_cell'][i])
        if sender_cell_barcode  == rcv_cell_barcode:
            continue # for now, skipping autocrine signals
            
        ligand_gene = ccc_pairs['ligand_gene'][i]
        rec_gene = ccc_pairs['rec_gene'][i]
        # need to find the index of gene nodes in cells

        if ligand_gene in gene_node_list_per_spot[sender_cell_barcode] and \
            rec_gene in gene_node_list_per_spot[rcv_cell_barcode] and \
            ligand_gene in X_protein_embedding and rec_gene in X_protein_embedding:
            
            ligand_node_index = gene_node_list_per_spot[sender_cell_barcode][ligand_gene]
            rec_node_index = gene_node_list_per_spot[rcv_cell_barcode][rec_gene]
            
            sender_set = cell_vs_gene_emb[sender_cell_barcode][ligand_node_index]
            rcvr_set = cell_vs_gene_emb[rcv_cell_barcode][rec_node_index]
            #score = ccc_pairs['attention_score'][i]
            dataset.append([sender_set, rcvr_set, -1, ligand_gene, rec_gene])
            record_index.append(i)

    print('len dataset: %d'%len(dataset))
    return dataset, record_index



def get_cellEmb_geneEmb_pairs(
    #cell_vs_index: dict(),
    barcode_info_gene: list(),
    #X_embedding = np.array,
    X_gene_embedding = np.array,
    X_protein_embedding = np.array
) -> defaultdict(dict):
    """

    Parameters:
    cell_vs_index: dictionary with key = cell_barcode, value = index of that cell 
    barcode_info_gene: list of [cell's barcode, cell's X, cell's Y, -1, gene_node_index, gene_name]
    X = 2D np.array having row = cell index, column = feature dimension
    X_g = 2D np.array having row = gene node index, column = feature dimension
    """
    
    cell_vs_gene_emb = defaultdict(dict)
    not_found = dict()
    for i in range (0, len(barcode_info_gene)):
        cell_index = i
        cell_barcode = barcode_info_gene[i][0]
        gene_index = barcode_info_gene[i][4]
            
        #cell_index_cellnest = cell_vs_index[cell_barcode]
        gene_name = barcode_info_gene[i][5]
        #if cell_barcode == 'GGCGCTCCTCATCAAT-1':
        #    print(gene_index)  
        if gene_name in X_protein_embedding:         
            cell_vs_gene_emb[cell_barcode][gene_index] = [np.zeros((1, 512)), X_gene_embedding[gene_index], X_protein_embedding[gene_name]]
            #X_embedding[cell_index_cellnest]
        else:
            not_found[gene_name] = 1
            
    return cell_vs_gene_emb



data_names = ['LRbind_Xenium_Prime_Human_Skin_FFPE_manualDB_geneLocalCorrKNN_bidir'
            ]

model_names = ['model_LRbind_Xenium_Prime_Human_Skin_FFPE_manualDB_geneLocalCorrKNN_bidir',                               
          ]
target_ligands = [
                 'TGFB1'
                 ]
target_receptors = [
                   'TGFBR2'
                   ]

if __name__ == "__main__":
    elbow_cut_flag = 0 #1 #0 #histogram
    knee_flag = 0 #1 #0 # pairwise
    file_name_suffix = "100" #'_elbow_' #'100_woHistElbowCut' # '_elbow' #'100' 
    ##########################################################
    # 4, 13
    for data_index in [0]: #range(0, len(data_names)):
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
        parser.add_argument( '--total_subgraphs', type=int, default=16, help='')
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
        ''''''
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
        args.model_name = model_names[data_index]

        ######################### LR database ###############################################
        df = pd.read_csv(args.database_path, sep=",")
        db_gene_nodes = dict()
        for i in range (0, df["Ligand"].shape[0]):
            ligand = df["Ligand"][i]
            receptor = df["Receptor"][i]
            db_gene_nodes[ligand] = '1'
            db_gene_nodes[receptor] = '1'


        ##################################################################
        fp = gzip.open('input_graph/LRbind_Xenium_Prime_Human_Skin_FFPE_manualDB_geneLocalCorrKNN_bidir/LRbind_Xenium_Prime_Human_Skin_FFPE_manualDB_geneLocalCorrKNN_bidir_adjacency_gene_records', 'rb')  
        row_col_gene, edge_weight, lig_rec, gene_node_type, gene_node_expression, total_num_gene_node, start_of_intra_edge = pickle.load(fp)
        ########### Merge the subgraphs ####################
        dict_cell_edge = defaultdict(list) # key = node. values = incoming edges
        dict_cell_neighbors = defaultdict(list) # key = node. value = nodes corresponding to incoming edges/neighbors

        nodes_active = dict()
        for i in range(0, len(row_col_gene)): 
            dict_cell_edge[row_col_gene[i][1]].append(i) # index of the edges (incoming)
            dict_cell_neighbors[row_col_gene[i][1]].append(row_col_gene[i][0]) # neighbor id
            nodes_active[row_col_gene[i][1]] = '' # to 
            nodes_active[row_col_gene[i][0]] = '' # from
        
        datapoint_size = len(nodes_active.keys())
        for i in range (0, datapoint_size):
            neighbor_list = dict_cell_neighbors[i]
            neighbor_list = list(set(neighbor_list))
            dict_cell_neighbors[i] = neighbor_list


        node_id_sorted_path = args.metadata_from + '/'+ args.data_name+'_'+'gene_node_id_sorted_xy'
        fp = gzip.open(node_id_sorted_path, 'rb')
        node_id_sorted_xy = pickle.load(fp)
        node_id_sorted_xy_temp = []
        for i in range(0, len(node_id_sorted_xy)):
            if node_id_sorted_xy[i][0] in nodes_active: # skip those which are not in our ROI
                node_id_sorted_xy_temp.append(node_id_sorted_xy[i])
        
        node_id_sorted_xy = node_id_sorted_xy_temp    
        
        ##################################################################################################################
        # split it into N set of edges    
        total_subgraphs = args.total_subgraphs 
        #edge_list = []
        graph_bag = []
        start_index = []
        id_map_old_new = [] # make an index array, so that existing node ids are mapped to new ids
        id_map_new_old = []
        
        for i in range (0, total_subgraphs+1):
            start_index.append((datapoint_size//total_subgraphs)*i)
            id_map_old_new.append(dict())
            id_map_new_old.append(dict())
    
        
        ##################################################################################################################
        set_id=-1
        suggraph_vs_gene_emb = defaultdict(list)
        for indx in range (0, len(start_index)-1):
            set_id = set_id + 1
            #print('graph id %d, node %d to %d'%(set_id,start_index[indx],start_index[indx+1]))
            set1_nodes = []
            set1_edges_index = []
            node_limit_set1 = start_index[indx+1]
            set1_direct_edges = []

            for i in range (start_index[indx], node_limit_set1):
                set1_nodes.append(node_id_sorted_xy[i][0])
                suggraph_vs_gene_emb[set_id].append(node_id_sorted_xy[i][0])
                # add it's incoming edges - first hop
                for edge_index in dict_cell_edge[node_id_sorted_xy[i][0]]: 
                    set1_edges_index.append(edge_index) # has both row_col_gene and edge_weight
                    set1_direct_edges.append(edge_index)
                # add it's neighbor's edges - second hop
                for neighbor in dict_cell_neighbors[node_id_sorted_xy[i][0]]:
                    if node_id_sorted_xy[i][0] == neighbor:
                        continue
                    for edge_index in dict_cell_edge[neighbor]:
                        set1_edges_index.append(edge_index) # has both row_col_gene and edge_weight
        
            set1_edges_index = list(set(set1_edges_index))

            #print('len of set1_edges_index %d'%len(set1_edges_index))
            #if len(set1_edges_index)==0:
            #    break

            # old to new mapping of the nodes
            # make an index array, so that existing node ids are mapped to new ids
            new_id = 0
            spot_list = []
            for k in set1_edges_index:
                i = row_col_gene[k][0]
                j = row_col_gene[k][1]
                if i not in id_map_old_new[set_id]:
                    id_map_old_new[set_id][i] = new_id
                    id_map_new_old[set_id][new_id] = i
                    spot_list.append(i) #new_id)
                    new_id = new_id + 1
        
                if j not in id_map_old_new[set_id]:
                    id_map_old_new[set_id][j] = new_id
                    id_map_new_old[set_id][new_id] = j
                    spot_list.append(j) #new_id)
                    new_id = new_id + 1
        
        
            #print('new id: %d'%new_id)
            set1_edges = []
            for i in set1_direct_edges:  #set1_edges_index:
                set1_edges.append([[id_map_old_new[set_id][row_col_gene[i][0]], id_map_old_new[set_id][row_col_gene[i][1]]], edge_weight[i]])
                #set1_edges.append([row_col_gene[i], edge_weight[i]])

            #edge_list.append(set1_edges)
            num_nodes = new_id
            row_col_gene_temp = []
            edge_weight_temp = []
            for i in range (0, len(set1_edges)):
                row_col_gene_temp.append(set1_edges[i][0])
                edge_weight_temp.append(set1_edges[i][1])
        
            print("subgraph %d: number of nodes %d, Total number of edges %d"%(set_id, num_nodes, len(row_col_gene_temp)))



            gc.collect()



        ######## embedding merge from subgraphs ########################################
        X_embedding = np.zeros((total_num_gene_node, 256)) # args.hidden_dimension
        for subgraph_id in range(0, args.total_subgraphs):
            X_embedding_filename =  args.embedding_path + args.model_name + '_r1_Embed_X' + '_subgraph'+str(subgraph_id)
            with gzip.open(X_embedding_filename, 'rb') as fp:  
                X_embedding_sub = pickle.load(fp)

            for old_id in suggraph_vs_gene_emb[subgraph_id]:
                new_id = id_map_old_new[subgraph_id][old_id]
                X_embedding[old_id, :] = X_embedding_sub[new_id, :]

            

        X_embedding_filename =  args.embedding_path + args.model_name + '_r1_Embed_X_subgraphs_combined' 
        with gzip.open(X_embedding_filename, 'wb') as fp:  
            pickle.dump(X_embedding, fp)


        for i in range (0, X_embedding.shape[0]):
            total_score_per_row = np.sum(X_embedding[i][:])
            X_embedding[i] = X_embedding[i]/total_score_per_row


            
        ############ attention scores ##############################
        attention_scores = defaultdict(dict)  
        for subgraph_id in range(0, args.total_subgraphs):
            layer = 3        
            distribution = []
            X_attention_filename = args.embedding_path +  args.model_name + '_r1_attention'+'_subgraph'+str(subgraph_id)
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
                
            percentage_value = 70
            th_70th = np.percentile(sorted(distribution), percentage_value) # higher attention score means stronger connection
            # Now keep only 
            
            for index in range (0, X_attention_bundle[0].shape[1]):
                i = X_attention_bundle[0][0][index]
                j = X_attention_bundle[0][1][index]
                scaled_score = (X_attention_bundle[layer][index][0]-min_value)/(max_value-min_value)
                if scaled_score >= 0.7: #th_70th:
                    old_i = id_map_new_old[subgraph_id][i]
                    old_j = id_map_new_old[subgraph_id][j]
                    attention_scores[old_i][old_j] = scaled_score


            
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
                print(i)
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
            data_list_pd.to_csv(args.output_path + args.model_name+'_allLR_nodeInfo.csv.gz', index=False, compression='gzip') #_negatome
            print(len(data_list_pd))

            if post_embFusion == 1:
                ccc_pairs = pd.read_csv(args.output_path + args.model_name+'_allLR_nodeInfo_LUAD_LYMPH_top20.csv.gz') #LUAD_LYMPH, LUAD_LYMPH_top20, LUADtraining_woNegatome
                lr_dict = defaultdict(list)
                for i in range(0, len(ccc_pairs['attention_score'])):
                    if ccc_pairs['pred_score'][i] <= 0: #< 0.7:
                        continue
                    
                    if ccc_pairs['attention_score'][i] < 0.7:
                        continue

                    #lr_dict[ccc_pairs['ligand_gene'][i]+'+'+ccc_pairs['rec_gene'][i]].append([ccc_pairs['score'][i], ccc_pairs['from_cell_index'], ccc_pairs['to_cell_index'], ccc_pairs['pred_score'][i], ccc_pairs['attention_score']])  # score, cell ids, gene_node ids   
                    lr_dict[ccc_pairs['ligand_gene'][i]+'+'+ccc_pairs['rec_gene'][i]].append([ccc_pairs['score'][i], ccc_pairs['from_cell_index'], ccc_pairs['to_cell_index'], ccc_pairs['attention_score'], -1])  # score, cell ids, gene_node ids   




            else:
                ccc_pairs = all_ccc_pairs
                lr_dict = defaultdict(list)
                for i in range(0, len(ccc_pairs['attention_score'])):
                    #if ccc_pairs['pred_score'][i] <= 0: #< 0.7:
                    #    continue
                    
                    if ccc_pairs['attention_score'][i] < 0.7:
                        continue

                    #lr_dict[ccc_pairs['ligand_gene'][i]+'+'+ccc_pairs['rec_gene'][i]].append([ccc_pairs['score'][i], ccc_pairs['from_cell_index'], ccc_pairs['to_cell_index'], ccc_pairs['pred_score'][i], ccc_pairs['attention_score']])  # score, cell ids, gene_node ids   
                    lr_dict[ccc_pairs['ligand_gene'][i]+'+'+ccc_pairs['rec_gene'][i]].append([ccc_pairs['score'][i], ccc_pairs['from_cell_index'], ccc_pairs['to_cell_index'], ccc_pairs['attention_score'], -1])  # score, cell ids, gene_node ids   


            #=====================================================================================
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
            ##############################################################################

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
            if post_embFusion == 1:
                data_list_pd.to_csv(args.output_path +args.model_name+'_lr_list_sortedBy_totalScore_top'+ file_name_suffix+'_allLR_predClass_LUAD_LYMPH_top20.csv', index=False) #_top20, LUAD_LYMPH, LUADtraining_woNegatome, LUADtraining_interNegatome
            else:
                data_list_pd.to_csv(args.output_path +args.model_name+'_lr_list_sortedBy_totalScore_top'+ file_name_suffix+'_allLR_wofilter.csv', index=False) #_negatome


##########################################################################################################
    parser = argparse.ArgumentParser()
    parser.add_argument( '--lr_lrbind_csv_path', type=str, 
                        default='/cluster/home/t116508uhn/LRbind_output/without_elbow_cut/LRbind_Xenium_Prime_Human_Skin_FFPE_manualDB_geneLocalCorrKNN_bidir/model_LRbind_Xenium_Prime_Human_Skin_FFPE_manualDB_geneLocalCorrKNN_bidir_allLR_nodeInfo.csv.gz', 
                        help='Name of the dataset') #, required=True)  #V1_Human_Lymph_Node_spatial_novelLR
    #parser.add_argument( '--barcode_info_cellnest_path', type=str, default='../NEST/metadata/LUAD_TD1_manualDB/LUAD_TD1_manualDB_barcode_info' , help='Path to the dataset to read from. Space Ranger outs/ folder is preferred. Otherwise, provide the *.mtx file of the gene expression matrix.') #,required=True) 
    parser.add_argument( '--barcode_info_gene_path', type=str, default='metadata/LRbind_Xenium_Prime_Human_Skin_FFPE_manualDB_geneLocalCorrKNN_bidir/LRbind_Xenium_Prime_Human_Skin_FFPE_manualDB_geneLocalCorrKNN_bidir_barcode_info_gene', help='Name of the dataset') 
    parser.add_argument( '--barcode_info_path', type=str, default='metadata/LRbind_Xenium_Prime_Human_Skin_FFPE_manualDB_geneLocalCorrKNN_bidir/LRbind_Xenium_Prime_Human_Skin_FFPE_manualDB_geneLocalCorrKNN_bidir_barcode_info', help='Name of the dataset')    
    #parser.add_argument( '--cell_emb_cellnest_path', type=str, default='../NEST/embedding_data/LUAD_TD1_manualDB/CellNEST_LUAD_TD1_manualDB_r1_Embed_X', help='Name of the dataset')
    parser.add_argument( '--gene_emb_path', type=str, default='embedding_data/LRbind_Xenium_Prime_Human_Skin_FFPE_manualDB_geneLocalCorrKNN_bidir/model_LRbind_Xenium_Prime_Human_Skin_FFPE_manualDB_geneLocalCorrKNN_bidir_r1_Embed_X', help='Name of the dataset')
    parser.add_argument( '--protein_emb_path', type=str, default='database/ligand_receptor_protein_embedding.pkl', help='Name of the dataset')
    args = parser.parse_args()

    #######################################################


    ccc_pairs = pd.read_csv(args.lr_lrbind_csv_path, sep=",")
    with gzip.open(args.barcode_info_path, 'rb') as fp:     
        barcode_info = pickle.load(fp)

    with gzip.open(args.barcode_info_gene_path, 'rb') as fp: 
        barcode_info_gene, na, na, gene_node_list_per_spot, na, na, na, na, na = pickle.load(fp)

    gene_node_list_per_spot_temp = defaultdict(dict)
    for cell_index in range(0, len(barcode_info)): 
        gene_node_list_per_spot_temp[barcode_info[cell_index][0]] = gene_node_list_per_spot[cell_index]
        
    gene_node_list_per_spot = gene_node_list_per_spot_temp 
    gene_node_list_per_spot_temp = 0
    gc.collect()
    
  
    X_gene_embedding = X_embedding

    gzip.open(args.protein_emb_path, 'rb') as fp:  
        X_protein_embedding = pickle.load(fp)

    #X_p = 
    
    #cell_vs_index = dict()
    #for i in range(0, len(barcode_info_cellnest)):
    #    cell_vs_index[barcode_info_cellnest[i][0]] = i

#    cell_vs_gene_emb = get_cellEmb_geneEmb_pairs(cell_vs_index, barcode_info_gene, X_embedding, X_gene_embedding, X_protein_embedding)
    cell_vs_gene_emb = get_cellEmb_geneEmb_pairs(barcode_info_gene, X_gene_embedding, X_protein_embedding)
    
    dataset, record_index = get_dataset(ccc_pairs, cell_vs_gene_emb, gene_node_list_per_spot, X_protein_embedding)
    print(len(dataset))
    # save it


    #with gzip.open('database/'+'LRbind_LUAD_1D_manualDB_geneCorrP7KNN_bidir'+'_dataset_results_to_embFusion.pkl', 'wb') as fp:
    #with gzip.open('database/'+'LRbind_LUAD_1D_manualDB_geneLocalCorrKNN_bidir_negatome'+'_dataset_results_to_embFusion.pkl', 'wb') as fp:
    #with gzip.open('database/'+'LRbind_V1_Human_Lymph_Node_spatial_1D_manualDB_geneLocalCorrKNN_bidir_prefiltered_tanh'+'_dataset_results_to_embFusion.pkl', 'wb') as fp:
    #with gzip.open('database/'+'Xenium_FFPE_Human_Breast_Cancer_Rep1_manualDB_geneLocalCorrKNN_bidir_full_tanh'+'_dataset_results_to_embFusion.pkl', 'wb') as fp:
    #with gzip.open('database/'+'Xenium_FFPE_Human_Breast_Cancer_Rep1_manualDB_geneLocalCorrKNN_bidir_removedLR_tanh'+'_dataset_results_to_embFusion.pkl', 'wb') as fp:
    with gzip.open('database/'+'Xenium_Prime_Human_Skin_FFPE_manualDB_geneLocalCorrKNN_bidir_tanh'+'_dataset_results_to_embFusion.pkl', 'wb') as fp:
    	pickle.dump([dataset, record_index], fp)
