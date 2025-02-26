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
    '''
    parser.add_argument( '--data_name', type=str, default='LRbind_PDAC_e2d1_64630_1D_manualDB', help='The name of dataset') #, required=True) # default='LRbind_V1_Human_Lymph_Node_spatial_1D_manualDB',
    parser.add_argument( '--model_name', type=str, default='model_LRbind_PDAC_e2d1_64630_1D_manualDB_dgi', help='Name of the trained model') #, required=True) 'LRbind_model_V1_Human_Lymph_Node_spatial_1D_manualDB'
    '''
    parser.add_argument( '--database_path', type=str, default='database/NEST_database.csv' , help='Provide your desired ligand-receptor database path here. Default database is a combination of CellChat and NicheNet database.')    
    parser.add_argument( '--data_name', type=str, default='LRbind_V1_Human_Lymph_Node_spatial_1D_manualDB_geneCorr', help='The name of dataset') #, required=True) # default='',
    parser.add_argument( '--model_name', type=str, default='model_LRbind_V1_Human_Lymph_Node_spatial_1D_manualDB_geneCorr', help='Name of the trained model') #, required=True) ''
    #_geneCorr_remFromDB
    
    parser.add_argument( '--total_runs', type=int, default=3, help='How many runs for ensemble (at least 2 are preferred)') #, required=True)
    #######################################################################################################
    parser.add_argument( '--embedding_path', type=str, default='embedding_data/', help='Path to grab the attention scores from')
    parser.add_argument( '--metadata_from', type=str, default='metadata/', help='Path to grab the metadata') 
    parser.add_argument( '--data_from', type=str, default='input_graph/', help='Path to grab the input graph from (to be passed to GAT)')
    parser.add_argument( '--output_path', type=str, default='/cluster/home/t116508uhn/LRbind_output/', help='Path to save the visualization results, e.g., histograms, graph etc.')
    parser.add_argument( '--top_percent', type=int, default=20, help='Top N percentage communications to pick')
    parser.add_argument( '--cutoff_MAD', type=int, default=-1, help='Set it to 1 to filter out communications having deviation higher than MAD')
    parser.add_argument( '--cutoff_z_score', type=float, default=-1, help='Set it to 1 to filter out communications having z_score less than 1.97 value')
    parser.add_argument( '--output_all', type=int, default=1, help='Set it to 1 to output all communications')
    args = parser.parse_args()

    args.metadata_from = args.metadata_from + args.data_name + '/'
    args.data_from = args.data_from + args.data_name + '/'
    args.embedding_path  = args.embedding_path + args.data_name + '/'
    args.output_path = args.output_path + args.data_name + '/'
    if not os.path.exists(args.output_path):
        os.makedirs(args.output_path)
        
    
##################### get metadata: barcode_info ###################################

    with gzip.open(args.metadata_from +args.data_name+'_barcode_info', 'rb') as fp:  #b, a:[0:5]   _filtered
        barcode_info = pickle.load(fp) 

    barcode_index = dict()
    for i in range (0, len(barcode_info)):
        barcode_index[barcode_info[i][0]] = i

    Tcell_zone = []
    df = pd.read_csv("../NEST/data/V1_Human_Lymph_Node_spatial_annotation.csv", sep=",")
    for i in range (0, df["Barcode"].shape[0]):
        if df["Type"][i] == 'T-cell':
            Tcell_zone.append(barcode_index[df["Barcode"][i]])

    
        
    
    with gzip.open(args.metadata_from +args.data_name+'_barcode_info_gene', 'rb') as fp:  #b, a:[0:5]   _filtered
        barcode_info_gene, ligand_list, receptor_list, gene_node_list_per_spot, dist_X, l_r_pair = pickle.load(fp)
    
    with gzip.open(args.metadata_from + args.data_name +'_test_set', 'rb') as fp:  
        target_LR_index, target_cell_pair = pickle.load(fp)

    ############# load output graph #################################################
    args.model_name = args.model_name + '_r1'
    X_embedding_filename =  args.embedding_path + args.model_name + '_Embed_X' #.npy
    with gzip.open(X_embedding_filename, 'rb') as fp:  
        X_embedding = pickle.load(fp)

    '''
    found_list = dict()
    input_cell_pair_list = dict() 
    top_N = 30
    for LR_target in target_cell_pair.keys():
        ligand = LR_target.split('+')[0]
        receptor = LR_target.split('+')[1]
        pair_list = target_cell_pair[LR_target]
        for pair in pair_list:
            i = pair[0]
            j = pair[1]
            input_cell_pair_list[i] = 1
            input_cell_pair_list[j] = 1
            ligand_node_index = []
            for gene in gene_node_list_per_spot[i]:
                if gene in ligand_list:
                    ligand_node_index.append([gene_node_list_per_spot[i][gene], gene])

            receptor_node_index = []
            for gene in gene_node_list_per_spot[j]:
                if gene in receptor_list:
                    receptor_node_index.append([gene_node_list_per_spot[j][gene], gene])

            dot_prod_list = []
            for i_gene in ligand_node_index:
                for j_gene in receptor_node_index:
                    if i_gene[1] in l_r_pair and j_gene[1] in l_r_pair[i_gene[1]]: # discard the existing ones
                        continue
                    dot_prod_list.append([np.dot(X_embedding[i_gene[0]], X_embedding[j_gene[0]]), i, j, i_gene[1], j_gene[1]])
                
            #dot_prod_list = sorted(dot_prod_list, key = lambda x: x[0], reverse=True)[0:top_N]    
            ########## knee find ###########
            score_list = []
            for item in dot_prod_list:
                score_list.append(item[0])

            score_list = sorted(score_list) # small to high
            y = score_list
            x = range(1, len(y)+1)
            kn = KneeLocator(x, y, curve='convex', direction='increasing')
            kn_value = y[kn.knee-1]    
            temp_dot_prod_list = []
            for item in dot_prod_list:
                if item[0] >= kn_value:
                    temp_dot_prod_list.append(item)

            dot_prod_list = temp_dot_prod_list
            ###########################
            for item in dot_prod_list:
                #print(item)
                if item[3] == ligand and item[4] == receptor:
                    found_list[i] = 1
                    found_list[j] = 1
                    break

        # plot found_list
        print("positive: %d out of %d"%(len(found_list), len(input_cell_pair_list)))
        # plot input_cell_pair_list
        
    ######### plot output #############################
        data_list=dict()
        data_list['X']=[]
        data_list['Y']=[]   
        data_list['gene_expression']=[] 
        
        for i in range (0, len(barcode_info)):
            data_list['X'].append(barcode_info[i][1])
            data_list['Y'].append(-barcode_info[i][2])
            if i in found_list:
                data_list['gene_expression'].append(1)
            else:
                data_list['gene_expression'].append(0)
        
        source= pd.DataFrame(data_list)
        
        chart = alt.Chart(source).mark_point(filled=True).encode(
            alt.X('X', scale=alt.Scale(zero=False)),
            alt.Y('Y', scale=alt.Scale(zero=False)),
            color=alt.Color('gene_expression:Q', scale=alt.Scale(scheme='magma'))
        )
        chart.save(args.output_path + args.model_name + '_output_' + ligand + '-' + receptor +'_top'+ str(top_N)  + '_novel.html')
        print(args.output_path + args.model_name + '_output_' + ligand + '-' + receptor +'_top'+ str(top_N)  + '_novel.html')
    ##################### plot input ###########################
    
        data_list=dict()
        data_list['X']=[]
        data_list['Y']=[]   
        data_list['gene_expression']=[] 
        
        for i in range (0, len(barcode_info)):
            data_list['X'].append(barcode_info[i][1])
            data_list['Y'].append(-barcode_info[i][2])
            if i in input_cell_pair_list:
                data_list['gene_expression'].append(1)
            else:
                data_list['gene_expression'].append(0)
        
        source= pd.DataFrame(data_list)
        
        chart = alt.Chart(source).mark_point(filled=True).encode(
            alt.X('X', scale=alt.Scale(zero=False)),
            alt.Y('Y', scale=alt.Scale(zero=False)),
            color=alt.Color('gene_expression:Q', scale=alt.Scale(scheme='magma'))
        )
        chart.save(args.output_path + args.model_name + '_input_' + ligand + '-' + receptor +'.html')
        print(args.output_path + args.model_name + '_input_' + ligand + '-' + receptor +'.html')
######################################################
    with gzip.open(args.data_from + args.data_name + '_adjacency_gene_records_1D', 'rb') as fp:  
        row_col_gene, edge_weight, lig_rec, gene_node_type, gene_node_expression, total_num_gene_node = pickle.load(fp)

    for i in range (0, len(lig_rec)):
        lr = lig_rec[i]
        if lr[0]=='CCL19' or lr[1]=='CCR7':
            print('found %d'%i)
            break
    '''
########## all ############################################# 
    top_lrp_count = 5000
    knee_flag = 0
    break_flag = 0
    for top_N in [100, 30, 10]:
        if break_flag == 1:  
            break
        if knee_flag == 1:
            top_N = 0
            break_flag = 1
        lr_dict = defaultdict(list)
        Tcell_zone_lr_dict = defaultdict(list)
        target_ligand = 'CCL19'
        target_receptor = 'CCR7'
        found_list = defaultdict(list)
        test_mode = 1
        for i in range (0, len(barcode_info)):
            for j in range (0, len(barcode_info)):
                
                if dist_X[i][j]==0 or i==j :
                    continue
                # from i to j
                ligand_node_index = []
                for gene in gene_node_list_per_spot[i]:
                    if gene in ligand_list:
                        ligand_node_index.append([gene_node_list_per_spot[i][gene], gene])
    
                receptor_node_index = []
                for gene in gene_node_list_per_spot[j]:
                    if gene in receptor_list:
                        receptor_node_index.append([gene_node_list_per_spot[j][gene], gene])
    
                dot_prod_list = []
                product_only = []
                for i_gene in ligand_node_index:
                    for j_gene in receptor_node_index:
                        if i_gene[1]==j_gene[1]:
                            continue
                        temp = np.dot(X_embedding[i_gene[0]], X_embedding[j_gene[0]])
                        dot_prod_list.append([temp, i, j, i_gene[1], j_gene[1]])
                        product_only.append(temp)

                if len(dot_prod_list) == 0:
                    continue
                # scale 
                
                max_prod = np.max(product_only)
                min_prod = np.min(product_only)
                for item_idx in range (0, len(dot_prod_list)):
                    scaled_prod = (dot_prod_list[item_idx][0]-min_prod)/(max_prod-min_prod)
                    dot_prod_list[item_idx][0] = scaled_prod
                
                if knee_flag == 0:
                    dot_prod_list = sorted(dot_prod_list, key = lambda x: x[0], reverse=True)[0:top_N]
                else:
                    ########## knee find ###########
                    score_list = []
                    for item in dot_prod_list:
                        score_list.append(item[0])
        
                    score_list = sorted(score_list) # small to high
                    y = score_list
                    x = range(1, len(y)+1)
                    kn = KneeLocator(x, y, direction='increasing')
                    kn_value_inc = y[kn.knee-1]
                    kn = KneeLocator(x, y, direction='decreasing')
                    kn_value_dec = y[kn.knee-1]            
                    kn_value = max(kn_value_inc, kn_value_dec)
                    
                    temp_dot_prod_list = []
                    for item in dot_prod_list:
                        if item[0] >= kn_value:
                            temp_dot_prod_list.append(item)
        
                    dot_prod_list = temp_dot_prod_list
                ###########################
                for item in dot_prod_list:
                    lr_dict[item[3]+'+'+item[4]].append([item[0], item[1], item[2]])
                    
                    if i in Tcell_zone and j in Tcell_zone:
                        Tcell_zone_lr_dict[item[3]+'+'+item[4]].append([item[0], item[1], item[2]])
                        
                    if test_mode == 1 and item[3] == target_ligand and item[4] == target_receptor:
                        found_list[i].append(item[0]) #= 1
                        found_list[j].append(item[0])
                        break
    
        # plot found_list
        print("positive: %d"%(len(found_list)))                
        # plot input_cell_pair_list  
        if test_mode==1:
        ######### plot output #############################
            # UPDATE # annottaion
            data_list=dict()
            data_list['X']=[]
            data_list['Y']=[]   
            data_list['total count']=[] 
            for i in range (0, len(barcode_info)):
                data_list['X'].append(barcode_info[i][1])
                data_list['Y'].append(-barcode_info[i][2])
                if i in found_list:
                    data_list['total count'].append(np.sum(found_list[i]))
                else:
                    data_list['total count'].append(0)
            
            source= pd.DataFrame(data_list)
            
            chart = alt.Chart(source).mark_point(filled=True).encode(
                alt.X('X', scale=alt.Scale(zero=False)),
                alt.Y('Y', scale=alt.Scale(zero=False)),
                color=alt.Color('total count:Q', scale=alt.Scale(scheme='magma'))
            )
            chart.save(args.output_path + args.model_name + '_output_' + target_ligand + '-' + target_receptor +'_top'+ str(top_N)  + '_wholeTissue_allLR.html')
            print(args.output_path + args.model_name + '_output_' + target_ligand + '-' + target_receptor +'_top'+ str(top_N)  + '_wholeTissue_allLR.html')    
        # save lr_dict that has info about gene node id as well
    
        ########## take top hits #################################### 
        #if top_N == 30:
        #    continue
        sort_lr_list = []
        for lr_pair in lr_dict:
            sum = 0
            cell_pair_list = lr_dict[lr_pair]
            for item in cell_pair_list:
                sum = sum + item[0]  
    
            sort_lr_list.append([lr_pair, sum])
    
        sort_lr_list = sorted(sort_lr_list, key = lambda x: x[1], reverse=True)
        print('len sort_lr_list %d'%len(sort_lr_list))
        # save = num_spots/cells * top_N pairs
        if knee_flag == 0:
            sort_lr_list = sort_lr_list[0: top_lrp_count]

        
        top_hit_lrp_dict = dict()
        for item in sort_lr_list:
            top_hit_lrp_dict[item[0]] = ''
        
        # now plot the histograms where X axis will show the name or LR pair and Y axis will show the score.
        data_list=dict()
        data_list['X']=[]
        data_list['Y']=[] 
        max_rows = min(500, len(sort_lr_list))
        for i in range (0, max_rows): #1000): #:
            data_list['X'].append(sort_lr_list[i][0])
            data_list['Y'].append(sort_lr_list[i][1])
            
        data_list_pd = pd.DataFrame({
            'Ligand-Receptor Pairs': data_list['X'],
            'Total Count': data_list['Y']
        })
        data_list_pd.to_csv(args.output_path +args.model_name+'_novel_lr_list_sortedBy_totalScore_top'+str(top_N)+'allLR.csv', index=False)
        print(args.output_path +args.model_name+'_novel_lr_list_sortedBy_totalScore_top'+str(top_N)+'allLR.csv')    
        # same as histogram plots
        chart = alt.Chart(data_list_pd).mark_bar().encode(
            x=alt.X("Ligand-Receptor Pairs:N", axis=alt.Axis(labelAngle=45), sort='-y'),
            y='Total Count'
        )
    
        chart.save(args.output_path +args.model_name+'_novel_lr_list_sortedBy_totalScore_top'+str(top_N)+'_histogramsallLR.html')
        print(args.output_path +args.model_name+'_novel_lr_list_sortedBy_totalScore_top'+str(top_N)+'_histogramsallLR.html')   
        ############################### novel only out of all LR ################
        sort_lr_list_temp = []
        for pair in sort_lr_list:                
            ligand = pair[0].split('+')[0]
            receptor = pair[0].split('+')[1]
            if ligand in l_r_pair and receptor in l_r_pair[ligand]:
                continue
                
            sort_lr_list_temp.append(pair) 
            
        print('novel LRP length %d out of top %d LRP'%(len(sort_lr_list_temp), top_lrp_count))
        # now plot the histograms where X axis will show the name or LR pair and Y axis will show the score.
        data_list=dict()
        data_list['X']=[]
        data_list['Y']=[] 
    
        max_rows = min(500, len(sort_lr_list_temp))
        for i in range (0, max_rows): #1000): #
            data_list['X'].append(sort_lr_list_temp[i][0])
            data_list['Y'].append(sort_lr_list_temp[i][1])
            
        data_list_pd = pd.DataFrame({
            'Ligand-Receptor Pairs': data_list['X'],
            'Total Count': data_list['Y']
        })
        data_list_pd.to_csv(args.output_path +args.model_name+'_novel_lr_list_sortedBy_totalScore_top'+str(top_N)+'_novelsOutOfallLR.csv', index=False)
        print(args.output_path +args.model_name+'_novel_lr_list_sortedBy_totalScore_top'+str(top_N)+'allLR.csv')    
        # same as histogram plots
        chart = alt.Chart(data_list_pd).mark_bar().encode(
            x=alt.X("Ligand-Receptor Pairs:N", axis=alt.Axis(labelAngle=45), sort='-y'),
            y='Total Count'
        )
    
        chart.save(args.output_path +args.model_name+'_novel_lr_list_sortedBy_totalScore_top'+str(top_N)+'_histograms_novelsOutOfallLR.html')
        print(args.output_path +args.model_name+'_novel_lr_list_sortedBy_totalScore_top'+str(top_N)+'_histograms_novelsOutOfallLR.html')   
        ################################# when not remFromDB ##########################################################################################################
        
        set_LRbind_novel = []
        for i in range (0, len(sort_lr_list_temp)):
            set_LRbind_novel.append(sort_lr_list_temp[i][0])
    
        print('ligand-receptor database reading.')
        df = pd.read_csv(args.database_path, sep=",")
        set_nichenet_novel = []
        for i in range (0, df["Ligand"].shape[0]):
            ligand = df["Ligand"][i] 
            receptor = df["Receptor"][i]
            if ligand in ligand_list and receptor in receptor_list and 'ppi' in df["Reference"][i]:
                set_nichenet_novel.append(ligand + '+' + receptor)
    
        set_nichenet_novel = np.unique(set_nichenet_novel)
        common_lr = list(set(set_LRbind_novel) & set(set_nichenet_novel))
        print('top_N:%d, Only LRbind %d, only nichenet %d, common %d'%(top_N, len(set_LRbind_novel), len(set_nichenet_novel)-len(common_lr), len(common_lr)))
        
    ##################################  ccl19 and ccr7 related #################
        '''
        print('top_N: %d'%top_N)
        set_LRbind_novel = []
        data_list=dict()
        data_list['X']=[]
        data_list['Y']=[] 
        for i in range (0, len(sort_lr_list_temp)):
            ligand = sort_lr_list_temp[i][0].split('+')[0]
            receptor =  sort_lr_list_temp[i][0].split('+')[1]
            if ligand == 'CCL19' or receptor == 'CCR7':
                set_LRbind_novel.append(sort_lr_list_temp[i][0])
                data_list['X'].append(sort_lr_list_temp[i][0])
                data_list['Y'].append(sort_lr_list_temp[i][1])
    
        set_LRbind_novel = np.unique(set_LRbind_novel)
        # plot set_LRbind_nov
        # now plot the histograms where X axis will show the name or LR pair and Y axis will show the score.
        data_list_pd = pd.DataFrame({
            'Ligand-Receptor Pairs': data_list['X'],
            'Total Count': data_list['Y']
        })
        #data_list_pd.to_csv(args.output_path +args.model_name+'_novel_lr_list_sortedBy_totalScore_top'+str(top_N)+'_novelccl19ccr7OutOfallLR.csv', index=False)
        #print(args.output_path +args.model_name+'_novel_lr_list_sortedBy_totalScore_top'+str(top_N)+'_novelccl19ccr7OutOfallLR.csv')    
        # same as histogram plots
        chart = alt.Chart(data_list_pd).mark_bar().encode(
            x=alt.X("Ligand-Receptor Pairs:N", axis=alt.Axis(labelAngle=45), sort='-y'),
            y='Total Count'
        )
        #chart.save(args.output_path +args.model_name+'_novel_lr_list_sortedBy_totalScore_top'+str(top_N)+'_histograms_novelccl19ccr7OutOfallLR.html')
        #print(args.output_path +args.model_name+'_novel_lr_list_sortedBy_totalScore_top'+str(top_N)+'_histograms_novelccl19ccr7OutOfallLR.html')   
        
        df = pd.read_csv(args.database_path, sep=",")
        set_manual = []
        for i in range (0, df["Ligand"].shape[0]):
            ligand = df["Ligand"][i] 
            receptor = df["Receptor"][i]
            if ((ligand==target_ligand and receptor in receptor_list) or (receptor == target_receptor and ligand in ligand_list)) and ('ppi' not in df["Reference"][i]):
                set_manual.append(ligand + '+' + receptor)
                
        set_manual = np.unique(set_manual)
        common_lr = list(set(set_LRbind_novel) & set(set_manual))
        print('CCL19/CCR7 related: Only LRbind %d, only manual %d, common %d'%(len(set_LRbind_novel), len(set_manual)-len(common_lr), len(common_lr)))
        '''
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
        print('Only LRbind %d, only manual %d, common %d'%(len(set_LRbind_novel), len(set_nichenet_novel)-len(common_lr), len(common_lr)))
        '''
         ############ only Tcell Zone plot ##############################################################################################################################
        Tcell_zone_sort_lr_list = []
        for lr_pair in Tcell_zone_lr_dict:
            if lr_pair not in top_hit_lrp_dict:
                continue
            sum = 0
            cell_pair_list = Tcell_zone_lr_dict[lr_pair]
            for item in cell_pair_list:
                sum = sum + item[0] # 
    
            Tcell_zone_sort_lr_list.append([lr_pair, sum])
    
        Tcell_zone_sort_lr_list = sorted(Tcell_zone_sort_lr_list, key = lambda x: x[1], reverse=True)
        
        # now plot the histograms where X axis will show the name or LR pair and Y axis will show the score.
        data_list=dict()
        data_list['X']=[]
        data_list['Y']=[] 
        max_rows = min(500, len(Tcell_zone_sort_lr_list))
        for i in range (0, max_rows): #1000): #:
            data_list['X'].append(Tcell_zone_sort_lr_list[i][0])
            data_list['Y'].append(Tcell_zone_sort_lr_list[i][1])
            
        data_list_pd = pd.DataFrame({
            'Ligand-Receptor Pairs': data_list['X'],
            'Total Count': data_list['Y']
        })
        data_list_pd.to_csv(args.output_path +args.model_name+'_novel_lr_list_sortedBy_totalScore_top'+str(top_N)+'Tcell_zone_allLR.csv', index=False)
        print(args.output_path +args.model_name+'_novel_lr_list_sortedBy_totalScore_top'+str(top_N)+'Tcell_zone_allLR.csv')    
        # same as histogram plots
        chart = alt.Chart(data_list_pd).mark_bar().encode(
            x=alt.X("Ligand-Receptor Pairs:N", axis=alt.Axis(labelAngle=45), sort='-y'),
            y='Total Count'
        )
    
        chart.save(args.output_path +args.model_name+'_novel_lr_list_sortedBy_totalScore_top'+str(top_N)+'Tcell_zone_histogramsallLR.html')
        print(args.output_path +args.model_name+'_novel_lr_list_sortedBy_totalScore_top'+str(top_N)+'Tcell_zone_histogramsallLR.html')   
        ############################### novel only out of all LR ################
        '''
        Tcell_zone_sort_lr_list = []
        for lr_pair in Tcell_zone_lr_dict:
            if lr_pair not in top_hit_lrp_dict:
                continue
            
            ligand = lr_pair.split('+')[0]
            receptor = lr_pair.split('+')[1]
            if ligand in l_r_pair and receptor in l_r_pair[ligand]:
                continue
            sum = 0
            cell_pair_list = Tcell_zone_lr_dict[lr_pair]
            for item in cell_pair_list:
                sum = sum + item[0] # 
    
            Tcell_zone_sort_lr_list.append([lr_pair, sum])
    
        Tcell_zone_sort_lr_list = sorted(Tcell_zone_sort_lr_list, key = lambda x: x[1], reverse=True)
        
        # now plot the histograms where X axis will show the name or LR pair and Y axis will show the score.
        data_list=dict()
        data_list['X']=[]
        data_list['Y']=[] 
    
        max_rows = min(500, len(Tcell_zone_sort_lr_list))
        for i in range (0, max_rows): #1000): #
            data_list['X'].append(Tcell_zone_sort_lr_list[i][0])
            data_list['Y'].append(Tcell_zone_sort_lr_list[i][1])
            
        data_list_pd = pd.DataFrame({
            'Ligand-Receptor Pairs': data_list['X'],
            'Total Count': data_list['Y']
        })
        #data_list_pd.to_csv(args.output_path +args.model_name+'_novel_lr_list_sortedBy_totalScore_top'+str(top_N)+'_Tcell_zone_novelsOutOfallLR.csv', index=False)
        #print(args.output_path +args.model_name+'_novel_lr_list_sortedBy_totalScore_top'+str(top_N)+'_Tcell_zone_novelsOutOfallLR.csv')    
        # same as histogram plots
        chart = alt.Chart(data_list_pd).mark_bar().encode(
            x=alt.X("Ligand-Receptor Pairs:N", axis=alt.Axis(labelAngle=45), sort='-y'),
            y='Total Count'
        )
    
        #chart.save(args.output_path +args.model_name+'_novel_lr_list_sortedBy_totalScore_top'+str(top_N)+'_histograms_Tcell_zone_novelsOutOfallLR.html')
        #print(args.output_path +args.model_name+'_novel_lr_list_sortedBy_totalScore_top'+str(top_N)+'_histograms_Tcell_zone_novelsOutOfallLR.html')       
        '''
    ########## novel only ############################################# ###########################################################################################  
        '''
        lr_dict = defaultdict(list)
        target_ligand = 'CCL19'
        target_receptor = 'CCR7'
        found_list = defaultdict(list)
        test_mode = 1
        for i in range (0, len(barcode_info)):
            for j in range (0, len(barcode_info)):
                
                if dist_X[i][j]==0 or i==j:
                    continue
                # from i to j
                ligand_node_index = []
                for gene in gene_node_list_per_spot[i]:
                    if gene in ligand_list:
                        ligand_node_index.append([gene_node_list_per_spot[i][gene], gene])
    
                receptor_node_index = []
                for gene in gene_node_list_per_spot[j]:
                    if gene in receptor_list:
                        receptor_node_index.append([gene_node_list_per_spot[j][gene], gene])
    
                dot_prod_list = []
                for i_gene in ligand_node_index:
                    for j_gene in receptor_node_index:
                        if i_gene[1] in l_r_pair and j_gene[1] in l_r_pair[i_gene[1]]: # discard the existing ones
                            continue
                        dot_prod_list.append([np.dot(X_embedding[i_gene[0]], X_embedding[j_gene[0]]), i, j, i_gene[1], j_gene[1]])
    
                
                if knee_flag == 0:
                    dot_prod_list = sorted(dot_prod_list, key = lambda x: x[0], reverse=True)[0:top_N]
                else:
                    if len(dot_prod_list) == 0:
                        continue
                    ########## knee find ###########
                    score_list = []
                    for item in dot_prod_list:
                        score_list.append(item[0])
        
                    score_list = sorted(score_list) # small to high
                    y = score_list
                    x = range(1, len(y)+1)
        
                    kn = KneeLocator(x, y, direction='increasing')
                    kn_value_inc = y[kn.knee-1]
                    kn = KneeLocator(x, y, direction='decreasing')
                    kn_value_dec = y[kn.knee-1]            
                    kn_value = max(kn_value_inc, kn_value_dec)
                    
                    temp_dot_prod_list = []
                    for item in dot_prod_list:
                        if item[0] >= kn_value:
                            temp_dot_prod_list.append(item)
        
                    dot_prod_list = temp_dot_prod_list
                ###########################
                for item in dot_prod_list:
                    lr_dict[item[3]+'+'+item[4]].append([item[0], item[1], item[2]])
                    if test_mode == 1 and item[3] == target_ligand and item[4] == target_receptor:
                        found_list[i].append(item[0]) #= 1
                        found_list[j].append(item[0])
                        break
    
        # plot found_list
        print("positive: %d"%(len(found_list)))
        # plot input_cell_pair_list  
        if test_mode==1:
        ######### plot output #############################
            data_list=dict()
            data_list['X']=[]
            data_list['Y']=[]   
            data_list['total count']=[] 
            for i in range (0, len(barcode_info)):
                data_list['X'].append(barcode_info[i][1])
                data_list['Y'].append(-barcode_info[i][2])
                if i in found_list:
                    data_list['total count'].append(np.sum(found_list[i]))
                else:
                    data_list['total count'].append(0)
            
            source= pd.DataFrame(data_list)
            
            chart = alt.Chart(source).mark_point(filled=True).encode(
                alt.X('X', scale=alt.Scale(zero=False)),
                alt.Y('Y', scale=alt.Scale(zero=False)),
                color=alt.Color('total count:Q', scale=alt.Scale(scheme='magma'))
            )
            chart.save(args.output_path + args.model_name + '_output_' + target_ligand + '-' + target_receptor +'_top'+ str(top_N)  + '_wholeTissue.html')
            print(args.output_path + args.model_name + '_output_' + target_ligand + '-' + target_receptor +'_top'+ str(top_N)  + '_wholeTissue.html')    
        # save lr_dict that has info about gene node id as well
    
        ########## take top hits #################################### 
        sort_lr_list = []
        for lr_pair in lr_dict:
            sum = 0
            cell_pair_list = lr_dict[lr_pair]
            for item in cell_pair_list:
                sum = sum + 1 #item[0] # 
    
            sort_lr_list.append([lr_pair, sum])
    
        sort_lr_list = sorted(sort_lr_list, key = lambda x: x[1], reverse=True)
    
        # save = num_spots/cells * top_N pairs
        if knee_flag == 0:
            sort_lr_list = sort_lr_list[0: top_lrp_count]
    
        # now plot the histograms where X axis will show the name or LR pair and Y axis will show the score.
        data_list=dict()
        data_list['X']=[]
        data_list['Y']=[] 
        max_rows = min(500, len(sort_lr_list))
        for i in range (0, max_rows): #1000): #:
            data_list['X'].append(sort_lr_list[i][0])
            data_list['Y'].append(sort_lr_list[i][1])
            
        data_list_pd = pd.DataFrame({
            'Ligand-Receptor Pairs': data_list['X'],
            'Total Count': data_list['Y']
        })
        data_list_pd.to_csv(args.output_path +args.model_name+'_novel_lr_list_sortedBy_totalScore_top'+str(top_N)+'.csv', index=False)
        print(args.output_path +args.model_name+'_novel_lr_list_sortedBy_totalScore_top'+str(top_N)+'.csv')    
        # same as histogram plots
        chart = alt.Chart(data_list_pd).mark_bar().encode(
            x=alt.X("Ligand-Receptor Pairs:N", axis=alt.Axis(labelAngle=45), sort='-y'),
            y='Total Count'
        )
    
        chart.save(args.output_path +args.model_name+'_novel_lr_list_sortedBy_totalScore_top'+str(top_N)+'_histograms.html')
        print(args.output_path +args.model_name+'_novel_lr_list_sortedBy_totalScore_top'+str(top_N)+'_histograms.html')    
        
        ##################### if not remFromDB #############################################
        
        set_LRbind_novel = []
        for i in range (0, len(sort_lr_list)):
            set_LRbind_novel.append(sort_lr_list[i][0])
    
        print('ligand-receptor database reading.')
        df = pd.read_csv(args.database_path, sep=",")
        set_nichenet_novel = [] 
        for i in range (0, df["Ligand"].shape[0]):
            ligand = df["Ligand"][i] 
            receptor = df["Receptor"][i]
            if ligand in ligand_list and receptor in receptor_list and 'ppi' in df["Reference"][i]:
                set_nichenet_novel.append(ligand + '+' + receptor)
    
        set_nichenet_novel = np.unique(set_nichenet_novel)
        common_lr = list(set(set_LRbind_novel) & set(set_nichenet_novel))
        print('top_N:%d, Only LRbind %d, only nichenet %d, common %d'%(top_N, len(set_LRbind_novel), len(set_nichenet_novel)-len(common_lr), len(common_lr)))
        '''
    ################################## remFromDB #################
        '''
        print('top_N: %d'%top_N)
        set_LRbind_novel = []
        data_list=dict()
        data_list['X']=[]
        data_list['Y']=[] 
        for i in range (0, len(sort_lr_list)):
            ligand = sort_lr_list[i][0].split('+')[0]
            receptor =  sort_lr_list[i][0].split('+')[1]
            if ligand == 'CCL19' or receptor == 'CCR7':
                set_LRbind_novel.append(sort_lr_list[i][0])
                data_list['X'].append(sort_lr_list[i][0])
                data_list['Y'].append(sort_lr_list[i][1])
    
        set_LRbind_novel = np.unique(set_LRbind_novel)
        # plot set_LRbind_nov
        # now plot the histograms where X axis will show the name or LR pair and Y axis will show the score.
        data_list_pd = pd.DataFrame({
            'Ligand-Receptor Pairs': data_list['X'],
            'Total Count': data_list['Y']
        })
        data_list_pd.to_csv(args.output_path +args.model_name+'_novel_lr_list_sortedBy_totalScore_top'+str(top_N)+'_novelccl19ccr7.csv', index=False)
        print(args.output_path +args.model_name+'_novel_lr_list_sortedBy_totalScore_top'+str(top_N)+'_novelccl19ccr7.csv')    
        # same as histogram plots
        chart = alt.Chart(data_list_pd).mark_bar().encode(
            x=alt.X("Ligand-Receptor Pairs:N", axis=alt.Axis(labelAngle=45), sort='-y'),
            y='Total Count'
        )
        chart.save(args.output_path +args.model_name+'_novel_lr_list_sortedBy_totalScore_top'+str(top_N)+'_histograms_novelccl19ccr7.html')
        print(args.output_path +args.model_name+'_novel_lr_list_sortedBy_totalScore_top'+str(top_N)+'_histograms_novelccl19ccr7.html')   
    
    ##############################  remFromDB #####################
    
        print('ligand-receptor database reading.')
        df = pd.read_csv(args.database_path, sep=",")
        set_manual = []
        for i in range (0, df["Ligand"].shape[0]):
            ligand = df["Ligand"][i] 
            receptor = df["Receptor"][i]
            if ((ligand==target_ligand and receptor in receptor_list) or (receptor == target_receptor and ligand in ligand_list)) and ('ppi' not in df["Reference"][i]):
                set_manual.append(ligand + '+' + receptor)
                
        set_manual = np.unique(set_manual)
        common_lr = list(set(set_LRbind_novel) & set(set_manual))
        print('ccl19 and ccr7 related: Only LRbind %d, only manual %d, common %d'%(len(set_LRbind_novel), len(set_manual)-len(common_lr), len(common_lr)))
        '''

    ###########################only Tcell Zone ###############################################################################################
         ############ only Tcell Zone plot ##############################################################################################################################
       ############ only Tcell Zone plot ##############################################################################################################################
        Tcell_zone_sort_lr_list = []
        for lr_pair in Tcell_zone_lr_dict:
            sum = 0
            cell_pair_list = Tcell_zone_lr_dict[lr_pair]
            for item in cell_pair_list:
                sum = sum + 1 #item[0] # 
    
            Tcell_zone_sort_lr_list.append([lr_pair, sum])
    
        Tcell_zone_sort_lr_list = sorted(Tcell_zone_sort_lr_list, key = lambda x: x[1], reverse=True)
    
        # save = num_spots/cells * top_N pairs
        if knee_flag == 0:
            Tcell_zone_sort_lr_list = Tcell_zone_sort_lr_list[0: top_lrp_count]
    
        # now plot the histograms where X axis will show the name or LR pair and Y axis will show the score.
        data_list=dict()
        data_list['X']=[]
        data_list['Y']=[] 
        max_rows = min(500, len(Tcell_zone_sort_lr_list))
        for i in range (0, max_rows): #1000): #:
            data_list['X'].append(Tcell_zone_sort_lr_list[i][0])
            data_list['Y'].append(Tcell_zone_sort_lr_list[i][1])
            
        data_list_pd = pd.DataFrame({
            'Ligand-Receptor Pairs': data_list['X'],
            'Total Count': data_list['Y']
        })
        data_list_pd.to_csv(args.output_path +args.model_name+'_novel_lr_list_sortedBy_totalScore_top'+str(top_N)+'Tcell_zone.csv', index=False)
        print(args.output_path +args.model_name+'_novel_lr_list_sortedBy_totalScore_top'+str(top_N)+'Tcell_zone.csv')    
        # same as histogram plots
        chart = alt.Chart(data_list_pd).mark_bar().encode(
            x=alt.X("Ligand-Receptor Pairs:N", axis=alt.Axis(labelAngle=45), sort='-y'),
            y='Total Count'
        )
    
        chart.save(args.output_path +args.model_name+'_novel_lr_list_sortedBy_totalScore_top'+str(top_N)+'Tcell_zone_histogram.html')
        print(args.output_path +args.model_name+'_novel_lr_list_sortedBy_totalScore_top'+str(top_N)+'Tcell_zone_histograms.html')   
        ############################### novel only out of all LR ################
        Tcell_zone_sort_lr_list = []
        for lr_pair in Tcell_zone_lr_dict:
            ligand = lr_pair.split('+')[0]
            receptor = lr_pair.split('+')[1]
            if ligand in l_r_pair and receptor in l_r_pair[ligand]:
                continue
            sum = 0
            cell_pair_list = Tcell_zone_lr_dict[lr_pair]
            for item in cell_pair_list:
                sum = sum + 1 #item[0] # 
    
            Tcell_zone_sort_lr_list.append([lr_pair, sum])
    
        Tcell_zone_sort_lr_list = sorted(Tcell_zone_sort_lr_list, key = lambda x: x[1], reverse=True)
    
        # save = num_spots/cells * top_N pairs
        if knee_flag == 0:
            Tcell_zone_sort_lr_list = Tcell_zone_sort_lr_list[0: top_lrp_count]
    
        # now plot the histograms where X axis will show the name or LR pair and Y axis will show the score.
        data_list=dict()
        data_list['X']=[]
        data_list['Y']=[] 
    
        max_rows = min(500, len(Tcell_zone_sort_lr_list))
        for i in range (0, max_rows): #1000): #
            data_list['X'].append(Tcell_zone_sort_lr_list[i][0])
            data_list['Y'].append(Tcell_zone_sort_lr_list[i][1])
            
        data_list_pd = pd.DataFrame({
            'Ligand-Receptor Pairs': data_list['X'],
            'Total Count': data_list['Y']
        })
        data_list_pd.to_csv(args.output_path +args.model_name+'_novel_lr_list_sortedBy_totalScore_top'+str(top_N)+'_Tcell_zone_novels.csv', index=False)
        print(args.output_path +args.model_name+'_novel_lr_list_sortedBy_totalScore_top'+str(top_N)+'_Tcell_zone_novels.csv')    
        # same as histogram plots
        chart = alt.Chart(data_list_pd).mark_bar().encode(
            x=alt.X("Ligand-Receptor Pairs:N", axis=alt.Axis(labelAngle=45), sort='-y'),
            y='Total Count'
        )
    
        chart.save(args.output_path +args.model_name+'_novel_lr_list_sortedBy_totalScore_top'+str(top_N)+'_histograms_Tcell_zone_novels.html')
        print(args.output_path +args.model_name+'_novel_lr_list_sortedBy_totalScore_top'+str(top_N)+'_histograms_Tcell_zone_novels.html')      
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
    print('Only LRbind %d, only manual %d, common %d'%(len(set_LRbind_novel), len(set_nichenet_novel)-len(common_lr), len(common_lr)))
    print(common_lr)
    pd.DataFrame(common_lr).to_csv(args.output_path +args.model_name+'_novel_lr_list_sortedBy_totalScore_top'+str(top_N)+'_commonWmanual_ccl19ccr7.csv', index=False)
    print(args.output_path +args.model_name+'_novel_lr_list_sortedBy_totalScore_top'+str(top_N)+'_commonWmanual_ccl19ccr7.csv')    
         

    ##################################################################
    for i in range (0, len(sort_lr_list)):
        if sort_lr_list[i][0] == ligand + '+' + receptor:
            print('index is %d'%i)
            break
            
    # ccl19-ccr7 index is 174 if sorted by total count
    # ccl19-ccr7 index is 196 if sorted by total frequency 
    # make a histogram plot
    '''
