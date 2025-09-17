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
        
        for sample_idx in range(0, len(sample_type)):
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
            if sample_idx==0:
                args.embedding_path  = args.embedding_path + args.data_name + '/'
                

            args.model_name = model_names[model_index] 
            args.model_name = args.model_name + '_r1' + '_' + sample_type[sample_idx]

        ##################### get metadata: barcode_info ##################################        

            with gzip.open(args.metadata_from +args.data_name+'_barcode_info_gene', 'rb') as fp:  #b, a:[0:5]   _filtered
                barcode_info_gene, ligand_list, receptor_list, gene_node_list_per_spot, dist_X, l_r_pair, gene_node_index_active, ligand_active, receptor_active = pickle.load(fp)


        
        ccc_pairs = pd.read_csv(args.output_path +args.model_name +'_allLR_nodeInfo_top20p.csv.gz') #_negatome
        #ccc_pairs['score'] = all_ccc_pairs['score']
        #ccc_pairs['from_cell_index'] = all_ccc_pairs['from_cell_index']
        #ccc_pairs['to_cell_index'] = all_ccc_pairs['to_cell_index']
        #ccc_pairs['attention_score'] = all_ccc_pairs['attention_score']
        
        lr_dict = defaultdict(list)
        for i in range(0, len(ccc_pairs)):
            if ccc_pairs['pred_score'][i] <= 0: #< 0.7:
                continue
            
            #if ccc_pairs['attention_score'][i] < 0.7:
            #    continue

            #lr_dict[ccc_pairs['ligand_gene'][i]+'+'+ccc_pairs['rec_gene'][i]].append([ccc_pairs['score'][i], ccc_pairs['from_cell_index'], ccc_pairs['to_cell_index'], ccc_pairs['pred_score'][i], ccc_pairs['attention_score']])  # score, cell ids, gene_node ids   
            lr_dict[ccc_pairs['ligand_gene'][i]+'+'+ccc_pairs['rec_gene'][i]].append([ccc_pairs['score'][i], ccc_pairs['from_cell_index'], ccc_pairs['to_cell_index'], ccc_pairs['attention_score'], -1])  # score, cell ids, gene_node ids   

            
        ####### sort the pairs based on total score ####################
        sort_lr_list = []
        for lr_pair in lr_dict:
            print(len)
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
        #data_list_pd.to_csv(args.output_path +model_name+'_lr_list_sortedBy_totalScore_top'+ file_name_suffix+'_allLR_predScore.csv', index=False) #_negatome
        data_list_pd.to_csv(args.output_path +args.model_name+'_lr_list_sortedBy_totalScore_top'+ file_name_suffix+'_allLR_predClass_top20p.csv', index=False) #_negatome
        print(len(data_list_pd))
        print('done')

