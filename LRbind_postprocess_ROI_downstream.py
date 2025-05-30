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
    
    with gzip.open(args.metadata_from + args.data_name +'_test_set', 'rb') as fp:  
        target_LR_index, target_cell_pair = pickle.load(fp)

    #####################################################################################
    
    with gzip.open(args.data_from + args.data_name + '_cell_vs_gene_quantile_transformed', 'rb') as fp:
        cell_vs_gene = pickle.load(fp)

    with gzip.open(args.data_from + args.data_name + '_gene_index', 'rb') as fp:
        gene_index = pickle.load(fp)

    with gzip.open('metadata/LRbind_LUAD_1D_manualDB_geneCorrP7KNN_bidir/'+args.data_name+'_receptor_intra_KG.pkl', 'rb') as fp:
        receptor_intraNW, TF_genes = pickle.load(fp)

    
    ############# load output graph #################################################
    model_names = [#'model_LRbind_V1_Human_Lymph_Node_spatial_1D_manualDB_geneCorr',
                   #'model_LRbind_V1_Human_Lymph_Node_spatial_1D_manualDB_geneCorr_vgae',
                   #'model_LRbind_V1_Human_Lymph_Node_spatial_1D_manualDB_vgae',
                   #'model_LRbind_V1_Human_Lymph_Node_spatial_1D_manualDB_vgae_gat',
                   #'model_LRbind_V1_Human_Lymph_Node_spatial_1D_manualDB_vgae_gat_wbce',
                   # 'LRbind_model_V1_Human_Lymph_Node_spatial_1D_manualDB',
                   #'model_LRbind_V1_Human_Lymph_Node_spatial_1D_manualDB_bidir',
                   #'model_LRbind_V1_Human_Lymph_Node_spatial_1D_manualDB_bidir_3L',
                   #'model_LRbind_V1_Human_Lymph_Node_spatial_1D_manualDB_geneCorr_bidir_3L',
                   #'model_LRbind_GSM6177599_NYU_BRCA0_Vis_processed_1D_manualDB_geneCorr_bidir_3L'
                   #'model_LRbind_CID44971_1D_manualDB_geneCorr_bidir_3L',
                   #'model_LRbind_CID44971_1D_manualDB_geneCorrKNN_bidir_3L'
                   #'model_LRbind_LUAD_1D_manualDB_geneCorr_bidir_3L'
                   #'model_LRbind_LUAD_1D_manualDB_geneCorr_signaling_bidir_3L'
                   #'model_LRbind_LUAD_1D_manualDB_geneCorrKNN_bidir_3L'
                   'model_LRbind_LUAD_1D_manualDB_geneCorrP7KNN_bidir_3L'
                   # 'model_LRbind_PDAC64630_1D_manualDB_geneCorrKNN_bidir_3L'
                   # 'model_LRbind_V1_Breast_Cancer_Block_A_Section_1_spatial_1D_manualDB_geneCorrKNN_bidir_3L'
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
            
        ''' '''   
            
    ########## all ############################################# 
        top_lrp_count = 1000
        knee_flag = 0
        break_flag = 0
        test_mode = 1
        target_ligand = args.target_ligand
        target_receptor = args.target_receptor
        for top_N in [100]: #, 30, 10]:
            print(top_N)
            if break_flag == 1:  
                break
            if knee_flag == 1:
                top_N = 0
                break_flag = 1

            ################
            lr_dict = defaultdict(list)
            Tcell_zone_lr_dict = defaultdict(list)
            found_list = defaultdict(list)
            for pair in target_cell_pair[target_ligand+'+'+target_receptor]:
                i = pair[0]
                j = pair[1]
                #if barcode_info[i][1] < 5000 or barcode_info[i][2] > 5000:
                #    continue
                print("%d, %d, found list: %d"%(i,j,len(found_list)))
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
                start_index = 0
                for i_gene in ligand_node_index:  
                    for j_gene in receptor_node_index:
                        if i_gene[1]==j_gene[1]:
                            continue
                        temp = distance.euclidean(X_embedding[i_gene[0]], X_embedding[j_gene[0]]) #(X_PCA[i_gene[0]], X_PCA[j_gene[0]]) #
                        # distance.euclidean(X_embedding[i_gene[0]], X_embedding[j_gene[0]]) 
                        # (X_embedding[i_gene[0]], X_embedding[j_gene[0]])
                        dot_prod_list.append([temp, i, j, i_gene[1], j_gene[1]])
                        product_only.append(temp)
                 

                
                if len(dot_prod_list) == 0:
                    continue
                    
                if knee_flag == 0:
                    
                    max_score = max(product_only)
                    for item_idx in range (0, len(dot_prod_list)):
                        scaled_prod = max_score - dot_prod_list[item_idx][0]
                        dot_prod_list[item_idx][0] = scaled_prod 
                    
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
                    
                    #if i in Tcell_zone and j in Tcell_zone:
                    #    Tcell_zone_lr_dict[item[3]+'+'+item[4]].append([item[0], item[1], item[2]])
                        
                    if test_mode == 1 and item[3] == target_ligand and item[4] == target_receptor:
                        found_list[i].append(item[0]) #= 1
                        found_list[j].append(item[0])
                        #break
    
            # plot found_list
            print("positive: %d"%(len(found_list)))                
            # plot input_cell_pair_list  
            
            ######### plot output #############################
            # UPDATE # annottaion
            '''
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
                data_list['Y'].append(barcode_info[i][2])
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
            chart.save(args.output_path + args.model_name + '_output_' + target_ligand + '-' + target_receptor +'_top'+ str(top_N)  + '_ROI_allLR.html')
            print(args.output_path + args.model_name + '_output_' + target_ligand + '-' + target_receptor +'_top'+ str(top_N)  + '_ROI_allLR.html') 
            '''
            #############################################################
            save_lr_dict = copy.deepcopy(lr_dict)
            ############################
            lr_dict = copy.deepcopy(save_lr_dict)
            print('before post process len %d'%len(lr_dict.keys()))
            # Set threshold gene percentile
            threshold_gene_exp = 90
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
            #####################
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
                target_list = []
                for rows in receptor_intraNW[receptor]:
                    target_list.append(rows[1][0])
                # what percent of them has the target genes expressed
            
               
                count = 0
                keep_receptor = dict()
                for cell in receptor_cell_list:
                    found = 0
                    for gene in target_list:
                        if cell_vs_gene[cell][gene_index[gene]] >= cell_percentile[cell]:
                            found = found + 1
                            
                            
                    if found>0 and len(target_list)/found >= 0.7:
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

            
            ########## take top hits #################################### 
            #if top_N == 30:
            #    continue
            sort_lr_list = []
            ligand_found_dict = defaultdict(list)
            for lr_pair in lr_dict:
                sum = 0
                cell_pair_list = lr_dict[lr_pair]
                for item in cell_pair_list:
                    sum = sum + item[0]  

                #sum = sum/len(cell_pair_list)
                sort_lr_list.append([lr_pair, sum])
                

                
                ligand_found_dict[lr_pair.split('+')[0]].append(lr_pair.split('+')[1])
        
            sort_lr_list = sorted(sort_lr_list, key = lambda x: x[1], reverse=True)
            
            #if knee_flag == 0:
            #    sort_lr_list = sort_lr_list[0: top_lrp_count]
    
            
            top_hit_lrp_dict = dict()
            i = 0
            for item in sort_lr_list:
                top_hit_lrp_dict[item[0]] = i
                i = i+1
            
            # now plot the histograms where X axis will show the name or LR pair and Y axis will show the score.
            ligand_found_dict = defaultdict(list)
            data_list=dict()
            data_list['X']=[]
            data_list['Y']=[] 
            max_rows = min(500, len(sort_lr_list))
            for i in range (0, max_rows): #len(sort_lr_list)): #1000): #:
                data_list['X'].append(sort_lr_list[i][0])
                data_list['Y'].append(sort_lr_list[i][1])
                ligand_found_dict[sort_lr_list[i][0].split('+')[0]].append(sort_lr_list[i][0].split('+')[1])
                if sort_lr_list[i][0].split('+')[0]=='WNT10A' and sort_lr_list[i][0].split('+')[1]=='FZD1':
                    print(i)

            
            data_list_pd = pd.DataFrame({
                'Ligand-Receptor Pairs': data_list['X'],
                'Score': data_list['Y']
            }) 
            print(data_list['X'][0:20])
            '''
            with gzip.open('output/'+args.data_name+'/' + args.model_name +'_lr_dict_pca', 'wb') as fp:  
                pickle.dump(lr_dict, fp)
            '''

            
            data_list_pd.to_csv(args.output_path +model_name+'_lr_list_sortedBy_totalScore_top'+str(top_N)+'_ROI_allLR.csv', index=False)
            print(args.output_path +args.model_name+'_lr_list_sortedBy_totalScore_top'+str(top_N)+'_ROI_allLR.csv')    
            # same as histogram plots
            chart = alt.Chart(data_list_pd).mark_bar().encode(
                x=alt.X("Ligand-Receptor Pairs:N", axis=alt.Axis(labelAngle=45), sort='-y'),
                y='Score'
            )
        
            chart.save(args.output_path +model_name+'_lr_list_sortedBy_totalScore_top'+str(top_N)+'_ROI_histogramsallLR.html')
            print(args.output_path +args.model_name+'_lr_list_sortedBy_totalScore_top'+str(top_N)+'_ROI_histogramsallLR.html')   
            if target_ligand + '+' + target_receptor  in list(data_list_pd['Ligand-Receptor Pairs']):
                print("found %s-%s: %d"%(target_ligand, target_receptor, top_hit_lrp_dict[target_ligand + '+' + target_receptor])) # for 100, position is 44, with 4L -- only 5 spots are detected and position - none
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
                
            #print('novel LRP length %d out of top %d LRP'%(len(sort_lr_list_temp), top_lrp_count))
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
                'Avg_dotProduct': data_list['Y']
            })
            data_list_pd.to_csv(args.output_path +args.model_name+'_novel_lr_list_sortedBy_totalScore_top'+str(top_N)+'_novelsOutOfallLR.csv', index=False)
            #print(args.output_path +args.model_name+'_novel_lr_list_sortedBy_totalScore_top'+str(top_N)+'novelsOutOfallLR.csv')    
            # same as histogram plots
            chart = alt.Chart(data_list_pd).mark_bar().encode(
                x=alt.X("Ligand-Receptor Pairs:N", axis=alt.Axis(labelAngle=45), sort='-y'),
                y='Avg_dotProduct'
            )
        
            chart.save(args.output_path +args.model_name+'_novel_lr_list_sortedBy_totalScore_top'+str(top_N)+'_histograms_novelsOutOfallLR.html')
            #print(args.output_path +args.model_name+'_novel_lr_list_sortedBy_totalScore_top'+str(top_N)+'_histograms_novelsOutOfallLR.html')   
            ################################# when not remFromDB ##########################################################################################################
            
            set_LRbind_novel = []
            for i in range (0, len(sort_lr_list_temp)):
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
            #print('top_N:%d, Only LRbind %d, only nichenet %d, common %d'%(top_N, len(set_LRbind_novel)-len(common_lr), len(set_nichenet_novel)-len(common_lr), len(common_lr)))
            pd.DataFrame(common_lr).to_csv(args.output_path +args.model_name+'_novel_lr_list_sortedBy_totalScore_top'+str(top_N)+'_common_with_nichenet.csv', index=False)
            #print(args.output_path +args.model_name+'_novel_lr_list_sortedBy_totalScore_top'+str(top_N)+'novelsOutOfallLR.csv') 
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
             ############ only Tcell Zone plot ##############################################################################################################################


            Tcell_zone_sort_lr_list = []
            for lr_pair in Tcell_zone_lr_dict:
                #if lr_pair not in top_hit_lrp_dict:
                #    continue
                sum = 0
                cell_pair_list = Tcell_zone_lr_dict[lr_pair]
                for item in cell_pair_list:
                    sum = sum + item[0] # 

                #sum = sum/len(cell_pair_list)
                Tcell_zone_sort_lr_list.append([lr_pair, sum])
        
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
                'Avg_dotProduct': data_list['Y']
            })
            #if 'CCL19+CCR7' in list(data_list_pd['Ligand-Receptor Pairs']):
            #    print("found CCL19-CCR7")
            
            data_list_pd.to_csv(args.output_path +args.model_name+'_novel_lr_list_sortedBy_totalScore_top'+str(top_N)+'Tcell_zone_allLR.csv', index=False)
            #print(args.output_path +args.model_name+'_novel_lr_list_sortedBy_totalScore_top'+str(top_N)+'Tcell_zone_allLR.csv')    
            # same as histogram plots
            chart = alt.Chart(data_list_pd).mark_bar().encode(
                x=alt.X("Ligand-Receptor Pairs:N", axis=alt.Axis(labelAngle=45), sort='-y'),
                y='Avg_dotProduct'
            )
        
            chart.save(args.output_path +args.model_name+'_novel_lr_list_sortedBy_totalScore_top'+str(top_N)+'Tcell_zone_histogramsallLR.html')
            #print(args.output_path +args.model_name+'_novel_lr_list_sortedBy_totalScore_top'+str(top_N)+'Tcell_zone_histogramsallLR.html')   


#################################
    node_ROI = dict()
    for pair in target_cell_pair[target_ligand+'+'+target_receptor]:
        i = pair[0]
        j = pair[1]
        node_ROI[i] = 1
        node_ROI[j] = 1

    if plot_input == 1:
        data_list=dict()
        data_list['X']=[]
        data_list['Y']=[]   
        data_list['input']=[] 
        for i in range (0, len(barcode_info)):
            

            data_list['X'].append(barcode_info[i][1])
            data_list['Y'].append(barcode_info[i][2])
            if i in node_ROI:
                data_list['input'].append(1)
            else:
                data_list['input'].append(0)
            
                
        source= pd.DataFrame(data_list)
        chart = alt.Chart(source).mark_point(filled=True).encode(
            alt.X('X', scale=alt.Scale(zero=False)),
            alt.Y('Y', scale=alt.Scale(zero=False)),
            color=alt.Color('input:Q', scale=alt.Scale(scheme='magma')),
            #shape = alt.Shape('label:N')
        )
        chart.save(args.output_path + args.data_name + '_input_' + target_ligand + '-' + target_receptor + '.html')
        print(args.output_path + args.data_name + '_input_' + target_ligand + '-' + target_receptor + '.html') 
    
####################################



