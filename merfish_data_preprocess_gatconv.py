import numpy as np
import csv
import pickle
from scipy import sparse
import scipy.io as sio
import scanpy as sc
import matplotlib
matplotlib.use('Agg')
#matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import stlearn as st
import numpy as np
from matplotlib.colors import LinearSegmentedColormap, to_hex, rgb2hex
from typing import List
import qnorm
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components
from collections import defaultdict
import pandas as pd
import gzip
from kneed import KneeLocator
import copy 
import pickle
import gc 
options = 'Female_Virgin_ParentingExcitatory'
import argparse
parser = argparse.ArgumentParser()
parser.add_argument( '--data_path', type=str, default="/cluster/projects/schwartzgroup/fatema/find_ccc/merfish_mouse_cortex/" , help='The path to dataset') 
parser.add_argument( '--embedding_data_path', type=str, default='new_alignment/Embedding_data_ccc_rgcn/' , help='The path to attention') 
parser.add_argument( '--data_name', type=str, default='messi_merfish_data_'+options, help='The name of dataset')
#parser.add_argument( '--slice', type=int, default=0, help='starting index of ligand')
args = parser.parse_args()
spot_diameter = 100 # micrometer # 0.2-μm-diameter carboxylate-modified orange fluorescent beads from org paper: https://www.science.org/doi/10.1126/science.aaa6090
threshold_distance = spot_diameter*3 # 100 µm for the MERFISH hypothalamus dataset used in MESSI
distance_measure = 'threshold_distance' #'knn' #
k_nn = 10
################
# The camera was configured so that a pixel corresponds to 167 nm in the sample plane. But the data gives coordinates in micro meter format

############
with gzip.open(args.data_path + args.data_name, 'rb') as fp:    
    data_sets_gatconv, lr_pairs, cell_cell_contact= pickle.load(fp) 
##############################################


############
# animal_id = 16
animal_id = 24 #data_sets_gatconv[0][4][0][0]
bregma = [0.11, 0.16, 0.21, 0.26] #data_sets_gatconv[0][4][0][3] []
for bregma_id in range (0, len(bregma)): #bregma:
    print('animal id:%d, bregma: %g'%(animal_id, bregma[bregma_id]))
    z_index_yes = 0
    barcode_info = []
    cell_vs_gene_list = []
    total_cell = 0
    sample_index = 0
    for index in range (0,len(data_sets_gatconv)):
        if data_sets_gatconv[index][4][0][0] == animal_id and data_sets_gatconv[index][4][0][3] == bregma[bregma_id]:
            sample_index = index
            cell_barcodes = data_sets_gatconv[index][0]
            coordinates = data_sets_gatconv[index][1]
            cell_vs_gene = data_sets_gatconv[index][2]
            cell_vs_gene_list.append(cell_vs_gene)
            total_cell = total_cell + cell_vs_gene.shape[0]
            z_index = data_sets_gatconv[index][4][0][3]
            print('index:%d, cell count: %d'%(index, cell_vs_gene.shape[0]))
            if z_index_yes == 1:
                for i in range (0, len(cell_barcodes)):
                    barcode_info.append([cell_barcodes[0], coordinates[i,0], coordinates[i,1], z_index,0])
                    i=i+1
            else:
                for i in range (0, len(cell_barcodes)):
                    barcode_info.append([cell_barcodes[0], coordinates[i,0], coordinates[i,1], 0])
                    i=i+1       

                break


    ########
    gene_index = dict()
    gene_list = data_sets_gatconv[sample_index][3].keys() # keys are the gene
    for gene in gene_list:
        gene_index[data_sets_gatconv[sample_index][3][gene]] = gene # we know which index has which gene. So we record gene_ids in 0, 1, 2, ... oder

    gene_ids = []
    gene_list = sorted(gene_index.keys()) # 0, 1, 2, ...
    for index in gene_list:
        gene_ids.append(gene_index[index])

    gene_info=dict()
    for gene in gene_ids:
        gene_info[gene]=''


    gene_index = data_sets_gatconv[sample_index][3]

    #############################
    if z_index_yes == 1:
        coordinates = np.zeros((total_cell, 3))
    else:
        coordinates = np.zeros((total_cell, 2))

    for i in range (0, len(barcode_info)): 
        coordinates[i][0] = barcode_info[i][1]
        coordinates[i][1] = barcode_info[i][2]
        if z_index_yes == 1:
            coordinates[i][2] = barcode_info[i][3]


    ##################################    
    cell_vs_gene = np.zeros((total_cell, len(gene_ids)))
    start_row = 0
    for i in range (0, len(cell_vs_gene_list)):
        cell_vs_gene[start_row : start_row+cell_vs_gene_list[i].shape[0], :] = cell_vs_gene_list[i]
        start_row = start_row + cell_vs_gene_list[i].shape[0]

    cell_vs_gene_list = 0
    gc.collect()




    #################### 
    print('min cell_vs_gene %g, max: %g'%(np.min(cell_vs_gene),np.max(cell_vs_gene)))
    temp = qnorm.quantile_normalize(np.transpose(cell_vs_gene))  
    cell_vs_gene = np.transpose(temp)  
    print('min cell_vs_gene %g, max: %g'%(np.min(cell_vs_gene),np.max(cell_vs_gene)))

    ####################
    cell_percentile = []
    for i in range (0, cell_vs_gene.shape[0]):
        y = sorted(cell_vs_gene[i])
        x = range(1, len(y)+1)
        kn = KneeLocator(x, y, curve='convex', direction='increasing')
        kn_value = y[kn.knee-1]
        cell_percentile.append([np.percentile(y, 10), np.percentile(y, 20),np.percentile(y, 90), np.percentile(y, 95), kn_value])
    #######################################################


    ligand_dict_dataset = defaultdict(list)
    cell_cell_contact = dict()
    cell_chat_file = '/cluster/home/t116508uhn/Human-2020-Jin-LR-pairs_cellchat.csv'
    df = pd.read_csv(cell_chat_file)
    for i in range (0, df["ligand_symbol"].shape[0]):
        ligand = df["ligand_symbol"][i]
        #if ligand not in gene_marker_ids:
        if ligand not in gene_info:
            continue

        if df["annotation"][i] == 'ECM-Receptor':    
            continue

        receptor_symbol_list = df["receptor_symbol"][i]
        receptor_symbol_list = receptor_symbol_list.split("&")
        for receptor in receptor_symbol_list:
            if receptor in gene_info:
            #if receptor in gene_marker_ids:
                ligand_dict_dataset[ligand].append(receptor)
                #######
                if df["annotation"][i] == 'Cell-Cell Contact':
                    cell_cell_contact[receptor] = ''
                #######                

    print(len(ligand_dict_dataset.keys()))

    nichetalk_file = '/cluster/home/t116508uhn/NicheNet-LR-pairs.csv'   
    df = pd.read_csv(nichetalk_file)
    for i in range (0, df["from"].shape[0]):
        ligand = df["from"][i]
        #if ligand not in gene_marker_ids:
        if ligand not in gene_info:
            continue
        receptor = df["to"][i]
        #if receptor not in gene_marker_ids:
        if receptor not in gene_info:
            continue
        ligand_dict_dataset[ligand].append(receptor)

    ##############################################################
    print('number of ligands %d '%len(ligand_dict_dataset.keys()))
    count_pair = 0
    for gene in list(ligand_dict_dataset.keys()): 
        ligand_dict_dataset[gene]=list(set(ligand_dict_dataset[gene]))
        gene_info[gene] = 'included'
        for receptor_gene in ligand_dict_dataset[gene]:
            gene_info[receptor_gene] = 'included'
            count_pair = count_pair + 1

    print('number of pairs %d '%count_pair)       

    count = 0
    included_gene=[]
    for gene in gene_info.keys(): 
        if gene_info[gene] == 'included':
            count = count + 1
            included_gene.append(gene)

    print('number of affected genes %d '%count)
    affected_gene_count = count



    '''	
    ligand_dict_dataset = defaultdict(list)
    for i in range (0, len(lr_pairs)):
        ligand = lr_pairs['ligand'][i]  
        if ligand not in gene_info:
            continue

        receptor = lr_pairs['receptor'][i]  
        if receptor not in gene_info:
            continue

        ligand_dict_dataset[ligand].append(receptor)


    print(len(ligand_dict_dataset.keys()))

    for gene in list(ligand_dict_dataset.keys()): 
        ligand_dict_dataset[gene]=list(set(ligand_dict_dataset[gene]))
        gene_info[gene] = 'included'
        for receptor_gene in ligand_dict_dataset[gene]:
            gene_info[receptor_gene] = 'included'

    count = 0
    for gene in gene_info.keys(): 
        if gene_info[gene] == 'included':
            count = count + 1
    print(count)
    '''
    ##################################################################

    ligand_list = list(ligand_dict_dataset.keys())  
    print('len ligand_list %d'%len(ligand_list))
    total_relation = 0
    l_r_pair = dict()
    count = 0
    lr_id = 0
    for gene in list(ligand_dict_dataset.keys()): 
        ligand_dict_dataset[gene]=list(set(ligand_dict_dataset[gene]))
        l_r_pair[gene] = dict()
        for receptor_gene in ligand_dict_dataset[gene]:
            l_r_pair[gene][receptor_gene] = lr_id 
            lr_id  = lr_id  + 1

    print('total type of l-r pairs found: %d'%lr_id )
    '''
    relation_id = dict()
    count = 0
    for gene in list(ligand_dict_dataset.keys()): 
        ligand_dict_dataset[gene]=list(set(ligand_dict_dataset[gene]))
        relation_id[gene] = dict()
        for receptor_gene in ligand_dict_dataset[gene]:
            relation_id[gene][receptor_gene] = count
            count = count + 1

    print('number of relations found %d'%count)        
    ##################################################################

    ligand_list = list(ligand_dict_dataset.keys())  
    '''
    print('creating distance matrix')
    from sklearn.metrics.pairwise import euclidean_distances
    distance_matrix = euclidean_distances(coordinates, coordinates)
    print('process distance matrix')
    dist_X = np.zeros((distance_matrix.shape[0], distance_matrix.shape[1]))
    for j in range(0, distance_matrix.shape[1]):
        max_value=np.max(distance_matrix[:,j])
        min_value=np.min(distance_matrix[:,j])
        for i in range(distance_matrix.shape[0]):
            dist_X[i,j] = 1-(distance_matrix[i,j]-min_value)/(max_value-min_value)

        #list_indx = list(np.argsort(dist_X[:,j]))
        #k_higher = list_indx[len(list_indx)-k_nn:len(list_indx)]
        '''
        for i in range(0, distance_matrix.shape[0]):
            if distance_matrix[i,j] > threshold_distance: #spot_diameter*4: #i not in k_higher:
                dist_X[i,j] = 0 #-1
        '''
    cell_rec_count = np.zeros((cell_vs_gene.shape[0]))

    ########
    ######################################
    ##############################################################################
    print('create cells_ligand_vs_receptor')

    count_total_edges = 0
    activated_cell_index = dict()

    cells_ligand_vs_receptor = []
    for i in range (0, cell_vs_gene.shape[0]):
        cells_ligand_vs_receptor.append([])

    for i in range (0, cell_vs_gene.shape[0]):
        for j in range (0, cell_vs_gene.shape[0]):
            cells_ligand_vs_receptor[i].append([])
            cells_ligand_vs_receptor[i][j] = []

    start_index = 0 #args.slice
    end_index = len(ligand_list) #min(len(ligand_list), start_index+100)
    for g in range(start_index, end_index): 
        gene = ligand_list[g]
        for i in range (0, cell_vs_gene.shape[0]): # ligand
            count_rec = 0    
            if cell_vs_gene[i][gene_index[gene]] < cell_percentile[i][3]:
                continue

            for j in range (0, cell_vs_gene.shape[0]): # receptor
                if distance_matrix[i,j] > threshold_distance: #spot_diameter*4:
                    continue

                #if gene in cell_cell_contact and distance_matrix[i,j] > spot_diameter:
                #    continue

                for gene_rec in ligand_dict_dataset[gene]:
                    if cell_vs_gene[j][gene_index[gene_rec]] >= cell_percentile[j][3]: # or cell_vs_gene[i][gene_index[gene]] >= cell_percentile[i][4] :#gene_list_percentile[gene_rec][1]: #global_percentile: #
                        if gene_rec in cell_cell_contact and distance_matrix[i,j] > spot_diameter:
                            continue

                        '''if gene_rec in cell_cell_contact and distance_matrix[i,j] < spot_diameter:
                            print(gene)'''

                        communication_score = cell_vs_gene[i][gene_index[gene]] * cell_vs_gene[j][gene_index[gene_rec]]

                        relation_id = l_r_pair[gene][gene_rec]
                        #print("%s - %s "%(gene, gene_rec))
                        if communication_score<=0:
                            print('zero valued ccc score found')
                            continue	
                        cells_ligand_vs_receptor[i][j].append([gene, gene_rec, communication_score, relation_id])
                        count_rec = count_rec + 1
                        count_total_edges = count_total_edges + 1
                        activated_cell_index[i] = ''
                        activated_cell_index[j] = ''


            cell_rec_count[i] =  count_rec   
            #print("%d - %d "%(i, count_rec))
            #print("%d - %d , max %g and min %g "%(i, count_rec, max_score, min_score))

        print(g)

    print('total number of edges in the input graph %d '%count_total_edges)


    ################################################################################
    ccc_index_dict = dict()
    row_col = []
    edge_weight = []
    lig_rec = []
    count_edge = 0
    max_local = 0
    #local_list = np.zeros((102))
    for i in range (0, len(cells_ligand_vs_receptor)):
        #ccc_j = []
        for j in range (0, len(cells_ligand_vs_receptor)):
            if distance_matrix[i][j] <= threshold_distance: #spot_diameter*4: 
                count_local = 0
                if len(cells_ligand_vs_receptor[i][j])>0:
                    for k in range (0, len(cells_ligand_vs_receptor[i][j])):
                        gene = cells_ligand_vs_receptor[i][j][k][0]
                        gene_rec = cells_ligand_vs_receptor[i][j][k][1]
                        # above 5th percentile only
                        #if cell_vs_gene[i][gene_index[gene]] >= cell_percentile[i][2] and cell_vs_gene[j][gene_index[gene_rec]] >= cell_percentile[j][2]:
                        count_edge = count_edge + 1
                        count_local = count_local + 1
    #print(count_edge)                      
                        mean_ccc = cells_ligand_vs_receptor[i][j][k][2]
                        row_col.append([i,j])
                        #if gene=='SERPINA1': # or gene=='MIF':
                        #    ccc_index_dict[i] = ''
                        #ccc_index_dict[j] = ''
                        edge_weight.append([dist_X[i,j], mean_ccc,cells_ligand_vs_receptor[i][j][k][3]])
                        #edge_weight.append([dist_X[i,j], mean_ccc])
                        lig_rec.append([gene, gene_rec])                      

                    if max_local < count_local:
                        max_local = count_local
                '''
                else:
                    row_col.append([i,j])
                    edge_weight.append([dist_X[i,j], 0])
                    lig_rec.append(['', ''])
                '''
                #local_list[count_local] = local_list[count_local] + 1

    print('len row col %d'%len(row_col))
    print('count local %d'%max_local) 


    with gzip.open("/cluster/projects/schwartzgroup/fatema/find_ccc/" +args.data_name+'_id'+str(animal_id)+'_bregma'+str(bregma[bregma_id])+'_adjacency_records_GAT_selective_lr_STnCCC_separate_'+'bothAbove_cell95th_3d', 'wb') as fp:  #b, a:[0:5]  _filtered 
        pickle.dump([row_col, edge_weight, lig_rec], fp)

    with gzip.open("/cluster/projects/schwartzgroup/fatema/find_ccc/" +args.data_name+'_id'+str(animal_id)+'_bregma'+str(bregma[bregma_id])+'_cell_vs_gene_quantile_transformed', 'wb') as fp:  #b, a:[0:5]   _filtered
        pickle.dump(cell_vs_gene, fp)

    with gzip.open("/cluster/projects/schwartzgroup/fatema/find_ccc/" +args.data_name+'_id'+str(animal_id)+'_bregma'+str(bregma[bregma_id])+'_barcode_info', 'wb') as fp:  #b, a:[0:5]   _filtered
        pickle.dump(barcode_info, fp)
    with gzip.open("/cluster/projects/schwartzgroup/fatema/find_ccc/" +args.data_name+'_id'+str(animal_id)+'_bregma'+str(bregma[bregma_id])+'_coordinates', 'wb') as fp:  #b, a:[0:5]   _filtered
        pickle.dump(coordinates, fp)
        

##########
with gzip.open("/cluster/projects/schwartzgroup/fatema/find_ccc/" +args.data_name+'_id'+str(animal_id)+'_adjacency_records_GAT_selective_lr_STnCCC_separate_'+'bothAbove_cell98th_xyz_3d_ThDistance500', 'wb') as fp:  #b, a:[0:5]  _filtered 
    pickle.dump([row_col, edge_weight, lig_rec], fp)
             
with gzip.open("/cluster/projects/schwartzgroup/fatema/find_ccc/" +args.data_name+'_id'+str(animal_id)+'_cell_vs_gene_xyz_quantile_transformed', 'wb') as fp:  #b, a:[0:5]   _filtered
	pickle.dump(cell_vs_gene, fp)



###########################################################Visualization starts ##################
datapoint_size = len(barcode_info)    

with gzip.open("/cluster/projects/schwartzgroup/fatema/find_ccc/" +args.data_name+'_id'+str(animal_id)+'_adjacency_records_GAT_selective_lr_STnCCC_separate_'+'bothAbove_cell98th_xyz_3d_ThDistance500', 'rb') as fp:  #b, a:[0:5]  _filtered 
    row_col, edge_weight, lig_rec = pickle.load(fp)

#####
barcode_type=dict()
for i in range (0, datapoint_size):
    barcode_type[barcode_info[i][0]] = 0 
    
lig_rec_dict = []
for i in range (0, datapoint_size):
    lig_rec_dict.append([])  
    for j in range (0, datapoint_size):	
        lig_rec_dict[i].append([])   
        lig_rec_dict[i][j] = []

total_type = np.zeros((2))        
for index in range (0, len(row_col)):
        i = row_col[index][0]
        j = row_col[index][1]
        lig_rec_dict[i][j].append(lig_rec[index])  
        
filename = ["r1", "r2", "r3", "r4", "r5", "r6", "r7", "r8", "r9", "r10"]
total_runs = 1
start_index = 0
csv_record_dict = defaultdict(list)
for run_time in range (start_index, start_index+total_runs):
    gc.collect()
    #run_time = 2
    run = run_time
    X_attention_filename = args.embedding_data_path + args.data_name + '/' + 'merfish_data_Female_Virgin_ParentingExcitatory_id24_cellchat_nichenet_threshold_distance_bothAbove_cell98th_tanh_3dim_xyz_'+filename[run_time]+'_th500_attention_l1.npy'   
    X_attention_bundle = np.load(X_attention_filename, allow_pickle=True) #_withFeature

    #l = 3
    for l in [2,3]:  # 3 = layer 1, 2 = layer 2
        distribution = []
        for index in range (0, X_attention_bundle[0].shape[1]):
            i = X_attention_bundle[0][0][index]
            j = X_attention_bundle[0][1][index]
            #if barcode_type[barcode_info[i][0]] != 1 or barcode_type[barcode_info[j][0]] != 1:
            #    continue
            distribution.append(X_attention_bundle[l][index][0])

        attention_scores = []
        for i in range (0, datapoint_size):
            attention_scores.append([])   
            for j in range (0, datapoint_size):	
                attention_scores[i].append([])   
                attention_scores[i][j] = []

        min_attention_score = 1000
        max_value = np.max(distribution)
        min_value = np.min(distribution)
        distribution = []
        for index in range (0, X_attention_bundle[0].shape[1]):
            i = X_attention_bundle[0][0][index]
            j = X_attention_bundle[0][1][index]
            #if barcode_type[barcode_info[i][0]] != 1 or barcode_type[barcode_info[j][0]] != 1:
            #    continue
            scaled_score = (X_attention_bundle[l][index][0]-min_value)/(max_value-min_value)
            attention_scores[i][j].append(scaled_score) #X_attention_bundle[2][index][0]
            if min_attention_score > scaled_score:
                min_attention_score = scaled_score
            distribution.append(scaled_score)


        if min_attention_score<0:
            min_attention_score = -min_attention_score
        else: 
            min_attention_score = 0

        print('min attention score %g'%min_attention_score)
        ##############
        #plt.clf()
        #plt.hist(distribution, color = 'blue',bins = int(len(distribution)/5))
        #save_path = '/cluster/home/t116508uhn/64630/'
        #plt.savefig(save_path+'distribution_region_of_interest_'+filename[run_time]+'_l2attention_score.svg', dpi=400) # _CCL19_CCR7     
        #plt.savefig(save_path+'dist_'+args.data_name+'_bothAbove98th_3dim_tanh_h512_l2attention_'+filename[run_time]+'attention_score.svg', dpi=400)
        #plt.clf()
        ##############

        ##############

        #hold_attention_score = copy.deepcopy(attention_scores)  
        #attention_scores = copy.deepcopy(hold_attention_score)  
        ####################################################################################

        ###########################

        ccc_index_dict = dict()
        threshold_down =  np.percentile(sorted(distribution), 90)
        threshold_up =  np.percentile(sorted(distribution), 100)
        connecting_edges = np.zeros((len(barcode_info),len(barcode_info)))
        for j in range (0, datapoint_size):
            #threshold =  np.percentile(sorted(attention_scores[:,j]), 97) #
            for i in range (0, datapoint_size):
                atn_score_list = attention_scores[i][j]
                #print(len(atn_score_list))
                #s = min(0,len(atn_score_list)-1)
                for k in range (0, len(atn_score_list)):
                    if attention_scores[i][j][k] >= threshold_down and attention_scores[i][j][k] <= threshold_up: #np.percentile(sorted(distribution), 50):
                        connecting_edges[i][j] = 1
                        ccc_index_dict[i] = ''
                        ccc_index_dict[j] = ''



        graph = csr_matrix(connecting_edges)
        n_components, labels = connected_components(csgraph=graph,directed=True, connection = 'weak',  return_labels=True) #
        print('number of component %d'%n_components)

        count_points_component = np.zeros((n_components))
        for i in range (0, len(labels)):
             count_points_component[labels[i]] = count_points_component[labels[i]] + 1

        print(count_points_component)

        id_label = 2 # initially all are zero. =1 those who have self edge but above threshold. >= 2 who belong to some component
        index_dict = dict()
        for i in range (0, count_points_component.shape[0]):
            if count_points_component[i]>1:
                index_dict[i] = id_label
                id_label = id_label+1

        print(id_label)


        for i in range (0, len(barcode_info)):
        #    if barcode_info[i][0] in barcode_label:
            if count_points_component[labels[i]] > 1:
                barcode_info[i][3] = index_dict[labels[i]] #2
            elif connecting_edges[i][i] == 1 and len(lig_rec_dict[i][i])>0: 
                barcode_info[i][3] = 1
            else:
                barcode_info[i][3] = 0

        #######################


        '''
        barcode_type=dict()
        for i in range (1, len(pathologist_label)):
            if pathologist_label[i][1] == 'tumor': #'Tumour':
                barcode_type[pathologist_label[i][0]] = '2_tumor'
            elif pathologist_label[i][1] =='stroma_deserted':
                barcode_type[pathologist_label[i][0]] = '0_stroma_deserted'
            elif pathologist_label[i][1] =='acinar_reactive':
                barcode_type[pathologist_label[i][0]] = '1_acinar_reactive'
            else:
                barcode_type[pathologist_label[i][0]] = 'zero' #0
        '''

        data_list=dict()
        data_list['pathology_label']=[]
        data_list['component_label']=[]
        data_list['X']=[]
        data_list['Y']=[]

        for i in range (0, len(barcode_info)):
            #if barcode_type[barcode_info[i][0]] == 'zero':
            #    continue
            data_list['pathology_label'].append(barcode_type[barcode_info[i][0]])
            data_list['component_label'].append(barcode_info[i][3])
            data_list['X'].append(barcode_info[i][1])
            data_list['Y'].append(barcode_info[i][2])

        '''
        data_list_pd = pd.DataFrame(data_list)
        #data_list_pd.to_csv('/cluster/home/t116508uhn/64630/omnipath_ccc_th95_tissue_plot_withFeature_woBlankEdges.csv', index=False)
        #df_test = pd.read_csv('/cluster/home/t116508uhn/64630/omnipath_ccc_th95_tissue_plot_withFeature_woBlankEdges.csv')
        #set1 = altairThemes.get_colour_scheme("Set1", len(data_list_pd["component_label"].unique()))
        set1 = altairThemes.get_colour_scheme("Set1", id_label)
        set1[0] = '#000000'

        chart = alt.Chart(data_list_pd).mark_point(filled=True, opacity = 1).encode(
            alt.X('X', scale=alt.Scale(zero=False)),
            alt.Y('Y', scale=alt.Scale(zero=False)),
            shape = alt.Shape('pathology_label:N'), #shape = "pathology_label",
            color=alt.Color('component_label:N', scale=alt.Scale(range=set1)),
            tooltip=['component_label']
        )#.configure_legend(labelFontSize=6, symbolLimit=50)
        # output 2
        save_path = '/cluster/home/t116508uhn/64630/'
        '''
        #chart.save(save_path+args.data_name+'_altair_plot_bothAbove98_3dim_tanh_3heads_l2attention_th95_'+filename[run_time]+'.html')
        #chart.save(save_path+args.data_name+'_filtered_CCL19_CCR7_input_graph.html') #
        #chart.save(save_path+args.data_name+'_CCL19_CCR7_th95_graph.html') #selective_
        #chart.save(save_path+args.data_name+'_IL21_IL21R_attention_only_th95_l2attention_'+filename[run_time]+'.html') #
        #chart.save(save_path+args.data_name+'_CCL19_CCR7_attention_only_th95_l1attention_'+filename[run_time]+'.html') #
        #chart.save(save_path+args.data_name+'_altair_plot_bothAbove98_th99p9_3dim_tanh_h512_l2attention_'+filename[run_time]+'.html') #filtered_l2attention_
        #chart.save(save_path+'altair_plot_98th_bothAbove98_3dim_tanh_h2048_'+filename[run_time]+'.html')
        #chart.save(save_path+'altair_plot_bothAbove98_3dim_'+filename[run_time]+'.html')
        #chart.save(save_path+'altair_plot_97th_bothAbove98_3d_input.html')
        #chart.save(save_path+'altair_plot_97th_bothAbove98_'+filename[run_time]+'.html')
        #chart.save(save_path+'region_of_interest_r1.html')
        #chart.save(save_path+'altair_plot_95_withlrFeature_bothAbove98_'+filename[run_time]+'.html')
        #chart.save(save_path+'altair_plot_'+'80'+'th_'+filename[run_time]+'.html')
        #chart.save(save_path+'altair_plot_'+'80'+'th_'+filename[run_time]+'.html')

        ##############
        '''
        region_list =[2, 3, 9, 11, 4, 5, 7]

        spot_interest_list = []
        for i in range (0, len(barcode_info)):
            if data_list['component_label'][i] in region_list:

                spot_interest_list.append(i)
        '''
        ###############
        csv_record = []
        csv_record.append(['from_cell', 'to_cell', 'ligand', 'receptor', 'attention_score', 'component', 'from_id', 'to_id'])
        '''
        for j in range (0, len(barcode_info)):
            for i in range (0, len(barcode_info)):
                for k in range (0, len(lig_rec_dict[i][j])):
                    csv_record.append([barcode_info[i][0], barcode_info[j][0], lig_rec_dict[i][j][k][0], lig_rec_dict[i][j][k][1], -1, barcode_info[i][3], i, j])

        '''

        for j in range (0, len(barcode_info)):
            for i in range (0, len(barcode_info)):

                if i==j:
                    if len(lig_rec_dict[i][j])==0:
                        continue

                atn_score_list = attention_scores[i][j]
                for k in range (0, len(atn_score_list)):
                    if attention_scores[i][j][k] >= threshold_down and attention_scores[i][j][k] <= threshold_up: 
                        if barcode_info[i][3]==0:
                            print('error')
                        elif barcode_info[i][3]==1:
                            csv_record.append([barcode_info[i][0], barcode_info[j][0], lig_rec_dict[i][j][k][0], lig_rec_dict[i][j][k][1], min_attention_score + attention_scores[i][j][k], '0-single', i, j])
                        else:
                            csv_record.append([barcode_info[i][0], barcode_info[j][0], lig_rec_dict[i][j][k][0], lig_rec_dict[i][j][k][1], min_attention_score + attention_scores[i][j][k], barcode_info[i][3], i, j])
        '''
        df = pd.DataFrame(csv_record)
        df.to_csv('/cluster/home/t116508uhn/64630/input_test.csv', index=False, header=False)
        ############
        alt.themes.register("publishTheme", altairThemes.publishTheme)
        # enable the newly registered theme
        alt.themes.enable("publishTheme")
        inFile = '/cluster/home/t116508uhn/64630/input_test.csv' #sys.argv[1]
        df = readCsv(inFile)
        df = preprocessDf(df)
        outPathRoot = inFile.split('.')[0]
        p = plot(df)
        #outPath = '/cluster/home/t116508uhn/64630/test_hist_'+args.data_name+'_'+filename[run_time]+'_th99p7_h512_l2attention_'+str(len(csv_record))+'edges.html' #filteredl2attention__ l2attention_
        outPath = '/cluster/home/t116508uhn/64630/test_hist_'+args.data_name+'_'+filename[run_time]+'_selective_only_Tcellzone_th90_h512_'+str(len(csv_record))+'edges.html' #filteredl2attention__ l2attention_
        p.save(outPath)	# output 3
        '''
        ###########	
        #run = 1
        #csv_record_dict = defaultdict(list)
        print('records found %d'%len(csv_record))
        for i in range (1, len(csv_record)):
            key_value = str(csv_record[i][6]) +'-'+ str(csv_record[i][7]) + '-' + csv_record[i][2] + '-' + csv_record[i][3]# + '-'  + str( csv_record[i][5])
            csv_record_dict[key_value].append([csv_record[i][4], run])


for key_value in csv_record_dict.keys():
    run_dict = defaultdict(list)
    for scores in csv_record_dict[key_value]:
        run_dict[scores[1]].append(scores[0])
    
    for runs in run_dict.keys():
        run_dict[runs] = np.mean(run_dict[runs])
        
        
    csv_record_dict[key_value] = []
    for runs in run_dict.keys():
        csv_record_dict[key_value].append([run_dict[runs],runs])

        
        
combined_score_distribution = []
csv_record = []
csv_record.append(['from_cell', 'to_cell', 'ligand', 'receptor', 'attention_score', 'component', 'from_id', 'to_id'])
csv_record_intersect_dict = defaultdict(dict) 
for key_value in csv_record_dict.keys():
    if len(csv_record_dict[key_value])>=1: #3: #((total_runs*80)/100):
        item = key_value.split('-')
        i = int(item[0])
        j = int(item[1])
        ligand = item[2]
        receptor = item[3]        
        ###
        
        score = 0
        for k in range (0, len(csv_record_dict[key_value])):
            score = score + csv_record_dict[key_value][k][0]
        score = score/len(csv_record_dict[key_value]) # take the average score
        ''''''
        ###        
        label = -1 #csv_record_dict[key_value][total_runs-1][1]
        #score = csv_record_dict[key_value][total_runs-1][0] #score/total_runs
        if ligand+'-'+receptor not in csv_record_intersect_dict or label not in csv_record_intersect_dict[ligand+'-'+receptor]:
            csv_record_intersect_dict[ligand+'-'+receptor][label] = []
        
        csv_record_intersect_dict[ligand+'-'+receptor][label].append(score)
        csv_record.append([barcode_info[i][0], barcode_info[j][0], ligand, receptor, score, label, i, j])
        combined_score_distribution.append(score)
        
print('common LR count %d'%len(csv_record))
            
threshold_value =  np.percentile(combined_score_distribution,50)
connecting_edges = np.zeros((len(barcode_info),len(barcode_info)))  
count = 0
for k in range (1, len(csv_record)):
    ligand = csv_record[k][2]
    receptor = csv_record[k][3]
    if csv_record[k][4] >= threshold_value:        
        i = csv_record[k][6]
        j = csv_record[k][7]
        connecting_edges[i][j]=1
        count = count+1
        
print("edges after thresholding further is %d"%count)

graph = csr_matrix(connecting_edges)
n_components, labels = connected_components(csgraph=graph,directed=True, connection = 'weak',  return_labels=True) #
print('number of component %d'%n_components)

count_points_component = np.zeros((n_components))
for i in range (0, len(labels)):
     count_points_component[labels[i]] = count_points_component[labels[i]] + 1

print(count_points_component)

id_label = 2 # initially all are zero. =1 those who have self edge but above threshold. >= 2 who belong to some component
index_dict = dict()
for i in range (0, count_points_component.shape[0]):
    if count_points_component[i]>1:
        index_dict[i] = id_label
        id_label = id_label+1

print(id_label)

for i in range (0, len(barcode_info)):
#    if barcode_info[i][0] in barcode_label:
    if count_points_component[labels[i]] > 1:
        barcode_info[i][3] = index_dict[labels[i]] #2
    elif connecting_edges[i][i] == 1 and len(lig_rec_dict[i][i])>0: 
        barcode_info[i][3] = 1
    else:
        barcode_info[i][3] = 0

# update the label based on new component numbers
#max opacity

for record in range (1, len(csv_record)):
    i = csv_record[record][6]
    label = barcode_info[i][3]
    csv_record[record][5] = label
    

###########	

exist_spot = defaultdict(list)
for record_idx in range (1, len(csv_record)):
    record = csv_record[record_idx]
    i = record[6]
    pathology_label = barcode_type[barcode_info[i][0]]
    component_label = record[5]
    X = barcode_info[i][1]
    Y = -barcode_info[i][2]
    opacity = record[4]
    exist_spot[i].append([pathology_label, component_label, X, Y, opacity])
    '''
    j = record[7]
    pathology_label = barcode_type[barcode_info[j][0]]
    component_label = record[5]
    X = barcode_info[j][1]
    Y = -barcode_info[j][2]
    opacity = record[4]   
    exist_spot[j].append([pathology_label, component_label, X, Y, opacity])
    '''
    
opacity_list = []
for i in exist_spot:
    sum_opacity = []
    for edges in exist_spot[i]:
        sum_opacity.append(edges[4])
        
    avg_opacity = np.max(sum_opacity) #np.mean(sum_opacity)
    opacity_list.append(avg_opacity)
    
    exist_spot[i]=[exist_spot[i][0][0], exist_spot[i][0][1], exist_spot[i][0][2], exist_spot[i][0][3], avg_opacity]

min_opacity = np.min(opacity_list)
max_opacity = np.max(opacity_list)
min_opacity = min_opacity - 5

data_list=dict()
data_list['pathology_label']=[]
data_list['component_label']=[]
data_list['X']=[]
data_list['Y']=[]   
data_list['opacity']=[] 

for i in range (0, len(barcode_info)):
    #if barcode_type[barcode_info[i][0]] == 'zero':
    #    continue
        
    if i in exist_spot:
        data_list['pathology_label'].append(exist_spot[i][0])
        data_list['component_label'].append(exist_spot[i][1])
        data_list['X'].append(exist_spot[i][2])
        data_list['Y'].append(exist_spot[i][3])
        data_list['opacity'].append((exist_spot[i][4]-min_opacity)/(max_opacity-min_opacity))
        
    else:
        data_list['pathology_label'].append(barcode_type[barcode_info[i][0]])
        data_list['component_label'].append(0)
        data_list['X'].append(barcode_info[i][1])
        data_list['Y'].append(-barcode_info[i][2])
        data_list['opacity'].append(0.1)


#id_label= len(list(set(data_list['component_label'])))
import altairThemes # assuming you have altairThemes.py at your current directoy or your system knows the path of this altairThemes.py.
data_list_pd = pd.DataFrame(data_list)
set1 = altairThemes.get_colour_scheme("Set1", id_label)
set1[0] = '#000000'
chart = alt.Chart(data_list_pd).mark_point(filled=True, opacity = 1).encode(
    alt.X('X', scale=alt.Scale(zero=False)),
    alt.Y('Y', scale=alt.Scale(zero=False)),
    shape = alt.Shape('pathology_label:N'), #shape = "pathology_label",
    color=alt.Color('component_label:N', scale=alt.Scale(range=set1)),
    #opacity=alt.Opacity('opacity:N'), #"opacity",
    tooltip=['component_label'] #,'opacity'
)#.configure_legend(labelFontSize=6, symbolLimit=50)

# output 6
save_path = '/cluster/home/t116508uhn/64630/'
chart.save(save_path+'altair_plot_test.html')
chart.save(save_path+'altair_plot_'+args.data_name+'_opacity_bothAbove98_th97_90_3dim_tanh_h512_l1l2attention_combined_5runs_'+str(len(csv_record))+'edges.html')

########################################################################################################################
threshold_value =  np.percentile(combined_score_distribution,0)
csv_record_temp = []
csv_record_temp.append(csv_record[0])
for k in range (1, len(csv_record)):
    if csv_record[k][4] >= threshold_value:    
        csv_record_temp.append(csv_record[k])
   

#for k in range (1, len(csv_record_temp)):
#    csv_record_temp[k][5] = ccc_too_many_cells_LUAD_dict[csv_record_temp[k][0]]
        
i=0
j=0
csv_record_temp.append([barcode_info[i][0], barcode_info[j][0], 'no-ligand', 'no-receptor', 0, 0, i, j])
df = pd.DataFrame(csv_record_temp) # output 4
#df.to_csv('/cluster/home/t116508uhn/64630/input_test_'+args.data_name+'_h512_filtered_l2attention_edges'+str(len(csv_record))+'_combined_th90_100percent_totalruns_'+str(total_runs)+'.csv', index=False, header=False) #
#df.to_csv('/cluster/home/t116508uhn/64630/input_test_'+args.data_name+'_h512_l2attention_edges'+str(len(csv_record))+'_combined_th98p5_100percent_totalruns_'+str(total_runs)+'.csv', index=False, header=False) #
df.to_csv('/cluster/home/t116508uhn/64630/input_test.csv', index=False, header=False)
            
            
            
            

