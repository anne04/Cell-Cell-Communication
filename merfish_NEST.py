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
import altairThemes
import altair as alt
import gc

alt.themes.register("publishTheme", altairThemes.publishTheme)
# enable the newly registered theme
alt.themes.enable("publishTheme")

##########################################################
# written by GW                                                                                                                                                                                                                                                                                                                                                              
import scipy.stats

#sys.path.append("/home/gw/code/utility/altairThemes/")
#if True:  # In order to bypass isort when saving
#    import altairThemes

def readCsv(x):
  """Parse file."""
  #colNames = ["method", "benchmark", "start", "end", "time", "memory"]
  df = pd.read_csv(x, sep=",")

  return df

def preprocessDf(df):
  """Transform ligand and receptor columns."""
  df["ligand-receptor"] = df["ligand"] + '-' + df["receptor"]
  df["component"] = df["component"] #.astype(str).str.zfill(2)

  return df

def statOrNan(xs, ys):
  if len(xs) == 0 or len(ys) == 0:
    return None
  else:
    return scipy.stats.mannwhitneyu(xs, ys)

def summarizeStats(df, feature):
  meanRes = df.groupby(["benchmark", "method"])[feature].mean()
  statRes = df.groupby("benchmark").apply(lambda x: post.posthoc_ttest(x, val_col = feature, group_col = "method", p_adjust = "fdr_bh"))

  return (meanRes, statRes)

def writeStats(stats, feature, outStatsPath):
  stats[0].to_csv(outStatsPath + "_feature_" + feature + "_mean.csv")
  stats[1].to_csv(outStatsPath + "_feature_" + feature + "_test.csv")

  return

def plot(df):
  set1 = altairThemes.get_colour_scheme("Set1", len(df["component"].unique()))
  set1[0] = '#000000'
  base = alt.Chart(df).mark_bar().encode(
            x=alt.X("ligand-receptor:N", axis=alt.Axis(labelAngle=45), sort='-y'),
            y=alt.Y("count()"),
            color=alt.Color("component:N", scale = alt.Scale(range=set1)),
            order=alt.Order("component:N", sort="ascending"),
            tooltip=["component"]
        )
  p = base

  return p

def totalPlot(df, features, outPath):

  p = alt.hconcat(*map(lambda x: plot(df, x), features))

  outPath = outPath + "_boxplot.html"

  p.save(outPath)

  return

###################

meta_file = '/cluster/projects/schwartzgroup/fatema/MESSI/input/merfish_meta_Virgin_Parenting_Female.csv'
df = pd.read_csv(meta_file)
microglia_cell_id = []
for i in range (0, df["Cell_ID"].shape[0]):
    if df["Cell_class"][i] == 'Microglia':
        microglia_cell_id.append(df["Cell_ID"][i])
        
with gzip.open("/cluster/projects/schwartzgroup/fatema/find_ccc/merfish_microglia", 'wb') as fp:  #b, a:[0:5]  _filtered 
    pickle.dump(microglia_cell_id, fp)
 



##########
options =  'Female_Naive_Excitatory' #'Female_Virgin_ParentingExcitatory' #  'Female_Parenting_Excitatory' # 
import argparse
parser = argparse.ArgumentParser()
parser.add_argument( '--data_path', type=str, default="/cluster/projects/schwartzgroup/fatema/find_ccc/merfish_mouse_cortex/" , help='The path to dataset') 
parser.add_argument( '--embedding_data_path', type=str, default='new_alignment/Embedding_data_ccc_rgcn/' , help='The path to attention') 
parser.add_argument( '--data_name', type=str, default='messi_merfish_data_'+options, help='The name of dataset') #
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
#animal_id = 16
#animal_id = 24 #data_sets_gatconv[0][4][0][0]
#bregma = [0.11, 0.16, 0.21, 0.26] #data_sets_gatconv[0][4][0][3] []
#bregma_id = 0
bregma_interest = [-0.09, 0.21]
#for bregma_id in range (0, len(bregma)): #bregma:
for animal_id in [1]:   
    print('animal id:%d'%(animal_id))
    #print('animal id:%d, bregma: %g'%(animal_id, bregma[bregma_id]))
    z_index_yes = 1
    barcode_info = []
    cell_vs_gene_list = []
    total_cell = 0
    sample_index = 0
    z_index = -100
    for index in range (0,len(data_sets_gatconv)):
        if data_sets_gatconv[index][4][0][0] == animal_id : #and data_sets_gatconv[index][4][0][3] in bregma_interest: #== bregma[bregma_id]:
            sample_index = index
            z_index = z_index + 100
            cell_barcodes = data_sets_gatconv[index][0]
            coordinates = data_sets_gatconv[index][1]
            # plot everything at 0,0 point
            
            add_x = -1*min(coordinates[:,0])
            add_y = -1*min(coordinates[:,1])
            for i in range (0, coordinates.shape[0]):
                coordinates[i,0] = coordinates[i,0] + add_x 
                coordinates[i,1] = coordinates[i,1] + add_y
            
            cell_vs_gene = data_sets_gatconv[index][2]
            cell_vs_gene_list.append(cell_vs_gene)
            total_cell = total_cell + cell_vs_gene.shape[0]
            bregma = data_sets_gatconv[index][4][0][3]
            neuron_type = data_sets_gatconv[index][5]
		
            print('animal id:%d, bregma: %g'%(animal_id, bregma))
            print('index:%d, cell count: %d'%(index, cell_vs_gene.shape[0]))
            if z_index_yes == 1:
                for i in range (0, len(cell_barcodes)):
                    barcode_info.append([cell_barcodes[i], coordinates[i,0], coordinates[i,1], z_index,0, neuron_type[i][0]])
                    i=i+1
            else:
                for i in range (0, len(cell_barcodes)):
                    barcode_info.append([cell_barcodes[i], coordinates[i,0], coordinates[i,1], 0, neuron_type[i][0]])
                    i=i+1       

                #break


    ##################################################################
    if z_index_yes == 1:
        coordinates = np.zeros((total_cell, 3))
    else:
        coordinates = np.zeros((total_cell, 2))


    temp_x = defaultdict(list)
    temp_y = defaultdict(list)
    for i in range (0, len(barcode_info)): 
        coordinates[i][0] = barcode_info[i][1]
        coordinates[i][1] = barcode_info[i][2]

        if z_index_yes == 1:
            coordinates[i][2] = barcode_info[i][3]
            temp_x[coordinates[i][2]].append(coordinates[i][0])
            temp_y[coordinates[i][2]].append(coordinates[i][1])


	
    '''
    plt.clf()
    node_xyz = np.array(coordinates)
    fig = plt.figure()
    ax = fig.add_subplot(projection="3d")
    # Plot the nodes - alpha is scaled by "depth" automatically
    

    
    ax.scatter(*node_xyz.T, s=.1, alpha=0.5, c="#4F4F4F")
    _format_axes(ax)
    #fig.tight_layout()
    save_path = '/cluster/home/t116508uhn/64630/'
    plt.savefig(save_path+'3d_merfish.svg', dpi=400) #
    plt.clf()    

    
    # color plotting of the bregma slides #
    color_list = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf', '#29aecf', '#8c014b']
    data_list=dict()
    data_list['label']=[]
    data_list['X']=[]
    data_list['Y']=[]   

    j = 0
    for bregma in temp_x.keys():
        for i in range (0, len(temp_x[bregma])):
            data_list['X'].append(temp_x[bregma][i])
            data_list['Y'].append(temp_y[bregma][i])
            data_list['label'].append(bregma)     
            
        j = j+1

    
    data_list_pd = pd.DataFrame(data_list)
    set1 = color_list 
    chart = alt.Chart(data_list_pd).mark_point(filled=True, opacity = 1).encode(
        alt.X('X', scale=alt.Scale(zero=False)),
        alt.Y('Y', scale=alt.Scale(zero=False)),
        color=alt.Color('label:N', scale=alt.Scale(range=set1)),
        tooltip=['label'] #,'opacity'
    )#.configure_legend(labelFontSize=6, symbolLimit=50)
    save_path = '/cluster/home/t116508uhn/64630/'
    chart.save(save_path+'merfish_temp_animalID'+str(animal_id)+'.html')
    '''
	
    ####################################################################
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
    cell_percentile = []
    for i in range (0, cell_vs_gene.shape[0]):
        y = sorted(cell_vs_gene[i])
        x = range(1, len(y)+1)
        kn = KneeLocator(x, y, curve='convex', direction='increasing')
        kn_value = y[kn.knee-1]
        cell_percentile.append([np.percentile(y, 10), np.percentile(y, 20),np.percentile(y, 98), np.percentile(y, 98), kn_value]) 
       
    #######################################################

    
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
                        if gene_rec in cell_cell_contact and distance_matrix[i,j] > spot_diameter//3: #because I set spot_diameter very big! And I don't know the actual size. but visually, it looks like spot_dia//3
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
with gzip.open("/cluster/projects/schwartzgroup/fatema/find_ccc/" +args.data_name+'_id'+str(animal_id)+'_adjacency_records_GAT_selective_lr_STnCCC_separate_'+'bothAbove_cell95th_xyz_3d', 'wb') as fp:  #b, a:[0:5]  _filtered 
    pickle.dump([row_col, edge_weight, lig_rec], fp)
             
with gzip.open("/cluster/projects/schwartzgroup/fatema/find_ccc/" +args.data_name+'_id'+str(animal_id)+'_cell_vs_gene_xyz_quantile_transformed', 'wb') as fp:  #b, a:[0:5]   _filtered
	pickle.dump(cell_vs_gene, fp)

########### 3D ###################################################################################

with gzip.open("/cluster/projects/schwartzgroup/fatema/find_ccc/" +args.data_name+'_id'+str(animal_id)+'_index_4_10_adjacency_records_GAT_selective_lr_STnCCC_separate_'+'bothAbove_cell95th_xyz_3d', 'wb') as fp:  #b, a:[0:5]  _filtered 
    pickle.dump([row_col, edge_weight, lig_rec], fp)
             
with gzip.open("/cluster/projects/schwartzgroup/fatema/find_ccc/" +args.data_name+'_id'+str(animal_id)+'_index_4_10_cell_vs_gene_xyz_quantile_transformed', 'wb') as fp:  #b, a:[0:5]   _filtered
	pickle.dump(cell_vs_gene, fp)

with gzip.open("/cluster/projects/schwartzgroup/fatema/find_ccc/" +args.data_name+'_id'+str(animal_id)+'_index_4_10_barcode_info', 'wb') as fp:  #b, a:[0:5]   _filtered
    pickle.dump(barcode_info, fp)
    
with gzip.open("/cluster/projects/schwartzgroup/fatema/find_ccc/" +args.data_name+'_id'+str(animal_id)+'_index_4_10_coordinates', 'wb') as fp:  #b, a:[0:5]   _filtered
    pickle.dump(coordinates, fp)


#### read for 3D  (x, y, bregma) #######################################################
animal_id = 1
with gzip.open("/cluster/projects/schwartzgroup/fatema/find_ccc/" +args.data_name+'_id'+str(animal_id)+'_index_4_10_adjacency_records_GAT_selective_lr_STnCCC_separate_'+'bothAbove_cell95th_xyz_3d', 'rb') as fp:  #b, a:[0:5]  _filtered 
    row_col, edge_weight, lig_rec = pickle.load(fp)
             

with gzip.open("/cluster/projects/schwartzgroup/fatema/find_ccc/" +args.data_name+'_id'+str(animal_id)+'_index_4_10_barcode_info', 'rb') as fp:  #b, a:[0:5]   _filtered
    barcode_info = pickle.load(fp)
    
with gzip.open("/cluster/projects/schwartzgroup/fatema/find_ccc/" +args.data_name+'_id'+str(animal_id)+'_index_4_10_coordinates', 'rb') as fp:  #b, a:[0:5]   _filtered
    coordinates = pickle.load(fp)

########## multiple slide ####################################################

with gzip.open('/cluster/projects/schwartzgroup/fatema/find_ccc/' +args.data_name+'_id'+str(animal_id)+'_adjacency_records_GAT_selective_lr_STnCCC_separate_'+'bothAbove_cell98th_xyz_3d', 'rb') as fp: 
    row_col, edge_weight, lig_rec = pickle.load(fp)

with gzip.open("/cluster/projects/schwartzgroup/fatema/find_ccc/" +args.data_name+'_id'+str(animal_id)+'_barcode_info', 'rb') as fp:  #b, a:[0:5]   _filtered
    barcode_info = pickle.load(fp)

with gzip.open("/cluster/projects/schwartzgroup/fatema/find_ccc/" +args.data_name+'_id'+str(animal_id)+'_coordinates', 'rb') as fp:  #b, a:[0:5]   _filtered
    coordinates = pickle.load(fp)



###########################################################Visualization starts ##################
animal_id = 24 #19 #
bregma_id = 0
bregma = [0.11, 0.16, 0.21, 0.26] #data_sets_gatconv[0][4][0][3] []

with gzip.open("/cluster/projects/schwartzgroup/fatema/find_ccc/" +args.data_name+'_id'+str(animal_id)+'_bregma'+str(bregma[bregma_id])+'_adjacency_records_GAT_selective_lr_STnCCC_separate_'+'bothAbove_cell95th_3d', 'rb') as fp:  #b, a:[0:5]  _filtered 
#with gzip.open("/cluster/projects/schwartzgroup/fatema/find_ccc/" +args.data_name+'_id'+str(animal_id)+'_adjacency_records_GAT_selective_lr_STnCCC_separate_'+'bothAbove_cell98th_xyz_3d_ThDistance500', 'rb') as fp:  #b, a:[0:5]  _filtered 
    row_col, edge_weight, lig_rec = pickle.load(fp)


with gzip.open("/cluster/projects/schwartzgroup/fatema/find_ccc/" +args.data_name+'_id'+str(animal_id)+'_bregma'+str(bregma[bregma_id])+'_barcode_info', 'rb') as fp:  #b, a:[0:5]   _filtered
    barcode_info = pickle.load(fp)
    
with gzip.open("/cluster/projects/schwartzgroup/fatema/find_ccc/" +args.data_name+'_id'+str(animal_id)+'_bregma'+str(bregma[bregma_id])+'_coordinates', 'rb') as fp:  #b, a:[0:5]   _filtered
    coordinates = pickle.load(fp)

      
###############################################
datapoint_size = len(barcode_info)    
barcode_type=dict()
for i in range (0, datapoint_size):
    barcode_type[barcode_info[i][0]] = 0 
    

'''lig_rec_dict = []
for i in range (0, datapoint_size):
    lig_rec_dict.append([])  
    for j in range (0, datapoint_size):	
        lig_rec_dict[i].append([])   
        lig_rec_dict[i][j] = []
'''
lig_rec_dict = defaultdict(dict)
total_type = np.zeros((2))        
for index in range (0, len(row_col)):
    i = row_col[index][0]
    j = row_col[index][1]
    #########################
    if i not in lig_rec_dict:
        lig_rec_dict[i][j] = []
    elif j not in  lig_rec_dict[i]:
        lig_rec_dict[i][j]=[]
    #########################
    lig_rec_dict[i][j].append(lig_rec[index])  
        

########################################################
'''
cell_kept = []
for cell_id in microglia_cell_id:
    if cell_id in barcode_type:
        cell_kept.append(cell_id)
'''     
###########
filename = ["r1", "r2", "r3", "r4", "r5", "r6", "r7", "r8", "r9", "r10"]
total_runs = 3 #5
start_index = 0

distribution_rank = []
all_edge_sorted_by_rank = []
for layer in range (0, 2):
    distribution_rank.append([])
    all_edge_sorted_by_rank.append([])

layer = -1
percentage_value = 0
for l in [2, 3]: # 2 = layer 2, 3 = layer 1
    layer = layer + 1
    csv_record_dict = defaultdict(list)
    for run_time in range (start_index, start_index+total_runs):
        gc.collect()
        #run_time = 2
        run = run_time
        print('run %d'%run)
        X_attention_filename = 'new_alignment/Embedding_data_ccc_rgcn/merfish_data_Female_Naive_Excitatory/'+'merfish_data_id1_cellchat_nichenet_threshold_distance_bothAbove_cell98th_tanh_3dim_xyz_manySlides_'+filename[run_time]+'_attention_l1.npy'
        #X_attention_filename = 'new_alignment/Embedding_data_ccc_rgcn/merfish_data_Female_Naive_Excitatory/'+'merfish_data_id1_cellchat_nichenet_threshold_distance_bothAbove_cell95th_tanh_3dim_xyz_'+filename[run_time]+'_attention_l1.npy' 
        #X_attention_filename = args.embedding_data_path + args.data_name + '/' + 'merfish_data_'+options+'_id'+str(animal_id)+'_bregma_p11_cellchat_nichenet_threshold_distance_bothAbove_cell95th_tanh_3dim_'+filename[run_time]+'_attention_l1.npy'   
        X_attention_bundle = np.load(X_attention_filename, allow_pickle=True) #_withFeature
    
        X_attention_bundle = np.load(X_attention_filename, allow_pickle=True) #_withFeature
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
        
        print('min attention score %g, total edges %d'%(min_attention_score, len(distribution)))
        ##############
        #plt.clf()
        #plt.hist(distribution, color = 'blue',bins = int(len(distribution)/5))
        #save_path = '/cluster/home/t116508uhn/64630/'
        #plt.savefig(save_path+'distribution_region_of_interest_'+filename[run_time]+'_l2attention_score.svg', dpi=400) # _CCL19_CCR7
        #plt.clf()

    
        ##############
        
        #hold_attention_score = copy.deepcopy(attention_scores)  
        #attention_scores = copy.deepcopy(hold_attention_score)  
        ####################################################################################
       
        ###########################
        
        ccc_index_dict = dict()
        threshold_down =  np.percentile(sorted(distribution), 0)
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
    
    
        data_list_pd = pd.DataFrame(data_list)
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
        #chart.save(save_path+args.data_name+'_altair_plot_bothAbove98_3dim_tanh_3heads_l2attention_th95_'+filename[run_time]+'.html')

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
  
    #######################################
    
    all_edge_list = []
    for key_value in csv_record_dict.keys():
        edge_score_runs = []
        edge_score_runs.append(key_value)
        for runs in csv_record_dict[key_value]:
            edge_score_runs.append(runs[0]) # 
            
        all_edge_list.append(edge_score_runs) # [key_value, score_by_run1, score_by_run2, etc.]

    ## Find the rank product #####################################################################
    ## all_edge_list has all the edges along with their scores for different runs in following format: 
    ## [edge_1_info, score_by_run1, score_by_run2, etc.], [edge_2_info, score_by_run1, score_by_run2, etc.], ..., [edge_N_info, score_by_run1, score_by_run2, etc.]
    edge_rank_dictionary = defaultdict(list)
    # sort the all_edge_list by each run's rank 
    print('total runs %d'%total_runs)
    for runs in range (0, total_runs):
        sorted_list_temp = sorted(all_edge_list, key = lambda x: x[runs+1], reverse=True) # sort based on attention score by current run: large to small
        for rank in range (0, len(sorted_list_temp)):
            edge_rank_dictionary[sorted_list_temp[rank][0]].append(rank+1) # small rank being high attention

    max_weight = len(sorted_list_temp)
    all_edge_vs_rank = []
    for key_val in edge_rank_dictionary.keys():
        rank_product = 1
        attention_score_list = csv_record_dict[key_value]
        avg_score = 0 #[]
        total_weight = 0
        for i in range (0, len(edge_rank_dictionary[key_val])):
            rank_product = rank_product * edge_rank_dictionary[key_val][i]
            weight_by_run = max_weight - edge_rank_dictionary[key_val][i]
            avg_score = avg_score + attention_score_list[i][0] * weight_by_run
            #avg_score.append(attention_score_list[i][0])  
            total_weight = total_weight + weight_by_run
            
        avg_score = avg_score/total_weight # lower weight being higher attention np.max(avg_score) #
        all_edge_vs_rank.append([key_val, rank_product**(1/total_runs), avg_score])  # small rank being high attention
        distribution_rank[layer].append(rank_product**(1/total_runs))
    
    all_edge_sorted_by_rank[layer] = sorted(all_edge_vs_rank, key = lambda x: x[1]) # small rank being high attention 
    

percentage_value = 20 # top 20th percentile rank, low rank means higher attention score
csv_record_intersect_dict = defaultdict(list)
edge_score_intersect_dict = defaultdict(list)
for layer in range (0, 2):
    threshold_up = np.percentile(distribution_rank[layer], percentage_value) #np.round(np.percentile(distribution_rank[layer], percentage_value),2)
    for i in range (0, len(all_edge_sorted_by_rank[layer])):
        if all_edge_sorted_by_rank[layer][i][1] <= threshold_up:
            csv_record_intersect_dict[all_edge_sorted_by_rank[layer][i][0]].append(i)
            edge_score_intersect_dict[all_edge_sorted_by_rank[layer][i][0]].append(all_edge_sorted_by_rank[layer][i][2])


###########################################################################################################################################
## get the aggregated rank for all the edges ##
distribution_temp = []
for key_value in csv_record_intersect_dict.keys():  
    arg_index = np.argmin(csv_record_intersect_dict[key_value])
    csv_record_intersect_dict[key_value] = np.min(csv_record_intersect_dict[key_value]) # smaller rank being the higher attention
    edge_score_intersect_dict[key_value] = edge_score_intersect_dict[key_value][arg_index]
    distribution_temp.append(csv_record_intersect_dict[key_value]) 
##############################################################################
csv_record_dict = copy.deepcopy(csv_record_intersect_dict)
combined_score_distribution = []
csv_record = []
csv_record.append(['from_cell', 'to_cell', 'ligand', 'receptor', 'edge_rank', 'component', 'from_id', 'to_id', 'attention_score'])
for key_value in csv_record_dict.keys():
    item = key_value.split('-')
    i = int(item[0])
    j = int(item[1])
    ligand = item[2]
    receptor = item[3]        
    edge_rank = csv_record_dict[key_value]        
    score = edge_score_intersect_dict[key_value] # weighted average attention score, where weight is the rank, lower rank being higher attention score
    label = -1 
    csv_record.append([barcode_info[i][0], barcode_info[j][0], ligand, receptor, edge_rank, label, i, j, score])
    combined_score_distribution.append(score)

  
        
print('common LR count %d'%len(csv_record))

score_distribution = []
for k in range (1, len(csv_record)):
    score_distribution.append(csv_record[k][8])

min_score = np.min(score_distribution)
max_score = np.max(score_distribution)
for k in range (1, len(csv_record)):
    scaled_score = (csv_record[k][8]-min_score)/(max_score-min_score) 
    csv_record[k][8] = scaled_score


##### save the file for downstream analysis ########
csv_record_final = []
csv_record_final.append(csv_record[0])
for k in range (1, len(csv_record)):
    ligand = csv_record[k][2]
    receptor = csv_record[k][3]
    #if ligand =='CCL19' and receptor == 'CCR7':
    csv_record_final.append(csv_record[k])


df = pd.DataFrame(csv_record_final) # output 4
df.to_csv('/cluster/home/t116508uhn/64630/NEST_combined_rank_product_output_'+args.data_name+'_id'+str(animal_id)+'_top20.csv', index=False, header=False)
#df.to_csv('/cluster/home/t116508uhn/64630/NEST_combined_rank_product_output_'+args.data_name+'_id'+str(animal_id)+'_bregma'+str(bregma[bregma_id])+'.csv', index=False, header=False)

######################### read the NEST output in csv format ####################################################
current_directory = '/cluster/home/t116508uhn/64630/'
filename_str = 'NEST_combined_rank_product_output_'+args.data_name+'_id'+str(animal_id)+'_top20.csv'
#filename_str = 'NEST_combined_rank_product_output_'+args.data_name+'_id'+str(animal_id)+'_bregma'+str(bregma[bregma_id])+'.csv'
#filename_str = 'NEST_combined_rank_product_output_'+args.data_name+'.csv'
inFile = current_directory +filename_str 
df = pd.read_csv(inFile, sep=",")
csv_record = df.values.tolist()

## sort the edges based on their rank (column 4 = score) column, low to high, low being higher attention score
top_edge_count = 10000 #len(csv_record) # 3000 = microglia and 1000=general
csv_record = sorted(csv_record, key = lambda x: x[4])

## add the column names and take first top_edge_count edges
# columns are: from_cell, to_cell, ligand_gene, receptor_gene, attention_score, component, from_id, to_id
df_column_names = list(df.columns)
csv_record_final = [df_column_names] + csv_record[0:top_edge_count]

## add a dummy row at the end for the convenience of histogram preparation (to keep the color same as altair plot)
i=0
j=0
csv_record_final.append([barcode_info[i][0], barcode_info[j][0], 'no-ligand', 'no-receptor', 0, 0, i, j]) # dummy for histogram

############################################################################################################################


############################################### Optional filtering ########################################################
'''
## change the csv_record_final here if you want histogram for specific components/regions only. e.g., if you want to plot only stroma region, or tumor-stroma regions etc.    ##
#region_of_interest = [...] 
csv_record_final_temp = []
csv_record_final_temp.append(csv_record_final[0])
for record_idx in range (1, len(csv_record_final)-1): #last entry is a dummy for histograms, so ignore it.
    # if one of them is microglia, then plot them
    i = csv_record_final[record_idx][6]
    j = csv_record_final[record_idx][7]
    if (barcode_info[i][4]=='Microglia' and barcode_info[j][4]!='Microglia') or (barcode_info[i][4]!='Microglia' and barcode_info[j][4]=='Microglia') :
        csv_record_final_temp.append(csv_record_final[record_idx])
        X_attention_filename = args.embedding_data_path + args.data_name + '/' + 'synthetic_data_ccc_uniform_th2p5_lrc105_cell5000_relu_3d_'+filename[run]+'_attention_l1.npy' #_th2p5_ #knn
   ...:         #X_attention_filename = args.embedding_data_path + args.data_name + '/' + 'synthetic_data_ccc_equidistant_th2p5_lrc105_cell3000_tanh_3d_'+filename[run]+'_attention_l1.npy' #split_ #dropou
   ...: t_
   ...:         X_attention_bundle = np.load(X_attention_filename, allow_pickle=True)
csv_record_final_temp.append(csv_record_final[len(csv_record_final)-1])
csv_record_final = copy.deepcopy(csv_record_final_temp)


## plotting only between different z axis ##
#region_of_interest = [...] 
csv_record_final_temp = []
csv_record_final_temp.append(csv_record_final[0])
for record_idx in range (1, len(csv_record_final)-1): #last entry is a dummy for histograms, so ignore it.
    # if they are from different z value, then plot them
    i = csv_record_final[record_idx][6]
    j = csv_record_final[record_idx][7]
    z_i = coordinates[i][2]
    z_j = coordinates[j][2]
    if z_i != z_j:
        csv_record_final_temp.append(csv_record_final[record_idx])
        
csv_record_final_temp.append(csv_record_final[len(csv_record_final)-1])
csv_record_final = copy.deepcopy(csv_record_final_temp)
'''

######################## connected component finding #################################

connecting_edges = np.zeros((len(barcode_info),len(barcode_info)))  
for k in range (1, len(csv_record_final)-1): # last record is a dummy for histogram preparation
    i = csv_record_final[k][6]
    j = csv_record_final[k][7]
    connecting_edges[i][j]=1
        
graph = csr_matrix(connecting_edges)
n_components, labels = connected_components(csgraph=graph,directed=True, connection = 'weak',  return_labels=True) # It assigns each SPOT to a component based on what pair it belongs to
print('number of components %d'%n_components) 

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
    if count_points_component[labels[i]] > 1:
        barcode_info[i][3] = index_dict[labels[i]] #2
    elif connecting_edges[i][i] == 1 and len(lig_rec_dict[i][i])>0: 
        barcode_info[i][3] = 1
    else: 
        barcode_info[i][3] = 0

# update the label based on found component numbers
#max opacity
for record in range (1, len(csv_record_final)-1):
    i = csv_record_final[record][6]
    label = barcode_info[i][3]
    csv_record_final[record][5] = label


component_list = dict()
for record_idx in range (1, len(csv_record_final)-1): #last entry is a dummy for histograms, so ignore it.
    record = csv_record_final[record_idx]
    i = record[6]
    j = record[7]
    component_label = record[5]
    barcode_info[i][3] = component_label 
    barcode_info[j][3] = component_label
    component_list[component_label] = ''

component_list[0] = ''
unique_component_count = len(component_list.keys())

unique_component_count = max(unique_component_count, id_label) 

##################################### Altair Plot ##################################################################
## dictionary of those spots who are participating in CCC ##
active_spot = defaultdict(list)
for record_idx in range (1, len(csv_record_final)-1): #last entry is a dummy for histograms, so ignore it.
    record = csv_record_final[record_idx]
    i = record[6]
    pathology_label = barcode_type[barcode_info[i][0]]
    component_label = record[5]
    X = barcode_info[i][1]
    Y = barcode_info[i][2]
    opacity = np.float(record[4])
    active_spot[i].append([pathology_label, component_label, X, Y, opacity])
    
    j = record[7]
    pathology_label = barcode_type[barcode_info[j][0]]
    component_label = record[5]
    X = barcode_info[j][1]
    Y = barcode_info[j][2]
    opacity = np.float(record[4])   
    active_spot[j].append([pathology_label, component_label, X, Y, opacity])
    ''''''
    
######### color the spots in the plot with opacity = attention score #################
opacity_list = []
for i in active_spot:
    sum_opacity = []
    for edges in active_spot[i]:
        sum_opacity.append(edges[4])
        
    avg_opacity = np.max(sum_opacity) #np.mean(sum_opacity)
    opacity_list.append(avg_opacity) 
    active_spot[i]=[active_spot[i][0][0], active_spot[i][0][1], active_spot[i][0][2], active_spot[i][0][3], avg_opacity]

min_opacity = np.min(opacity_list)
max_opacity = np.max(opacity_list)
#min_opacity = min_opacity - 5

#### making dictionary for converting to pandas dataframe to draw altair plot ###########
data_list=dict()
data_list['pathology_label']=[]
data_list['component_label']=[]
data_list['X']=[]
data_list['Y']=[]   
data_list['opacity']=[]  

for i in range (0, len(barcode_info)):        
    if i in active_spot:
        data_list['pathology_label'].append(active_spot[i][0])
        data_list['component_label'].append(active_spot[i][1])
        data_list['X'].append(active_spot[i][2])
        data_list['Y'].append(active_spot[i][3])
        data_list['opacity'].append((active_spot[i][4]-min_opacity)/(max_opacity-min_opacity))
        
    else:
        data_list['pathology_label'].append(barcode_type[barcode_info[i][0]])
        data_list['component_label'].append(0) # make it zero so it is black
        data_list['X'].append(barcode_info[i][1])
        data_list['Y'].append(barcode_info[i][2])
        data_list['opacity'].append(0.1)



# converting to pandas dataframe

data_list_pd = pd.DataFrame(data_list)
id_label = len(list(set(data_list['component_label'])))#unique_component_count
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

chart.save(current_directory +'altair_plot_test.html')
###################################  Histogram plotting #################################################################################

df = pd.DataFrame(csv_record_final)
df.to_csv(current_directory+'temp_csv.csv', index=False, header=False)
df = readCsv(current_directory+'temp_csv.csv')
os.remove(current_directory+'temp_csv.csv') # delete the intermediate file
df = preprocessDf(df)
p = plot(df)
outPath = current_directory+'histogram_test.html'
p.save(outPath)	

######################### 3D plotting #####################################################################################################
for i in range (0, coordinates.shape[0]):
    for d in range (0, coordinates.shape[1]):
        coordinates[i][d] = coordinates[i][d]*10

from mpl_toolkits.mplot3d import Axes3D
def _format_axes(ax):
    """Visualization options for the 3D axes."""
    # Turn gridlines off
    ax.grid(False)
    # Suppress tick labels
    for dim in (ax.xaxis, ax.yaxis, ax.zaxis):
        dim.set_ticks([])
    # Set axes labels
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("z")
	
###############

plt.clf()
node_xyz = np.array(coordinates)
edge_xyz = []
for k in range (1, len(csv_record_final)-1): # last record is a dummy for histogram preparation
    i = csv_record_final[k][6]
    j = csv_record_final[k][7]
    coordinate_i = coordinates[i]
    coordinate_j = coordinates[j]
    edge_i_j = (coordinate_i, coordinate_j)
    edge_xyz.append(edge_i_j)

edge_xyz = np.array(edge_xyz)
################
fig = plt.figure()
ax = fig.add_subplot(projection="3d")
# Plot the nodes - alpha is scaled by "depth" automatically
ax.scatter(*node_xyz.T, s=.1, alpha=0.5, c="#4F4F4F")
# Plot the edges
for vizedge in edge_xyz:
    ax.plot(*vizedge.T, color="tab:green", linewidth=.6 )

_format_axes(ax)
#fig.tight_layout()
save_path = '/cluster/home/t116508uhn/64630/'
plt.savefig(save_path+'3d_merfish.svg', dpi=400) #
plt.clf()



############################  Network Plot ##############################################################################################
import altairThemes # assuming you have altairThemes.py at your current directoy or your system knows the path of this altairThemes.py.
set1 = altairThemes.get_colour_scheme("Set1", unique_component_count)
colors = set1
colors[0] = '#000000'
ids = []
x_index=[]
y_index=[]
colors_point = []
for i in range (0, len(barcode_info)):    
    ids.append(i)
    x_index.append(barcode_info[i][1]*3)
    y_index.append(-barcode_info[i][2]*3) 
	z_index.append(barcode_info[i][3]*3) 
    colors_point.append(colors[barcode_info[i][3]]) 
  
max_x = np.max(x_index)
max_y = np.max(y_index)

from pyvis.network import Network
import networkx as nx

g = nx.MultiDiGraph(directed=True) #nx.Graph()
for i in range (0, len(barcode_info)):
    marker_size = 'circle'
    label_str =  str(i)+'_c:'+str(barcode_info[i][3]) # barcode_info[i][4] #
    g.add_node(int(ids[i]), x=int(x_index[i]), y=int(y_index[i]), z=int(z_index[i]) label = label_str, pos = str(x_index[i])+","+str(y_index[i])+" !", physics=False, shape = marker_size, color=matplotlib.colors.rgb2hex(colors_point[i]))    

nt = Network( directed=True, height='1000px', width='100%') #"500px", "500px",, filter_menu=True

count_edges = 0
for k in range (1, len(csv_record_final)-1):
    i = csv_record_final[k][6]
    j = csv_record_final[k][7]    
    ligand = csv_record_final[k][2]
    receptor = csv_record_final[k][3]
    title_str =  "L:"+ligand+", R:"+receptor
    edge_score = csv_record_final[k][4]
    g.add_edge(int(i), int(j), label = title_str, color=colors_point[i] ) #, value=np.float64(edge_score) 
    count_edges = count_edges + 1
print('edges: %d'%count_edges)
nt.from_nx(g)
nt.show('mygraph.html')
cp mygraph.html /cluster/home/t116508uhn/64630/mygraph.html

from networkx.drawing.nx_agraph import write_dot
write_dot(g, "/cluster/home/t116508uhn/64630/test_interactive.dot")
            

###########################################################################################################################################################################################
#########################################################################################################################################################################################

import altairThemes # assuming you have altairThemes.py at your current directoy or your system knows the path of this altairThemes.py.
set1 = altairThemes.get_colour_scheme("Set1", unique_component_count)
colors = set1
colors[0] = '#000000'
ids = []
x_index=[]
y_index=[]
colors_point = []
for i in range (0, len(barcode_info)):    
    ids.append(i)
    x_index.append(barcode_info[i][1]*3)
    y_index.append(barcode_info[i][2]*3)    
    colors_point.append(colors[barcode_info[i][3]]) 
  
max_x = np.max(x_index)
max_y = np.max(y_index)

from pyvis.network import Network
import networkx as nx

g = nx.MultiDiGraph(directed=True) #nx.Graph()
for i in range (0, len(barcode_info)):
    marker_size = 'circle'
    label_str = barcode_info[i][4] # str(i)+'_c:'+str(barcode_info[i][3]) # 
    g.add_node(int(ids[i]), x=int(x_index[i]), y=int(y_index[i]), label = label_str, pos = str(x_index[i])+","+str(y_index[i])+" !", physics=False, shape = marker_size, color=matplotlib.colors.rgb2hex(colors_point[i]))    

nt = Network( directed=True, height='1000px', width='100%') #"500px", "500px",, filter_menu=True

count_edges = 0
for k in range (1, len(csv_record_final)-1):
    i = csv_record_final[k][6]
    j = csv_record_final[k][7]    
    ligand = csv_record_final[k][2]
    receptor = csv_record_final[k][3]
    title_str =  "L:"+ligand+", R:"+receptor
    edge_score = csv_record_final[k][4]
    g.add_edge(int(i), int(j), label = title_str, color=colors_point[i] ) #, value=np.float64(edge_score) 
    count_edges = count_edges + 1
print('edges: %d'%count_edges)
nt.from_nx(g)
nt.show('mygraph.html')
cp mygraph.html /cluster/home/t116508uhn/64630/mygraph.html

from networkx.drawing.nx_agraph import write_dot
write_dot(g, "/cluster/home/t116508uhn/64630/test_interactive.dot")
         
  

